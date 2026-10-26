import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import pytest

from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.lfs import LfsEvidenceProvider
from core.compaction.policy import CompactionPolicy
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore


EPISODES = "## Matching (turn 1)\nPreserve tabs.\n\n## Tests (turn 1)\nWhitespace regression passed."


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=next(self.responses))


def event(turn, content):
    return SessionEvent(type=EventType.USER, turn=turn, payload={"role": "user", "content": content})


def engine(manager, responses):
    model = Model(responses)
    return Compactor(
        policy=CompactionPolicy(keep_last_turns=1, compactor_model=model, critic_model=model),
        evidence_provider=LfsEvidenceProvider(manager), session_manager=manager,
    ), model


def document(path):
    text = path.read_text(encoding="utf-8")
    header, body = text.removeprefix("---\n").split("\n---\n\n", 1)
    date = header.removeprefix('date: "').removesuffix('"')
    datetime.strptime(date, "%Y-%m-%d-%H-%M-%S")
    assert header == f'date: "{date}"'
    return date, body.removesuffix("\n")


def test_lfs_archives_final_revision_and_exact_extraction_with_matching_metadata(tmp_path):
    manager = SessionManager(session_id="example", root=tmp_path)
    compactor, model = engine(manager, [EPISODES, "Draft", "Preserve tests", "Final arc", "APPROVE"])
    source = [event(1, "Keep tabs"), event(2, "Retained tail")]
    result = compactor.compact(source)
    root = tmp_path / "memories/example"
    arc, = (root / "arcs").glob("*.md")
    trajectory, = (root / "trajectories").glob("*.md")
    UUID(arc.stem)
    assert arc.name == trajectory.name
    arc_date, arc_body = document(arc)
    trajectory_date, trajectory_body = document(trajectory)
    assert arc_date == trajectory_date
    assert arc_body == result.memory_document
    assert arc_body.startswith("Final arc\n")
    assert trajectory_body == result.segmentation == EPISODES
    extraction_request = json.loads(model.calls[0][1].content)
    assert "Retained tail" not in json.dumps(extraction_request)
    arc_request = json.loads(model.calls[1][1].content)
    assert [item["reference"] for item in arc_request["task_episodes"]] == [str(trajectory)] * 2
    assert str(trajectory) in arc_body
    assert result.events[1:] == source[1:]
    assert len(model.calls) == 5


def test_repeated_compactions_append_pairs_and_preserve_previous_references(tmp_path):
    manager = SessionManager(root=tmp_path)
    compactor, model = engine(manager, [EPISODES, "Arc one", "APPROVE", "New episode", "Arc two", "APPROVE"])
    first = compactor.compact([event(1, "First task"), event(2, "Second task")])
    root = compactor.evidence_provider.root
    original = {path: path.read_bytes() for path in root.rglob("*.md")}
    second = compactor.compact([*first.events, event(3, "Current task")])
    assert len(list(root.glob("arcs/*.md"))) == 2
    assert len(list(root.glob("trajectories/*.md"))) == 2
    assert all(path.read_bytes() == content for path, content in original.items())
    references = second.events[0].payload["evidence_references"]
    assert len(references) == 2
    assert all(reference in second.memory_document for reference in references)
    assert all(reference in model.calls[4][1].content for reference in references)
    assert sorted(document(path)[1] for path in root.glob("trajectories/*.md")) == sorted([EPISODES, "New episode"])


@pytest.mark.parametrize("responses", [[""], [EPISODES, ""]])
def test_model_failure_does_not_create_archives(tmp_path, responses):
    compactor, _ = engine(SessionManager(root=tmp_path), responses)
    with pytest.raises(ValueError, match="empty"):
        compactor.compact([event(1, "Task"), event(2, "Tail")])
    assert not compactor.evidence_provider.root.exists()


def test_noop_does_not_create_archives_or_call_model(tmp_path):
    compactor, model = engine(SessionManager(root=tmp_path), [])
    source = [event(1, "Retained")]
    assert compactor.compact(source).events == source
    assert not compactor.evidence_provider.root.exists()
    assert not model.calls


def test_write_failure_removes_partial_pair_and_keeps_session(tmp_path):
    manager = SessionManager(root=tmp_path)
    source = [event(1, "Task"), event(2, "Tail")]
    manager.append(source)
    original = manager.curated_path.read_bytes()
    compactor, _ = engine(manager, [EPISODES, "Arc", "APPROVE"])
    notifications = []
    coordinator = CompactionCoordinator(
        manager, compactor, repo_root=tmp_path,
        telemetry_store=TelemetryStore(tmp_path / "telemetry.jsonl"),
        on_compaction_event=lambda phase, payload: notifications.append((phase, payload)),
    )
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path.parent.name == "trajectories" and args == ("x",):
            raise OSError("Disk full")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", failing_open):
        coordinator._run_compaction(source, 100, 2, {"trigger": "manual", "counts": {}})
    assert notifications[-1][0] == "error"
    assert notifications[-1][1]["error"] == "Disk full"
    assert manager.curated_path.read_bytes() == original
    assert not list(compactor.evidence_provider.root.rglob("*.md"))


def test_existing_files_are_never_overwritten(tmp_path):
    manager = SessionManager(root=tmp_path)
    compactor, _ = engine(manager, [EPISODES, "Arc", "APPROVE"])
    existing = compactor.evidence_provider.root / "trajectories/fixed-id.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("Previous archive", encoding="utf-8")
    with patch("core.compaction.lfs.uuid4", return_value="fixed-id"):
        with pytest.raises(FileExistsError):
            compactor.compact([event(1, "Task"), event(2, "Tail")])
    assert existing.read_text() == "Previous archive"
    assert list(compactor.evidence_provider.root.rglob("*.md")) == [existing]


def test_default_root_is_under_home_sessions(tmp_path):
    with patch("core.session.io.Path.home", return_value=tmp_path):
        provider = LfsEvidenceProvider(SessionManager(session_id="example"))
    assert provider.root == tmp_path / ".quasipilot/sessions/memories/example"


@pytest.mark.parametrize("trigger", ["manual", "policy"])
def test_coordinator_publishes_archived_arc_without_changing_raw_session(tmp_path, trigger):
    manager = SessionManager(root=tmp_path)
    source = [event(1, "Completed task"), event(2, "Keep this turn")]
    manager.append(source)
    original_dump = manager.dump_path.read_bytes()
    compactor, _ = engine(manager, [EPISODES, "Reviewed arc", "APPROVE"])
    notifications = []
    coordinator = CompactionCoordinator(
        manager, compactor, repo_root=tmp_path,
        telemetry_store=TelemetryStore(tmp_path / "telemetry.jsonl"),
        on_compaction_event=lambda phase, payload: notifications.append((phase, payload)),
    )
    coordinator._run_compaction(source, 100, 2, {"trigger": trigger, "counts": {}})
    assert [phase for phase, _ in notifications] == ["start", "end"]
    restored, retained = manager.read_curated()
    assert retained == source[-1]
    assert restored.payload["session_arc"] == "Reviewed arc"
    trajectory = Path(restored.payload["evidence_references"][0])
    arc = trajectory.parent.parent / "arcs" / trajectory.name
    assert document(arc)[1] in restored.payload["content"]
    assert document(trajectory)[1] == EPISODES
    assert manager.dump_path.read_bytes() == original_dump

import json
from types import SimpleNamespace

import pytest

from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.evidence import PreparedEvidence
from core.compaction.lfs import LfsCompactionOffloadingStrategy
from core.compaction.offloading import BaseCompactionOffloadingStrategy, CompactionOffloadingStack
from core.compaction.policy import CompactionPolicy
from core.compaction.prompts import ARC_PROMPT, ARC_REVIEW_PROMPT, TASK_EPISODES_PROMPT
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore


class RecordingStrategy(BaseCompactionOffloadingStrategy):
    def __init__(self, name, timeline, *, fail=False):
        self.name = name
        self.timeline = timeline
        self.fail = fail
        self.results = []

    def validate(self, manifest):
        self.timeline.append(f"{self.name}:validate")

    def prepare(self, events, turns, episodes, manifest):
        self.timeline.append(f"{self.name}:prepare")

        def persist(result):
            self.timeline.append(f"{self.name}:persist")
            if self.fail:
                raise OSError("Destination unavailable")
            self.results.append(result)

        return PreparedEvidence(
            manifest={"batch": [self.name]},
            episode_references=[f"{self.name}://episode/{index}" for index in range(len(episodes))],
            references=[f"{self.name}://batch", "shared://batch"],
            persist=persist,
        )


class Model:
    def __init__(self, timeline):
        self.timeline = timeline
        self.calls = []

    def invoke(self, messages):
        system, user = [message.content for message in messages]
        self.calls.append((system, user))
        self.timeline.append(system)
        return SimpleNamespace(content={
            TASK_EPISODES_PROMPT: "## One\nFinding.\n## Two\nNext finding.",
            ARC_PROMPT: "Reviewed arc",
            ARC_REVIEW_PROMPT: "APPROVE",
        }[system])


def events():
    return [SessionEvent(type=EventType.USER, turn=turn, payload={"content": text})
            for turn, text in [(1, "Completed task"), (2, "Retained tail")]]


def engine(offloading, timeline, manager=None):
    model = Model(timeline)
    return Compactor(
        policy=CompactionPolicy(keep_last_turns=1, compactor_model=model, critic_model=model),
        offloading=offloading, session_manager=manager,
    ), model


def test_base_strategy_requires_prepare_but_validation_is_optional():
    with pytest.raises(TypeError, match="abstract"):
        BaseCompactionOffloadingStrategy()

    class MinimalStrategy(BaseCompactionOffloadingStrategy):
        def prepare(self, events, turns, episodes, manifest):
            return PreparedEvidence(manifest, [""] * len(episodes), lambda result: None)

    assert MinimalStrategy().validate({}) is None


def test_stack_combines_references_and_persists_same_reviewed_result_in_order():
    timeline = []
    first, second = [RecordingStrategy(name, timeline) for name in ("first", "second")]
    compactor, model = engine([first, second], timeline)
    result = compactor.compact(events())
    assert timeline == [
        "first:validate", "second:validate", TASK_EPISODES_PROMPT,
        "first:prepare", "second:prepare", ARC_PROMPT, ARC_REVIEW_PROMPT,
        "first:persist", "second:persist",
    ]
    assert first.results[0] is second.results[0] is result
    assert len(model.calls) == 3
    assert result.events[0].payload["trajectory_manifest"] == {"batch": ["first", "second"]}
    assert result.events[0].payload["evidence_references"] == ["first://batch", "shared://batch", "second://batch"]
    request = json.loads(model.calls[1][1])
    assert [episode["reference"] for episode in request["task_episodes"]] == [
        "first://episode/0\nsecond://episode/0", "first://episode/1\nsecond://episode/1",
    ]


def test_single_strategy_and_singleton_stack_have_identical_requests_and_documents():
    single, first_model = engine(RecordingStrategy("only", []), [])
    stacked, second_model = engine([RecordingStrategy("only", [])], [])
    source = events()
    first, second = single.compact(source), stacked.compact(source)
    assert first_model.calls == second_model.calls
    assert first.memory_document == second.memory_document
    assert first.segmentation == second.segmentation
    assert first.events[0].payload == second.events[0].payload


def test_empty_stack_is_equivalent_to_no_offloading():
    empty, first_model = engine([], [])
    disabled, second_model = engine(None, [])
    source = events()
    first, second = empty.compact(source), disabled.compact(source)
    assert first_model.calls == second_model.calls
    assert first.memory_document == second.memory_document
    assert first.events[0].payload == second.events[0].payload


def test_stack_merges_without_losing_existing_ids_or_duplicating_references():
    strategy = RecordingStrategy("same", [])
    manifest = {"old": ["old-record"], "batch": ["same", "existing-record"]}
    prepared = CompactionOffloadingStack([strategy, strategy]).prepare([], [1], ["Task"], manifest)
    assert prepared.manifest == manifest
    assert prepared.manifest is not manifest
    assert prepared.episode_references == ["same://episode/0"]
    assert prepared.references == ["same://batch", "shared://batch"]


def test_stack_rejects_invalid_strategy_reference_count_before_synthesis():
    class InvalidStrategy(BaseCompactionOffloadingStrategy):
        def prepare(self, events, turns, episodes, manifest):
            return PreparedEvidence(manifest, [], lambda result: None)

    compactor, model = engine([InvalidStrategy()], [])
    with pytest.raises(ValueError, match="one reference per episode"):
        compactor.compact(events())
    assert len(model.calls) == 1


def test_all_validation_precedes_model_calls():
    class MissingEvidence(RecordingStrategy):
        def validate(self, manifest):
            raise ValueError("Missing evidence")

    compactor, model = engine([RecordingStrategy("first", []), MissingEvidence("second", [])], [])
    with pytest.raises(ValueError, match="Missing evidence"):
        compactor.compact(events())
    assert model.calls == []


@pytest.mark.parametrize("fail", [False, True])
def test_lfs_and_second_destination_stack_publication(tmp_path, fail):
    manager = SessionManager(root=tmp_path)
    source = events()
    manager.append(source)
    original = manager.curated_path.read_bytes()
    timeline = []
    lfs = LfsCompactionOffloadingStrategy(manager)
    second = RecordingStrategy("second", timeline, fail=fail)
    third = RecordingStrategy("third", timeline)
    compactor, _ = engine([lfs, second, third], timeline, manager)
    notifications = []
    coordinator = CompactionCoordinator(
        manager, compactor, repo_root=tmp_path,
        telemetry_store=TelemetryStore(tmp_path / "telemetry.jsonl"),
        on_compaction_event=lambda phase, payload: notifications.append((phase, payload)),
    )
    coordinator._run_compaction(source, 100, 2, {"trigger": "manual", "counts": {}})
    arc, = (lfs.root / "arcs").glob("*.md")
    assert (lfs.root / "trajectories" / arc.name).exists()
    if fail:
        assert notifications[-1][0] == "error"
        assert "Destination unavailable" in notifications[-1][1]["error"]
        assert manager.curated_path.read_bytes() == original
        assert not second.results
        assert not third.results
        assert "third:persist" not in timeline
    else:
        assert notifications[-1][0] == "end"
        result, = second.results
        assert third.results[0] is result
        assert result.memory_document in arc.read_text()
        assert manager.read_curated()[0].payload == result.events[0].payload
        assert "second://batch" in result.memory_document

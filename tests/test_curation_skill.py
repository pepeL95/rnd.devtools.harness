import importlib.util
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.middleware.skills import SkillsMiddleware, create_read_skill_tool, discover_skills
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.trajectory_bank.evidence import content_hash, fingerprint
from core.trajectory_bank.models import TaskArc, TrajectoryRecord
from core.trajectory_bank.offloading import ChromaCompactionOffloadingStrategy
from core.trajectory_bank.store import TrajectoryBank

SKILL = Path(__file__).resolve().parents[1] / "skills/trajectory-bank-curation"
spec = importlib.util.spec_from_file_location("bank_guard", SKILL / "scripts/bank_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class EmbeddingModel(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def archive(tmp_path):
    manager = SessionManager(session_id="session-1", root=tmp_path / "sessions")
    bank = TrajectoryBank(tmp_path / "workspace", embedder=EmbeddingModel(), embedding_identity="test-original")
    records = []
    for turn in (1, 2):
        event = SessionEvent(type=EventType.USER, turn=turn, payload={"content": f"tabs turn {turn}"})
        manager.append([event])
        document = f"User:\nFix tabs {turn}\n\nTrajectory:\nI preserved tabs.\n\nAssistant:\nVerified."
        records.append(TrajectoryRecord(f"record-{turn}", document, {
            "schema_version": 1, "session_id": manager.session_id, "arc_id": "arc-tabs", "turn": turn,
            "source_path": str(manager.dump_path), "source_event_ids": json.dumps([event.id]),
            "source_fingerprint": fingerprint([event]), "content_hash": content_hash(document),
            "started_at": event.timestamp, "keywords": '["tabs"]',
        }))
    arc = TaskArc(arc_id="arc-tabs", title="Tabs", status="completed", intent="Fix tabs",
                  implementation="Preserve tabs", findings="Exact matches", constraints="No fuzzy matching",
                  validation="Tests", remaining="None", turns=[1])
    manager.append([SessionEvent(type=EventType.USER, turn=0, payload={
        "kind": "memory_restore", "task_arcs": [arc.model_dump()],
        "trajectory_manifest": {"arc-tabs": [records[0].id]}, "content": "Evidence",
    })], curated=True)
    bank.archive(records)
    return bank, manager, records


def test_audit_reports_arc_publication_without_embedding_calls(archive):
    bank, manager, _ = archive
    with patch.object(bank.embedder, "embed_documents", side_effect=AssertionError("no embeddings")):
        report = guard.inspect_bank(bank.root, manager.root)
    assert report["issues"] == []
    assert [row["published"] for row in report["records"]] == [True, False]
    assert report["arcs"][manager.session_id]["arc-tabs"]["title"] == "Tabs"


def test_audit_accepts_prose_session_with_batch_references(tmp_path):
    from core.compaction.compactor import Compactor
    from core.compaction.policy import CompactionPolicy

    manager = SessionManager(session_id="prose", root=tmp_path / "sessions")
    manager.append([
        SessionEvent(type=EventType.USER, turn=1, payload={"content": "Keep tabs"}),
        SessionEvent(type=EventType.REASONING, turn=1, payload={"content": "I confirmed exact matching preserves tabs."}),
        SessionEvent(type=EventType.USER, turn=2, payload={"content": "Continue"}),
        SessionEvent(type=EventType.USER, turn=3, payload={"content": "Tail"}),
    ])
    bank = TrajectoryBank(tmp_path / "workspace", embedder=EmbeddingModel(), embedding_identity="test-original")
    engine = Compactor(policy=CompactionPolicy(keep_last_turns=1, max_critic_loops=0,
        compactor_model=FakeMessagesListChatModel(responses=[AIMessage(content="Preserve exact tabs (turn 1).") ])),
        offloading=ChromaCompactionOffloadingStrategy(bank, manager), session_manager=manager)
    result = engine.compact(manager.read_curated())
    manager.apply_compaction_result(result.events, snapshot_latest_turn=3)
    report = guard.inspect_bank(bank.root, manager.root)
    assert not report["issues"]
    assert report["records"][0]["published"]
    assert "task_arcs" not in result.events[0].payload
    assert json.loads(bank.get()[0].metadata["source_turns"]) == [1, 2]


def test_audit_never_creates_missing_bank_or_session(archive, tmp_path):
    bank, _, _ = archive
    with pytest.raises(ValueError, match="No existing"):
        guard.inspect_bank(tmp_path / "missing", tmp_path / "absent-sessions")
    with pytest.raises(ValueError, match="Missing session"):
        guard.inspect_bank(bank.root, tmp_path / "absent-sessions")
    assert not (tmp_path / "missing").exists()
    assert not (tmp_path / "absent-sessions").exists()


@pytest.mark.parametrize("damage", ["document", "source", "arc", "missing", "pending"])
def test_audit_detects_integrity_failures(archive, damage):
    bank, manager, records = archive
    collection = guard.open_bank(bank.root)
    if damage == "document":
        collection.update(ids=[records[0].id], documents=["changed"], embeddings=[[1.0, 0.0]])
    elif damage == "source":
        events = manager.read_dump()
        events[0].payload["content"] = "changed"
        from core.session.io import replace_events
        replace_events(manager.dump_path, events)
    elif damage == "arc":
        collection.update(ids=[records[0].id], metadatas=[{"arc_id": "wrong-arc"}])
    elif damage == "missing":
        collection.delete(ids=[records[0].id])
    else:
        pending = manager.root / "pending_trajectories" / f"{manager.session_id}.json"
        pending.parent.mkdir()
        pending.write_text("{}")
    assert guard.inspect_bank(bank.root, manager.root)["issues"]


def test_prune_backs_up_and_only_deletes_reviewed_unpublished_record(archive, tmp_path):
    bank, manager, records = archive
    original_dump, original_curated = manager.dump_path.read_bytes(), manager.curated_path.read_bytes()
    report = guard.inspect_bank(bank.root, manager.root)
    destination = tmp_path / "backup"
    result = guard.prune_unreferenced(bank.root, manager.root, ids=[records[1].id],
                                      expected_digest=report["digest"], backup_dir=destination, writers_stopped=True)
    assert [row["id"] for row in result["records"]] == [records[0].id]
    assert not result["issues"]
    assert set(guard._records(guard.open_bank(destination / "bank"))) == {item.id for item in records}
    assert (destination / "sessions/dump/session-1.jsonl").read_bytes() == original_dump
    assert manager.dump_path.read_bytes() == original_dump
    assert manager.curated_path.read_bytes() == original_curated


@pytest.mark.parametrize("failure", ["published", "stale", "online", "pending", "output"])
def test_prune_refuses_unsafe_operations(archive, tmp_path, failure):
    bank, manager, records = archive
    report = guard.inspect_bank(bank.root, manager.root)
    if failure == "stale":
        manager.append([SessionEvent(type=EventType.USER, turn=3, payload={"content": "new direction"})])
    if failure == "pending":
        pending = manager.root / "pending_trajectories/session-1.json"
        pending.parent.mkdir()
        pending.write_text("{}")
    with pytest.raises(ValueError):
        guard.prune_unreferenced(bank.root, manager.root,
                                  ids=[records[0 if failure == "published" else 1].id],
                                  expected_digest=report["digest"], writers_stopped=failure != "online",
                                  backup_dir=bank.root if failure == "output" else tmp_path / "backup")
    assert bank.get() == records


def test_reindex_preserves_records_and_supports_changed_identity(archive, tmp_path):
    bank, manager, records = archive
    before = guard.inspect_bank(bank.root, manager.root)
    with patch("core.trajectory_bank.defaults.get_default_embedding_model", return_value=EmbeddingModel()), \
         patch("core.trajectory_bank.defaults.DEFAULT_EMBEDDING_IDENTITY", "test-replacement"):
        output = guard.reindex(bank.root, manager.root, tmp_path / "candidate", writers_stopped=True)
    assert guard.inspect_bank(output, manager.root)["embedding_identity"] == "test-replacement"
    assert guard._records(guard.open_bank(output)) == guard._records(guard.open_bank(bank.root))
    assert guard.inspect_bank(bank.root, manager.root) == before
    reopened = TrajectoryBank(tmp_path / "workspace", embedder=EmbeddingModel(), embedding_identity="test-original")
    assert reopened.get() == records


@pytest.mark.parametrize("failure", ["nan", "count", "query-dimension", "source-change"])
def test_failed_reindex_leaves_original_bank_usable(archive, tmp_path, failure):
    bank, manager, records = archive

    class BrokenModel(EmbeddingModel):
        def embed_documents(self, texts):
            if failure == "nan":
                return [[float("nan"), 0.0] for _ in texts]
            if failure == "count":
                return []
            if failure == "source-change":
                manager.append([SessionEvent(type=EventType.USER, turn=3, payload={"content": "new"})])
            return super().embed_documents(texts)

        def embed_query(self, text):
            return [1.0] if failure == "query-dimension" else super().embed_query(text)

    with pytest.raises(ValueError):
        guard.reindex(bank.root, manager.root, tmp_path / "candidate", writers_stopped=True,
                      embedder=BrokenModel(), embedding_identity="bad-test")
    assert bank.get() == records


def test_global_skill_catalog_and_read_skill_work_from_unrelated_cwd(tmp_path, monkeypatch):
    home = tmp_path / "home"
    installed = home / ".quasipilot/skills" / SKILL.name
    shutil.copytree(SKILL, installed)
    cwd = tmp_path / "unrelated"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    skills = discover_skills(cwd)
    assert next(item for item in skills if item.name == SKILL.name).path == installed / "SKILL.md"
    middleware = SkillsMiddleware(cwd=cwd)
    request = ModelRequest(model=None, messages=[], system_message=SystemMessage(content="Base"))
    captured = middleware.wrap_model_call(request, lambda updated: updated.system_message.text)
    assert SKILL.name in captured
    assert "Apply and Verify" not in captured  # Only metadata enters the system catalog.
    tool = create_read_skill_tool(cwd=cwd)
    loaded = tool.invoke({"name": SKILL.name})
    assert str(installed) in loaded
    assert "scripts/bank_guard.py" in loaded
    assert "references/contract.md" in loaded
    assert "Apply and Verify" in loaded

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "read_skill", "args": {"name": SKILL.name}, "id": "load"}]),
        AIMessage(content="Ready to inspect"),
    ])
    agent = create_agent(model, tools=[tool], middleware=[middleware])
    result = agent.invoke({"messages": [HumanMessage(content="Help curate my trajectory bank")]})
    response = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert response.status == "success" and "scripts/bank_guard.py" in response.content

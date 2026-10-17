import asyncio
import json
from math import sqrt
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from core.compaction.compactor import Compactor
from core.compaction.policy import CompactionPolicy
from core.middleware.reasoning import reasoning_tool
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.trajectory_bank import TrajectoryBankMiddleware
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager, StaleCompactionError
from core.trajectory_bank.evidence import content_hash, fingerprint
from core.trajectory_bank.models import TrajectoryRecord, arc_ref, record_ref
from core.trajectory_bank.policy import BankPolicy, load_policy
from core.trajectory_bank.retrieval import BankRetrieval
from core.trajectory_bank.store import TrajectoryBank


class FakeEmbeddings(Embeddings):

    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [float(term in text.lower()) for term in ("tabs", "compaction", "other")]
            if not any(vector):
                vector[-1] = 1.0
            norm = sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        return vectors


@pytest.fixture
def bank(tmp_path):
    return TrajectoryBank(tmp_path, embedder=FakeEmbeddings(), embedding_identity="test-keyword-v1")


@pytest.fixture
def manager(tmp_path):
    return SessionManager(session_id="test-session", root=tmp_path / "sessions")


def record(manager, text="tabs are significant", *, turn=1, arc="arc-tabs"):
    event = SessionEvent(type=EventType.USER, turn=turn, payload={"role": "user", "content": text})
    manager.append([event])
    item = TrajectoryRecord(f"record-{event.id}", text, {
        "schema_version": 1, "session_id": manager.session_id, "arc_id": arc, "turn": turn,
        "started_at": event.timestamp, "ended_at": event.timestamp,
        "source_event_ids": json.dumps([event.id]), "source_fingerprint": fingerprint([event]),
        "content_hash": content_hash(text), "keywords": '["tabs"]',
    })
    manager.replace_curated([*manager.read_curated(), SessionEvent(type=EventType.USER, turn=0, payload={
        "kind": "memory_restore", "trajectory_manifest": {arc: [item.id]}, "content": "Published evidence",
    })])
    return item


def test_lazy_persistent_archive_and_embedding_identity(bank, manager, tmp_path):
    assert bank.get() == []
    assert not bank.root.exists()
    item = record(manager)
    bank.archive([item])
    calls = bank.embedder.calls
    bank.archive([item])
    assert bank.embedder.calls == calls
    reopened = TrajectoryBank(tmp_path, embedder=FakeEmbeddings(), embedding_identity="test-keyword-v1")
    assert reopened.get(ids=[item.id]) == [item]
    assert reopened.search("tabs", where={"session_id": manager.session_id}, limit=2)[0].similarity > 0.99
    wrong = FakeEmbeddings()
    with pytest.raises(ValueError, match="reindexing"):
        TrajectoryBank(tmp_path, embedder=wrong, embedding_identity="incompatible-model").get()


def test_initialization_is_idempotent_and_preserves_records(bank, manager, tmp_path):
    bank.initialize()
    bank.initialize()
    assert bank.embedder.calls == 0
    item = record(manager)
    bank.archive([item])
    reopened = TrajectoryBank(tmp_path, embedder=FakeEmbeddings(), embedding_identity="test-keyword-v1")
    reopened.initialize()
    assert reopened.embedder.calls == 0
    assert reopened.get(ids=[item.id]) == [item]


def test_threshold_scope_exact_references_and_deleted_sources(bank, manager, tmp_path):
    current = record(manager)
    other_manager = SessionManager(session_id="other-session", root=manager.root)
    other = record(other_manager, "compaction evidence")
    bank.archive([current, other])
    retrieval = BankRetrieval(bank, manager)
    assert retrieval.retrieve(query="compaction", automatic=True)[1] == []
    assert retrieval.retrieve(query="compaction", scope="workspace")[1] == [other.id]
    assert retrieval.retrieve(query="tabs", automatic=True)[1] == [current.id]
    assert retrieval.retrieve(query="tabs", automatic=True, exclude={current.id})[1] == []
    assert retrieval.retrieve(ref=arc_ref(manager.session_id, "arc-tabs"))[1] == [current.id]
    with pytest.raises(ValueError, match="scope"):
        retrieval.retrieve(ref=record_ref(other.id))
    assert retrieval.retrieve(ref=record_ref(other.id), scope="workspace")[1] == [other.id]
    manager.pop_turn()
    record(manager, "tabs now mean something else", turn=1)
    assert retrieval.retrieve(query="tabs", automatic=True)[1] == []
    with pytest.raises(ValueError, match="removed or changed"):
        retrieval.retrieve(ref=record_ref(current.id))
    other_manager.dump_path.unlink()
    assert retrieval.retrieve(query="compaction", scope="workspace")[1] == []


def test_lower_manual_threshold_and_bounded_paginated_output(bank, manager):
    item = record(manager, "tabs compaction " * 1000)
    bank.archive([item])
    retrieval = BankRetrieval(bank, manager, policy=BankPolicy(automatic_min_similarity=0.8, search_min_similarity=0.5,
                                                           search_tokens=300))
    assert retrieval.retrieve(query="tabs", automatic=True)[1] == []
    text, ids = retrieval.retrieve(query="tabs")
    assert ids == [item.id]
    assert "Truncated" in text
    assert retrieval.counter.count_text(text) <= 300
    page, _ = retrieval.retrieve(ref=record_ref(item.id), offset=1000)
    assert "Historical" in page and "Truncated" in page


def test_policy_validates_thresholds(tmp_path):
    root = tmp_path / "bank"
    root.mkdir()
    (root / "config.toml").write_text("automatic_min_similarity = 0.2\nsearch_min_similarity = 0.8\n")
    with pytest.raises(ValueError):
        load_policy(root)


def add_turn(manager, turn, text):
    events = [SessionEvent(type=EventType.USER, turn=turn, payload={"role": "user", "content": text}),
              SessionEvent(type=EventType.REASONING, turn=turn, payload={"role": "assistant", "kind": "trajectory_memory",
                                                                       "content": f"Evidence for {text}: keep exact matching."}),
              SessionEvent(type=EventType.ASSISTANT, turn=turn, payload={"role": "assistant", "content": "Implemented and verified."}),
              SessionEvent(type=EventType.TURN_END, turn=turn, payload={})]
    manager.append(events)
    return events


def arc(turns, *, id="", title="Filesystem"):
    return dict(arc_id=id, title=title, status="completed", intent="Reduce tabs mismatch failures",
                implementation="Updated filesystem.py", findings="Exact matching protects whitespace",
                constraints="No fuzzy replacement", validation="Tests passed", remaining="None known",
                turns=turns, keywords=["tabs", "invented-keyword"])


def compact_model(arcs):
    return FakeMessagesListChatModel(responses=[AIMessage(content=json.dumps({
        "narrative": "Preserved safe edits before improving memory", "current_state": "Continue memory work", "arcs": arcs,
    }))])


def compactor(bank, manager, arcs):
    model = compact_model(arcs)
    return Compactor(policy=CompactionPolicy(keep_last_turns=1, compactor_model=model, max_critic_loops=0),
                     trajectory_bank=bank, session_manager=manager)


def test_compaction_archives_groups_and_preserves_arcs_on_repeated_compaction(bank, manager):
    for turn, text in enumerate(["tabs", "compaction", "tabs", "other pending"], 1):
        add_turn(manager, turn, text)
    original = manager.read_curated()
    result = compactor(bank, manager, [arc([1, 3]), arc([2], title="Memory")]).compact(original)
    payload = result.events[0].payload
    ids = [item["arc_id"] for item in payload["task_arcs"]]
    assert len(bank.get()) == 3
    assert "trajectory://test-session/" in result.memory_document
    assert all("invented-keyword" not in item["keywords"] for item in payload["task_arcs"])
    assert [event.turn for event in result.events[1:]] == [4] * 4
    archived = bank.get()
    assert all("Evidence for" in item.document for item in archived)
    manager.apply_compaction_result(result.events, snapshot_latest_turn=4)
    add_turn(manager, 5, "new tail")
    updated = compactor(bank, manager, [arc([4], id=ids[0])]).compact(manager.read_curated())
    assert [item["arc_id"] for item in updated.events[0].payload["task_arcs"]] == ids
    assert updated.events[0].payload["task_arcs"][1] == payload["task_arcs"][1]
    assert len(bank.get()) == 4
    assert {item.id for item in archived}.issubset({item.id for item in bank.get()})
    assert all(item in bank.get() for item in archived)


@pytest.mark.parametrize("assignments", [[arc([])], [arc([1, 1])], [arc([2])], [arc([1], id="fabricated")]])
def test_compaction_rejects_uncovered_duplicate_tail_and_fabricated_ids(bank, manager, assignments):
    add_turn(manager, 1, "tabs")
    add_turn(manager, 2, "tail")
    original = manager.read_curated()
    with pytest.raises(ValueError):
        compactor(bank, manager, assignments).compact(original)
    assert manager.read_curated() == original
    assert bank.get() == []


def test_failed_archive_and_missing_references_never_replace_context(bank, manager):
    add_turn(manager, 1, "tabs")
    add_turn(manager, 2, "tail")
    before = manager.read_curated()
    with patch.object(bank, "archive", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            compactor(bank, manager, [arc([1])]).compact(before)
    assert manager.read_curated() == before
    result = compactor(bank, manager, [arc([1])]).compact(before)
    manager.apply_compaction_result(result.events, snapshot_latest_turn=2)
    add_turn(manager, 3, "tail")
    with patch.object(bank, "get", return_value=[]):
        with pytest.raises(ValueError, match="references are missing"):
            compactor(bank, manager, [arc([2])]).compact(manager.read_curated())


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def runtime():
    return ToolRuntime(state={}, context=None, config={}, stream_writer=lambda _: None, tool_call_id="test", store=None)


@pytest.mark.parametrize("use_async", [False, True])
def test_post_reasoning_recall_and_mixed_batch_deferral(bank, manager, use_async):
    item = record(manager)
    bank.archive([item])
    middleware = TrajectoryBankMiddleware(BankRetrieval(bank, manager))
    calls = [{"name": "reasoning", "args": {"reasoning": "tabs"}, "id": "r"},
             {"name": "edit_file", "args": {}, "id": "e"}]
    state = {"messages": [AIMessage(content="", tool_calls=calls)]}
    executed = []

    def handler(request):
        executed.append(request.tool_call["name"])
        return ToolMessage(content="checkpoint", name=request.tool_call["name"], tool_call_id=request.tool_call["id"])

    async def ahandler(request):
        return handler(request)

    def invoke(call):
        request = ToolCallRequest(tool_call=call, tool=None, state=state, runtime=runtime())
        return asyncio.run(middleware.awrap_tool_call(request, ahandler)) if use_async else middleware.wrap_tool_call(request, handler)

    recalled = invoke(calls[0])
    assert item.id in recalled.content
    assert recalled.additional_kwargs["trajectory_record_ids"] == [item.id]
    assert invoke(calls[1]).status == "error"
    assert executed == ["reasoning"]
    state["messages"].append(recalled)
    assert "Historical" not in invoke(calls[0]).content
    state["messages"] = []
    assert "Historical" in invoke(calls[0]).content  # Compaction removed the loaded evidence.


def test_real_graph_records_recall_once_and_defers_actions(bank, manager):
    item = record(manager)
    bank.archive([item])
    executed = []

    @tool
    def edit_example() -> str:
        """Make a test edit."""
        executed.append(True)
        return "edited"

    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "reasoning", "args": {"reasoning": "tabs"}, "id": "r"},
                                          {"name": "edit_example", "args": {}, "id": "deferred"}]),
        AIMessage(content="", tool_calls=[{"name": "edit_example", "args": {}, "id": "actual"}]),
        AIMessage(content="done"),
    ])
    graph = create_agent(model, tools=[reasoning_tool, edit_example], middleware=[
        SessionDumpMiddleware(manager), TrajectoryBankMiddleware(BankRetrieval(bank, manager)),
    ])
    inputs = {"messages": [HumanMessage(content="fix tabs")]}
    result = graph.invoke(inputs)
    assert executed == [True]
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.status for message in tool_messages] == ["success", "error", "success"]
    persisted = [event for event in manager.read_dump() if event.payload.get("trajectory_record_ids")]
    assert len(persisted) == 1
    restored = [message for message in manager.load_curated_messages() if message.additional_kwargs.get("trajectory_record_ids")]
    assert len(restored) == 1
    assert restored[0].additional_kwargs["trajectory_record_ids"] == [item.id]


def test_unpublished_compaction_is_not_retrievable(bank, manager):
    add_turn(manager, 1, "tabs")
    add_turn(manager, 2, "tail")
    result = compactor(bank, manager, [arc([1])]).compact(manager.read_curated())
    item = bank.get()[0]
    retrieval = BankRetrieval(bank, manager)
    assert retrieval.retrieve(query="tabs", automatic=True)[1] == []
    with pytest.raises(ValueError, match="not published"):
        retrieval.retrieve(ref=record_ref(item.id))
    manager.apply_compaction_result(result.events, snapshot_latest_turn=2)
    assert retrieval.retrieve(query="tabs", automatic=True)[1] == [item.id]


def test_stale_compaction_cannot_overwrite_new_session_or_same_turn_events(manager):
    source = add_turn(manager, 1, "tabs")
    manager.append([SessionEvent(type=EventType.USER, turn=1, payload={"content": "live correction"})])
    current = manager.read_curated()
    with pytest.raises(StaleCompactionError):
        manager.apply_compaction_result([], snapshot_latest_turn=1, source_event_ids=[event.id for event in source])
    assert manager.read_curated() == current
    manager.sync_system_instruction("latest", kind="agents_md", source="AGENTS.md", turn=1)
    add_turn(manager, 2, "newer turn")
    manager.apply_compaction_result(source, snapshot_latest_turn=1, source_event_ids=[event.id for event in current])
    assert manager.read_curated()[0].payload["content"] == "latest"
    assert any(event.turn == 2 for event in manager.read_curated())


def test_stale_pending_trajectory_is_discarded_after_restart(manager, tmp_path):
    from core.trajectory.compactor import TrajectoryCompactor
    from core.trajectory.coordinator import PendingTrajectoryCompaction, TrajectoryCompactionCoordinator
    from core.telemetry.store import TelemetryStore

    source = add_turn(manager, 1, "tabs")
    telemetry = TelemetryStore(tmp_path / "telemetry.jsonl")
    coordinator = TrajectoryCompactionCoordinator(manager, TrajectoryCompactor(), repo_root=tmp_path, telemetry_store=telemetry)
    pending = PendingTrajectoryCompaction(result_events=source, snapshot_latest_turn=1,
        payload={"counts": {}, "token_usage": {}}, compacted_event_count=1, compacted_turn_count=1,
        result_event_count=len(source), model_names={}, token_usage={}, duration_ms=1,
        source_event_ids=[event.id for event in source])
    coordinator._store_pending(pending)
    manager.replace_curated([SessionEvent(type=EventType.USER, turn=1, payload={"content": "new session arc"})])
    restarted = TrajectoryCompactionCoordinator(manager, TrajectoryCompactor(), repo_root=tmp_path, telemetry_store=telemetry)
    assert restarted.prepare_for_agent() == "stale"
    assert restarted.prepare_for_agent() == "not_ready"
    assert not restarted._pending_path.exists()
    assert manager.read_curated()[0].payload["content"] == "new session arc"


def test_legacy_memory_is_preserved_without_manufactured_references(bank, manager):
    old = SessionEvent(type=EventType.USER, turn=1, payload={"kind": "memory_restore", "content": "Legacy task details"})
    manager.append([old])
    add_turn(manager, 2, "tabs")
    add_turn(manager, 3, "tail")
    result = compactor(bank, manager, [arc([2])]).compact(manager.read_curated())
    assert "Legacy task details" in result.memory_document
    arcs = result.events[0].payload["task_arcs"]
    assert arcs[0]["arc_id"].startswith("legacy-")
    assert arcs[0]["arc_id"] not in result.events[0].payload["trajectory_manifest"]


@pytest.mark.parametrize("use_async", [False, True])
def test_explicit_retrieval_tool_is_available_in_graph(bank, manager, use_async):
    item = record(manager)
    bank.archive([item])
    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "trajectory_bank", "args": {"ref": record_ref(item.id)}, "id": "lookup"}]),
        AIMessage(content="done"),
    ])
    graph = create_agent(model, middleware=[TrajectoryBankMiddleware(BankRetrieval(bank, manager))])
    inputs = {"messages": [HumanMessage(content="recall") ]}
    result = asyncio.run(graph.ainvoke(inputs)) if use_async else graph.invoke(inputs)
    evidence = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert evidence.status == "success"
    assert evidence.additional_kwargs["trajectory_record_ids"] == [item.id]


def test_recall_failure_does_not_abort_reasoning(bank, manager):
    retrieval = BankRetrieval(bank, manager)
    middleware = TrajectoryBankMiddleware(retrieval)
    call = {"name": "reasoning", "args": {"reasoning": "tabs"}, "id": "r"}
    request = ToolCallRequest(tool_call=call, tool=None, state={"messages": []}, runtime=runtime())
    with patch.object(retrieval, "retrieve", side_effect=OSError("unavailable")):
        result = middleware.wrap_tool_call(request, lambda _: ToolMessage(content="checkpoint", tool_call_id="r"))
    assert result.status == "success"
    assert result.content.startswith("checkpoint") and "recall unavailable" in result.content


def test_invented_arc_id_is_repaired_and_schema_constrains_ids(bank, manager):
    from core.compaction.compactor import _invoke_text

    add_turn(manager, 1, "tabs")
    add_turn(manager, 2, "tail")
    engine = compactor(bank, manager, [arc([1])])
    engine.policy.compactor_model.responses = [
        compact_model([arc([1], id="invented")]).responses[0],
        compact_model([arc([1])]).responses[0],
    ]
    with patch("core.compaction.compactor._invoke_text", wraps=_invoke_text) as invoke:
        result = engine.compact(manager.read_curated())
    schema = json.loads(invoke.call_args_list[0].args[2])["schema"]
    assert schema["$defs"]["TaskArc"]["properties"]["arc_id"]["enum"] == [""]
    assert "invented" in invoke.call_args_list[1].args[2]
    assert result.revisions == 1 and result.token_usage["call_count"] == 2
    assert len(bank.get()) == 1


def test_raw_turn_fallback_and_empty_scope_avoid_embedding(bank, manager):
    manager.append([SessionEvent(type=EventType.USER, turn=1, payload={"content": "tabs exact evidence"}),
                    SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"content": "Checked source"})])
    add_turn(manager, 2, "tail")
    compactor(bank, manager, [arc([1])]).compact(manager.read_curated())
    item = bank.get()[0]
    assert item.metadata["source_kind"] == "raw_turn"
    assert "tabs exact evidence" in item.document and "Checked source" in item.document
    calls = bank.embedder.calls
    assert bank.search("tabs", where={"session_id": "empty-session"}, limit=2) == []
    assert bank.embedder.calls == calls


def test_legacy_pending_rewrite_is_discarded(manager, tmp_path):
    from core.trajectory.compactor import TrajectoryCompactor
    from core.trajectory.coordinator import TrajectoryCompactionCoordinator

    pending_path = manager.root / "pending_trajectories" / f"{manager.session_id}.json"
    pending_path.parent.mkdir(parents=True)
    pending_path.write_text(json.dumps({"result_events": [], "snapshot_latest_turn": 1}))
    coordinator = TrajectoryCompactionCoordinator(manager, TrajectoryCompactor(), repo_root=tmp_path)
    assert coordinator.prepare_for_agent() == "not_ready"
    assert not pending_path.exists()


def test_cli_names_recall_without_surfacing_reasoning_evidence():
    from cli.components.streams import ToolStream, _format_output_block

    assert ToolStream._pretty_name(None, "trajectory_bank") == "Recalled"
    assert _format_output_block("reasoning", "Historical trajectory evidence") == ""


@pytest.mark.parametrize("approved", [True, False])
def test_fidelity_review_repairs_or_preserves_original_context(bank, manager, approved):
    add_turn(manager, 1, "Reread AGENTS.md each turn; do not cache stale instructions")
    add_turn(manager, 2, "tail")
    source = manager.read_curated()
    incorrect = arc([1])
    incorrect["findings"] = "Rereading is inefficient"
    corrected = arc([1])
    corrected["findings"] = "Reread AGENTS.md each turn; do not cache stale instructions"
    synthesis = FakeMessagesListChatModel(responses=[compact_model([incorrect]).responses[0], compact_model([corrected]).responses[0]])
    critic = FakeMessagesListChatModel(responses=[AIMessage(content="Remove invented inefficiency; preserve reread-each-turn requirement."),
                                                AIMessage(content="APPROVE" if approved else "Still missing source constraints")])
    engine = Compactor(policy=CompactionPolicy(keep_last_turns=1, compactor_model=synthesis, critic_model=critic,
                                               max_critic_loops=1), trajectory_bank=bank, session_manager=manager)
    if approved:
        result = engine.compact(source)
        assert "Reread AGENTS.md each turn" in result.memory_document
        assert "inefficient" not in result.memory_document
        assert result.revisions == 1 and result.token_usage["call_count"] == 4
        assert len(result.critiques) == 2 and result.critiques[-1].approved
        assert len(bank.get()) == 1
    else:
        with pytest.raises(ValueError, match="fidelity review failed"):
            engine.compact(source)
        assert manager.read_curated() == source
        assert bank.get() == []


def test_local_embeddings_include_text_beyond_first_window(monkeypatch):
    import numpy as np
    from tokenizers import Tokenizer, models, pre_tokenizers
    from core.trajectory_bank.embeddings import LocalEmbeddings

    class FakeONNX:
        def __init__(self, **kwargs):
            self.tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2}, unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            self.tokenizer.enable_truncation(256)

        def __call__(self, chunks):
            return np.array([[max(1, chunk.count("alpha")), chunk.count("beta")] for chunk in chunks], dtype=float)

    monkeypatch.setattr("chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2", FakeONNX)
    vector = LocalEmbeddings().embed_query("alpha " * 500 + "beta " * 100)
    assert vector[1] > 0.1
    assert np.linalg.norm(vector) == pytest.approx(1)


def test_legacy_compaction_keeps_published_evidence_for_reenabling(bank, manager):
    item = record(manager)
    bank.archive([item])
    add_turn(manager, 2, "tail")
    model = FakeMessagesListChatModel(responses=[AIMessage(content="Legacy task memory")])
    legacy = Compactor(policy=CompactionPolicy(keep_last_turns=1, task_extractor_model=model,
                                              compactor_model=model, max_critic_loops=0))
    with patch("core.compaction.compactor.quality_report", return_value=None):
        result = legacy.compact(manager.read_curated())
    assert result.events[0].payload["trajectory_manifest"] == {"arc-tabs": [item.id]}
    manager.apply_compaction_result(result.events, snapshot_latest_turn=2)
    assert BankRetrieval(bank, manager).retrieve(ref=record_ref(item.id))[1] == [item.id]
    add_turn(manager, 3, "new tail")
    resumed = compactor(bank, manager, [arc([2])]).compact(manager.read_curated())
    manager.apply_compaction_result(resumed.events, snapshot_latest_turn=3)
    assert "Legacy task memory" in resumed.memory_document
    assert BankRetrieval(bank, manager).retrieve(ref=record_ref(item.id))[1] == [item.id]


def test_child_directory_archive_initializes_parent_chroma_workspace(tmp_path, manager):
    (tmp_path / ".quasipilot").mkdir()
    child = tmp_path / "src"
    child.mkdir()
    bank = TrajectoryBank(child, embedder=FakeEmbeddings(), embedding_identity="test-keyword-v1")
    item = record(manager)
    bank.archive([item])
    assert (tmp_path / ".quasipilot/.trajectorybank/chroma.sqlite3").exists()
    assert not (child / ".quasipilot").exists()
    assert TrajectoryBank(tmp_path, embedder=FakeEmbeddings(), embedding_identity="test-keyword-v1").get(ids=[item.id]) == [item]


def test_standard_embeddings_use_separate_document_and_query_methods(tmp_path, manager):
    class AsymmetricEmbeddings(Embeddings):
        def __init__(self):
            self.documents = []
            self.queries = []

        def embed_documents(self, texts):
            self.documents.extend(texts)
            return [[1.0, 0.0] if text == "first" else [0.0, 1.0] for text in texts]

        def embed_query(self, text):
            self.queries.append(text)
            return [0.0, 1.0]

    embeddings = AsymmetricEmbeddings()  # No harness-specific attributes required.
    bank = TrajectoryBank(tmp_path, embedder=embeddings, embedding_identity="asymmetric-test-v1")
    bank.initialize()
    assert not embeddings.documents and not embeddings.queries
    first = record(manager, "first")
    second = record(manager, "second", turn=2)
    bank.archive([first, second])
    bank.archive([first, second])
    assert embeddings.documents == ["first", "second"]
    assert bank.search("first", where=None, limit=2)[0].record.id == second.id
    assert embeddings.queries == ["first"]
    assert bank.get(ids=[first.id]) == [first]
    assert embeddings.queries == ["first"]


@pytest.mark.parametrize("arguments", [{"embedder": FakeEmbeddings()}, {"embedding_identity": "id-only"},
    {"embedder": FakeEmbeddings(), "embedding_identity": " "}])
def test_custom_embeddings_require_an_explicit_nonempty_identity(tmp_path, arguments):
    with pytest.raises(ValueError):
        TrajectoryBank(tmp_path, **arguments)
    assert not (tmp_path / ".quasipilot").exists()


def test_embedding_identity_mismatch_fails_before_model_use(bank, manager, tmp_path):
    bank.archive([record(manager)])
    model = FakeEmbeddings()
    replacement = TrajectoryBank(tmp_path, embedder=model, embedding_identity="changed-prompts-v2")
    with pytest.raises(ValueError, match="reindexing"):
        replacement.initialize()
    assert model.calls == 0


def test_local_model_implements_langchain_sync_and_async_interfaces():
    from core.trajectory_bank.embeddings import LocalEmbeddings

    embeddings = LocalEmbeddings()
    assert isinstance(embeddings, Embeddings)
    assert embeddings.embed_documents([]) == []
    assert embeddings._model is None
    with patch.object(embeddings, "embed_documents", return_value=[[0.5, 0.5]]) as documents:
        assert embeddings.embed_query("query") == [0.5, 0.5]
        documents.assert_called_once_with(["query"])
        assert asyncio.run(embeddings.aembed_query("async query")) == [0.5, 0.5]
        assert asyncio.run(embeddings.aembed_documents(["async document"])) == [[0.5, 0.5]]

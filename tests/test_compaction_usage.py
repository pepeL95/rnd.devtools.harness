import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from core.compaction.compactor import Compactor, _invoke_text as session_invoke
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.compaction.token_counter import TokenCounter
from core.compaction.usage import TokenUsage
from core.compaction.telemetry import context_token_breakdown
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore
from core.trajectory.compactor import _invoke_text as trajectory_invoke


@pytest.mark.parametrize("invoke", [session_invoke, trajectory_invoke])
def test_provider_counts_replace_estimates_in_both_compactors(invoke):
    response = AIMessage(content="Short visible text", usage_metadata={
        "input_tokens": 1234, "output_tokens": 500, "total_tokens": 1734,
        "output_token_details": {"reasoning": 490},
        "input_token_details": {"cache_read": 1000},
    })
    counter = Mock(spec=TokenCounter)
    counter.count_text.side_effect = AssertionError("Do not estimate reported usage")
    usage = TokenUsage()
    assert invoke(Mock(invoke=Mock(return_value=response)), "system", "source",
                  token_counter=counter, usage=usage) == "Short visible text"
    assert usage.as_dict() == {
        "input_tokens": 1234, "output_tokens": 500, "total_tokens": 1734,
        "call_count": 1, "reported_calls": 1, "estimated_calls": 0,
    }


@pytest.mark.parametrize("metadata", [
    None, {}, {"input_tokens": 10}, {"input_tokens": -1, "output_tokens": 3, "total_tokens": 2},
    {"input_tokens": True, "output_tokens": 3, "total_tokens": 4},
])
def test_unavailable_or_invalid_usage_is_explicitly_estimated(metadata):
    response = SimpleNamespace(content="answer", usage_metadata=metadata)
    counter = Mock(spec=TokenCounter)
    counter.count_text.side_effect = [20, 4]
    usage = TokenUsage()
    session_invoke(Mock(invoke=Mock(return_value=response)), "system", "source",
                   token_counter=counter, usage=usage)
    assert usage.as_dict() == {
        "input_tokens": 20, "output_tokens": 4, "total_tokens": 24,
        "call_count": 1, "reported_calls": 0, "estimated_calls": 1,
    }


def test_zero_usage_is_reported_not_estimated():
    usage = TokenUsage()
    response = AIMessage(content="", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    session_invoke(Mock(invoke=Mock(return_value=response)), "", "", token_counter=TokenCounter(), usage=usage)
    assert usage.as_dict()["reported_calls"] == 1
    assert usage.as_dict()["total_tokens"] == 0
    assert TokenUsage().as_dict()["call_count"] == 0


@pytest.mark.parametrize("trigger", ["manual", "policy"])
def test_context_delta_reaches_cli_while_model_usage_stays_in_telemetry(tmp_path, trigger):
    model = FakeMessagesListChatModel(responses=[
        AIMessage(content=text, usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
        for text in ("## Task\nPreserve findings.", "First arc", "Missing constraint", "Revised arc", "APPROVE")
    ])
    manager = SessionManager(session_id="usage", root=tmp_path / "sessions")
    events = [SessionEvent(type=EventType.USER, turn=turn, payload={"content": text})
              for turn, text in [(1, "Task"), (2, "Tail")]]
    manager.append(events)
    observed = []
    path = tmp_path / "telemetry.jsonl"
    coordinator = CompactionCoordinator(
        manager, Compactor(policy=CompactionPolicy(keep_last_turns=1, max_critic_loops=1,
                                                   compactor_model=model, critic_model=model)),
        on_compaction_event=lambda phase, payload: observed.append((phase, payload)),
        telemetry_store=TelemetryStore(path), repo_root=tmp_path,
    )
    coordinator._run_compaction(events, 999, 2, {"trigger": trigger, "counts": {}})
    phase, payload = observed[-1]
    assert phase == "end"
    assert payload["token_usage"] == {
        "source_tokens": 999, "input_tokens": 500, "output_tokens": 100, "total_tokens": 600,
        "call_count": 5, "reported_calls": 5, "estimated_calls": 0,
    }
    counter = TokenCounter()
    start = observed[0][1]
    breakdown = payload["context_tokens"]
    injected = manager.read_curated()[0].payload["content"]
    assert start["context_tokens"]["compacted_tokens"] == counter.count_text("Task")
    assert "injected_tokens" not in start["context_tokens"]
    assert breakdown["compacted_tokens"] == counter.count_text("Task")
    assert breakdown["retained_tokens"] == counter.count_text("Tail")
    assert breakdown["injected_tokens"] == counter.count_text(injected)
    assert breakdown["injected_tokens"] > counter.count_text("Revised arc")  # Wrapper is injected too.
    assert breakdown["saved_tokens"] == breakdown["compacted_tokens"] - breakdown["injected_tokens"]
    assert breakdown["before_tokens"] == breakdown["compacted_tokens"] + breakdown["retained_tokens"]
    assert breakdown["after_tokens"] == breakdown["injected_tokens"] + breakdown["retained_tokens"]
    assert "grew by" in payload["content"]  # Tiny source grows; do not claim savings.
    assert "input /" not in payload["content"]
    logged = [json.loads(line) for line in path.read_text().splitlines()][-1]["payload"]
    assert logged["context_tokens"] == breakdown
    assert logged["token_usage"] == payload["token_usage"]

    from cli.run import QuasipilotApp
    from cli.components import StatusBubble
    app = QuasipilotApp()
    mounted = []
    app._mount_chat = mounted.append
    app._sync_compaction_ui = lambda: None
    app.notify = lambda *args, **kwargs: None
    app._handle_compaction_event("start", start)
    app._handle_compaction_event(phase, payload)
    assert len(mounted) == 2 and all(isinstance(item, StatusBubble) for item in mounted)
    assert f"{breakdown['compacted_tokens']:,} estimated tokens to compact" in str(mounted[0].render())
    assert f"-> {breakdown['injected_tokens']:,} estimated tokens" in str(mounted[1].render())


def test_mixed_counts_track_reported_and_estimated_calls():
    usage = TokenUsage()
    counter = Mock(spec=TokenCounter)
    counter.count_text.side_effect = [10, 3]
    for metadata in (None, {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}):
        session_invoke(Mock(invoke=Mock(return_value=SimpleNamespace(content="text", usage_metadata=metadata))),
                       "system", "source", token_counter=counter, usage=usage)
    assert usage.as_dict()["total_tokens"] == 133
    assert usage.as_dict()["reported_calls"] == usage.as_dict()["estimated_calls"] == 1


def test_context_estimate_excludes_duplicate_storage_fields_and_pinned_instructions():
    counter = TokenCounter()
    memory = SessionEvent(type=EventType.USER, turn=1, payload={
        "kind": "memory_restore", "content": "Visible memory",
        "session_arc": "DUPLICATE " * 1000, "trajectory_manifest": {"batch": ["id"] * 1000},
    })
    pinned = [SessionEvent(type=EventType.SYSTEM, turn=0, payload={"kind": kind, "content": "INSTRUCTIONS " * 1000})
              for kind in ("agents_md", "system_prompt", "preferences")]
    noise = SessionEvent(type=EventType.META, turn=1, payload={"content": "BOOKKEEPING " * 1000})
    assert counter.count_context([*pinned, memory, noise]) == counter.count_text("Visible memory")
    assert counter.count_context([]) == 0
    assert counter.count_events([memory]) > counter.count_context([memory])


def test_context_estimate_includes_tool_calls_results_and_reasoning(tmp_path):
    from langchain_core.messages import HumanMessage, ToolMessage
    manager = SessionManager(session_id="tools", root=tmp_path)
    messages = [
        HumanMessage(content="Inspect"),
        AIMessage(content=[{"type": "thinking", "thinking": "Exact matching matters."}],
                  tool_calls=[{"id": "t", "name": "execute", "args": {"command": "rg tabs"}}]),
        ToolMessage(content="Found tabs.", tool_call_id="t"),
        AIMessage(content="Verified."),
    ]
    events = manager.events_from_messages(messages, turn=1)
    counter = TokenCounter()
    assert counter.count_context(events) > sum(counter.count_text(t) for t in ("Inspect", "Found tabs.", "Verified."))


def test_context_breakdown_accounts_only_for_snapshot_replacements():
    old = SessionEvent(type=EventType.USER, turn=1, payload={"content": "Old finding " * 200})
    tail = SessionEvent(type=EventType.USER, turn=2, payload={"content": "Retain me"})
    memory = SessionEvent(type=EventType.USER, turn=1, payload={"content": "Compact finding"})
    counter = TokenCounter()
    breakdown = context_token_breakdown([old, tail], [memory, tail], counter)
    assert breakdown["saved_tokens"] > 0
    assert breakdown["saved_tokens"] == breakdown["before_tokens"] - breakdown["after_tokens"]
    no_op = context_token_breakdown([old, tail], [old, tail], counter)
    assert no_op["compacted_tokens"] == no_op["injected_tokens"] == no_op["saved_tokens"] == 0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from core.compaction.compactor import Compactor, _invoke_text as session_invoke
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.compaction.token_counter import TokenCounter
from core.compaction.usage import TokenUsage, format_token_usage
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
    assert format_token_usage(usage.as_dict()) == (
        "1,234 input / 500 output tokens (provider-reported, 1 call)"
    )


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
    assert format_token_usage(usage.as_dict()).endswith("(estimated, 1 call)")


def test_zero_usage_is_reported_not_estimated():
    usage = TokenUsage()
    response = AIMessage(content="", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    session_invoke(Mock(invoke=Mock(return_value=response)), "", "", token_counter=TokenCounter(), usage=usage)
    assert usage.as_dict()["reported_calls"] == 1
    assert format_token_usage(usage.as_dict()) == "0 input / 0 output tokens (provider-reported, 1 call)"
    assert format_token_usage(TokenUsage().as_dict()) == "no model calls"


def test_extraction_synthesis_review_and_revision_usage_reaches_cli_event_and_telemetry(tmp_path):
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
    coordinator._run_compaction(events, 999, 2, {"trigger": "manual", "counts": {}})
    phase, payload = observed[-1]
    assert phase == "end"
    assert payload["token_usage"] == {
        "source_tokens": 999, "input_tokens": 500, "output_tokens": 100, "total_tokens": 600,
        "call_count": 5, "reported_calls": 5, "estimated_calls": 0,
    }
    assert "500 input / 100 output tokens (provider-reported, 5 calls)" in payload["content"]
    assert '"reported_calls": 5' in path.read_text()

    from cli.run import QuasipilotApp
    from cli.components import StatusBubble
    app = QuasipilotApp()
    mounted = []
    app._mount_chat = mounted.append
    app._sync_compaction_ui = lambda: None
    app.notify = lambda *args, **kwargs: None
    app._handle_compaction_event(phase, payload)
    assert len(mounted) == 1 and isinstance(mounted[0], StatusBubble)
    assert "500 input / 100 output tokens" in str(mounted[0].render())


def test_mixed_counts_are_not_labeled_actual():
    usage = TokenUsage()
    counter = Mock(spec=TokenCounter)
    counter.count_text.side_effect = [10, 3]
    for metadata in (None, {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}):
        session_invoke(Mock(invoke=Mock(return_value=SimpleNamespace(content="text", usage_metadata=metadata))),
                       "system", "source", token_counter=counter, usage=usage)
    assert usage.as_dict()["total_tokens"] == 133
    assert usage.as_dict()["reported_calls"] == usage.as_dict()["estimated_calls"] == 1
    assert format_token_usage(usage.as_dict()) == (
        "110 input / 23 output tokens (mixed; 1/2 calls estimated, 2 calls)"
    )

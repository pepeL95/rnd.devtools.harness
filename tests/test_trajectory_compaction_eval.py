import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.compaction.token_counter import TokenCounter
from core.session.events import EventType
from core.trajectory import compactor
from evals.trajectory_compaction_ab import RecordedModel, case_events, run_case, score_checks, summarize


FIXTURE = Path(__file__).resolve().parents[1] / "evals/trajectory_compaction_set.json"


def test_small_eval_set_has_unique_cases_and_source_grounded_checks():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) == len({case["id"] for case in cases}) == 5
    for case in cases:
        events = case_events(case)
        assert events == case_events(case)
        assert events[0].type == EventType.USER
        assert events[-1].type == EventType.TURN_END
        source = "\n".join(str(event.payload.get("content", "")) + str(event.payload.get("args", ""))
                           for event in events)
        assert all(score_checks(case["checks"], source).values()), case["id"]
        assert all(score_checks(case["probe_checks"], source).values()), case["id"]


def test_checks_require_all_patterns_not_just_keywords():
    checks = [{"id": "cause", "patterns": ["tabs", "indentation"]}]
    assert score_checks(checks, "Tabs were present") == {"cause": False}
    assert score_checks(checks, "Tabs changed indentation") == {"cause": True}


def test_probe_is_blind_to_raw_events_and_answer_key_and_prompt_patch_is_scoped():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    calls = []

    class Model:
        def invoke(self, messages):
            calls.append(messages)
            text = (json.dumps({"turns": [{"turn": 1, "synthesis": "A compact note.", "live_edge": "Check next."}]})
                    if len(calls) == 1 else "The details are missing.")
            return SimpleNamespace(content=text)

    original = compactor.SYNTHESIS_PROMPT
    row = run_case(case, case_events(case), "Candidate prompt", Model(), TokenCounter())
    assert compactor.SYNTHESIS_PROMPT == original
    assert len(calls) == 2
    assert calls[0][0].content == "Candidate prompt"
    assert "probe_checks" not in calls[0][1].content
    probe = json.loads(calls[1][1].content)
    assert set(probe) == {"user", "assistant", "memory", "question"}
    assert "_edit_diagnostic" not in calls[1][1].content
    assert row["synthesis_usage"]["call_count"] == row["probe_usage"]["call_count"] == 1
    assert row["memory_tokens"] > 0


def test_summary_does_not_hide_failed_runs():
    summary = summarize([{"variant": "baseline", "error": {"message": "quota"},
                          "context_before": 100, "check_count": 5, "probe_check_count": 3}],
                        ["baseline", "candidate"])
    assert summary["baseline"]["runs"] == summary["baseline"]["errors"] == 1
    assert summary["baseline"]["total_checks"] == 5
    assert summary["baseline"]["total_probe_checks"] == 3
    assert summary["baseline"]["context_after"] == summary["baseline"]["context_before"] == 100
    assert summary["candidate"]["runs"] == 0


def test_recorder_keeps_malformed_output_and_usage():
    model = SimpleNamespace(model="fake", invoke=lambda messages: SimpleNamespace(content='{"broken'))
    recorded = RecordedModel(model, TokenCounter())
    case = json.loads(FIXTURE.read_text())["cases"][0]
    with pytest.raises(json.JSONDecodeError):
        run_case(case, case_events(case), "Candidate", recorded, TokenCounter())
    assert recorded.calls[0]["content"] == '{"broken'
    assert recorded.calls[0]["usage"]["output_tokens"] > 0

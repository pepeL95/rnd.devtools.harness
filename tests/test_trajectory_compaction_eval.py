import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.compaction.token_counter import TokenCounter
from core.session.events import EventType
from core.trajectory import compactor
from evals.trajectory_compaction_ab import RecordedModel, case_events, render_comparison, run_case, summarize


FIXTURE = Path(__file__).resolve().parents[1] / "evals/trajectory_compaction_set.json"


def test_small_eval_set_has_five_reproducible_cases_without_keyword_rubrics():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) == len({case["id"] for case in cases}) == 5
    for case in cases:
        events = case_events(case)
        assert events == case_events(case)
        assert events[0].type == EventType.USER
        assert events[-1].type == EventType.TURN_END
        assert set(case) == {"id", "user", "steps", "assistant", "probe"}


def test_comparison_includes_source_memories_probes_and_failures_without_verdicts():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    rows = [
        {"case_id": case["id"], "repeat": 0, "variant": "baseline", "memory": "A saved diagnosis.",
         "context_before": 100, "context_after": 50, "memory_tokens": 30, "probe_answer": "Inspect the exact string."},
        {"case_id": case["id"], "repeat": 0, "variant": "candidate", "error": {"message": "Invalid JSON"}},
    ]
    report = render_comparison([case], rows)
    assert case["user"] in report and "inspect_match.py" in report
    assert case["probe"] in report
    assert "A saved diagnosis." in report and "Inspect the exact string." in report
    assert "original context retained" in report and "Invalid JSON" in report
    assert "100 -> 50" in report
    assert "winner" not in report and "passed_checks" not in report


def test_probe_is_blind_to_raw_events_and_prompt_patch_is_scoped():
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
    assert case["probe"] not in calls[0][1].content
    probe = json.loads(calls[1][1].content)
    assert set(probe) == {"user", "assistant", "memory", "question"}
    assert "_edit_diagnostic" not in calls[1][1].content
    assert row["synthesis_usage"]["call_count"] == row["probe_usage"]["call_count"] == 1
    assert row["memory_tokens"] > 0
    assert "checks" not in row and "probe_checks" not in row


def test_summary_does_not_hide_failed_runs():
    summary = summarize([{"variant": "baseline", "error": {"message": "quota"},
                          "context_before": 100}],
                        ["baseline", "candidate"])
    assert summary["baseline"]["runs"] == summary["baseline"]["errors"] == 1
    assert not any("checks" in key for key in summary["baseline"])
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

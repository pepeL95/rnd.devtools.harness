import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.compaction.token_counter import TokenCounter
from core.session.events import EventType
from core.trajectory import compactor
from evals import trajectory_compaction_ab as evaluation
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


def test_captured_cases_preserve_native_events_and_turn_numbers():
    from evals.common import load_session

    cases = json.loads(FIXTURE.with_name("trajectory_compaction_captured_set.json").read_text())["cases"]
    for case in cases:
        events = case_events(case)
        assert events == [event for event in load_session(Path(case["source_session"]))
                          if event.turn == case["source_turn"]]
        assert {event.turn for event in events} == {case["source_turn"]}
        assert "source_session" not in render_comparison([case], [])
        assert events[0].id in render_comparison([case], [])
    with pytest.raises(ValueError, match="Missing completed source turn"):
        case_events({**cases[0], "source_turn": 999})


def test_captured_probe_uses_retained_messages_not_fixture_answers():
    case = json.loads(FIXTURE.with_name("trajectory_compaction_captured_set.json").read_text())["cases"][-1]
    calls = []

    class Model:
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content=("Environment verified." if len(calls) == 1
                                            else "Use the verified environment."))

    run_case(case, case_events(case), "Candidate", Model(), TokenCounter())
    probe = json.loads(calls[1][1].content)
    assert probe["user"] == "conda activate quasipilot-harness"
    assert probe["assistant"] == "The `quasipilot-harness` environment is already active."
    assert "conda env list" not in calls[1][1].content


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
            text = ("A compact note."
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


def test_no_savings_probe_receives_retained_source_not_an_empty_note():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    calls = []

    def invoke(messages):
        calls.append(messages)
        return SimpleNamespace(content="Expanded note. " * 1000 if len(calls) == 1 else "Continuation.")

    row = run_case(case, case_events(case), "Candidate", SimpleNamespace(invoke=invoke), TokenCounter())
    assert row["skipped"] is True
    assert row["memory"] == "" and row["memory_tokens"] == 0
    assert row["context_after"] == row["context_before"]
    assert "repr(source)" in json.loads(calls[1][1].content)["memory"]
    assert summarize([{**row, "variant": "candidate"}], ["candidate"])["candidate"]["skipped"] == 1


def test_recorder_keeps_malformed_output_and_usage():
    model = SimpleNamespace(model="fake", invoke=lambda messages: SimpleNamespace(content='{"broken'))
    recorded = RecordedModel(model, TokenCounter())
    case = json.loads(FIXTURE.read_text())["cases"][0]
    with pytest.raises(json.JSONDecodeError):
        run_case(case, case_events(case), "Baseline", recorded, TokenCounter(), legacy_json=True)
    assert recorded.calls[0]["content"] == '{"broken'
    assert recorded.calls[0]["usage"]["output_tokens"] > 0


def test_synthesis_effort_experiment_does_not_change_probe_model():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    synthesis_calls, probe_calls = [], []

    def synthesize(messages):
        synthesis_calls.append(messages)
        return SimpleNamespace(content="Evidence.")

    def probe(messages):
        probe_calls.append(messages)
        return SimpleNamespace(content="Continuation.")

    row = run_case(case, case_events(case), "Candidate", SimpleNamespace(invoke=synthesize),
                   TokenCounter(), probe_model=SimpleNamespace(invoke=probe))
    assert len(synthesis_calls) == len(probe_calls) == 1
    assert row["probe_answer"] == "Continuation."


def test_cli_records_effort_and_distinguishes_legacy_and_decoded_input(monkeypatch, tmp_path):
    models = []

    class Model:
        model = "fake"
        temperature = 0
        reasoning_effort = "minimal"

        @property
        def thinking_level(self):
            return self.reasoning_effort

        def invoke(self, messages):
            if messages[0].content == evaluation.PROBE_PROMPT:
                return SimpleNamespace(content="Continuation.")
            if messages[0].content == evaluation.SYNTHESIS_PROMPT:
                return SimpleNamespace(content="Evidence.")
            return SimpleNamespace(content=json.dumps({"turns": [{
                "turn": 1, "synthesis": "Evidence.", "live_edge": "No pending action."
            }]}))

    def factory():
        model = Model()
        models.append(model)
        return model

    monkeypatch.setattr(evaluation, "get_default_trajectory_compactor_model", factory)
    monkeypatch.setattr("sys.argv", ["eval", "--case-id", "exact_whitespace", "--synthesis-thinking", "low",
                                     "--candidate-model", "fake-candidate", "--candidate-thinking", "medium",
                                     "--output-root", str(tmp_path)])
    evaluation.main()
    directory = next(tmp_path.iterdir())
    metadata = json.loads((directory / "experiment.json").read_text())
    assert metadata["thinking_level"] == "low"
    assert metadata["probe_thinking_level"] == "minimal"
    assert metadata["candidate_input"] == "decoded event content, retained user/final anchors"
    assert metadata["baseline_format"] == "json" and metadata["candidate_format"] == "markdown"
    assert models[0].reasoning_effort == "low" and models[1].reasoning_effort == "minimal"
    assert models[2].model == metadata["candidate_model"] == "fake-candidate"
    assert models[2].reasoning_effort == metadata["candidate_thinking_level"] == "medium"
    rows = json.loads((directory / "results.json").read_text())["runs"]
    source = case_events(json.loads(FIXTURE.read_text())["cases"][0])
    for row in rows:
        request = row["calls"][0]["messages"][1]["content"]
        if row["variant"] == "baseline":
            payload = json.loads(request)["turns"][0]
            assert payload["assistant_message"] == source[-2].payload["content"]
            assert "payload:" in payload["internal_trajectory"]
        else:
            assert "END OF HISTORICAL SOURCE" in request
            assert source[-2].payload["content"] in request
            for event in compactor._compacted_source_events(source):
                if event.payload.get("content"):
                    assert event.payload["content"] in request
    assert compactor.events_to_internal_trajectory is evaluation.events_to_internal_trajectory


def test_cli_rejects_unknown_case_before_model_setup(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["eval", "--case-id", "missing"])
    monkeypatch.setattr(evaluation, "get_default_trajectory_compactor_model",
                        lambda: pytest.fail("Unknown case must fail before model setup"))
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
    assert "Unknown or empty case selection" in capsys.readouterr().err

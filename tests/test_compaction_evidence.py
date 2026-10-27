import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.compaction.compactor import Compactor
from core.compaction.evidence import PreparedEvidence
from core.compaction.offloading import BaseCompactionOffloadingStrategy
from core.compaction.policy import CompactionPolicy
from core.compaction.prompts import ARC_PROMPT, ARC_REVIEW_PROMPT, TASK_EPISODES_PROMPT
from core.compaction.serialization import events_to_trajectory
from core.session.events import EventType, SessionEvent


class RecordingModel:
    def __init__(self, timeline):
        self.calls = []
        self.timeline = timeline

    def invoke(self, messages):
        system, user = [message.content for message in messages]
        self.calls.append((system, user))
        self.timeline.append(system)
        return SimpleNamespace(content={
            TASK_EPISODES_PROMPT: "## Matching\nTabs stay exact (turn 1).",
            ARC_PROMPT: "Exact matching is required; preserve tabs.",
            ARC_REVIEW_PROMPT: "APPROVE",
        }[system])


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_evidence_preserves_the_arc_request_contract(enabled):
    timeline = []
    model = RecordingModel(timeline)
    events = [SessionEvent(type=EventType.USER, turn=turn, payload={"content": content})
              for turn, content in [(1, "Keep tabs"), (2, "Retained tail")]]

    class Provider(BaseCompactionOffloadingStrategy):
        def validate(self, manifest):
            assert manifest == {}
            timeline.append("validate")

        def prepare(self, source, turns, episodes, manifest):
            assert source == events[:1] and turns == [1]
            assert episodes == ["## Matching\nTabs stay exact (turn 1)."]
            timeline.append("prepare")
            return PreparedEvidence({"batch": ["record-1"]}, ["trajectory-record://record-1"],
                                    lambda result: timeline.append("persist"))

    engine = Compactor(policy=CompactionPolicy(keep_last_turns=1, compactor_model=model, critic_model=model),
                       offloading=Provider() if enabled else None)
    with patch.dict(sys.modules, {"core.trajectory_bank": None, "chromadb": None}):
        result = engine.compact(events)
    source = {"previous_session_arc": [], "trajectory": events_to_trajectory(events[:1])}
    assert model.calls[0] == (TASK_EPISODES_PROMPT, json.dumps(source, ensure_ascii=False))
    prompt = json.dumps({
        **source,
        "task_episodes": [{"content": result.segmentation,
                           "reference": "trajectory-record://record-1" if enabled else ""}],
        "retained_tail_for_orientation_only": events_to_trajectory(events[1:]),
        "evidence_references": ["trajectory-record://record-1"] if enabled else [],
    }, ensure_ascii=False)
    assert model.calls[1] == (ARC_PROMPT, prompt)
    assert model.calls[2] == (ARC_REVIEW_PROMPT, prompt + "\nProposed session arc:\nExact matching is required; preserve tabs.")
    assert result.events[0].payload["session_arc"] == "Exact matching is required; preserve tabs."
    assert result.events[1:] == events[1:]
    assert result.token_usage["call_count"] == 3
    assert timeline == (["validate", TASK_EPISODES_PROMPT, "prepare", ARC_PROMPT, ARC_REVIEW_PROMPT, "persist"]
                        if enabled else [TASK_EPISODES_PROMPT, ARC_PROMPT, ARC_REVIEW_PROMPT])


def test_evidence_validation_failure_precedes_model_calls():
    timeline = []
    model = RecordingModel(timeline)

    class MissingEvidence(BaseCompactionOffloadingStrategy):
        def validate(self, manifest):
            raise ValueError("Missing evidence")

        def prepare(self, events, turns, episodes, manifest):
            raise AssertionError("Validation must run first")

    engine = Compactor(policy=CompactionPolicy(keep_last_turns=0, compactor_model=model, critic_model=model),
                       offloading=MissingEvidence())
    with pytest.raises(ValueError, match="Missing evidence"):
        engine.compact([SessionEvent(type=EventType.USER, turn=1, payload={"content": "Task"})])
    assert model.calls == []

from __future__ import annotations

import json

from core.compaction.llm import LangChainTextGenerator, TextGenerator
from core.session.events import EventType, SessionEvent
from core.trajectory_compaction.models import TrajectoryCompactionResult, TurnTrajectorySynthesis
from core.trajectory_compaction.policy import TrajectoryCompactionPolicy
from core.trajectory_compaction.prompts import SYNTHESIS_PROMPT
from core.trajectory_compaction.serialization import (
    events_to_internal_trajectory,
    trajectory_memory_message,
)

INTERNAL_EVENT_TYPES = {EventType.REASONING, EventType.TOOL, EventType.TOOL_OUTPUT, EventType.META, EventType.RUNTIME}
SYNTHETIC_KINDS = {"memory_restore", "trajectory_memory"}


class TrajectoryCompactor:
    """Compact tool and reasoning trajectories while preserving human-visible turns."""

    def __init__(
        self,
        generator: TextGenerator | None = None,
        policy: TrajectoryCompactionPolicy | None = None,
    ) -> None:
        self.policy = policy or TrajectoryCompactionPolicy()
        self.policy.validate()
        self.generator = generator or LangChainTextGenerator.from_chat_model(self.policy.model)

    def compact(self, events: list[SessionEvent]) -> TrajectoryCompactionResult:
        compacted_turns = self._compactable_turns(events)
        if not compacted_turns:
            return TrajectoryCompactionResult(
                events=list(events),
                compacted_turns=[],
                compacted_event_count=0,
                memory_document="",
                turn_syntheses=[],
            )

        turn_payloads = []
        compacted_event_count = 0
        for turn in compacted_turns:
            turn_events = [event for event in events if event.turn == turn]
            source_events = [event for event in turn_events if event.type in INTERNAL_EVENT_TYPES]
            compacted_event_count += len(source_events)
            turn_payloads.append(
                {
                    "turn": turn,
                    "user_message": _turn_user_message(turn_events),
                    "assistant_message": _turn_assistant_message(turn_events),
                    "internal_trajectory": events_to_internal_trajectory(source_events),
                }
            )
        turn_syntheses = self._synthesize_batch(turn_payloads)
        synthesis_by_turn = {item.turn: item for item in turn_syntheses}
        turn_memory_events: dict[int, SessionEvent] = {}
        for turn in compacted_turns:
            turn_synthesis = synthesis_by_turn[turn]
            turn_memory_events[turn] = SessionEvent(
                type=EventType.USER,
                turn=turn,
                payload={
                    "role": "user",
                    "content": trajectory_memory_message(turn_synthesis),
                    "kind": "trajectory_memory",
                    "turns": [turn],
                    "compacted_event_count": len(
                        [
                            event
                            for event in events
                            if event.turn == turn and event.type in INTERNAL_EVENT_TYPES
                        ]
                    ),
                },
            )
        rewritten = _rewrite_events(events, compacted_turns, turn_memory_events)
        return TrajectoryCompactionResult(
            events=rewritten,
            compacted_turns=compacted_turns,
            compacted_event_count=compacted_event_count,
            memory_document="\n\n".join(item.synthesis for item in turn_syntheses),
            turn_syntheses=turn_syntheses,
        )

    def should_compact(self, events: list[SessionEvent]) -> bool:
        turns = sorted({event.turn for event in events})
        latest_turn = turns[-1] if turns else 0
        decision = self.policy.compaction_decision(len(turns), latest_turn)
        return decision.should_compact and bool(self._compactable_turns(events))

    def _compactable_turns(self, events: list[SessionEvent]) -> list[int]:
        turns = sorted({event.turn for event in events})
        window_size = self.policy.trigger_every_turns
        if len(turns) < window_size * 2:
            return []
        candidate_turns = turns[-(window_size * 2) : -window_size]
        turns_with_internal_events: list[int] = []
        for turn in candidate_turns:
            turn_events = [event for event in events if event.turn == turn]
            if any(event.payload.get("kind") in SYNTHETIC_KINDS for event in turn_events):
                continue
            if any(event.type in INTERNAL_EVENT_TYPES for event in turn_events):
                turns_with_internal_events.append(turn)
        return turns_with_internal_events

    def _synthesize_batch(self, turn_payloads: list[dict[str, str | int]]) -> list[TurnTrajectorySynthesis]:
        raw = self.generator.generate(
            SYNTHESIS_PROMPT,
            json.dumps({"turns": turn_payloads}, ensure_ascii=False, indent=2),
        )
        data = json.loads(raw)
        items = data.get("turns")
        if not isinstance(items, list):
            raise ValueError("Trajectory compaction output must include a 'turns' list.")
        syntheses: list[TurnTrajectorySynthesis] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each trajectory synthesis item must be an object.")
            turn = item.get("turn")
            synthesis = item.get("synthesis")
            live_edge = item.get("live_edge")
            if not isinstance(turn, int):
                raise ValueError("Each trajectory synthesis item must include an integer 'turn'.")
            if not isinstance(synthesis, str) or not synthesis.strip():
                raise ValueError(f"Trajectory synthesis for turn {turn} is missing 'synthesis'.")
            if not isinstance(live_edge, str) or not live_edge.strip():
                raise ValueError(f"Trajectory synthesis for turn {turn} is missing 'live_edge'.")
            syntheses.append(
                TurnTrajectorySynthesis(
                    turn=turn,
                    synthesis=synthesis.strip(),
                    live_edge=live_edge.strip(),
                )
            )
        expected_turns = {int(item["turn"]) for item in turn_payloads}
        observed_turns = {item.turn for item in syntheses}
        if observed_turns != expected_turns:
            raise ValueError(
                f"Trajectory synthesis turns mismatch. Expected {sorted(expected_turns)}, got {sorted(observed_turns)}."
            )
        return sorted(syntheses, key=lambda item: item.turn)


def _rewrite_events(
    events: list[SessionEvent],
    compacted_turns: list[int],
    memory_events: dict[int, SessionEvent],
) -> list[SessionEvent]:
    if not compacted_turns:
        return list(events)
    compacted_turn_set = set(compacted_turns)
    rewritten: list[SessionEvent] = []
    inserted_turns: set[int] = set()
    last_index_by_turn = {
        turn: max(index for index, event in enumerate(events) if event.turn == turn)
        for turn in compacted_turns
    }
    for index, event in enumerate(events):
        if event.turn in compacted_turn_set and event.type in INTERNAL_EVENT_TYPES:
            pass
        else:
            if (
                event.turn in compacted_turn_set
                and event.type == EventType.ASSISTANT
                and event.turn not in inserted_turns
            ):
                rewritten.append(memory_events[event.turn])
                inserted_turns.add(event.turn)
            rewritten.append(event)
        if event.turn in compacted_turn_set and index == last_index_by_turn[event.turn] and event.turn not in inserted_turns:
            rewritten.append(memory_events[event.turn])
            inserted_turns.add(event.turn)
    return rewritten


def _turn_user_message(events: list[SessionEvent]) -> str:
    for event in events:
        if event.type == EventType.USER and event.payload.get("kind") not in SYNTHETIC_KINDS:
            return str(event.payload.get("content", "")).strip()
    return ""


def _turn_assistant_message(events: list[SessionEvent]) -> str:
    for event in reversed(events):
        if event.type == EventType.ASSISTANT:
            return str(event.payload.get("content", "")).strip()
    return ""

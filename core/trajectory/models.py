from __future__ import annotations

from dataclasses import dataclass, field

from core.session.events import SessionEvent


@dataclass(frozen=True)
class TurnTrajectorySynthesis:
    turn: int
    synthesis: str
    live_edge: str


@dataclass(frozen=True)
class TrajectoryCompactionResult:
    events: list[SessionEvent]
    compacted_turns: list[int]
    compacted_event_count: int
    memory_document: str
    model_names: dict[str, str] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    turn_syntheses: list[TurnTrajectorySynthesis] = field(default_factory=list)

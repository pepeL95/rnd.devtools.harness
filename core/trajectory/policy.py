from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel

from core.utilities.defaults import get_default_trajectory_compactor_model


@dataclass(frozen=True)
class TrajectoryCompactionDecision:
    should_compact: bool
    reason: str | None = None


@dataclass(frozen=True)
class TrajectoryCompactionPolicy:
    trigger_every_turns: int = 2
    compactor_model: BaseChatModel = field(default_factory=get_default_trajectory_compactor_model)

    def compaction_decision(self, turn_count: int, latest_turn: int) -> TrajectoryCompactionDecision:
        if latest_turn <= 0 or turn_count <= 0:
            return TrajectoryCompactionDecision(False)
        if turn_count < self.trigger_every_turns * 2:
            return TrajectoryCompactionDecision(False)
        if latest_turn % self.trigger_every_turns == 0:
            return TrajectoryCompactionDecision(True, "turn_interval")
        return TrajectoryCompactionDecision(False)

    def validate(self) -> None:
        if self.trigger_every_turns <= 0:
            raise ValueError("trigger_every_turns must be positive.")

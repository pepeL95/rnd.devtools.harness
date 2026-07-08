from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState

from core.trajectory.coordinator import TrajectoryCompactionCoordinator


class TrajectoryCompactionMiddleware(AgentMiddleware):
    """Apply completed trajectory syntheses at run boundaries and schedule the next one."""

    def __init__(self, coordinator: TrajectoryCompactionCoordinator) -> None:
        self.coordinator = coordinator

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self.coordinator.prepare_for_agent()
        self.coordinator.request_compaction()
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return None

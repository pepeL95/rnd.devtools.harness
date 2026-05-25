from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState

from core.trajectory.coordinator import TrajectoryCompactionCoordinator


class TrajectoryCompactionMiddleware(AgentMiddleware):
    """Schedule high-frequency internal-trajectory compaction after completed turns."""

    def __init__(self, coordinator: TrajectoryCompactionCoordinator) -> None:
        self.coordinator = coordinator

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self.coordinator.request_compaction()
        return None

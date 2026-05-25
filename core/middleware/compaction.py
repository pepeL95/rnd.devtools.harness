from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from core.compaction.coordinator import CompactionCoordinator


class CompactionMiddleware(AgentMiddleware):
    """Schedule curated session compaction after agent turns."""

    def __init__(self, coordinator: CompactionCoordinator) -> None:
        self.coordinator = coordinator

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self.coordinator.request_policy_compaction(runtime)
        return None

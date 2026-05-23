from __future__ import annotations

from typing import Any

from core.compaction.compactor import Compactor
from core.compaction.token_counter import TokenCounter
from langchain.agents.middleware import AgentMiddleware, AgentState
from core.session.session_manager import SessionManager


class CompactionMiddleware(AgentMiddleware):
    """Compact curated session history after an agent turn crosses policy limits."""

    def __init__(
        self,
        manager: SessionManager,
        compactor: Compactor,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.manager = manager
        self.compactor = compactor
        self.token_counter = token_counter or TokenCounter()

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self._compact_if_needed()
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self._compact_if_needed()
        return None

    def _compact_if_needed(self) -> None:
        events = self.manager.read_curated()
        estimated = self.token_counter.count_events(events)
        if not self.compactor.policy.should_compact(estimated):
            return
        result = self.compactor.compact(events, token_estimate=estimated)
        self.manager.replace_curated(result.events)

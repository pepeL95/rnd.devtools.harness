from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState

from core.session.restore import reset_messages_update
from core.session.manager import SessionManager


class SessionLoadMiddleware(AgentMiddleware):
    """Restore curated session history into the agent message state."""

    def __init__(self, manager: SessionManager) -> None:
        self.manager = manager

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        restored = self.manager.load_curated_messages()
        if not restored:
            return None
        current = list(state.get("messages", []))
        return {"messages": reset_messages_update([*restored, *current])}

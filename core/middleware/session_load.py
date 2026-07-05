from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, AgentState

from core.session.restore import reset_messages_update
from core.session.manager import SessionManager

if TYPE_CHECKING:
    from core.middleware.session_dump import SessionDumpMiddleware


class SessionLoadMiddleware(AgentMiddleware):
    """Restore curated session history into the agent message state."""

    def __init__(
        self,
        manager: SessionManager,
        session_dump: "SessionDumpMiddleware | None" = None,
    ) -> None:
        self.manager = manager
        self._session_dump = session_dump

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        if self._session_dump is not None and self._session_dump.is_interrupted:
            # Re-entering after a live-steering interrupt: LangGraph state already
            # contains the full message history from the interrupted run, including
            # tool calls and results. Restoring curated history would wipe that
            # context and cause the agent to redo already-completed work.
            return None
        restored = self.manager.load_curated_messages()
        if not restored:
            return None
        current = list(state.get("messages", []))
        return {"messages": reset_messages_update([*restored, *current])}

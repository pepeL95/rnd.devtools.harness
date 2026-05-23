from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.session_manager import SessionManager


class SessionDumpMiddleware(AgentMiddleware):
    """Append full-fidelity agent state to dump and curated session streams."""

    def __init__(self, manager: SessionManager) -> None:
        self.manager = manager
        self._seen_message_keys: set[tuple[str, str]] = set()
        self._active_turn: int | None = None

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self._active_turn = self.manager.next_turn()
        self.manager.append(
            [
                SessionEvent(type=EventType.TURN_BEGIN, turn=self._active_turn, payload={}),
                self._runtime_event(runtime),
            ]
        )
        self._dump_new_messages(state)
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        turn = self._active_turn or self.manager.next_turn()
        self._dump_new_messages(state)
        self.manager.append([SessionEvent(type=EventType.TURN_END, turn=turn, payload={})])
        self._active_turn = None
        return None

    def _dump_new_messages(self, state: AgentState) -> None:
        turn = self._active_turn or self.manager.next_turn()
        unseen = []
        for message in state.get("messages", []):
            key = (str(getattr(message, "type", "")), str(getattr(message, "content", message)))
            if key not in self._seen_message_keys:
                self._seen_message_keys.add(key)
                unseen.append(message)
        self.manager.append(self.manager.events_from_messages(unseen, turn=turn))

    def _runtime_event(self, runtime: Any) -> SessionEvent:
        context = getattr(runtime, "context", None)
        cwd = getattr(context, "cwd", None) or (context.get("cwd") if isinstance(context, dict) else None)
        snapshot = RuntimeSnapshot(cwd=str(cwd or "unknown"))
        return SessionEvent(
            type=EventType.RUNTIME,
            turn=self._active_turn or self.manager.next_turn(),
            payload={
                "cwd": snapshot.cwd,
                "git_branch": snapshot.git_branch,
                "git_dirty": snapshot.git_dirty,
            },
        )


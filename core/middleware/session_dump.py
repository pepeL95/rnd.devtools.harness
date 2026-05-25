from __future__ import annotations

from typing import Any
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware, AgentState
from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.manager import SessionManager
from core.utilities.git import git_branch, git_dirty

class SessionDumpMiddleware(AgentMiddleware):
    """Append full-fidelity agent state to dump and curated session streams."""

    def __init__(self, manager: SessionManager) -> None:
        self.manager = manager
        self._seen_event_keys: set[tuple[Any, ...]] = set()
        self._active_turn: int | None = None
        self._prime_seen_events()

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
        unseen_events: list[SessionEvent] = []
        for message in state.get("messages", []):
            if _is_restored_memory_message(message):
                continue
            for event in self.manager.events_from_messages([message], turn=turn):
                key = self._event_key(event)
                if key in self._seen_event_keys:
                    continue
                self._seen_event_keys.add(key)
                unseen_events.append(event)
        self.manager.append(unseen_events)

    def _event_key(self, event: SessionEvent) -> tuple[Any, ...]:
        payload = event.payload
        return (
            event.type.value,
            payload.get("role"),
            payload.get("index"),
            payload.get("content"),
            payload.get("name"),
            repr(payload.get("args")),
            payload.get("tool_call_id"),
            payload.get("reasoning_format"),
            payload.get("signature"),
        )

    def _prime_seen_events(self) -> None:
        for event in self.manager.read_dump():
            if event.type in {EventType.TURN_BEGIN, EventType.TURN_END, EventType.RUNTIME}:
                continue
            self._seen_event_keys.add(self._event_key(event))

    def _runtime_event(self, runtime: Any) -> SessionEvent:
        cwd = self._resolve_cwd(runtime)
        snapshot = RuntimeSnapshot(cwd=str(cwd or "unknown"), git_branch=git_branch(cwd), git_dirty=git_dirty(cwd))
        return SessionEvent(
            type=EventType.RUNTIME,
            turn=self._active_turn or self.manager.next_turn(),
            payload={
                "cwd": snapshot.cwd,
                "git_branch": snapshot.git_branch,
                "git_dirty": snapshot.git_dirty,
            },
        )

    def _resolve_cwd(self, runtime: Any) -> str:
        context = getattr(runtime, "context", {})
        cwd_str = context.get("cwd") if isinstance(context, dict) else getattr(context, "cwd", None)
        return Path(cwd_str).expanduser().resolve() if cwd_str else str(Path.cwd().expanduser().resolve())

def _is_restored_memory_message(message: Any) -> bool:
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    return additional_kwargs.get("session_kind") in {"memory_restore", "trajectory_memory"}

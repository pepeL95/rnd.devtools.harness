from __future__ import annotations

from dataclasses import asdict
from typing import Any
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware import ModelRequest, ModelResponse
from langgraph.types import Command
from core.live_steering import CancellationInterrupt, LiveSteeringInterrupt, format_cancellation_introspection, format_steering_introspection
from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.manager import SessionManager
from core.utilities.git import git_branch, git_dirty

class SessionDumpMiddleware(AgentMiddleware):
    """Append full-fidelity agent state to dump and curated session streams."""

    def __init__(self, manager: SessionManager, python_interpreter: str | Path | None = None) -> None:
        self.manager = manager
        self.python_interpreter = (
            Path(python_interpreter).expanduser().resolve() if python_interpreter else None
        )
        self._seen_event_keys: set[tuple[Any, ...]] = set()
        self._active_turn: int | None = None
        self._interrupted: bool = False
        self._prime_seen_events()

    @property
    def is_interrupted(self) -> bool:
        """True when the last turn was interrupted and the agent is re-entering."""
        return self._interrupted

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        if self._interrupted and self._active_turn is not None:
            # Re-entering after a live-steering interrupt: continue the same turn,
            # no new TURN_BEGIN or RUNTIME event needed.
            self._interrupted = False
        else:
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
        self._end_turn(turn)
        self._active_turn = None
        return None

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> ModelResponse:
        try:
            response = handler(request)
        except Exception as exc:
            self._record_failure("model_error", {"error_type": exc.__class__.__name__, "error": str(exc)})
            raise
        self._append_messages(getattr(response, "result", []) or [])
        return response

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        try:
            result = handler(request)
        except CancellationInterrupt:
            self._record_cancellation()
            raise
        except LiveSteeringInterrupt as exc:
            self._record_interrupt("live_steering_interrupt", {"steering": exc.steering})
            raise
        except Exception as exc:
            tool_call = getattr(request, "tool_call", None) or {}
            self._record_failure(
                "tool_error",
                {
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "tool_name": tool_call.get("name"),
                    "tool_call_id": tool_call.get("id"),
                },
            )
            raise
        self._append_messages(_tool_result_messages(result))
        return result

    def _dump_new_messages(self, state: AgentState) -> None:
        self._append_messages(state.get("messages", []))

    def _append_messages(self, messages: Any) -> None:
        turn = self._active_turn or self.manager.next_turn()
        unseen_events: list[SessionEvent] = []
        for message in messages or []:
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

    def _record_failure(self, kind: str, payload: dict[str, Any]) -> None:
        turn = self._active_turn
        if turn is None:
            return
        self.manager.append(
            [
                SessionEvent(type=EventType.META, turn=turn, payload={"kind": kind, **payload}),
                SessionEvent(type=EventType.TURN_END, turn=turn, payload={"status": "error"}),
            ]
        )
        self._active_turn = None

    def _record_cancellation(self) -> None:
        turn = self._active_turn
        if turn is None:
            return
        event = SessionEvent(
            type=EventType.REASONING,
            turn=turn,
            payload={
                "role": "assistant",
                "content": format_cancellation_introspection(),
                "reasoning_format": "cancellation",
                "signature": None,
                "index": 0,
            },
        )
        self._seen_event_keys.add(self._event_key(event))
        self.manager.append([
            event,
            SessionEvent(type=EventType.TURN_END, turn=turn, payload={"status": "cancelled"}),
        ])
        self._active_turn = None

    def _record_interrupt(self, kind: str, payload: dict[str, Any]) -> None:
        turn = self._active_turn
        if turn is None:
            return
        steering = payload.get("steering", "")
        user_event = SessionEvent(
            type=EventType.USER,
            turn=turn,
            payload={"role": "user", "content": steering, "kind": kind},
        )
        reasoning_event = SessionEvent(
            type=EventType.REASONING,
            turn=turn,
            payload={
                "role": "assistant",
                "content": format_steering_introspection(steering),
                "reasoning_format": "live_steering",
                "signature": None,
                "index": 0,
            },
        )
        # Register both events before writing so _dump_new_messages on re-entry
        # doesn't see the steering message in LangGraph state and write it again.
        self._seen_event_keys.add(self._event_key(user_event))
        self._seen_event_keys.add(self._event_key(reasoning_event))
        self.manager.append([user_event, reasoning_event])
        # Turn stays open; _active_turn is preserved for re-entry via before_agent.
        self._interrupted = True

    def _end_turn(self, turn: int) -> None:
        self.manager.append([SessionEvent(type=EventType.TURN_END, turn=turn, payload={})])

    def _runtime_event(self, runtime: Any) -> SessionEvent:
        cwd = self._resolve_cwd(runtime)
        snapshot = RuntimeSnapshot(
            cwd=str(cwd or "unknown"),
            git_branch=git_branch(cwd),
            git_dirty=git_dirty(cwd),
            python_interpreter=str(self.python_interpreter) if self.python_interpreter else None,
        )
        return SessionEvent(
            type=EventType.RUNTIME,
            turn=self._active_turn or self.manager.next_turn(),
            payload=asdict(snapshot),
        )

    def _resolve_cwd(self, runtime: Any) -> str:
        context = getattr(runtime, "context", {})
        cwd_str = context.get("cwd") if isinstance(context, dict) else getattr(context, "cwd", None)
        return Path(cwd_str).expanduser().resolve() if cwd_str else str(Path.cwd().expanduser().resolve())

def _is_restored_memory_message(message: Any) -> bool:
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    return additional_kwargs.get("session_kind") in {"memory_restore", "trajectory_memory"}


def _tool_result_messages(result: Any) -> list[Any]:
    if result is None:
        return []
    if _is_message_like(result):
        return [result]
    if isinstance(result, Command):
        update = getattr(result, "update", None) or {}
        messages = update.get("messages", [])
        if isinstance(messages, list):
            return [message for message in messages if _is_message_like(message)]
    return []


def _is_message_like(value: Any) -> bool:
    return hasattr(value, "content") or hasattr(value, "tool_call_id")

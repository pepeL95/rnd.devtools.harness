from __future__ import annotations

from threading import RLock
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from core.hooks.commands import parse_command
from core.hooks.dispatcher import HookDispatcher
from core.live_steering import CancellationInterrupt, LiveSteeringInterrupt
from core.middleware.session_dump import SessionDumpMiddleware


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Command):
        return _json_value(value.update)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class HooksMiddleware(AgentMiddleware):
    """Adapt synchronous harness execution to configured lifecycle hooks.

    Register after SessionDump: turn hooks run inside its lifecycle, while
    wrappers run before its output recording. Deferred notes are delivered only
    after tool results, keeping provider transcripts structurally valid.
    """

    def __init__(self, dispatcher: HookDispatcher, session_dump: SessionDumpMiddleware) -> None:
        self.dispatcher = dispatcher
        self.session_dump = session_dump
        self._pending: list[HumanMessage] = []
        self._lock = RLock()

    def _drain(self) -> list[HumanMessage]:
        with self._lock:
            messages, self._pending = self._pending, []
            self.dispatcher.persist(messages)
            return messages

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        turn = self.session_dump.active_turn
        if turn is not None and self.dispatcher.start_turn(turn):
            self._pending.clear()
            self._pending.extend(self.dispatcher.dispatch("before.turn", {"input": _json_value(state.get("messages", []))}))
        messages = self._drain()
        return {"messages": messages} if messages else None

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._pending.extend(self.dispatcher.dispatch("before.model", {"input": _json_value(state.get("messages", []))}))
        messages = self._drain()
        return {"messages": messages} if messages else None

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        data = {"input": _json_value(request.messages), "model": str(getattr(request.model, "model_name", None) or getattr(request.model, "model", "unknown"))}
        try:
            result = handler(request)
        except Exception as exc:
            self._pending.extend(self.dispatcher.dispatch("after.model", {**data, **_failure(exc)}))
            self._drain()
            self.dispatcher.finish_turn()
            raise
        self._pending.extend(self.dispatcher.dispatch("after.model", {**data, "status": "success", "result": _json_value(result.result)}))
        return result

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = next((message for message in reversed(state.get("messages", [])) if isinstance(message, AIMessage)), None)
        if last is not None and last.tool_calls:
            return None
        messages = self._drain()
        return {"messages": messages} if messages else None

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # SessionDump.after_agent has not closed the logical turn yet.
        if self.session_dump.active_turn is None:
            return None
        last = next((message for message in reversed(state.get("messages", [])) if isinstance(message, AIMessage)), None)
        if last is None or last.tool_calls:
            return None
        self._pending.extend(self.dispatcher.dispatch("after.turn", {"status": "success", "result": _json_value(last)}))
        messages = self._drain()
        if messages:
            return {"messages": messages, "jump_to": "model"}
        self.dispatcher.finish_turn()
        return None

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        # Serialize each tool's hook sequence; defer all notes until results exist.
        with self._lock:
            tool = dict(request.tool_call)
            data = {"tool": _json_value(tool)}
            parsed = None
            if tool.get("name") == "execute":
                raw = tool.get("args", {}).get("command")
                if isinstance(raw, str):
                    parsed = parse_command(raw)
                    data["command"] = {"raw": raw, "cmd": parsed.cmd if parsed else None, "args": list(parsed.args) if parsed else None}
                if parsed is None and any(hook.config.trigger.endswith(".command") for hook in self.dispatcher.hooks):
                    self.dispatcher.diagnostic(None, "before.command", {"status": "skipped", "reason": "unsupported shell syntax", **data})
            self._pending.extend(self.dispatcher.dispatch("before.tool", data))
            if parsed:
                self._pending.extend(self.dispatcher.dispatch("before.command", data, parsed))
            try:
                result = handler(request)
            except Exception as exc:
                outcome = {**data, **_failure(exc)}
                if parsed:
                    self._pending.extend(self.dispatcher.dispatch("after.command", outcome, parsed))
                self._pending.extend(self.dispatcher.dispatch("after.tool", outcome))
                self._drain()
                if not isinstance(exc, LiveSteeringInterrupt):
                    self.dispatcher.finish_turn()
                raise
            artifact = getattr(result, "artifact", None)
            exit_code = artifact.get("exit_code") if isinstance(artifact, dict) else None
            failed = getattr(result, "status", None) == "error" or (exit_code is not None and exit_code != 0)
            outcome = {**data, "status": "error" if failed else "success", "result": _json_value(result)}
            if parsed:
                self._pending.extend(self.dispatcher.dispatch("after.command", outcome, parsed))
            self._pending.extend(self.dispatcher.dispatch("after.tool", outcome))
            return result


def _failure(exc: Exception) -> dict[str, str]:
    status = "cancelled" if isinstance(exc, CancellationInterrupt) else (
        "interrupted" if isinstance(exc, LiveSteeringInterrupt) else "error"
    )
    return {"status": status, "error_type": type(exc).__name__, "error": str(exc)}

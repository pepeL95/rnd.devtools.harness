from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from core.live_steering import LiveSteeringInterrupt
from core.compaction.token_counter import TokenCounter
from core.telemetry.events import TelemetryEvent
from core.telemetry.store import TelemetryStore


class TelemetryMiddleware(AgentMiddleware):
    """Record lightweight lifecycle telemetry without owning session history."""

    def __init__(self, store: TelemetryStore, token_counter: TokenCounter | None = None) -> None:
        self.store = store
        self.token_counter = token_counter or TokenCounter()

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self.store.record(TelemetryEvent(name="agent.start", payload={"messages": len(state.get("messages", []))}))
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self.store.record(TelemetryEvent(name="agent.end", payload={"messages": len(state.get("messages", []))}))
        return None

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> ModelResponse:
        messages = list(getattr(request, "messages", []) or [])
        input_token_estimate = _estimate_messages_tokens(messages, self.token_counter)
        self.store.record(
            TelemetryEvent(
                name="model.start",
                payload={
                    "messages": len(messages),
                    "model": _model_name(getattr(request, "model", None)),
                    "input_tokens_estimate": input_token_estimate,
                },
            )
        )
        try:
            response = handler(request)
        except Exception as exc:
            self.store.record(
                TelemetryEvent(
                    name="model.error",
                    payload={
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "exception": _exception_payload(exc),
                        "input_tokens_estimate": input_token_estimate,
                    },
                )
            )
            raise
        usage = _response_usage(response)
        self.store.record(
            TelemetryEvent(
                name="model.end",
                payload={
                    "input_tokens_estimate": input_token_estimate,
                    "usage": usage,
                },
            )
        )
        return response

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        payload = _tool_call_payload(request)
        self.store.record(TelemetryEvent(name="tool.start", payload=payload))
        try:
            result = handler(request)
        except LiveSteeringInterrupt as exc:
            self.store.record(
                TelemetryEvent(
                    name="tool.interrupt",
                    payload={
                        **payload,
                        "kind": "live_steering",
                        "steering": exc.steering,
                    },
                )
            )
            raise
        except Exception as exc:
            self.store.record(
                TelemetryEvent(
                    name="tool.error",
                    payload={
                        **payload,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "exception": _exception_payload(exc),
                    },
                )
            )
            raise
        self.store.record(
            TelemetryEvent(
                name="tool.end",
                payload={
                    **payload,
                    "result": _tool_result_payload(result),
                },
            )
        )
        return result


def _model_name(model: Any) -> str:
    if model is None:
        return ""
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if name:
        return str(name)
    return model.__class__.__name__


def _estimate_messages_tokens(messages: list[Any], token_counter: TokenCounter) -> int:
    if not messages:
        return 0
    text = "\n\n".join(_message_to_text(message) for message in messages)
    return token_counter.count_text(text)


def _message_to_text(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None) or "message"
    content = getattr(message, "content", message)
    return f"{role}: {_content_to_text(content)}"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("reasoning") or block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _response_usage(response: ModelResponse) -> dict[str, Any]:
    messages = list(getattr(response, "result", []) or [])
    usage_items = [getattr(message, "usage_metadata", None) for message in messages]
    usage_items = [item for item in usage_items if isinstance(item, dict)]
    if not usage_items:
        return {}

    totals: dict[str, Any] = {}
    for item in usage_items:
        _merge_usage_item(totals, item)
    return totals


def _merge_usage_item(target: dict[str, Any], item: dict[str, Any]) -> None:
    for key, value in item.items():
        if isinstance(value, int):
            target[key] = int(target.get(key, 0)) + value
        elif isinstance(value, dict):
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
                target[key] = existing
            _merge_usage_item(existing, value)
        elif key not in target:
            target[key] = value


def _exception_payload(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    payload: dict[str, Any] = {
        "type": exc.__class__.__name__,
        "message": message,
        "chain": _exception_chain(exc),
    }

    status_code = _first_attr(exc, "status_code", "code")
    if status_code is not None:
        payload["status_code"] = status_code
    retry_delay = _retry_delay_seconds(message)
    if retry_delay is not None:
        payload["retry_delay_seconds"] = retry_delay
    if "RESOURCE_EXHAUSTED" in message:
        payload["provider_status"] = "RESOURCE_EXHAUSTED"
    return payload


def _exception_chain(exc: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": current.__class__.__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return chain


def _first_attr(exc: BaseException, *names: str) -> Any:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for name in names:
            value = getattr(current, name, None)
            if value is not None:
                return value
        current = current.__cause__ or current.__context__
    return None


def _retry_delay_seconds(message: str) -> int | None:
    match = re.search(r"retry(?:Delay| in)?['\": ]+(\d+)(?:s| seconds)?", message, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _tool_call_payload(request: Any) -> dict[str, Any]:
    tool_call = getattr(request, "tool_call", None) or {}
    tool = getattr(request, "tool", None)
    args = _tool_call_value(tool_call, "args") or {}
    return {
        "tool_call_id": _tool_call_value(tool_call, "id"),
        "tool_name": _tool_call_value(tool_call, "name") or getattr(tool, "name", None) or "tool",
        "args": _truncate_value(args),
    }


def _tool_result_payload(result: Any) -> dict[str, Any]:
    content = getattr(result, "content", None)
    if content is None:
        content = str(result)
    text = _content_to_text(content)
    return {
        "type": result.__class__.__name__,
        "content_chars": len(text),
        "content_preview": _truncate_text(text),
    }


def _tool_call_value(tool_call: Any, key: str) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _truncate_value(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, dict):
        return {str(key): _truncate_value(item, limit=limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, limit=limit) for item in value]
    if isinstance(value, str):
        return _truncate_text(value, limit=limit)
    return value


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

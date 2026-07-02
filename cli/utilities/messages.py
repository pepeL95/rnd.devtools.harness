from __future__ import annotations

from typing import Any

from core.utilities.messages import (
    message_reasoning_blocks as core_message_reasoning_blocks,
    message_tool_calls as core_message_tool_calls,
)


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def message_reasoning(message: Any) -> str | None:
    parts = [block["text"] for block in core_message_reasoning_blocks(message) if block.get("text")]
    if parts:
        return "\n".join(parts).strip()
    return None


def message_tool_calls(message: Any) -> list[dict[str, Any]]:
    return core_message_tool_calls(message)


def format_tool_call(call: dict[str, Any]) -> tuple[str, str]:
    name = str(call.get("name") or call.get("type") or "tool")
    args = call.get("args") or call.get("arguments") or {}
    if not isinstance(args, dict):
        return name, str(args)
    if not args:
        return name, ""
    parts = [f"{key}={value!r}" for key, value in args.items()]
    return name, " ".join(parts)


def format_tool_input(call: dict[str, Any]) -> tuple[str, str]:
    name = str(call.get("name") or call.get("type") or "tool")
    args = call.get("args") or call.get("arguments") or {}
    if not isinstance(args, dict):
        return name, str(args).strip()
    if not args:
        return name, ""

    if name in {"execute", "shell", "bash"}:
        command = args.get("cmd") or args.get("command")
        if command:
            return name, str(command).strip()

    preferred_keys = (
        "path",
        "file_path",
        "pattern",
        "query",
        "q",
        "url",
        "session_id",
        "location",
        "text",
        "prompt",
    )
    if len(args) == 1:
        only_key = next(iter(args))
        return name, _tool_value_to_text(args[only_key])

    ordered_keys = [key for key in preferred_keys if key in args]
    ordered_keys.extend(key for key in args if key not in ordered_keys)
    parts: list[str] = []
    for key in ordered_keys:
        value = args[key]
        rendered = _tool_value_to_text(value)
        if key in {"path", "file_path", "pattern", "query", "q", "url", "session_id", "location", "text", "prompt"}:
            parts.append(rendered)
        else:
            parts.append(f"{key}={rendered}")
    return name, " ".join(part for part in parts if part)


def _tool_value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    return str(value)

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def normalize_message_content(content: Any) -> str | list[Any]:
    """Coerce persisted session content into LangChain-friendly form."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if not content:
            return ""
        if all(isinstance(block, dict) and block.get("type") == "text" for block in content):
            parts = [str(block.get("text", "")) for block in content if block.get("text")]
            return "\n".join(parts)
        return content
    if content is None:
        return ""
    return str(content)


def message_role(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    if role == "ai":
        return "assistant"
    if role == "human":
        return "user"
    return str(role or "message")


def message_reasoning_blocks(message: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content = getattr(message, "content", "")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type not in {"reasoning", "thinking"}:
                continue
            text = block.get("reasoning") or block.get("thinking") or block.get("text")
            if not text:
                continue
            extras = block.get("extras") or {}
            signature = block.get("signature") or extras.get("signature")
            blocks.append(
                {
                    "type": "reasoning",
                    "text": str(text).strip(),
                    "format": str(block_type),
                    "signature": str(signature) if signature else None,
                }
            )
    additional = getattr(message, "additional_kwargs", None) or {}
    reasoning = additional.get("reasoning") or additional.get("reasoning_content")
    if reasoning:
        blocks.append(
            {
                "type": "reasoning",
                "text": str(reasoning).strip(),
                "format": "additional_kwargs",
                "signature": None,
            }
        )
    return blocks


def message_text_content(message: Any) -> str:
    """Extract only assistant-visible text from a message content payload."""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def message_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None) or []
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            normalized.append(call)
            continue
        normalized.append(
            {
                "name": getattr(call, "name", "tool"),
                "args": getattr(call, "args", {}),
                "id": getattr(call, "id", None),
            }
        )
    return normalized


def make_message(role: str, content: Any, *, additional_kwargs: dict[str, Any] | None = None) -> BaseMessage:
    normalized = normalize_message_content(content)
    extras = dict(additional_kwargs or {})
    if role == "user":
        return HumanMessage(content=normalized, additional_kwargs=extras)
    if role == "assistant":
        return AIMessage(content=normalized, additional_kwargs=extras)
    if role == "system":
        return SystemMessage(content=normalized, additional_kwargs=extras)
    if role in {"tool", "tool_output"}:
        tool_call_id = ""
        if isinstance(content, dict):
            tool_call_id = str(content.get("tool_call_id") or "")
        return ToolMessage(
            content=str(normalized),
            tool_call_id=tool_call_id or "restored",
            additional_kwargs=extras,
        )
    raise ValueError(f"Unsupported message role for agent restore: {role}")


def system_message_with_appended_text(message: SystemMessage | None, text: str) -> SystemMessage:
    if message is None:
        blocks: list[dict[str, Any]] = []
    else:
        content_blocks = getattr(message, "content_blocks", None)
        if content_blocks is not None:
            blocks = list(content_blocks)
        else:
            blocks = [{"type": "text", "text": str(message.content)}]
    blocks.append({"type": "text", "text": text})
    return SystemMessage(content=blocks)

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


def make_message(role: str, content: Any) -> BaseMessage:
    normalized = normalize_message_content(content)
    if role == "user":
        return HumanMessage(content=normalized)
    if role == "assistant":
        return AIMessage(content=normalized)
    if role == "system":
        return SystemMessage(content=normalized)
    if role in {"tool", "tool_output"}:
        tool_call_id = ""
        if isinstance(content, dict):
            tool_call_id = str(content.get("tool_call_id") or "")
        return ToolMessage(content=str(normalized), tool_call_id=tool_call_id or "restored")
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

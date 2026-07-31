from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

MISSING_TOOL_RESULT = "Tool execution did not produce a result before the session was interrupted."


def reset_messages_update(messages: list[BaseMessage]) -> list[BaseMessage | RemoveMessage]:
    """Reset the LangGraph message channel, then apply the intended sequence.

    Agent state uses message reducers; prepending restored history requires
    clearing the channel before writing the merged list.
    """

    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]


def normalize_tool_transcript(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return a provider-valid transcript with every tool call paired exactly once."""

    normalized: list[BaseMessage] = []
    pending_call_ids: list[str] = []
    used_call_ids: set[str] = set()
    generated_id = 0

    def close_pending_calls() -> None:
        for tool_call_id in pending_call_ids:
            normalized.append(
                ToolMessage(
                    content=MISSING_TOOL_RESULT,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"session_kind": "transcript_repair"},
                )
            )
        pending_call_ids.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            if not pending_call_ids:
                continue
            tool_call_id = str(message.tool_call_id or "").strip()
            if tool_call_id in pending_call_ids:
                matched_id = tool_call_id
            elif not tool_call_id or tool_call_id == "restored":
                matched_id = pending_call_ids[0]
                message = message.model_copy(update={"tool_call_id": matched_id})
            else:
                continue
            normalized.append(message)
            pending_call_ids.remove(matched_id)
            continue

        close_pending_calls()
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls: list[dict[str, Any]] = []
            for call in message.tool_calls:
                tool_call_id = str(call.get("id") or "").strip()
                if not tool_call_id or tool_call_id in used_call_ids:
                    generated_id += 1
                    tool_call_id = f"restored-call-{generated_id}"
                used_call_ids.add(tool_call_id)
                pending_call_ids.append(tool_call_id)
                tool_calls.append({**call, "id": tool_call_id})
            message = message.model_copy(update={"tool_calls": tool_calls})
        normalized.append(message)

    close_pending_calls()
    return normalized

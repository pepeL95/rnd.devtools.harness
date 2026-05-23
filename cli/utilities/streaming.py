from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from cli.utilities.messages import format_tool_call, message_reasoning, message_text, message_tool_calls


StreamCallback = Callable[[str, dict[str, Any]], None]


def iter_agent_turn(
    agent: Any,
    user_text: str,
    on_event: StreamCallback,
) -> str:
    """Stream a single user turn and return the final assistant text."""

    inputs = {"messages": [HumanMessage(content=user_text)]}
    assistant_text = ""
    seen_message_ids: set[str] = set()

    for item in agent.stream(inputs, stream_mode=["updates"]):
        if isinstance(item, tuple) and len(item) == 2:
            mode, chunk = item
            if mode != "updates":
                continue
        else:
            chunk = item
        if not isinstance(chunk, dict):
            continue
        for update in chunk.values():
            if not isinstance(update, dict):
                continue
            messages = update.get("messages") or []
            for message in messages:
                message_id = str(getattr(message, "id", id(message)))
                if message_id in seen_message_ids:
                    continue
                seen_message_ids.add(message_id)

                if isinstance(message, AIMessage):
                    for call in message_tool_calls(message):
                        name, args = format_tool_call(call)
                        on_event("tool", {"name": name, "args": args})
                    reasoning = message_reasoning(message)
                    if reasoning:
                        on_event("reason", {"text": reasoning})
                    text = message_text(message)
                    if text:
                        assistant_text = text

    return assistant_text

from __future__ import annotations

from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def reset_messages_update(messages: list[BaseMessage]) -> list[BaseMessage | RemoveMessage]:
    """Reset the LangGraph message channel, then apply the intended sequence.

    Agent state uses message reducers; prepending restored history requires
    clearing the channel before writing the merged list.
    """

    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]

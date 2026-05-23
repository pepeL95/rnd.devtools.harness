"""Small compatibility boundary around LangChain imports.

The real harness uses LangChain classes when installed. The fallback classes keep
local contract tests importable in a fresh checkout before dependencies are
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol


try:  # pragma: no cover - exercised once project dependencies are installed.
    from langchain.agents.middleware import AgentMiddleware, AgentState
    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except (ImportError, ModuleNotFoundError):  # pragma: no cover - covered by tests in this repo.
    AgentState = dict[str, Any]
    ModelResponse = Any

    class AgentMiddleware:
        """Fallback stand-in for import-time tests."""

    @dataclass(frozen=True)
    class BaseMessage:
        content: Any
        type: str = "message"
        additional_kwargs: dict[str, Any] | None = None

    @dataclass(frozen=True)
    class HumanMessage(BaseMessage):
        type: str = "human"

    @dataclass(frozen=True)
    class AIMessage(BaseMessage):
        type: str = "ai"

    @dataclass(frozen=True)
    class SystemMessage(BaseMessage):
        type: str = "system"

        @property
        def content_blocks(self) -> list[dict[str, Any]]:
            if isinstance(self.content, list):
                return list(self.content)
            return [{"type": "text", "text": str(self.content)}]

    @dataclass(frozen=True)
    class ModelRequest:
        system_message: SystemMessage
        messages: list[BaseMessage]
        runtime: Any = None
        state: dict[str, Any] | None = None
        tools: list[Any] | None = None
        model: Any = None

        def override(self, **changes: Any) -> "ModelRequest":
            return replace(self, **changes)


class ModelHandler(Protocol):
    def __call__(self, request: ModelRequest) -> ModelResponse: ...


def message_role(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    if role == "ai":
        return "assistant"
    if role == "human":
        return "user"
    return str(role or "message")


def make_message(role: str, content: Any) -> BaseMessage:
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return BaseMessage(content=content, type=role)


def system_message_with_appended_text(message: SystemMessage, text: str) -> SystemMessage:
    blocks = list(getattr(message, "content_blocks", [{"type": "text", "text": str(message.content)}]))
    blocks.append({"type": "text", "text": text})
    return SystemMessage(content=blocks)


def reset_messages_update(messages: list[BaseMessage]) -> list[Any]:
    """Build a LangGraph message reset when available, else return messages.

    LangChain agent state uses message reducers, so prepending restored history
    is safest when we first remove the current message channel and then write
    the intended ordered sequence.
    """

    try:  # pragma: no cover - dependency path.
        from langchain_core.messages import RemoveMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES
    except (ImportError, ModuleNotFoundError):
        return messages
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]

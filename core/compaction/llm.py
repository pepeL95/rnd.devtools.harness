from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel


class TextGenerator(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class LangChainTextGenerator:
    """Thin adapter from the compaction pipeline to any LangChain chat model."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    @classmethod
    def from_chat_model(cls, model: BaseChatModel) -> "LangChainTextGenerator":
        return cls(model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        return _stringify_content(content)


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv


class TextGenerator(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class LangChainTextGenerator:
    """Thin adapter from the compaction pipeline to any LangChain chat model."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def from_model_name(cls, model_name: str) -> "LangChainTextGenerator":
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        from langchain.chat_models import init_chat_model

        return cls(init_chat_model(model_name, temperature=0))

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

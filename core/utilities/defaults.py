from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def get_default_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    model_name = "gemini-3.1-flash-lite"
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        retries=3,
        include_thoughts=True,
        thinking_level="low",
    )


def get_default_compaction_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    model_name = "gemini-3.1-flash-lite"
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        retries=1,
        include_thoughts=False,
        thinking_level="minimal",
        timeout=30,
    )


def get_model_name(model: BaseChatModel) -> str:
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if name:
        return str(name)
    return model.__class__.__name__

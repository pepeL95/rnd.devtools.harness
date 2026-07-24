from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_DRIVER_MODEL_NAME = "gemini-3.1-flash-lite"


def create_driver_model(model_name: str) -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        retries=3,
        include_thoughts=True,
        thinking_level="low",
    )


def get_default_driver_model() -> BaseChatModel:
    return create_driver_model(DEFAULT_DRIVER_MODEL_NAME)

def get_model_name(model: BaseChatModel) -> str:
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if name:
        return str(name)
    return model.__class__.__name__

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def get_default_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return ChatGoogleGenerativeAI(
        model=_default_google_model_name(),
        temperature=0,
        retries=int(os.getenv("QUASIPILOT_MODEL_RETRIES", "0")),
    )


def get_model_name(model: BaseChatModel) -> str:
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if name:
        return str(name)
    return model.__class__.__name__


def _default_google_model_name() -> str:
    configured = os.getenv("QUASIPILOT_GOOGLE_MODEL") or os.getenv("QUASIPILOT_DRIVER_MODEL")
    if not configured:
        return "gemini-3.5-flash"
    return configured.removeprefix("google_genai:")

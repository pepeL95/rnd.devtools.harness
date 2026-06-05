from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def get_default_model() -> BaseChatModel:
    return get_default_compactor_model()

def get_default_driver_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        retries=3,
        include_thoughts=True,
        thinking_level="low",
    )


def get_default_task_extractor_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        retries=3,
        include_thoughts=False,
        thinking_level="minimal",
        timeout=30,
    )


def get_default_compactor_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        retries=3,
        include_thoughts=False,
        thinking_level="minimal",
        timeout=30,
    )


def get_default_critic_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        retries=3,
        include_thoughts=False,
        thinking_level="minimal",
        timeout=30,
    )


def get_default_trajectory_compactor_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        retries=3,
        include_thoughts=False,
        thinking_level="minimal",
        timeout=30,
    )

def get_model_name(model: BaseChatModel) -> str:
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if name:
        return str(name)
    return model.__class__.__name__

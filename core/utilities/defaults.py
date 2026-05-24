from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.data._profiles import _PROFILES


def get_default_model() -> BaseChatModel:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    model_name = _default_google_model_name()
    return configure_model_for_reasoning(
        ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            retries=int(os.getenv("QUASIPILOT_MODEL_RETRIES", "0")),
        )
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


def _default_google_reasoning_kwargs(model_name: str) -> dict[str, object]:
    if not _env_flag("QUASIPILOT_REASONING_ENABLED", default=True):
        return {}
    if not _google_model_supports_reasoning(model_name):
        return {}

    kwargs: dict[str, object] = {
        "include_thoughts": _env_flag("QUASIPILOT_INCLUDE_THOUGHTS", default=True),
    }
    if _is_gemini_25_model(model_name):
        kwargs["thinking_budget"] = int(os.getenv("QUASIPILOT_THINKING_BUDGET", "-1"))
    else:
        kwargs["thinking_level"] = os.getenv("QUASIPILOT_THINKING_LEVEL", "low")
    return kwargs


def configure_model_for_reasoning(model: BaseChatModel) -> BaseChatModel:
    if not _env_flag("QUASIPILOT_REASONING_ENABLED", default=True):
        return model

    updates = _reasoning_updates(model)
    if not updates:
        return model

    copier = getattr(model, "model_copy", None)
    if callable(copier):
        return copier(update=updates)
    return model.bind(**updates)


def _reasoning_updates(model: BaseChatModel) -> dict[str, object]:
    field_names = _model_field_names(model)
    if "reasoning" in field_names and getattr(model, "reasoning", None) is None:
        return {
            "reasoning": {
                "effort": os.getenv("QUASIPILOT_REASONING_EFFORT", "low"),
                "summary": os.getenv("QUASIPILOT_REASONING_SUMMARY", "auto"),
            }
        }

    if "thinking" in field_names and getattr(model, "thinking", None) is None:
        budget = int(os.getenv("QUASIPILOT_REASONING_BUDGET_TOKENS", "2000"))
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    model_name = getattr(model, "model", None) or getattr(model, "model_name", None) or ""
    if {"thinking_level", "include_thoughts"} & field_names:
        return _google_reasoning_updates(model, str(model_name))

    return {}


def _google_reasoning_updates(model: BaseChatModel, model_name: str) -> dict[str, object]:
    if not _google_model_supports_reasoning(model_name):
        return {}

    updates: dict[str, object] = {}
    include_thoughts = getattr(model, "include_thoughts", None)
    if include_thoughts is None:
        updates["include_thoughts"] = _env_flag("QUASIPILOT_INCLUDE_THOUGHTS", default=True)

    if _is_gemini_25_model(model_name):
        if getattr(model, "thinking_budget", None) is None:
            updates["thinking_budget"] = int(
                os.getenv("QUASIPILOT_THINKING_BUDGET", os.getenv("QUASIPILOT_REASONING_BUDGET_TOKENS", "-1"))
            )
    else:
        if getattr(model, "thinking_level", None) is None:
            updates["thinking_level"] = os.getenv(
                "QUASIPILOT_THINKING_LEVEL",
                os.getenv("QUASIPILOT_REASONING_EFFORT", "low"),
            )
    return updates


def _model_field_names(model: BaseChatModel) -> set[str]:
    model_fields = getattr(type(model), "model_fields", None)
    if isinstance(model_fields, dict):
        return set(model_fields)
    fields = getattr(type(model), "__fields__", None)
    if isinstance(fields, dict):
        return set(fields)
    return set()


def _google_model_supports_reasoning(model_name: str) -> bool:
    profile = _PROFILES.get(model_name)
    if profile is not None:
        return bool(profile.get("reasoning_output", False))
    normalized = model_name.lower()
    return normalized.startswith("gemini-2.5") or normalized.startswith("gemini-3")


def _is_gemini_25_model(model_name: str) -> bool:
    return model_name.lower().startswith("gemini-2.5")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

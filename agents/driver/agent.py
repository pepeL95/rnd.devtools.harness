from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.driver.prompt import DRIVER_SYSTEM_PROMPT
from core.compaction.compactor import Compactor
from core.compaction.policy import CompactionPolicy
from core.middleware.compaction import CompactionMiddleware
from core.middleware.runtime import RuntimeContextMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.session_load import SessionLoadMiddleware
from core.middleware.system_prompt import SystemPromptMiddleware
from core.middleware.telemetry import TelemetryMiddleware
from core.session.session_manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path


@dataclass(frozen=True)
class DriverAgentConfig:
    cwd: Path
    model: str = field(default_factory=lambda: os.getenv("QUASIPILOT_DRIVER_MODEL", "google_genai:gemini-3.5-flash"))
    model_retries: int = field(default_factory=lambda: int(os.getenv("QUASIPILOT_MODEL_RETRIES", "0")))
    session_id: str | None = None


def create_driver_agent(config: DriverAgentConfig) -> Any:
    """Create the first coding driver agent.

    This uses LangChain's `create_agent` directly and attaches Deep Agents'
    filesystem middleware with a local shell backend, per the specs.
    """

    from deepagents.backends import LocalShellBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.agents import create_agent

    cwd = config.cwd.expanduser().resolve()
    manager = SessionManager(session_id=config.session_id)
    backend = _local_shell_backend(LocalShellBackend, cwd)
    model = _init_driver_model(config.model, retries=config.model_retries)
    middleware = [
        # Middleware order is load-bearing. LangChain runs before_* hooks
        # first-to-last, after_* hooks last-to-first, and wrap hooks as nested
        # wrappers. Telemetry is outermost; compaction runs before history load
        # to prevent oversized restored context, and after SessionDump writes
        # the completed turn so curated history can be compacted immediately.
        # Source: https://docs.langchain.com/oss/python/langchain/middleware/custom
        TelemetryMiddleware(TelemetryStore(telemetry_session_path(manager.session_id))),
        CompactionMiddleware(manager, Compactor(policy=CompactionPolicy())),
        SessionLoadMiddleware(manager),
        SystemPromptMiddleware(prompt=DRIVER_SYSTEM_PROMPT),
        RuntimeContextMiddleware(cwd=cwd),
        FilesystemMiddleware(backend=backend),
        SessionDumpMiddleware(manager),
    ]
    return create_agent(model=model, tools=[], middleware=middleware)


def _init_driver_model(model_name: str, retries: int) -> Any:
    if model_name.startswith("google_genai:"):
        from dotenv import load_dotenv
        from langchain_google_genai import ChatGoogleGenerativeAI

        load_dotenv(dotenv_path=Path.cwd() / ".env")
        return ChatGoogleGenerativeAI(
            model=model_name.removeprefix("google_genai:"),
            temperature=0,
            retries=retries,
        )
    return model_name


def _local_shell_backend(backend_cls: type[Any], cwd: Path) -> Any:
    kwargs: dict[str, Any] = {
        "root_dir": str(cwd),
        "inherit_env": True,
    }
    parameters = inspect.signature(backend_cls).parameters
    if "virtual_env" in parameters:
        kwargs["virtual_env"] = False
    if "virtual_mode" in parameters:
        kwargs["virtual_mode"] = True
    return backend_cls(**kwargs)

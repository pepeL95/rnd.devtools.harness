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
from core.session.session_manager import SessionManager


@dataclass(frozen=True)
class DriverAgentConfig:
    cwd: Path
    model: str = field(default_factory=lambda: os.getenv("QUASIPILOT_DRIVER_MODEL", "google_genai:gemini-3.5-flash"))
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
    middleware = [
        # >>> [note block]
        # before_* hooks run first-to-last, 
        # after_* hooks run last-to-first,
        # wrap hooks nest.
        # SessionDumpMiddleware.after_agent() writes the completed
        # turn before CompactionMiddleware.after_agent() reads curated history
        # Source: https://docs.langchain.com/oss/python/langchain/middleware/custom
        # <<< [note block]
        SessionLoadMiddleware(manager),
        SystemPromptMiddleware(prompt=DRIVER_SYSTEM_PROMPT),
        RuntimeContextMiddleware(cwd=cwd),
        FilesystemMiddleware(backend=backend),
        CompactionMiddleware(manager, Compactor(policy=CompactionPolicy())),
        SessionDumpMiddleware(manager),
    ]
    return create_agent(model=config.model, tools=[], middleware=middleware)


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

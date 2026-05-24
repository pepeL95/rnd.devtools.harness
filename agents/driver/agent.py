from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

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
from core.utilities.defaults import configure_model_for_reasoning, get_default_model

@dataclass(frozen=True)
class DriverAgentConfig:
    cwd: Path
    model: BaseChatModel = field(default_factory=get_default_model)
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
    return create_agent(model=configure_model_for_reasoning(config.model), tools=[], middleware=middleware)


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

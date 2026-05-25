from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agents.driver.prompt import DRIVER_SYSTEM_PROMPT
from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.middleware.compaction import CompactionMiddleware
from core.middleware.reasoning import ReasoningEagerness, ReasoningMiddleware, reasoning_tool
from core.middleware.runtime import RuntimeContextMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.session_load import SessionLoadMiddleware
from core.middleware.system_prompt import SystemPromptMiddleware
from core.middleware.trajectory import TrajectoryCompactionMiddleware
from core.middleware.telemetry import TelemetryMiddleware
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path
from core.trajectory.compactor import TrajectoryCompactor
from core.trajectory.coordinator import TrajectoryCompactionCoordinator
from core.trajectory.policy import TrajectoryCompactionPolicy
from core.utilities.defaults import get_default_model

@dataclass(frozen=True)
class DriverAgentConfig:
    cwd: Path
    model: BaseChatModel = field(default_factory=get_default_model)
    session_id: str | None = None
    session_manager: SessionManager | None = None
    reasoning_eagerness: ReasoningEagerness = "low"
    on_compaction_event: Callable[[str, dict[str, Any]], None] | None = None
    compaction_coordinator: CompactionCoordinator | None = None
    trajectory_compaction_coordinator: TrajectoryCompactionCoordinator | None = None


def create_driver_agent(config: DriverAgentConfig) -> Any:
    """Create the first coding driver agent.

    This uses LangChain's `create_agent` directly and attaches Deep Agents'
    filesystem middleware with a local shell backend, per the specs.
    """

    from deepagents.backends import LocalShellBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.agents import create_agent

    cwd = config.cwd.expanduser().resolve()
    manager = config.session_manager or SessionManager(session_id=config.session_id)
    backend = _local_shell_backend(LocalShellBackend, cwd)
    coordinator = config.compaction_coordinator or CompactionCoordinator(
        manager,
        Compactor(policy=CompactionPolicy()),
        on_compaction_event=config.on_compaction_event,
    )
    trajectory_coordinator = config.trajectory_compaction_coordinator or TrajectoryCompactionCoordinator(
        manager,
        TrajectoryCompactor(policy=TrajectoryCompactionPolicy()),
    )
    middleware = [
        # Middleware order is load-bearing. LangChain runs before_* hooks
        # first-to-last, after_* hooks last-to-first, and wrap hooks as nested
        # wrappers. Telemetry is outermost; the curated-session rewrite middlewares
        # run after SessionDump so they compact a completed turn from a stable
        # snapshot. Trajectory compaction is placed closer to SessionLoad so its
        # after_agent hook runs before the broader compaction middleware.
        # Source: https://docs.langchain.com/oss/python/langchain/middleware/custom
        TelemetryMiddleware(TelemetryStore(telemetry_session_path(manager.session_id))),
        ReasoningMiddleware(eagerness=config.reasoning_eagerness),
        CompactionMiddleware(coordinator),
        TrajectoryCompactionMiddleware(trajectory_coordinator),
        SessionLoadMiddleware(manager),
        SystemPromptMiddleware(prompt=DRIVER_SYSTEM_PROMPT),
        RuntimeContextMiddleware(cwd=cwd),
        FilesystemMiddleware(backend=backend),
        SessionDumpMiddleware(manager),
    ]
    return create_agent(model=config.model, tools=[reasoning_tool], middleware=middleware)


def _local_shell_backend(backend_cls: type[Any], cwd: Path) -> Any:
    kwargs: dict[str, Any] = {
        "root_dir": str(cwd),
        "inherit_env": True,
    }
    parameters = inspect.signature(backend_cls).parameters
    if "virtual_env" in parameters:
        kwargs["virtual_env"] = False
    if "virtual_mode" in parameters:
        kwargs["virtual_mode"] = False
    return backend_cls(**kwargs)

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agents.driver.prompt import DRIVER_SYSTEM_PROMPT
from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.live_steering import LiveSteeringController
from core.middleware.cancellation import CancellationMiddleware
from core.middleware.compaction import CompactionMiddleware
from core.middleware.filesystem import HarnessFilesystemMiddleware
from core.middleware.live_steering import LiveSteeringMiddleware
from core.middleware.reasoning import ReasoningEagerness, ReasoningMiddleware, reasoning_tool
from core.middleware.runtime import RuntimeContextMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.session_load import SessionLoadMiddleware
from core.middleware.skills import SkillsMiddleware, create_read_skill_tool
from core.middleware.system_prompt import SystemPromptMiddleware
from core.middleware.trajectory import TrajectoryCompactionMiddleware
from core.middleware.telemetry import TelemetryMiddleware
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path
from core.trajectory.compactor import TrajectoryCompactor
from core.trajectory.coordinator import TrajectoryCompactionCoordinator
from core.trajectory.policy import TrajectoryCompactionPolicy
from core.utilities.defaults import get_default_driver_model

@dataclass(frozen=True)
class DriverAgentConfig:
    cwd: Path
    model: BaseChatModel = field(default_factory=get_default_driver_model)
    python_interpreter: Path | None = None
    session_id: str | None = None
    session_manager: SessionManager | None = None
    reasoning_eagerness: ReasoningEagerness = "high"
    on_compaction_event: Callable[[str, dict[str, Any]], None] | None = None
    session_compaction_coordinator: CompactionCoordinator | None = None
    trajectory_compaction_coordinator: TrajectoryCompactionCoordinator | None = None
    live_steering_controller: LiveSteeringController | None = None
    cancel_event: Event | None = None


def create_driver_agent(config: DriverAgentConfig) -> Any:
    """Create the first coding driver agent.

    This uses LangChain's `create_agent` directly and attaches Deep Agents'
    filesystem middleware with a local shell backend, per the specs.
    """

    from deepagents.backends import LocalShellBackend
    from langchain.agents import create_agent

    cwd = config.cwd.expanduser().resolve()
    manager = config.session_manager or SessionManager(session_id=config.session_id)
    telemetry_store = TelemetryStore(telemetry_session_path(manager.session_id))
    backend = _local_shell_backend(LocalShellBackend, cwd)
    session_compaction_coordinator = config.session_compaction_coordinator or CompactionCoordinator(
        manager,
        Compactor(policy=CompactionPolicy()),
        on_compaction_event=config.on_compaction_event,
        telemetry_store=telemetry_store,
    )
    trajectory_compaction_coordinator = config.trajectory_compaction_coordinator or TrajectoryCompactionCoordinator(
        manager,
        TrajectoryCompactor(policy=TrajectoryCompactionPolicy()),
        telemetry_store=telemetry_store,
    )
    live_steering_controller = config.live_steering_controller or LiveSteeringController()
    session_dump = SessionDumpMiddleware(manager, python_interpreter=config.python_interpreter)
    middleware = [
        # Middleware order is load-bearing. LangChain runs before_* hooks
        # first-to-last, after_* hooks last-to-first, and wrap hooks as nested
        # wrappers. Telemetry is outermost; the curated-session rewrite middlewares
        # run after SessionDump so they compact a completed turn from a stable
        # snapshot. Trajectory compaction is placed closer to SessionLoad so its
        # after_agent hook runs before the broader compaction middleware.
        # Source: https://docs.langchain.com/oss/python/langchain/middleware/custom
        TelemetryMiddleware(telemetry_store),
        ReasoningMiddleware(eagerness=config.reasoning_eagerness),
        CompactionMiddleware(session_compaction_coordinator),
        SessionLoadMiddleware(manager, session_dump=session_dump),
        SystemPromptMiddleware(prompt=DRIVER_SYSTEM_PROMPT),
        SkillsMiddleware(cwd=cwd),
        RuntimeContextMiddleware(cwd=cwd, python_interpreter=config.python_interpreter),
        HarnessFilesystemMiddleware(backend=backend),
        session_dump,
        LiveSteeringMiddleware(live_steering_controller),
        *([CancellationMiddleware(config.cancel_event)] if config.cancel_event is not None else []),
        TrajectoryCompactionMiddleware(trajectory_compaction_coordinator),
    ]
    return create_agent(
        model=config.model,
        tools=[reasoning_tool, create_read_skill_tool(cwd=cwd)],
        middleware=middleware,
    )


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

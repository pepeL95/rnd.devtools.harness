"""LangChain middleware for the harness."""

from importlib import import_module
from typing import Any

__all__ = [
    "CompactionMiddleware",
    "ReasoningMiddleware",
    "RuntimeContextMiddleware",
    "SessionDumpMiddleware",
    "SessionLoadMiddleware",
    "SystemPromptMiddleware",
    "TrajectoryCompactionMiddleware",
]

_EXPORTS = {
    "CompactionMiddleware": "core.middleware.compaction",
    "ReasoningMiddleware": "core.middleware.reasoning",
    "RuntimeContextMiddleware": "core.middleware.runtime",
    "SessionDumpMiddleware": "core.middleware.session_dump",
    "SessionLoadMiddleware": "core.middleware.session_load",
    "SystemPromptMiddleware": "core.middleware.system_prompt",
    "TrajectoryCompactionMiddleware": "core.middleware.trajectory_compaction",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    return getattr(module, name)

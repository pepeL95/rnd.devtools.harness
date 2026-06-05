"""LangChain middleware for the harness."""

from importlib import import_module
from typing import Any

__all__ = [
    "CompactionMiddleware",
    "ReasoningMiddleware",
    "RuntimeContextMiddleware",
    "SessionDumpMiddleware",
    "SessionLoadMiddleware",
    "SkillsMiddleware",
    "SystemPromptMiddleware",
    "TrajectoryCompactionMiddleware",
]

_EXPORTS = {
    "CompactionMiddleware": "core.middleware.compaction",
    "ReasoningMiddleware": "core.middleware.reasoning",
    "RuntimeContextMiddleware": "core.middleware.runtime",
    "SessionDumpMiddleware": "core.middleware.session_dump",
    "SessionLoadMiddleware": "core.middleware.session_load",
    "SkillsMiddleware": "core.middleware.skills",
    "SystemPromptMiddleware": "core.middleware.system_prompt",
    "TrajectoryCompactionMiddleware": "core.middleware.trajectory",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    return getattr(module, name)

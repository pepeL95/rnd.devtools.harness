from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from core.utilities.messages import system_message_with_appended_text
from core.session.events import RuntimeSnapshot
from core.utilities.git import git_branch, git_dirty


class RuntimeContextMiddleware(AgentMiddleware):
    """Probe and inject runtime context before each model call."""

    def __init__(
        self,
        cwd: str | Path | None = None,
        python_interpreter: str | Path | None = None,
    ) -> None:
        self.cwd = Path(cwd).expanduser().resolve() if cwd else None
        self.python_interpreter = (
            Path(python_interpreter).expanduser().resolve() if python_interpreter else None
        )

    def snapshot(self, runtime: Any = None) -> RuntimeSnapshot:
        cwd = self._runtime_cwd(runtime) or self.cwd or Path.cwd()
        cwd = Path(cwd).expanduser().resolve()
        return RuntimeSnapshot(
            cwd=str(cwd),
            git_branch=git_branch(cwd),
            git_dirty=git_dirty(cwd),
            python_interpreter=str(self.python_interpreter) if self.python_interpreter else None,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        snapshot = self.snapshot(getattr(request, "runtime", None))
        updated = request.override(
            system_message=system_message_with_appended_text(
                request.system_message,
                snapshot.to_prompt_block(),
            )
        )
        return handler(updated)

    @staticmethod
    def _runtime_cwd(runtime: Any) -> str | Path | None:
        context = getattr(runtime, "context", None)
        return getattr(context, "cwd", None) or (context.get("cwd") if isinstance(context, dict) else None)

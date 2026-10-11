from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from core.utilities.messages import system_message_with_appended_text


class AgentsMdMiddleware(AgentMiddleware):
    """Append only the workspace's AGENTS.md, reloading on each model call."""

    def __init__(self, cwd: str | Path) -> None:
        self.path = Path(cwd).expanduser().resolve() / "AGENTS.md"

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        try:
            content = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return request
        if not content.strip():
            return request
        return request.override(
            system_message=system_message_with_appended_text(request.system_message, content)
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        updated = await asyncio.to_thread(self._prepare_request, request)
        return await handler(updated)

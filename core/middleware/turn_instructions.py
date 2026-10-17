from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from core.session.manager import SessionManager
from core.utilities.messages import system_message_with_appended_text

if TYPE_CHECKING:
    from core.middleware.session_dump import SessionDumpMiddleware


class TurnInstructionsMiddleware(AgentMiddleware):
    """Share snapshot, persistence, and injection for file-backed instructions."""

    def __init__(
        self, *, kind: str, loader: Callable[[], tuple[str, str]],
        manager: SessionManager | None = None,
        session_dump: SessionDumpMiddleware | None = None,
    ) -> None:
        self.kind = kind
        self._loader = loader
        self.manager = manager
        self.session_dump = session_dump
        self._content = ""
        self._source = ""
        self._pending = False

    def before_agent(self, state: Any, runtime: Any) -> None:
        if self.session_dump is not None and self.session_dump.is_interrupted:
            return
        content, source = self._loader()
        self._content = content if content.strip() else ""
        self._source = source
        self._pending = True

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        await asyncio.to_thread(self.before_agent, state, runtime)

    def before_model(self, state: Any, runtime: Any) -> None:
        # SessionDump has opened the turn by now; do not allocate another turn.
        if self._pending and self.manager is not None:
            turn = self.session_dump.active_turn if self.session_dump is not None else None
            self.manager.sync_system_instruction(
                self._content, kind=self.kind, source=self._source,
                turn=turn or self.manager.next_turn(),
            )
        self._pending = False

    async def abefore_model(self, state: Any, runtime: Any) -> None:
        await asyncio.to_thread(self.before_model, state, runtime)

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        if not self._content:
            return request
        return request.override(
            system_message=system_message_with_appended_text(request.system_message, self._content)
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._prepare_request(request))

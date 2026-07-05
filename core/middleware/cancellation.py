from __future__ import annotations

from threading import Event
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from core.live_steering import CancellationInterrupt


class CancellationMiddleware(AgentMiddleware):
    """Abort the active turn at the next tool boundary when cancel_event is set."""

    def __init__(self, cancel_event: Event) -> None:
        self.cancel_event = cancel_event

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        if self.cancel_event.is_set():
            raise CancellationInterrupt()
        return handler(request)

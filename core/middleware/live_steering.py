from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware

from core.live_steering import LiveSteeringController, LiveSteeringInterrupt


class LiveSteeringMiddleware(AgentMiddleware):
    """Interrupt the active turn when queued steering reaches a tool boundary."""

    def __init__(self, controller: LiveSteeringController) -> None:
        self.controller = controller

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        steering = self.controller.drain()
        if steering:
            raise LiveSteeringInterrupt(steering)
        return handler(request)

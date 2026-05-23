from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from core.telemetry.events import TelemetryEvent
from core.telemetry.store import TelemetryStore


class TelemetryMiddleware(AgentMiddleware):
    """Record lightweight lifecycle telemetry without owning session history."""

    def __init__(self, store: TelemetryStore) -> None:
        self.store = store

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self.store.record(TelemetryEvent(name="agent.start", payload={"messages": len(state.get("messages", []))}))
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self.store.record(TelemetryEvent(name="agent.end", payload={"messages": len(state.get("messages", []))}))
        return None

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> ModelResponse:
        self.store.record(
            TelemetryEvent(
                name="model.start",
                payload={
                    "messages": len(getattr(request, "messages", []) or []),
                    "model": _model_name(getattr(request, "model", None)),
                },
            )
        )
        try:
            response = handler(request)
        except Exception as exc:
            self.store.record(
                TelemetryEvent(
                    name="model.error",
                    payload={
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
            )
            raise
        self.store.record(TelemetryEvent(name="model.end", payload={}))
        return response


def _model_name(model: Any) -> str:
    if model is None:
        return ""
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if name:
        return str(name)
    return model.__class__.__name__

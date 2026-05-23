from __future__ import annotations

from typing import Any

from core.middleware._compat import AgentMiddleware, AgentState
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


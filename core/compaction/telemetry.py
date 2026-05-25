from __future__ import annotations

from typing import Any

from core.telemetry.events import TelemetryEvent
from core.telemetry.store import TelemetryStore


class CompactionTelemetry:
    """Record compact, lifecycle-oriented compaction telemetry."""

    def __init__(self, store: TelemetryStore) -> None:
        self.store = store

    def start(
        self,
        *,
        trigger: str,
        reason: str,
        estimated_tokens: int,
        event_count: int,
        latest_turn: int,
    ) -> None:
        self._record(
            "compaction.start",
            {
                "trigger": trigger,
                "reason": reason,
                "estimated_tokens": estimated_tokens,
                "event_count": event_count,
                "latest_turn": latest_turn,
            },
        )

    def skip(
        self,
        *,
        trigger: str,
        reason: str,
        event_count: int,
        latest_turn: int,
        estimated_tokens: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "trigger": trigger,
            "reason": reason,
            "event_count": event_count,
            "latest_turn": latest_turn,
        }
        if estimated_tokens is not None:
            payload["estimated_tokens"] = estimated_tokens
        self._record("compaction.skip", payload)

    def end(
        self,
        *,
        trigger: str,
        estimated_tokens: int,
        compacted_event_count: int,
        retained_event_count: int,
        curated_event_count: int,
        revisions: int,
        duration_ms: int,
    ) -> None:
        self._record(
            "compaction.end",
            {
                "trigger": trigger,
                "estimated_tokens": estimated_tokens,
                "compacted_event_count": compacted_event_count,
                "retained_event_count": retained_event_count,
                "curated_event_count": curated_event_count,
                "revisions": revisions,
                "duration_ms": duration_ms,
            },
        )

    def error(
        self,
        *,
        trigger: str,
        estimated_tokens: int,
        error: str,
        exception_type: str,
    ) -> None:
        self._record(
            "compaction.error",
            {
                "trigger": trigger,
                "estimated_tokens": estimated_tokens,
                "error": error,
                "exception_type": exception_type,
            },
        )

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        self.store.record(TelemetryEvent(name=name, payload=payload))

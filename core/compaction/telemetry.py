from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

from core.compaction.token_counter import TokenCounter
from core.session.events import SessionEvent
from core.telemetry.events import TelemetryEvent
from core.telemetry.store import TelemetryStore


class CompactionTelemetry:
    """Record normalized lifecycle telemetry and logs for compaction flows."""

    def __init__(
        self,
        *,
        kind: str,
        store: TelemetryStore,
        logger: logging.Logger | None = None,
    ) -> None:
        self.kind = kind
        self.store = store
        self.logger = logger

    def start(self, payload: dict[str, Any]) -> None:
        self._record("start", payload)

    def skip(self, payload: dict[str, Any]) -> None:
        self._record("skip", payload)

    def end(self, payload: dict[str, Any]) -> None:
        self._record("end", payload)

    def error(self, payload: dict[str, Any]) -> None:
        self._record("error", payload)

    def _record(self, status: str, payload: dict[str, Any]) -> None:
        normalized = {"kind": self.kind, "status": status, **payload}
        self.store.record(TelemetryEvent(name=f"compaction.{status}", payload=normalized))
        if self.logger is not None:
            level = logging.ERROR if status == "error" else logging.INFO
            self.logger.log(level, json.dumps(normalized, ensure_ascii=False, sort_keys=True))


def context_token_breakdown(
    source: list[SessionEvent], result: list[SessionEvent], counter: TokenCounter, *, completed: bool = True,
) -> dict[str, Any]:
    """Compare replaced and inserted message content within the compaction snapshot."""
    source_ids = {event.id for event in source}
    result_ids = {event.id for event in result}
    removed = [event for event in source if event.id not in result_ids]
    retained = [event for event in source if event.id in result_ids]
    compacted_tokens = counter.count_context(removed)
    retained_tokens = counter.count_context(retained)
    breakdown = {
        "estimated": True,
        "basis": "restored_message_content",
        "encoding": counter.encoding_name,
        "fallback": "characters_divided_by_4",
        "compacted_tokens": compacted_tokens,
        "retained_tokens": retained_tokens,
        "before_tokens": compacted_tokens + retained_tokens,
    }
    if completed:
        injected = [event for event in result if event.id not in source_ids]
        injected_tokens = counter.count_context(injected)
        breakdown.update(
            injected_tokens=injected_tokens,
            after_tokens=retained_tokens + injected_tokens,
            saved_tokens=compacted_tokens - injected_tokens,
        )
    return breakdown


def compaction_logger(repo_root: Path, *, name: str, log_dir: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_filename = datetime.datetime.now().strftime("%m-%d-%Y.log")
    log_path = (repo_root / ".logs" / log_dir / log_filename).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path:
            return logger

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def compaction_time_span(events: list[SessionEvent]) -> dict[str, Any]:
    timestamps = [_parse_timestamp(event.timestamp) for event in events]
    resolved = [timestamp for timestamp in timestamps if timestamp is not None]
    if not resolved:
        return {
            "time_span_start": None,
            "time_span_end": None,
            "time_span_seconds": None,
        }
    start = min(resolved)
    end = max(resolved)
    return {
        "time_span_start": start.isoformat(),
        "time_span_end": end.isoformat(),
        "time_span_seconds": int((end - start).total_seconds()),
    }


def _parse_timestamp(value: str) -> datetime.datetime | None:
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)

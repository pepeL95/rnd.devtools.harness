from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.telemetry.events import TelemetryEvent


def default_telemetry_root() -> Path:
    return Path.home() / ".quasipilot" / "telemetry"


def telemetry_session_path(
    session_id: str,
    root: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Return the per-session telemetry path.

    The directory name intentionally follows the current product spec spelling:
    ~/.quasipilot/telemetry/[yyyy]/[mm]/[dd]/[session-id].jsonl.
    """

    instant = timestamp or datetime.now(timezone.utc)
    base = root or default_telemetry_root()
    return base / f"{instant.year:04d}" / f"{instant.month:02d}" / f"{instant.day:02d}" / f"{session_id}.jsonl"


class TelemetryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: TelemetryEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_json_dict(), ensure_ascii=False) + "\n")

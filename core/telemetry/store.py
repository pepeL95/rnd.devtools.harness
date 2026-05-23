from __future__ import annotations

import json
from pathlib import Path

from core.telemetry.events import TelemetryEvent


class TelemetryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: TelemetryEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_json_dict(), ensure_ascii=False) + "\n")


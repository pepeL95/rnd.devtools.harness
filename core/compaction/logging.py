from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompactionLogEntry:
    timestamp: str
    session_id: str
    run_id: str
    phase: str
    status: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "session_id": self.session_id,
                "run_id": self.run_id,
                "phase": self.phase,
                "status": self.status,
                "payload": self.payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


class CompactionLogger:
    """Append structured compaction runtime logs to a session-scoped JSONL file."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def log(self, session_id: str, run_id: str, phase: str, status: str, payload: dict[str, Any]) -> Path:
        entry = CompactionLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            run_id=run_id,
            phase=phase,
            status=status,
            payload=payload,
        )
        path = self.root / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json() + "\n")
        return path

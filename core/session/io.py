from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.session.events import SessionEvent


def default_session_root() -> Path:
    return Path.home() / ".quasipilot" / "sessions"


def session_paths(session_id: str, root: Path | None = None) -> tuple[Path, Path]:
    base = root or default_session_root()
    return base / "dump" / f"{session_id}.jsonl", base / "curated" / f"{session_id}.jsonl"


def append_events(path: Path, events: Iterable[SessionEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_json_dict(), ensure_ascii=False) + "\n")


def read_events(path: Path) -> list[SessionEvent]:
    if not path.exists():
        return []
    events: list[SessionEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                events.append(SessionEvent.from_json_dict(json.loads(stripped)))
    return events


def replace_events(path: Path, events: Iterable[SessionEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_json_dict(), ensure_ascii=False) + "\n")
    temp_path.replace(path)

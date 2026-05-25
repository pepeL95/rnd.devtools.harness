from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from core.session.events import SessionEvent
from core.session.io import read_events


def load_session(path: Path) -> list[SessionEvent]:
    return read_events(path)


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, events: Iterable[SessionEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_json_dict(), ensure_ascii=False) + "\n")


def render_event(event: SessionEvent) -> str:
    kind = event.payload.get("kind")
    prefix = f"turn={event.turn} type={event.type.value}"
    if kind:
        prefix += f" kind={kind}"
    content = str(event.payload.get("content", "")).strip()
    if not content:
        return prefix
    return f"{prefix}\n{content}"


def render_events(events: Iterable[SessionEvent]) -> str:
    return "\n\n---\n\n".join(render_event(event) for event in events)


def render_turn_table(events: Iterable[SessionEvent]) -> str:
    rows = ["| turn | type | kind | preview |", "| --- | --- | --- | --- |"]
    for event in events:
        content = " ".join(str(event.payload.get("content", "")).split())
        preview = content[:96] + ("..." if len(content) > 96 else "")
        rows.append(
            f"| {event.turn} | {event.type.value} | {event.payload.get('kind', '')} | {preview.replace('|', '/')} |"
        )
    return "\n".join(rows)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cli.utilities.display import content_to_plaintext
from core.session.events import EventType
from core.session.io import default_session_root, read_events, session_paths


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    path: Path
    modified_at: datetime
    preview: str


def _latest_user_preview(events: list) -> str:
    for event in reversed(events):
        if event.type == EventType.USER:
            text = content_to_plaintext(event.payload.get("content", ""))
            return _truncate(text)
    return ""


def _truncate(text: str, limit: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def list_sessions(root: Path | None = None) -> list[SessionSummary]:
    base = root or default_session_root()
    curated_dir = base / "curated"
    if not curated_dir.exists():
        return []

    summaries: list[SessionSummary] = []
    for path in curated_dir.glob("*.jsonl"):
        stat = path.stat()
        events = read_events(path)
        summaries.append(
            SessionSummary(
                session_id=path.stem,
                path=path,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                preview=_latest_user_preview(events),
            )
        )
    summaries.sort(key=lambda item: item.modified_at, reverse=True)
    return summaries


def clear_session_files(session_id: str, root: Path | None = None) -> None:
    dump_path, curated_path = session_paths(session_id, root)
    for path in (dump_path, curated_path):
        if path.exists():
            path.unlink()

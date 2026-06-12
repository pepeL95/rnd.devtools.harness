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
    cwd_matches: bool = False


def _first_user_preview(events: list) -> str:
    for event in events:
        if event.type == EventType.USER:
            text = content_to_plaintext(event.payload.get("content", ""))
            return _truncate(text)
    return ""


def _truncate(text: str, limit: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _latest_session_cwd(events: list) -> str:
    for event in reversed(events):
        if event.type != EventType.RUNTIME:
            continue
        cwd = event.payload.get("cwd")
        if cwd:
            return str(cwd)
    return ""


def _normalized_path(path: Path | str | None) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve())


def list_sessions(root: Path | None = None, current_cwd: Path | None = None) -> list[SessionSummary]:
    base = root or default_session_root()
    curated_dir = base / "curated"
    if not curated_dir.exists():
        return []

    normalized_current_cwd = _normalized_path(current_cwd)
    summaries: list[SessionSummary] = []
    for path in curated_dir.glob("*.jsonl"):
        stat = path.stat()
        dump_path, _ = session_paths(path.stem, root)
        events = read_events(dump_path)
        session_cwd = _normalized_path(_latest_session_cwd(events))
        summaries.append(
            SessionSummary(
                session_id=path.stem,
                path=path,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                preview=_first_user_preview(events),
                cwd_matches=bool(normalized_current_cwd and session_cwd == normalized_current_cwd),
            )
        )
    summaries.sort(key=lambda item: (not item.cwd_matches, -item.modified_at.timestamp()))
    return summaries


def clear_session_files(session_id: str, root: Path | None = None) -> None:
    dump_path, curated_path = session_paths(session_id, root)
    for path in (dump_path, curated_path):
        if path.exists():
            path.unlink()

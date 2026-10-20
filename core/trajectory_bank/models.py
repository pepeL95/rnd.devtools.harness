from __future__ import annotations

from dataclasses import dataclass
from core.compaction.models import ArcUpdate, TaskArc  # Compatibility for installed curation skills.


@dataclass(frozen=True)
class TrajectoryRecord:
    id: str
    document: str
    metadata: dict


@dataclass(frozen=True)
class Hit:
    record: TrajectoryRecord
    similarity: float | None = None


def arc_ref(session_id: str, arc_id: str) -> str:
    return f"trajectory://{session_id}/{arc_id}"


def record_ref(record_id: str) -> str:
    return f"trajectory-record://{record_id}"

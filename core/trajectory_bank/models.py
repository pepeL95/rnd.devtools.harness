from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskArc(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    arc_id: str = ""
    title: str = Field(min_length=1)
    status: Literal["active", "completed", "blocked", "abandoned", "superseded"]
    intent: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    findings: str = Field(min_length=1)
    constraints: str = Field(min_length=1)
    validation: str = Field(min_length=1)
    remaining: str = Field(min_length=1)
    turns: list[int] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list, max_length=12)


class ArcUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    narrative: str = Field(min_length=1)
    current_state: str = Field(min_length=1)
    arcs: list[TaskArc]


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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.session.events import SessionEvent


@dataclass(frozen=True)
class CompactionWindow:
    compacted: list[SessionEvent]
    retained: list[SessionEvent]

    @property
    def should_compact(self) -> bool:
        return bool(self.compacted)


@dataclass(frozen=True)
class Critique:
    text: str

    @property
    def approved(self) -> bool:
        return self.text.strip() == "APPROVE"


@dataclass(frozen=True)
class CompactionResult:
    events: list[SessionEvent]
    memory_document: str
    segmentation: str  # Raw arc-update JSON retained for evaluation tooling.
    critiques: list[Critique] = field(default_factory=list)
    model_names: dict[str, str] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    revisions: int = 0
    token_estimate_before: int = 0
    compacted_event_count: int = 0
    retained_event_count: int = 0


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

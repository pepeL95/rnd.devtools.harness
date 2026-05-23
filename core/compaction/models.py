from __future__ import annotations

from dataclasses import dataclass, field

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
        normalized = self.text.upper()
        return (
            "RECOMMENDED ACTION: APPROVE AS-IS" in normalized
            or "RECOMMENDED ACTION: APPROVE" in normalized
            or "VIOLATIONS FOUND: 0" in normalized
        )


@dataclass(frozen=True)
class CompactionResult:
    events: list[SessionEvent]
    memory_document: str
    segmentation: str
    critiques: list[Critique] = field(default_factory=list)
    revisions: int = 0
    token_estimate_before: int = 0
    compacted_event_count: int = 0
    retained_event_count: int = 0


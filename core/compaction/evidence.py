from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from core.compaction.models import CompactionResult
from core.session.events import SessionEvent


@dataclass(frozen=True)
class PreparedEvidence:
    """References prepared before synthesis, persisted before session publication."""

    manifest: dict[str, list[str]]
    episode_references: list[str]
    persist: Callable[[CompactionResult], None]
    references: list[str] = field(default_factory=list)


class CompactionEvidenceProvider(Protocol):
    """Optional storage boundary; session-arc generation owns all model calls."""

    def validate(self, manifest: dict[str, list[str]]) -> None: ...

    def prepare(
        self, events: list[SessionEvent], turns: list[int], episodes: list[str],
        manifest: dict[str, list[str]],
    ) -> PreparedEvidence: ...

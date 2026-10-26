from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.compaction.models import CompactionResult


@dataclass(frozen=True)
class PreparedEvidence:
    """References prepared before synthesis, persisted before session publication."""

    manifest: dict[str, list[str]]
    episode_references: list[str]
    persist: Callable[[CompactionResult], None]
    references: list[str] = field(default_factory=list)

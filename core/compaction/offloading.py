from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from core.compaction.evidence import PreparedEvidence
from core.compaction.models import CompactionResult
from core.session.events import SessionEvent
from core.session.manager import SessionManager


class BaseCompactionOffloadingStrategy(ABC):
    """Storage lifecycle hooks; strategies must not mutate source inputs or arc results."""

    def validate(self, manifest: dict[str, list[str]]) -> None:
        """Optionally verify published references before any model calls."""

    @abstractmethod
    def prepare(
        self, events: list[SessionEvent], turns: list[int], episodes: list[str],
        manifest: dict[str, list[str]],
    ) -> PreparedEvidence:
        """Prepare references and a persistence callback; do not write yet."""


class CompactionOffloadingStack(BaseCompactionOffloadingStrategy):
    """Compose independent destinations, persisting in order before publication."""

    def __init__(self, strategies: Sequence[BaseCompactionOffloadingStrategy]) -> None:
        self.strategies = tuple(strategies)

    def validate(self, manifest: dict[str, list[str]]) -> None:
        for strategy in self.strategies:
            strategy.validate({group: list(ids) for group, ids in manifest.items()})

    def prepare(
        self, events: list[SessionEvent], turns: list[int], episodes: list[str],
        manifest: dict[str, list[str]],
    ) -> PreparedEvidence:
        prepared = []
        merged = {group: list(ids) for group, ids in manifest.items()}
        for strategy in self.strategies:
            evidence = strategy.prepare(
                events, turns, episodes, {group: list(ids) for group, ids in manifest.items()},
            )
            if len(evidence.episode_references) != len(episodes):
                raise ValueError(f"{type(strategy).__name__} must supply one reference per episode")
            prepared.append(evidence)
            for group, ids in evidence.manifest.items():
                merged[group] = list(dict.fromkeys([*merged.get(group, []), *ids]))

        def persist(result: CompactionResult) -> None:
            for evidence in prepared:
                evidence.persist(result)

        return PreparedEvidence(
            manifest=merged,
            episode_references=[
                "\n".join(dict.fromkeys(item.episode_references[index] for item in prepared
                                       if item.episode_references[index]))
                for index in range(len(episodes))
            ],
            persist=persist,
            references=list(dict.fromkeys(reference for item in prepared for reference in item.references)),
        )


def build_compaction_offloading(
    modes: str | Sequence[str], manager: SessionManager,
) -> list[BaseCompactionOffloadingStrategy]:
    """Resolve configured destinations in one place for the CLI and Python driver."""
    from core.compaction.lfs import LfsCompactionOffloadingStrategy

    if isinstance(modes, str):
        modes = [] if modes == "none" else [modes]
    factories = {"lfs": LfsCompactionOffloadingStrategy}
    strategies = []
    for name in modes:
        if name not in factories:
            raise ValueError(f"Unknown compaction offloading strategy: {name}")
        strategies.append(factories[name](manager))
    return strategies

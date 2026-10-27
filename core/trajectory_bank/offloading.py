from __future__ import annotations

from core.compaction.evidence import PreparedEvidence
from core.compaction.offloading import BaseCompactionOffloadingStrategy, CompactionOffloadingStack
from core.session.events import SessionEvent
from core.session.manager import SessionManager
from core.trajectory_bank.archival import prepare_trajectories
from core.trajectory_bank.store import TrajectoryBank


class ChromaCompactionOffloadingStrategy(BaseCompactionOffloadingStrategy):
    """Archive extracted task episodes without owning synthesis or agent retrieval."""

    def __init__(self, bank: TrajectoryBank, manager: SessionManager) -> None:
        self.bank = bank
        self.manager = manager

    def validate(self, manifest: dict[str, list[str]]) -> None:
        ids = {id for records in manifest.values() for id in records}
        if ids and {record.id for record in self.bank.get(ids=sorted(ids))} != ids:
            raise ValueError("Previously archived trajectory references are missing; refusing to compact")

    def prepare(
        self, events: list[SessionEvent], turns: list[int], episodes: list[str],
        manifest: dict[str, list[str]],
    ) -> PreparedEvidence:
        records = prepare_trajectories(self.manager, events, turns, episodes)
        updated = {group: list(ids) for group, ids in manifest.items()}
        for record in records:
            group = record.metadata["arc_id"]
            updated[group] = list(dict.fromkeys([*updated.get(group, []), record.id]))
        return PreparedEvidence(
            manifest=updated,
            episode_references=[f"trajectory-record://{record.id}" for record in records],
            persist=lambda result: self.bank.archive(records),
        )


def find_bank_strategy(
    strategy: BaseCompactionOffloadingStrategy | None,
) -> ChromaCompactionOffloadingStrategy | None:
    """Find the driver's retrieval store, including inside a supplied strategy stack."""
    if isinstance(strategy, ChromaCompactionOffloadingStrategy):
        return strategy
    if isinstance(strategy, CompactionOffloadingStack):
        matches = [match for child in strategy.strategies if (match := find_bank_strategy(child)) is not None]
        if len(matches) > 1:
            raise ValueError("Driver retrieval requires a single trajectory-bank destination")
        return matches[0] if matches else None
    return None

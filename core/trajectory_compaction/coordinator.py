from __future__ import annotations

from threading import Lock, Thread

from core.session.session_manager import CuratedCompactionLease, SessionManager
from core.trajectory_compaction.compactor import TrajectoryCompactor


class TrajectoryCompactionCoordinator:
    """Run trajectory compaction in the background against the curated session stream."""

    def __init__(self, manager: SessionManager, compactor: TrajectoryCompactor) -> None:
        self.manager = manager
        self.compactor = compactor
        self._mutex = Lock()
        self._worker: Thread | None = None

    def is_running(self) -> bool:
        with self._mutex:
            worker = self._worker
        return (worker is not None and worker.is_alive()) or self.manager.is_curated_locked()

    def request_compaction(self) -> str:
        with self._mutex:
            worker = self._worker
            if worker is not None and worker.is_alive():
                return "running"
            snapshot = self.manager.read_curated(include_pending=True)
            if not self.compactor.should_compact(snapshot):
                return "not_needed"
            lease = self.manager.begin_curated_compaction(
                trigger="trajectory",
                metadata={"reason": "turn_interval"},
            )
            if lease is None:
                return "running"
            self._worker = Thread(
                target=self._run_compaction,
                args=(lease,),
                name=f"trajectory-compaction-{self.manager.session_id}",
                daemon=True,
            )
            self._worker.start()
            return "started"

    def _run_compaction(self, lease: CuratedCompactionLease) -> None:
        try:
            result = self.compactor.compact(lease.snapshot_events)
            self.manager.finalize_curated_compaction(lease, result.events)
        except Exception:
            self.manager.abort_curated_compaction(lease)
        finally:
            with self._mutex:
                self._worker = None

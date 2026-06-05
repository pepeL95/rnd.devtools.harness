from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from time import perf_counter

from core.compaction.telemetry import CompactionTelemetry, compaction_logger, compaction_time_span
from core.compaction.token_counter import TokenCounter
from core.locks.session import SessionLease
from core.session.events import SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path
from core.trajectory.compactor import TrajectoryCompactor
from core.utilities.defaults import get_model_name


class TrajectoryCompactionCoordinator:
    """Run trajectory compaction in the background against the curated session stream."""

    def __init__(
        self,
        manager: SessionManager,
        compactor: TrajectoryCompactor,
        token_counter: TokenCounter | None = None,
        telemetry_store: TelemetryStore | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.manager = manager
        self.compactor = compactor
        self.token_counter = token_counter or TokenCounter()
        resolved_root = (repo_root or _default_repo_root()).expanduser().resolve()
        self.telemetry = CompactionTelemetry(
            kind="trajectory",
            store=telemetry_store or TelemetryStore(telemetry_session_path(manager.session_id)),
            logger=compaction_logger(resolved_root),
        )
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
            payload = self._payload(snapshot, trigger="trajectory", reason="turn_interval")
            if not self.compactor.should_compact(snapshot):
                self.telemetry.skip(payload)
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

    def _run_compaction(self, lease: SessionLease) -> None:
        payload = self._payload(lease.snapshot_events, trigger="trajectory", reason="turn_interval")
        started = perf_counter()
        self.telemetry.start(payload)
        try:
            result = self.compactor.compact(lease.snapshot_events)
            self.manager.finalize_curated_compaction(lease, result.events)
            self.telemetry.end(
                {
                    **payload,
                    "counts": {
                        **dict(payload["counts"]),
                        "compacted_event_count": result.compacted_event_count,
                        "compacted_turn_count": len(result.compacted_turns),
                        "result_event_count": len(result.events),
                    },
                    "model_names": result.model_names,
                    "token_usage": {
                        **dict(payload["token_usage"]),
                        **result.token_usage,
                    },
                    "duration_ms": int((perf_counter() - started) * 1000),
                }
            )
        except Exception as exc:
            self.manager.abort_curated_compaction(lease)
            self.telemetry.error(
                {
                    **payload,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
        finally:
            with self._mutex:
                self._worker = None

    def _payload(self, events: list[SessionEvent], *, trigger: str, reason: str) -> dict[str, object]:
        session_events = list(events)
        turns = sorted({event.turn for event in session_events})
        return {
            "trigger": trigger,
            "reason": reason,
            "model_names": {
                "compactor": get_model_name(self.compactor.policy.compactor_model),
            },
            "counts": {
                "source_event_count": len(session_events),
                "turn_count": len(turns),
                "latest_turn": turns[-1] if turns else 0,
            },
            "token_usage": {
                "source_tokens": self.token_counter.count_events(session_events) if session_events else 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
            },
            **compaction_time_span(session_events),
        }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

from __future__ import annotations

from collections.abc import Callable
from threading import Lock, Thread
from typing import Any

from core.compaction.compactor import Compactor
from core.compaction.token_counter import TokenCounter
from core.session.session_manager import CuratedCompactionLease, SessionManager

CompactionEventCallback = Callable[[str, dict[str, Any]], None]


class CompactionCoordinator:
    """Run session compaction in the background without blocking chat turns."""

    def __init__(
        self,
        manager: SessionManager,
        compactor: Compactor,
        token_counter: TokenCounter | None = None,
        on_compaction_event: CompactionEventCallback | None = None,
    ) -> None:
        self.manager = manager
        self.compactor = compactor
        self.token_counter = token_counter or TokenCounter()
        self.on_compaction_event = on_compaction_event
        self._mutex = Lock()
        self._worker: Thread | None = None

    def is_running(self) -> bool:
        with self._mutex:
            worker = self._worker
        return (worker is not None and worker.is_alive()) or self.manager.is_curated_locked()

    def request_manual_compaction(self, runtime: Any = None) -> str:
        return self._schedule(trigger="manual", runtime=runtime, force=True)

    def request_policy_compaction(self, runtime: Any = None) -> str:
        return self._schedule(trigger="policy", runtime=runtime, force=False)

    def _schedule(self, trigger: str, runtime: Any, force: bool) -> str:
        with self._mutex:
            worker = self._worker
            if worker is not None and worker.is_alive():
                return "running"

            visible_events = self.manager.read_curated(include_pending=True)
            estimated = self.token_counter.count_events(visible_events)
            if not force:
                decision = self.compactor.policy.compaction_decision(
                    estimated,
                    events=visible_events,
                    now=_runtime_now(runtime),
                )
                if not decision.should_compact:
                    return "not_needed"

            lease = self.manager.begin_curated_compaction(
                trigger=trigger,
                metadata={
                    "estimated_tokens": estimated,
                    "reason": "manual" if force else decision.reason,
                },
            )
            if lease is None:
                return "running"

            payload = {
                "trigger": trigger,
                "reason": "manual" if force else decision.reason,
                "estimated_tokens": estimated,
                "trigger_tokens": self.compactor.policy.trigger_tokens,
                "keep_last_turns": self.compactor.policy.keep_last_turns,
                "event_count": len(visible_events),
            }
            self._worker = Thread(
                target=self._run_compaction,
                args=(lease, estimated, payload),
                name=f"compaction-{self.manager.session_id}",
                daemon=True,
            )
            self._worker.start()
            return "started"

    def _run_compaction(
        self,
        lease: CuratedCompactionLease,
        estimated: int,
        payload: dict[str, Any],
    ) -> None:
        self._emit_compaction_event("start", payload)
        try:
            result = self.compactor.compact(lease.snapshot_events, token_estimate=estimated)
            merged = self.manager.finalize_curated_compaction(lease, result.events)
            self._emit_compaction_event(
                "end",
                {
                    **payload,
                    "compacted_event_count": result.compacted_event_count,
                    "retained_event_count": result.retained_event_count,
                    "revisions": result.revisions,
                    "memory_chars": len(result.memory_document),
                    "visible_event_count": len(merged),
                },
            )
        except Exception as exc:
            self.manager.abort_curated_compaction(lease)
            self._emit_compaction_event(
                "error",
                {
                    **payload,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )
        finally:
            with self._mutex:
                self._worker = None

    def _emit_compaction_event(self, phase: str, payload: dict[str, Any]) -> None:
        if self.on_compaction_event is None:
            return
        self.on_compaction_event(phase, payload)


def _runtime_now(runtime: Any) -> Any:
    context = getattr(runtime, "context", None)
    if isinstance(context, dict):
        return context.get("now")
    return getattr(context, "now", None)

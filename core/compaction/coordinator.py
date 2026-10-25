from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
from typing import Any

from core.compaction.compactor import Compactor
from core.compaction.telemetry import CompactionTelemetry, compaction_logger, compaction_time_span, context_token_breakdown
from core.compaction.token_counter import TokenCounter
from core.compaction.window import split_compaction_window
from core.session.events import SessionEvent, is_pinned_instruction
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path
from core.utilities.defaults import get_model_name

CompactionEventCallback = Callable[[str, dict[str, Any]], None]


class CompactionCoordinator:
    """Run full session compaction in the background against the curated session."""

    def __init__(
        self,
        manager: SessionManager,
        compactor: Compactor,
        token_counter: TokenCounter | None = None,
        on_compaction_event: CompactionEventCallback | None = None,
        telemetry_store: TelemetryStore | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.manager = manager
        self.compactor = compactor
        self.token_counter = token_counter or TokenCounter()
        self.on_compaction_event = on_compaction_event
        resolved_root = (repo_root or _default_repo_root()).expanduser().resolve()
        self.telemetry = CompactionTelemetry(
            kind="session",
            store=telemetry_store or TelemetryStore(telemetry_session_path(manager.session_id)),
            logger=compaction_logger(resolved_root, name="quasipilot.compaction", log_dir="compaction"),
        )
        self._mutex = Lock()
        self._worker: Thread | None = None

    def is_running(self) -> bool:
        with self._mutex:
            worker = self._worker
        return worker is not None and worker.is_alive()

    def request_manual_compaction(self, runtime: Any = None) -> str:
        return self._schedule(trigger="manual", runtime=runtime, force=True)

    def request_policy_compaction(self, runtime: Any = None) -> str:
        return self._schedule(trigger="policy", runtime=runtime, force=False)

    def _schedule(self, trigger: str, runtime: Any, force: bool) -> str:
        with self._mutex:
            worker = self._worker
            if worker is not None and worker.is_alive():
                return "running"
            source_events = [event for event in self.manager.read_curated() if not is_pinned_instruction(event)]
            latest_turn = max((event.turn for event in source_events), default=0)
            source_tokens = self.token_counter.count_events(source_events) if source_events else 0
            base_payload = {
                "trigger": trigger,
                "reason": "manual",
                "model_names": {
                    "compactor": get_model_name(self.compactor.policy.compactor_model),
                    "critic": get_model_name(self.compactor.policy.critic_model),
                },
                "counts": {
                    "source_event_count": len(source_events),
                    "latest_turn": latest_turn,
                },
                "token_usage": {
                    "source_tokens": source_tokens,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                },
                **compaction_time_span(source_events),
            }
            if not source_events:
                self.telemetry.skip({**base_payload, "reason": "empty_session"})
                return "not_needed"
            if not force:
                decision = self.compactor.policy.compaction_decision(
                    source_tokens,
                    events=source_events,
                    now=_runtime_now(runtime),
                )
                if not decision.should_compact:
                    self.telemetry.skip({**base_payload, "reason": decision.reason or "not_needed"})
                    return "not_needed"
                base_payload["reason"] = decision.reason or "policy"
            self._worker = Thread(
                target=self._run_compaction,
                args=(source_events, source_tokens, latest_turn, base_payload),
                name=f"session-compaction-{self.manager.session_id}",
                daemon=True,
            )
            self._worker.start()
            return "started"

    def _run_compaction(
        self,
        source_events: list[SessionEvent],
        source_tokens: int,
        snapshot_latest_turn: int,
        payload: dict[str, Any],
    ) -> None:
        started = perf_counter()
        try:
            window = split_compaction_window(source_events, self.compactor.policy.keep_last_turns)
            context_tokens = context_token_breakdown(
                source_events, window.retained, self.compactor.token_counter, completed=False,
            )
            payload = {**payload, "context_tokens": context_tokens}
            start_payload = {
                **payload,
                "content": self._format_compaction_message(
                    "start", str(payload["trigger"]), estimated=context_tokens["compacted_tokens"],
                ),
            }
            self.telemetry.start(start_payload)
            self._emit_compaction_event("start", start_payload)
            result = self.compactor.compact(source_events, token_estimate=source_tokens)
            context_tokens = context_token_breakdown(source_events, result.events, self.compactor.token_counter)
            merged_events = self.manager.apply_compaction_result(
                result.events,
                snapshot_latest_turn=snapshot_latest_turn,
                source_event_ids=[event.id for event in source_events],
            )
            duration_ms = int((perf_counter() - started) * 1000)
            end_payload = {
                **payload,
                "context_tokens": context_tokens,
                "counts": {
                    **dict(payload["counts"]),
                    "compacted_event_count": result.compacted_event_count,
                    "retained_event_count": result.retained_event_count,
                    "result_event_count": len(result.events),
                    "curated_event_count": len(merged_events),
                    "revisions": result.revisions,
                },
                "model_names": result.model_names,
                "token_usage": {
                    "source_tokens": source_tokens,
                    **result.token_usage,
                },
                "duration_ms": duration_ms,
                "content": self._format_compaction_message(
                    "end",
                    str(payload["trigger"]),
                    context_tokens=context_tokens,
                ),
            }
            self.telemetry.end(end_payload)
            self._emit_compaction_event("end", end_payload)
        except Exception as exc:
            error_payload = {
                **payload,
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "content": self._format_compaction_message("error", str(payload["trigger"]), error=str(exc)),
            }
            self.telemetry.error(error_payload)
            self._emit_compaction_event("error", error_payload)
        finally:
            with self._mutex:
                self._worker = None

    def _emit_compaction_event(self, phase: str, payload: dict[str, Any]) -> None:
        if self.on_compaction_event is None:
            return
        try:
            self.on_compaction_event(phase, payload)
        except Exception:
            return

    def _format_compaction_message(
        self,
        phase: str,
        trigger: str,
        *,
        estimated: int | None = None,
        context_tokens: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        prefix = "manual" if trigger == "manual" else "policy"
        if phase == "start":
            suffix = f" ({estimated:,} estimated tokens to compact)" if isinstance(estimated, int) else ""
            return f"{prefix} compaction started{suffix}"
        if phase == "end":
            suffix = ""
            if context_tokens is not None:
                before, after = context_tokens["compacted_tokens"], context_tokens["injected_tokens"]
                saved = context_tokens["saved_tokens"]
                delta = f"saved {saved:,}" if saved >= 0 else f"grew by {-saved:,}"
                suffix = f" ({before:,} -> {after:,} estimated tokens; {delta})"
            return f"{prefix} compaction finished{suffix}"
        if error:
            return f"{prefix} compaction failed: {error}"
        return f"{prefix} compaction failed"


def _runtime_now(runtime: Any) -> Any:
    context = getattr(runtime, "context", None)
    if isinstance(context, dict):
        return context.get("now")
    return getattr(context, "now", None)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

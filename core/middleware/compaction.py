from __future__ import annotations

from typing import Any
from collections.abc import Callable

from core.compaction.compactor import Compactor
from core.compaction.token_counter import TokenCounter
from langchain.agents.middleware import AgentMiddleware, AgentState
from core.session.session_manager import SessionManager

CompactionEventCallback = Callable[[str, dict[str, Any]], None]


class CompactionMiddleware(AgentMiddleware):
    """Compact curated session history after an agent turn crosses policy limits."""

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

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self._compact_if_needed(runtime)
        return None

    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        self._compact_if_needed(runtime)
        return None

    def _compact_if_needed(self, runtime: Any = None) -> None:
        events = self.manager.read_curated()
        estimated = self.token_counter.count_events(events)
        decision = self.compactor.policy.compaction_decision(
            estimated,
            events=events,
            now=_runtime_now(runtime),
        )
        if not decision.should_compact:
            return
        self._emit_compaction_event(
            "start",
            {
                "estimated_tokens": estimated,
                "trigger_tokens": self.compactor.policy.trigger_tokens,
                "keep_last_turns": self.compactor.policy.keep_last_turns,
                "event_count": len(events),
            },
        )
        result = self.compactor.compact(events, token_estimate=estimated)
        self.manager.replace_curated(result.events)
        self._emit_compaction_event(
            "end",
            {
                "estimated_tokens": estimated,
                "compacted_event_count": result.compacted_event_count,
                "retained_event_count": result.retained_event_count,
                "revisions": result.revisions,
                "memory_chars": len(result.memory_document),
            },
        )

    def _emit_compaction_event(self, phase: str, payload: dict[str, Any]) -> None:
        if self.on_compaction_event is None:
            return
        self.on_compaction_event(phase, payload)


def _runtime_now(runtime: Any) -> Any:
    context = getattr(runtime, "context", None)
    if isinstance(context, dict):
        return context.get("now")
    return getattr(context, "now", None)

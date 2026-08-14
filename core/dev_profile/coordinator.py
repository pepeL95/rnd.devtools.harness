from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from core.dev_profile.agent import create_dev_profile_agent
from core.dev_profile.store import DevProfileStore
from core.dev_profile.tools import completed_dump_snapshot
from core.session.events import SessionEvent
from core.session.manager import SessionManager

DevProfileEventCallback = Callable[[str, dict[str, Any]], None]
DevProfileAgentFactory = Callable[..., Any]


class DevProfileCoordinator:
    """Run developer-profile maintenance off the main agent path."""

    def __init__(
        self,
        manager: SessionManager,
        cwd: str | Path,
        *,
        model: BaseChatModel | None = None,
        on_event: DevProfileEventCallback | None = None,
        agent_factory: DevProfileAgentFactory = create_dev_profile_agent,
    ) -> None:
        self.manager = manager
        self.store = DevProfileStore(cwd)
        self.model = model
        self.on_event = on_event
        self.agent_factory = agent_factory
        self._lock = Lock()
        self._worker: Thread | None = None

    def is_running(self) -> bool:
        with self._lock:
            worker = self._worker
        return worker is not None and worker.is_alive()

    def request_update(self, focus: str = "") -> str:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return "running"
            snapshot = completed_dump_snapshot(self.manager.read_dump())
            if not snapshot:
                return "not_needed"
            self._worker = Thread(
                target=self._run,
                args=(snapshot, focus.strip()),
                name=f"dev-profile-{self.manager.session_id}",
                daemon=True,
            )
            self._worker.start()
            return "started"

    def _run(self, events: tuple[SessionEvent, ...], focus: str) -> None:
        self._emit("start", {"session_id": self.manager.session_id, "turn_count": len({event.turn for event in events})})
        try:
            before_revision = self.store.read().revision
            agent = self.agent_factory(events, self.store, model=self.model)
            instruction = "Review the completed session and evolve DEVPROFILE.md when durable evidence warrants it."
            if focus:
                instruction += f" The developer asked you to focus on: {focus}"
            agent.invoke({"messages": [{"role": "user", "content": instruction}]})
            after_revision = self.store.read().revision
            self._emit(
                "end",
                {
                    "session_id": self.manager.session_id,
                    "changed": before_revision != after_revision,
                    "path": str(self.store.path),
                },
            )
        except Exception as exc:
            self._emit(
                "error",
                {
                    "session_id": self.manager.session_id,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )
        finally:
            with self._lock:
                self._worker = None

    def _emit(self, phase: str, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(phase, payload)
        except Exception:
            return

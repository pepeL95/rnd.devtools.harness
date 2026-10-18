from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.preferences.store import PreferencesStore
from core.middleware.turn_instructions import TurnInstructionsMiddleware
from core.session.events import PREFERENCES_KIND
from core.session.manager import SessionManager

if TYPE_CHECKING:
    from core.middleware.session_dump import SessionDumpMiddleware

PREFERENCES_CONTEXT = """[PREFERENCES]

The following is an evolving set of durable developer preferences learned from prior interactions. Apply it unless it conflicts with the developer's current request or higher-priority repository or system instructions.

{preferences}

Do not modify PREFERENCES.md during normal agent work. It is maintained through the `/preferences` workflow.

[END PREFERENCES]"""


class PreferencesMiddleware(TurnInstructionsMiddleware):
    """Persist and inject the latest preferences at turn boundaries."""

    def __init__(
        self, cwd: str | Path, *, manager: SessionManager | None = None,
        session_dump: SessionDumpMiddleware | None = None,
    ) -> None:
        self.store = PreferencesStore(cwd)
        super().__init__(kind=PREFERENCES_KIND, loader=self._load_instruction, manager=manager, session_dump=session_dump)

    def _load_instruction(self) -> tuple[str, str]:
        snapshot = self.store.read()
        content = PREFERENCES_CONTEXT.format(preferences=snapshot.content) if snapshot.content else ""
        return content, str(snapshot.path)

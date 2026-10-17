from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.middleware.turn_instructions import TurnInstructionsMiddleware
from core.session.events import AGENTS_MD_KIND
from core.session.manager import SessionManager

if TYPE_CHECKING:
    from core.middleware.session_dump import SessionDumpMiddleware


class AgentsMdMiddleware(TurnInstructionsMiddleware):
    """Snapshot only the optional workspace AGENTS.md at turn boundaries."""

    def __init__(
        self, cwd: str | Path, *, manager: SessionManager | None = None,
        session_dump: SessionDumpMiddleware | None = None,
    ) -> None:
        self.path = Path(cwd).expanduser().resolve() / "AGENTS.md"
        super().__init__(kind=AGENTS_MD_KIND, loader=self._load_instruction, manager=manager, session_dump=session_dump)

    def _load_instruction(self) -> tuple[str, str]:
        try:
            content = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            content = ""
        return content, str(self.path)

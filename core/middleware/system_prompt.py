from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.middleware.turn_instructions import TurnInstructionsMiddleware
from core.session.events import SYSTEM_PROMPT_KIND
from core.session.manager import SessionManager

if TYPE_CHECKING:
    from core.middleware.session_dump import SessionDumpMiddleware


class SystemPromptMiddleware(TurnInstructionsMiddleware):
    """Snapshot required harness instructions, preferring workspace over home."""

    def __init__(
        self,
        prompt: str | None = None,
        prompt_path: str | Path | None = None,
        cwd: str | Path | None = None,
        *,
        manager: SessionManager | None = None,
        session_dump: SessionDumpMiddleware | None = None,
    ) -> None:
        self.prompt = prompt
        self.prompt_path = Path(prompt_path).expanduser().resolve() if prompt_path else None
        self.cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        super().__init__(kind=SYSTEM_PROMPT_KIND, loader=self._load_instruction, manager=manager, session_dump=session_dump)

    def _load_instruction(self, cwd: Path | None = None) -> tuple[str, str]:
        if self.prompt:
            return self.prompt.strip(), "inline"
        if self.prompt_path:
            return self.prompt_path.read_text(encoding="utf-8").strip(), str(self.prompt_path)
        local_path = (cwd or self.cwd) / ".quasipilot" / "SYSTEM.md"
        home_path = Path.home() / ".quasipilot" / "SYSTEM.md"
        for path in (local_path, home_path):
            try:
                return path.read_text(encoding="utf-8").strip(), str(path)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            "No system prompt found. Please provide a prompt, prompt_path, or ensure "
            ".quasipilot/SYSTEM.md exists in the current working or home directory."
        )

    def load_prompt(self, cwd: Path | None = None) -> str:
        return self._load_instruction(cwd)[0]

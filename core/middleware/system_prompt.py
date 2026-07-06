from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from core.utilities.messages import system_message_with_appended_text


class SystemPromptMiddleware(AgentMiddleware):
    """Append configured harness instructions to the model system message."""

    def __init__(
        self,
        prompt: str | None = None,
        prompt_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.prompt = prompt
        self.prompt_path = Path(prompt_path).expanduser() if prompt_path else None
        self.cwd = Path(cwd).expanduser().resolve() if cwd else None

    def load_prompt(self, cwd: Path | None = None) -> str:
        if self.prompt:
            return self.prompt.strip()
        
        if self.prompt_path:
            return self.prompt_path.read_text(encoding="utf-8").strip()
        
        # Fallback to .quasipilot/SYSTEM.md
        search_cwd = cwd or self.cwd or Path.cwd()
        local_path = search_cwd / ".quasipilot" / "SYSTEM.md"
        if local_path.exists():
            return local_path.read_text(encoding="utf-8").strip()
        
        home_path = Path.home() / ".quasipilot" / "SYSTEM.md"
        if home_path.exists():
            return home_path.read_text(encoding="utf-8").strip()
            
        raise FileNotFoundError(
            "No system prompt found. Please provide a prompt, prompt_path, or ensure "
            ".quasipilot/SYSTEM.md exists in the current working or home directory."
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        prompt = self.load_prompt(cwd=self.cwd)
        if not prompt:
            return handler(request)
        updated = request.override(
            system_message=system_message_with_appended_text(request.system_message, prompt)
        )
        return handler(updated)


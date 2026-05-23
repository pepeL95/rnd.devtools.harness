from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from core.utilities.messages import system_message_with_appended_text


class SystemPromptMiddleware(AgentMiddleware):
    """Append configured harness instructions to the model system message."""

    def __init__(self, prompt: str | None = None, prompt_path: str | Path | None = None) -> None:
        if not prompt and not prompt_path:
            raise ValueError("SystemPromptMiddleware requires prompt or prompt_path.")
        self.prompt = prompt
        self.prompt_path = Path(prompt_path).expanduser() if prompt_path else None

    def load_prompt(self) -> str:
        if self.prompt_path:
            return self.prompt_path.read_text(encoding="utf-8").strip()
        return (self.prompt or "").strip()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        prompt = self.load_prompt()
        if not prompt:
            return handler(request)
        updated = request.override(
            system_message=system_message_with_appended_text(request.system_message, prompt)
        )
        return handler(updated)


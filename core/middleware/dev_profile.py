from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse

from core.dev_profile.store import DevProfileStore
from core.utilities.messages import system_message_with_appended_text

DEV_PROFILE_CONTEXT = """[DEVELOPER PROFILE]

The following is an evolving set of durable developer preferences learned from prior interactions. Apply it unless it conflicts with the developer's current request or higher-priority repository or system instructions.

{profile}

Do not modify DEVPROFILE.md during normal agent work. It is maintained through the `/devprofile` workflow.

[END DEVELOPER PROFILE]"""


class DevProfileMiddleware(AgentMiddleware):
    """Inject one stable developer-profile snapshot for each agent run."""

    def __init__(self, cwd: str | Path) -> None:
        self.store = DevProfileStore(cwd)
        self._lock = Lock()
        self._run_profile = ""

    def before_agent(self, state: AgentState, runtime: object) -> dict[str, object] | None:
        try:
            profile = self.store.read().content
        except (OSError, UnicodeError):
            profile = ""
        with self._lock:
            self._run_profile = profile
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        with self._lock:
            profile = self._run_profile
        if not profile:
            return handler(request)
        updated = request.override(
            system_message=system_message_with_appended_text(
                request.system_message,
                DEV_PROFILE_CONTEXT.format(profile=profile),
            )
        )
        return handler(updated)

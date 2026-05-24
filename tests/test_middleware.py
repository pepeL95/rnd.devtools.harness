from pathlib import Path
from dataclasses import dataclass, replace
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

from langchain_core.messages import SystemMessage
from core.middleware.runtime import RuntimeContextMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.session_load import SessionLoadMiddleware
from core.middleware.system_prompt import SystemPromptMiddleware
from core.session.events import EventType, SessionEvent
from core.session.session_manager import SessionManager


@dataclass(frozen=True)
class FakeModelRequest:
    system_message: SystemMessage | None
    messages: list[Any]
    runtime: Any = None

    def override(self, **changes: Any) -> "FakeModelRequest":
        return replace(self, **changes)


class MiddlewareTests(TestCase):
    def test_system_prompt_middleware_appends_prompt(self) -> None:
        middleware = SystemPromptMiddleware(prompt="Use concise answers.")
        request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

        self.assertIn("Use concise answers.", str(response.content))

    def test_system_prompt_middleware_handles_missing_system_message(self) -> None:
        middleware = SystemPromptMiddleware(prompt="Use concise answers.")
        request = FakeModelRequest(system_message=None, messages=[])

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

        self.assertIn("Use concise answers.", str(response.content))

    def test_runtime_middleware_injects_cwd(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            middleware = RuntimeContextMiddleware(cwd=path)
            request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

            response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

            self.assertIn(str(path.resolve()), str(response.content))

    def test_runtime_middleware_handles_missing_system_message(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            middleware = RuntimeContextMiddleware(cwd=path)
            request = FakeModelRequest(system_message=None, messages=[])

            response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

            self.assertIn(str(path.resolve()), str(response.content))

    def test_session_load_prepends_curated_history(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append([SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "prior"})])
            middleware = SessionLoadMiddleware(manager)

            update = middleware.before_agent({"messages": []}, runtime=None)

            self.assertIsNotNone(update)
            assert update is not None
            restored_contents = [getattr(message, "content", None) for message in update["messages"]]
            self.assertIn("prior", restored_contents)

    def test_session_dump_does_not_reappend_restored_assistant_text_on_later_turns(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "prior user"}),
                    SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"role": "assistant", "content": "prior reply"}),
                ]
            )
            middleware = SessionDumpMiddleware(manager)

            middleware.before_agent({"messages": [manager.load_curated_messages()[1]]}, runtime=None)

            assistant_events = [event for event in manager.read_curated() if event.type == EventType.ASSISTANT]
            self.assertEqual(len(assistant_events), 1)
            self.assertEqual(assistant_events[0].payload["content"], "prior reply")

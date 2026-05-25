from pathlib import Path
from dataclasses import dataclass, replace
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from core.middleware.reasoning import ReasoningMiddleware, reasoning_tool
from core.middleware.runtime import RuntimeContextMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.session_load import SessionLoadMiddleware
from core.middleware.system_prompt import SystemPromptMiddleware
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager


@dataclass(frozen=True)
class FakeModelRequest:
    system_message: SystemMessage | None
    messages: list[Any]
    runtime: Any = None

    def override(self, **changes: Any) -> "FakeModelRequest":
        return replace(self, **changes)


class MiddlewareTests(TestCase):
    def test_reasoning_middleware_appends_low_eagerness_steering_prompt(self) -> None:
        middleware = ReasoningMiddleware()
        request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

        self.assertIn("Use the `reasoning` tool selectively", str(response.content))
        self.assertNotIn("Reason often", str(response.content))

    def test_reasoning_middleware_appends_high_eagerness_steering_prompt(self) -> None:
        middleware = ReasoningMiddleware(eagerness="high")
        request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

        self.assertIn("Use the `reasoning` tool often", str(response.content))
        self.assertIn("Reason often", str(response.content))

    def test_reasoning_middleware_rejects_invalid_eagerness(self) -> None:
        with self.assertRaises(ValueError):
            ReasoningMiddleware(eagerness="aggressive")  # type: ignore[arg-type]

    def test_reasoning_middleware_adds_one_shot_reminder_after_tool_failure(self) -> None:
        middleware = ReasoningMiddleware()
        request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

        with self.assertRaises(RuntimeError):
            middleware.wrap_tool_call(
                object(),
                lambda _: (_ for _ in ()).throw(RuntimeError("tool failed")),
            )

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)
        self.assertIn("A tool just failed.", str(response.content))

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)
        self.assertEqual(str(response.content).count("A tool just failed."), 0)

    def test_reasoning_tool_returns_visible_tool_output(self) -> None:
        self.assertIsInstance(reasoning_tool, BaseTool)

        result = reasoning_tool.invoke({"reasoning": "Need to inspect the middleware order before retrying."})

        self.assertEqual(
            result,
            "Reasoning recorded: Need to inspect the middleware order before retrying.",
        )

    def test_reasoning_middleware_requests_synthesis_after_long_read_file_output(self) -> None:
        middleware = ReasoningMiddleware()
        request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

        middleware.wrap_tool_call(
            type("Req", (), {"tool_call": {"name": "read_file"}})(),
            lambda _: ToolMessage(content="x" * 4500, tool_call_id="call-1"),
        )

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

        self.assertIn("long `read_file` result", str(response.content))
        self.assertIn("preserve signal for later compaction", str(response.content))

    def test_reasoning_middleware_does_not_trigger_for_short_read_file_output(self) -> None:
        middleware = ReasoningMiddleware()
        request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

        middleware.wrap_tool_call(
            type("Req", (), {"tool_call": {"name": "read_file"}})(),
            lambda _: ToolMessage(content="short", tool_call_id="call-1"),
        )

        response = middleware.wrap_model_call(request, lambda updated: updated.system_message)

        self.assertNotIn("long `read_file` result", str(response.content))

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

    def test_session_dump_skips_restored_memory_restore_messages(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(
                        type=EventType.USER,
                        turn=1,
                        payload={"role": "user", "content": "[MEMORY RESTORE]\n...", "kind": "memory_restore"},
                    )
                ]
            )
            middleware = SessionDumpMiddleware(manager)

            middleware.before_agent({"messages": manager.load_curated_messages()}, runtime=None)

            memory_restore_events = [
                event
                for event in manager.read_dump()
                if event.payload.get("kind") == "memory_restore"
            ]
            self.assertEqual(len(memory_restore_events), 1)

    def test_session_dump_skips_restored_trajectory_memory_messages(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(
                        type=EventType.USER,
                        turn=1,
                        payload={"role": "user", "content": "[TRAJECTORY MEMORY]\n...", "kind": "trajectory_memory"},
                    )
                ]
            )
            middleware = SessionDumpMiddleware(manager)

            middleware.before_agent({"messages": manager.load_curated_messages()}, runtime=None)

            trajectory_memory_events = [
                event
                for event in manager.read_dump()
                if event.payload.get("kind") == "trajectory_memory"
            ]
            self.assertEqual(len(trajectory_memory_events), 1)

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from core.session.events import EventType, SessionEvent
from core.session.session_manager import SessionManager


class SessionIOTests(TestCase):
    def test_session_manager_writes_dump_and_curated(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            event = SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "hello"})

            manager.append([event])

            self.assertEqual(manager.read_dump()[0].payload["content"], "hello")
            self.assertEqual(manager.read_curated()[0].payload["content"], "hello")

    def test_load_curated_messages_round_trips_roles(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "hello"}),
                    SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"role": "assistant", "content": "hi"}),
                ]
            )

            messages = manager.load_curated_messages()

            self.assertEqual([message.content for message in messages], ["hello", "hi"])

    def test_load_curated_messages_skips_tool_output(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "run git status"}),
                    SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"role": "assistant", "content": "ok"}),
                    SessionEvent(
                        type=EventType.TOOL_OUTPUT,
                        turn=1,
                        payload={"role": "tool", "content": "on branch main"},
                    ),
                ]
            )

            messages = manager.load_curated_messages()

            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0].content, "run git status")
            self.assertEqual(messages[1].content, "ok")


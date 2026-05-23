from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from core.session.events import EventType, SessionEvent
from core.session.io import append_events, session_paths
from cli.components.session_picker import _item_dom_id, _session_id_from_dom_id
from cli.utilities.messages import format_tool_call, message_text
from cli.utilities.sessions import clear_session_files, list_sessions


class CliSessionUtilityTests(TestCase):
    def test_list_sessions_orders_by_mtime(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older_id, newer_id = "older", "newer"
            for session_id in (older_id, newer_id):
                _, curated = session_paths(session_id, root)
                append_events(
                    curated,
                    [SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": session_id})],
                )

            older_curated = session_paths(older_id, root)[1]
            newer_curated = session_paths(newer_id, root)[1]
            older_curated.touch()
            newer_curated.touch()

            summaries = list_sessions(root)
            self.assertEqual([item.session_id for item in summaries], [newer_id, older_id])
            self.assertEqual(summaries[0].preview, newer_id)

    def test_clear_session_files_removes_dump_and_curated(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session_id = "abc"
            dump, curated = session_paths(session_id, root)
            append_events(dump, [SessionEvent(type=EventType.META, turn=0, payload={})])
            append_events(curated, [SessionEvent(type=EventType.USER, turn=1, payload={"content": "x"})])

            clear_session_files(session_id, root)

            self.assertFalse(dump.exists())
            self.assertFalse(curated.exists())


class SessionPickerIdTests(TestCase):
    def test_session_dom_ids_are_textual_safe(self) -> None:
        session_id = "7f7de5f3753a4c69ba2f8926189fafe9"
        dom_id = _item_dom_id(session_id)
        self.assertTrue(dom_id[0].isalpha())
        self.assertEqual(_session_id_from_dom_id(dom_id), session_id)


class CliMessageUtilityTests(TestCase):
    def test_message_text_from_blocks(self) -> None:
        class FakeMessage:
            content = [{"type": "text", "text": "hello"}]

        self.assertEqual(message_text(FakeMessage()), "hello")

    def test_format_tool_call(self) -> None:
        name, args = format_tool_call({"name": "read", "args": {"path": "/tmp"}})
        self.assertEqual(name, "read")
        self.assertIn("path=", args)

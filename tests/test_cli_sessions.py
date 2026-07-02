from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import os

from core.session.events import EventType, SessionEvent
from core.session.io import append_events, session_paths
from cli.components.session_picker import SessionPickerScreen, _item_dom_id, _session_id_from_dom_id
from cli.utilities.messages import format_tool_call, format_tool_input, message_text
from cli.utilities.sessions import SessionSummary, clear_session_files, list_sessions


class CliSessionUtilityTests(TestCase):
    def test_list_sessions_orders_by_mtime(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older_id, newer_id = "older", "newer"
            for session_id in (older_id, newer_id):
                dump, curated = session_paths(session_id, root)
                append_events(
                    dump,
                    [SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": session_id})],
                )
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

    def test_list_sessions_prioritizes_current_cwd_matches(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            current_cwd = root / "repo-a"
            other_cwd = root / "repo-b"
            current_cwd.mkdir()
            other_cwd.mkdir()

            dump, curated = session_paths("older-match", root)
            append_events(
                dump,
                [
                    SessionEvent(type=EventType.RUNTIME, turn=1, payload={"cwd": str(current_cwd)}),
                    SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "match"}),
                ],
            )
            append_events(curated, [SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "match"})])

            dump, curated = session_paths("newer-other", root)
            append_events(
                dump,
                [
                    SessionEvent(type=EventType.RUNTIME, turn=1, payload={"cwd": str(other_cwd)}),
                    SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "other"}),
                ],
            )
            append_events(curated, [SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "other"})])

            older_curated = session_paths("older-match", root)[1]
            newer_curated = session_paths("newer-other", root)[1]
            os.utime(older_curated, (1, 1))
            os.utime(newer_curated, (2, 2))

            summaries = list_sessions(root, current_cwd=current_cwd)

            self.assertEqual([item.session_id for item in summaries], ["older-match", "newer-other"])
            self.assertTrue(summaries[0].cwd_matches)
            self.assertFalse(summaries[1].cwd_matches)

    def test_list_sessions_uses_first_user_message_for_preview(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session_id = "s1"
            dump, curated = session_paths(session_id, root)
            append_events(
                curated,
                [
                    SessionEvent(type=EventType.META, turn=1, payload={"kind": "memory_restore", "content": "summary"}),
                    SessionEvent(type=EventType.USER, turn=2, payload={"role": "user", "content": "latest prompt"}),
                ],
            )
            append_events(
                dump,
                [
                    SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "first prompt"}),
                    SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"role": "assistant", "content": "reply"}),
                    SessionEvent(type=EventType.USER, turn=2, payload={"role": "user", "content": "latest prompt"}),
                ],
            )

            summaries = list_sessions(root)

            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].preview, "first prompt")

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

    def test_session_picker_inserts_divider_after_sponsored_results(self) -> None:
        from datetime import datetime, timezone

        screen = SessionPickerScreen(
            [
                SessionSummary("s1", Path("/tmp/s1.jsonl"), datetime.now(timezone.utc), "one", cwd_matches=True),
                SessionSummary("s2", Path("/tmp/s2.jsonl"), datetime.now(timezone.utc), "two", cwd_matches=False),
            ]
        )

        items = screen._items()

        self.assertEqual(len(items), 3)
        self.assertEqual(items[1].id, "session-divider")


class CliMessageUtilityTests(TestCase):
    def test_message_text_from_blocks(self) -> None:
        class FakeMessage:
            content = [{"type": "text", "text": "hello"}]

        self.assertEqual(message_text(FakeMessage()), "hello")

    def test_format_tool_call(self) -> None:
        name, args = format_tool_call({"name": "read", "args": {"path": "/tmp"}})
        self.assertEqual(name, "read")
        self.assertIn("path=", args)

    def test_format_tool_input_prefers_human_command_shape(self) -> None:
        name, args = format_tool_input({"name": "execute", "args": {"cmd": "pytest tests/test_cli_display.py"}})
        self.assertEqual(name, "execute")
        self.assertEqual(args, "pytest tests/test_cli_display.py")

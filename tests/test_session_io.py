from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from langchain_core.messages import AIMessage

from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.manager import SessionManager


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

    def test_load_curated_messages_marks_memory_restore_messages(self) -> None:
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

            messages = manager.load_curated_messages()

            self.assertEqual(messages[0].additional_kwargs.get("session_kind"), "memory_restore")

    def test_read_display_history_uses_dump_and_hides_memory_restore(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(
                        type=EventType.USER,
                        turn=1,
                        payload={"role": "user", "content": "[MEMORY RESTORE]\n...", "kind": "memory_restore"},
                    ),
                    SessionEvent(type=EventType.USER, turn=2, payload={"role": "user", "content": "real user"}),
                    SessionEvent(type=EventType.ASSISTANT, turn=2, payload={"role": "assistant", "content": "real assistant"}),
                ],
                curated=False,
            )

            history = manager.read_display_history()

            self.assertEqual([event.payload["content"] for event in history], ["real user", "real assistant"])

    def test_read_display_history_hides_trajectory_memory(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(
                        type=EventType.USER,
                        turn=1,
                        payload={"role": "user", "content": "[TRAJECTORY MEMORY]\n...", "kind": "trajectory_memory"},
                    ),
                    SessionEvent(type=EventType.USER, turn=2, payload={"role": "user", "content": "real user"}),
                    SessionEvent(type=EventType.ASSISTANT, turn=2, payload={"role": "assistant", "content": "real assistant"}),
                ],
                curated=False,
            )

            history = manager.read_display_history()

            self.assertEqual([event.payload["content"] for event in history], ["real user", "real assistant"])

    def test_events_from_messages_emits_reasoning_events_for_assistant_messages(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            message = AIMessage(
                content=[
                    {"type": "thinking", "thinking": "Need to inspect middleware order.", "signature": "abc123"},
                    {"type": "text", "text": "I found the bug."},
                ]
            )

            events = manager.events_from_messages([message], turn=3)

            self.assertEqual([event.type for event in events], [EventType.REASONING, EventType.ASSISTANT])
            self.assertEqual(events[0].payload["content"], "Need to inspect middleware order.")
            self.assertEqual(events[0].payload["signature"], "abc123")
            self.assertEqual(events[1].payload["content"], "I found the bug.")

    def test_events_from_messages_emits_tool_events_for_assistant_tool_calls(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            message = AIMessage(
                content=[{"type": "text", "text": "Reading the file now."}],
                tool_calls=[{"name": "read_file", "args": {"path": "/tmp/x"}, "id": "call-1"}],
            )

            events = manager.events_from_messages([message], turn=2)

            self.assertEqual([event.type for event in events], [EventType.TOOL, EventType.ASSISTANT])
            self.assertEqual(events[0].payload["name"], "read_file")
            self.assertEqual(events[0].payload["args"], {"path": "/tmp/x"})
            self.assertEqual(events[0].payload["tool_call_id"], "call-1")
            self.assertEqual(events[1].payload["content"], "Reading the file now.")

    def test_events_from_messages_does_not_duplicate_reasoning_inside_assistant_payload(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            message = AIMessage(
                content=[
                    {"type": "thinking", "thinking": "trace the call graph", "signature": "sig-1"},
                    {"type": "text", "text": "hi"},
                ]
            )

            events = manager.events_from_messages([message], turn=1)

            self.assertEqual(len([event for event in events if event.type == EventType.REASONING]), 1)
            self.assertEqual(len([event for event in events if event.type == EventType.ASSISTANT]), 1)
            self.assertEqual(events[1].payload["content"], "hi")

    def test_load_curated_messages_restores_assistant_text_without_reasoning_blocks(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "hello"}),
                    SessionEvent(
                        type=EventType.ASSISTANT,
                        turn=1,
                        payload={
                            "role": "assistant",
                            "content": "hi",
                        },
                    ),
                ]
            )

            messages = manager.load_curated_messages()

            self.assertEqual(messages[1].content, "hi")

    def test_apply_compaction_result_preserves_newer_turns(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append([SessionEvent(type=EventType.USER, turn=2, payload={"role": "user", "content": "new"})])
            merged = manager.apply_compaction_result(
                [SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "memory"})],
                snapshot_latest_turn=1,
            )

            self.assertEqual([event.payload["content"] for event in merged], ["memory", "new"])
            self.assertEqual([event.payload["content"] for event in manager.read_curated()], ["memory", "new"])

    def test_latest_runtime_snapshot_returns_latest_runtime_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.record_runtime(
                RuntimeSnapshot(
                    cwd="/tmp/one",
                    git_branch="main",
                    git_dirty=False,
                    python_interpreter="/tmp/one/.venv/bin/python",
                ),
                turn=1,
            )
            manager.record_runtime(
                RuntimeSnapshot(
                    cwd="/tmp/two",
                    git_branch="feature",
                    git_dirty=True,
                    python_interpreter="/tmp/two/.venv/bin/python",
                ),
                turn=2,
            )

            snapshot = manager.latest_runtime_snapshot()

            assert snapshot is not None
            self.assertEqual(snapshot.cwd, "/tmp/two")
            self.assertEqual(snapshot.git_branch, "feature")
            self.assertIs(snapshot.git_dirty, True)
            self.assertEqual(snapshot.python_interpreter, "/tmp/two/.venv/bin/python")

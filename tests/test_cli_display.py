from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cli.components import ChatInput
from cli.components import ToolStream
from cli.components import RuntimeBar
from cli.components import StatusBubble
from cli.run import QuasipilotApp
from cli.utilities.display import content_to_plaintext
from core.live_steering import format_live_steering_message
from core.session.events import RuntimeSnapshot
from core.session.manager import SessionManager


class DisplayUtilityTests(TestCase):
    def test_content_to_plaintext_from_blocks(self) -> None:
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        self.assertEqual(content_to_plaintext(content), "hello\nworld")

    def test_content_to_plaintext_preserves_markup_like_characters(self) -> None:
        raw = "args path='x' code [{'key': '='}}]"
        self.assertEqual(content_to_plaintext(raw), raw)


class SessionPickerMarkupTests(TestCase):
    def test_label_with_special_chars_uses_plain_markup(self) -> None:
        import asyncio

        from textual.app import App
        from textual.widgets import Label

        class Probe(App):
            def compose(self):
                yield Label("preview [{'key': '='}}]", markup=False)

        async def run() -> None:
            app = Probe()
            async with app.run_test():
                return

        asyncio.run(run())


class CompactionUiTests(TestCase):
    def test_handle_compaction_event_mounts_status_bubble(self) -> None:
        app = QuasipilotApp()
        mounted: list[object] = []
        notifications: list[str] = []
        app._mount_chat = mounted.append  # type: ignore[method-assign]
        app._sync_compaction_ui = lambda: None  # type: ignore[method-assign]
        app.notify = lambda message, **_: notifications.append(message)  # type: ignore[method-assign]
        app.notify_warning = lambda message: notifications.append(message)  # type: ignore[method-assign]

        app._handle_compaction_event(
            "end",
            {
                "kind": "compaction_event",
                "phase": "end",
                "content": "manual session compaction finished (4 events compacted, 2 events retained)",
            },
        )

        self.assertEqual(len(mounted), 1)
        self.assertIsInstance(mounted[0], StatusBubble)
        self.assertEqual(notifications, ["session compaction finished"])


class ToolStreamTests(TestCase):
    def test_tool_stream_labels_ls_as_listed(self) -> None:
        rendered = ToolStream("execute", "ls -la")._build_content("execute", "ls -la", None)

        self.assertTrue(rendered.plain.startswith("Listed ls -la"))

    def test_tool_stream_labels_find_as_found(self) -> None:
        command = 'find . -name "*.py"'
        rendered = ToolStream("execute", command)._build_content("execute", command, None)

        self.assertTrue(rendered.plain.startswith(f"Found {command}"))

    def test_tool_stream_labels_grep_as_searched(self) -> None:
        command = 'grep -R "SessionManager" core'
        rendered = ToolStream("execute", command)._build_content("execute", command, None)

        self.assertTrue(rendered.plain.startswith(f"Searched {command}"))

    def test_tool_stream_labels_mkdir_as_created(self) -> None:
        command = "mkdir -p tmp/cache"
        rendered = ToolStream("execute", command)._build_content("execute", command, None)

        self.assertTrue(rendered.plain.startswith(f"Created {command}"))

    def test_tool_stream_labels_rg_file_discovery_as_discovered(self) -> None:
        rendered = ToolStream("execute", "rg --files core")._build_content("execute", "rg --files core", None)

        self.assertTrue(rendered.plain.startswith("Discovered rg --files core"))

    def test_tool_stream_labels_rg_search_as_searched(self) -> None:
        command = 'rg -n --no-heading --color never "SessionManager" core tests'
        rendered = ToolStream("execute", command)._build_content("execute", command, None)

        self.assertTrue(rendered.plain.startswith(f"Searched {command}"))

    def test_tool_stream_formats_compact_header_and_indented_output(self) -> None:
        stream = ToolStream(
            "execute",
            "pytest tests/test_cli_display.py",
        )

        rendered = stream._build_content("execute", "pytest tests/test_cli_display.py", None)

        self.assertTrue(rendered.plain.endswith("pytest tests/test_cli_display.py"))

    def test_tool_stream_formats_muted_output_continuation(self) -> None:
        output = "\n".join(
            [
                "============================= test session starts ==============================",
                "platform darwin -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0",
                "cachedir: .pytest_cache",
                "rootdir: /tmp/project",
                "plugins: anyio-4.13.0",
                "============================== 12 passed in 2.00s ==============================",
            ]
        )

        rendered = ToolStream("execute", "", output)._build_content("execute", "", output)

        self.assertIn("  \u2514 ============================= test session starts ==============================", rendered.plain)
        self.assertIn("    platform darwin -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0", rendered.plain)
        self.assertIn("    \u2026 +2 lines", rendered.plain)
        self.assertIn("    ============================== 12 passed in 2.00s ==============================", rendered.plain)

    def test_tool_stream_set_output_updates_existing_row(self) -> None:
        stream = ToolStream("execute", "pytest tests/test_cli_display.py")
        stream.update = lambda content: None  # type: ignore[method-assign]

        stream.set_output("line 1\nline 2")

        rendered = stream._build_content(stream._tool_name, stream._tool_input_text, stream._tool_output)
        self.assertTrue(rendered.plain.startswith(("Dispatched", "Yeeted", "Ran", "Slammed", "Ramrodded")))
        self.assertIn("  \u2514 line 1", rendered.plain)


class AgentStreamUiTests(TestCase):
    def test_tool_output_updates_existing_tool_stream(self) -> None:
        app = QuasipilotApp()
        mounted: list[object] = []
        def mount_chat(widget: object) -> None:
            if isinstance(widget, ToolStream):
                widget.update = lambda content: None  # type: ignore[method-assign]
            mounted.append(widget)

        app._mount_chat = mount_chat  # type: ignore[method-assign]
        app._scroll_chat_to_bottom = lambda: None  # type: ignore[method-assign]

        app.on_agent_stream(
            type(
                "Event",
                (),
                {
                    "kind": "tool",
                    "payload": {"name": "execute", "input": "pytest tests/test_cli_display.py", "tool_call_id": "call-1"},
                },
            )()
        )
        app.on_agent_stream(
            type(
                "Event",
                (),
                {
                    "kind": "tool_output",
                    "payload": {
                        "name": "execute",
                        "input": "pytest tests/test_cli_display.py",
                        "output": "============================== 12 passed in 2.00s ==============================",
                        "tool_call_id": "call-1",
                    },
                },
            )()
        )

        self.assertEqual(len(mounted), 1)
        self.assertIsInstance(mounted[0], ToolStream)
        rendered = mounted[0]._build_content(
            mounted[0]._tool_name,
            mounted[0]._tool_input_text,
            mounted[0]._tool_output,
        )
        self.assertIn("12 passed in 2.00s", rendered.plain)


class ChatInputTests(TestCase):
    def test_enter_submits_message(self) -> None:
        import asyncio

        submitted: list[str] = []

        async def run() -> None:
            app = QuasipilotApp()
            app._mount_chat = lambda widget: None  # type: ignore[method-assign]
            app._set_busy = lambda busy, disable_input=True: None  # type: ignore[method-assign]
            app._show_spinner = lambda: None  # type: ignore[method-assign]
            app.run_turn = submitted.append  # type: ignore[method-assign]

            async with app.run_test() as pilot:
                await pilot.click("#chat-input")
                await pilot.press("h", "i", "enter")

        asyncio.run(run())

        self.assertEqual(submitted, ["hi"])

    def test_ctrl_j_inserts_newline_without_submitting(self) -> None:
        import asyncio

        snapshots: list[str] = []
        submitted: list[str] = []

        async def run() -> None:
            app = QuasipilotApp()
            app._mount_chat = lambda widget: None  # type: ignore[method-assign]
            app._set_busy = lambda busy, disable_input=True: None  # type: ignore[method-assign]
            app._show_spinner = lambda: None  # type: ignore[method-assign]
            app.run_turn = submitted.append  # type: ignore[method-assign]

            async with app.run_test() as pilot:
                await pilot.click("#chat-input")
                await pilot.press("h", "i", "ctrl+j", "t", "h", "e", "r", "e")
                snapshots.append(app.query_one(ChatInput).text)

        asyncio.run(run())

        self.assertEqual(submitted, [])
        self.assertEqual(snapshots, ["hi\nthere"])

    def test_busy_agent_submission_is_queued_as_live_steering(self) -> None:
        queued: list[str] = []
        notifications: list[str] = []

        app = QuasipilotApp()
        app._busy = True
        app._agent_active = True
        app._live_steering.submit = queued.append  # type: ignore[method-assign]
        app.notify = lambda message, **_: notifications.append(message)  # type: ignore[method-assign]

        event = ChatInput.Submitted(type("StubInput", (), {"load_text": lambda self, _: None})(), "change direction")
        app.on_chat_input_submitted(event)

        self.assertEqual(queued, ["change direction"])
        self.assertEqual(notifications, ["steering queued"])

    def test_interrupted_turn_restarts_with_formatted_steering_message(self) -> None:
        app = QuasipilotApp()
        started: list[str] = []
        focused: list[bool] = []
        app._start_agent_turn = started.append  # type: ignore[method-assign]
        app.query_one = lambda *_args, **_kwargs: type("FocusStub", (), {"focus": lambda self: focused.append(True)})()  # type: ignore[method-assign]
        app._mount_chat_batch = lambda *widgets: None  # type: ignore[method-assign]

        app._finish_agent_turn("partial", None, "tighten scope")

        self.assertEqual(started, [format_live_steering_message("tighten scope")])
        self.assertEqual(focused, [])


class RuntimeBarTests(TestCase):
    def test_update_runtime_renders_curated_path_when_session_active(self) -> None:
        import asyncio

        async def run() -> None:
            app = QuasipilotApp()
            async with app.run_test():
                bar = app.query_one(RuntimeBar)
                bar.update_runtime(curated_path="/tmp/sessions/curated/abc.jsonl")
                rendered = bar.render()
                self.assertEqual(
                    str(rendered),
                    "gemini-3.1-flash-lite  ·  /Users/pepelopez/Documents/Programming/rnd.devtools.harness  ·  /tmp/sessions/curated/abc.jsonl",
                )
                self.assertTrue(all(not span.style or "link " not in str(span.style) for span in rendered.spans))

        asyncio.run(run())


class RuntimeConfigTests(TestCase):
    def test_configure_python_interpreter_updates_app_state(self) -> None:
        app = QuasipilotApp()

        app.configure_python_interpreter("~/venv/bin/python")

        self.assertEqual(app._python_interpreter, Path("~/venv/bin/python").expanduser().resolve())

    def test_load_session_restores_python_interpreter_from_latest_runtime_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = SessionManager(session_id="s1", root=root)
            manager.record_runtime(
                RuntimeSnapshot(
                    cwd=str(root),
                    git_branch="main",
                    git_dirty=False,
                    python_interpreter=str((root / ".venv/bin/python").resolve()),
                ),
                turn=1,
            )

            app = QuasipilotApp()
            app._build_compaction_coordinator = lambda manager: None  # type: ignore[method-assign]
            app._build_agent = lambda session_id: object()  # type: ignore[method-assign]
            app._clear_chat = lambda: None  # type: ignore[method-assign]
            app._render_history = lambda: None  # type: ignore[method-assign]
            app._sync_compaction_ui = lambda: None  # type: ignore[method-assign]

            with patch("cli.run.SessionManager", side_effect=lambda session_id: SessionManager(session_id=session_id, root=root)):
                app.load_session("s1")

            self.assertEqual(app._python_interpreter, (root / ".venv/bin/python").resolve())

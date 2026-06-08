from unittest import TestCase

from cli.components import ChatInput
from cli.components import RuntimeBar
from cli.components import StatusBubble
from cli.run import QuasipilotApp
from cli.utilities.display import content_to_plaintext
from core.live_steering import format_live_steering_message


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

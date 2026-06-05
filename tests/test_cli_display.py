from unittest import TestCase

from cli.components import ChatInput
from cli.components import RuntimeBar
from cli.components import StatusBubble
from cli.run import QuasipilotApp
from cli.utilities.display import content_to_plaintext


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
            app._set_busy = lambda busy: None  # type: ignore[method-assign]
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
            app._set_busy = lambda busy: None  # type: ignore[method-assign]
            app._show_spinner = lambda: None  # type: ignore[method-assign]
            app.run_turn = submitted.append  # type: ignore[method-assign]

            async with app.run_test() as pilot:
                await pilot.click("#chat-input")
                await pilot.press("h", "i", "ctrl+j", "t", "h", "e", "r", "e")
                snapshots.append(app.query_one(ChatInput).text)

        asyncio.run(run())

        self.assertEqual(submitted, [])
        self.assertEqual(snapshots, ["hi\nthere"])


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

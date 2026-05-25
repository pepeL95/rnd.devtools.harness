from unittest import TestCase

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

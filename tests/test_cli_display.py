from unittest import TestCase

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

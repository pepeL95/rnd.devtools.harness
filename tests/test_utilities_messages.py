from unittest import TestCase

from core.utilities.messages import (
    make_message,
    message_role,
    normalize_message_content,
    system_message_with_appended_text,
)
from langchain_core.messages import SystemMessage


class UtilitiesMessagesTests(TestCase):
    def test_message_role_maps_langchain_types(self) -> None:
        self.assertEqual(message_role(make_message("user", "hi")), "user")
        self.assertEqual(message_role(make_message("assistant", "hi")), "assistant")

    def test_system_message_with_appended_text_handles_none(self) -> None:
        updated = system_message_with_appended_text(None, "extra")
        self.assertIn("extra", str(updated.content))

    def test_normalize_message_content_from_text_blocks(self) -> None:
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        self.assertEqual(normalize_message_content(content), "hello\nworld")

    def test_system_message_with_appended_text_preserves_existing_blocks(self) -> None:
        base = SystemMessage(content="Base")
        updated = system_message_with_appended_text(base, "extra")
        self.assertIn("Base", str(updated.content))
        self.assertIn("extra", str(updated.content))

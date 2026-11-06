from unittest import TestCase

from core.utilities.messages import (
    make_message,
    message_role,
    message_text_content,
    normalize_message_content,
    system_message_with_appended_text,
)
from cli.utilities.messages import message_text
from langchain_core.messages import AIMessage, SystemMessage


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

    def test_message_text_content_skips_reasoning_blocks(self) -> None:
        class FakeMessage:
            content = [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]

        self.assertEqual(message_text_content(FakeMessage()), "hello\nworld")

    def test_refusal_is_visible_in_core_and_cli_without_reasoning(self) -> None:
        message = AIMessage(content=[
            {"type": "reasoning", "reasoning": "private"},
            {"type": "refusal", "refusal": "I cannot fulfill that request."},
        ])
        for extract in (message_text_content, message_text):
            self.assertEqual(extract(message), "I cannot fulfill that request.")

    def test_gemini_metadata_only_refusal_is_not_silent(self) -> None:
        for metadata in (
            {"finish_reason": "SAFETY"},
            {"prompt_feedback": {"block_reason": "SAFETY"}},
        ):
            message = AIMessage(content="", response_metadata={"model_provider": "google_genai", **metadata})
            for extract in (message_text_content, message_text):
                with self.subTest(metadata=metadata, extract=extract):
                    self.assertEqual(extract(message), "The model declined this request.")

    def test_system_message_with_appended_text_preserves_existing_blocks(self) -> None:
        base = SystemMessage(content="Base")
        updated = system_message_with_appended_text(base, "extra")
        self.assertIn("Base", str(updated.content))
        self.assertIn("extra", str(updated.content))

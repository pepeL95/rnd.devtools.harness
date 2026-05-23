from unittest import TestCase
from unittest.mock import MagicMock

from cli.slash_commands.registry import SlashCommandRegistry


class SlashCommandRegistryTests(TestCase):
    def test_recognizes_and_dispatches_exit(self) -> None:
        registry = SlashCommandRegistry()
        app = MagicMock()

        self.assertTrue(registry.is_known_command("/exit"))
        should_exit = registry.dispatch(app, "/exit")

        self.assertTrue(should_exit)
        app.exit.assert_called_once()

    def test_unknown_slash_is_not_known(self) -> None:
        registry = SlashCommandRegistry()

        self.assertTrue(registry.is_slash("/session"))
        self.assertFalse(registry.is_known_command("/session"))
        self.assertEqual(registry.slash_name("/session"), "session")

    def test_plain_slash_is_unknown(self) -> None:
        registry = SlashCommandRegistry()

        self.assertTrue(registry.is_slash("/"))
        self.assertFalse(registry.is_known_command("/"))

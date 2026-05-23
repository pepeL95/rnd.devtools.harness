from unittest import TestCase
from unittest.mock import MagicMock

from cli.slash_commands.registry import SlashCommandRegistry


class SlashCommandRegistryTests(TestCase):
    def test_recognizes_and_dispatches_exit(self) -> None:
        registry = SlashCommandRegistry()
        app = MagicMock()

        self.assertTrue(registry.is_command("/exit"))
        registry.dispatch(app, "/exit")

        app.exit.assert_called_once()

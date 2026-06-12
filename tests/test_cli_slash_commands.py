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

    def test_compact_dispatches_manual_compaction(self) -> None:
        registry = SlashCommandRegistry()
        app = MagicMock()
        app._compaction_coordinator.request_manual_compaction.return_value = "started"

        should_exit = registry.dispatch(app, "/compact")

        self.assertFalse(should_exit)
        app.trigger_manual_compaction.assert_called_once()

    def test_python_dispatches_runtime_reconfiguration(self) -> None:
        registry = SlashCommandRegistry()
        app = MagicMock()

        should_exit = registry.dispatch(app, "/python ~/venv/bin/python")

        self.assertFalse(should_exit)
        app.configure_python_interpreter.assert_called_once()
        app.notify.assert_called_once()

    def test_python_requires_path_argument(self) -> None:
        registry = SlashCommandRegistry()
        app = MagicMock()

        should_exit = registry.dispatch(app, "/python")

        self.assertFalse(should_exit)
        app.notify_warning.assert_called_once_with("usage: /python /absolute/path/to/python")

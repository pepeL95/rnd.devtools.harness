from __future__ import annotations

from cli.slash_commands.base import SlashCommand
from cli.slash_commands.compact import CompactCommand
from cli.slash_commands.copy import CopyCommand
from cli.slash_commands.clear import ClearCommand
from cli.slash_commands.exit import ExitCommand
from cli.slash_commands.pop import PopCommand
from cli.slash_commands.python import PythonCommand
from cli.slash_commands.sessions import SessionsCommand


class SlashCommandRegistry:
    """Dispatch slash commands by name."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        for command in (SessionsCommand(), PythonCommand(), CompactCommand(), ClearCommand(), PopCommand(), ExitCommand(), CopyCommand()):
            self._commands[command.name] = command

    @staticmethod
    def is_slash(raw: str) -> bool:
        return raw.strip().startswith("/")

    def slash_name(self, raw: str) -> str:
        body = raw.strip()[1:].strip()
        if not body:
            return ""
        return body.split(maxsplit=1)[0].lower()

    def is_known_command(self, raw: str) -> bool:
        if not self.is_slash(raw):
            return False
        return self.slash_name(raw) in self._commands

    def dispatch(self, app, raw: str) -> bool:
        """Run a known slash command. Returns True when the app should exit."""

        name = self.slash_name(raw)
        command = self._commands.get(name)
        if command is None:
            return False
        _, _, args = raw.strip()[1:].strip().partition(" ")
        return command.run(app, args.strip())

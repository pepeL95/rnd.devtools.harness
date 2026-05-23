from __future__ import annotations

from cli.slash_commands.base import SlashCommand
from cli.slash_commands.clear import ClearCommand
from cli.slash_commands.exit import ExitCommand
from cli.slash_commands.sessions import SessionsCommand


class SlashCommandRegistry:
    """Dispatch slash commands by name."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        for command in (SessionsCommand(), ClearCommand(), ExitCommand()):
            self._commands[command.name] = command

    def dispatch(self, app, raw: str) -> bool:
        stripped = raw.strip()
        if not stripped.startswith("/"):
            return False
        body = stripped[1:].strip()
        if not body:
            return False
        name, _, args = body.partition(" ")
        command = self._commands.get(name.lower())
        if command is None:
            return False
        return command.run(app, args.strip())

    def is_command(self, raw: str) -> bool:
        stripped = raw.strip()
        if not stripped.startswith("/"):
            return False
        name = stripped[1:].strip().split(maxsplit=1)[0].lower()
        return name in self._commands

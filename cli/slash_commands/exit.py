from __future__ import annotations

from cli.slash_commands.base import SlashCommand


class ExitCommand(SlashCommand):
    name = "exit"

    def run(self, app, args: str) -> bool:
        app.exit()
        return True

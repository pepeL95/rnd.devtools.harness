from __future__ import annotations

from cli.slash_commands.base import SlashCommand


class NewCommand(SlashCommand):
    name = "new"

    def run(self, app, args: str) -> bool:
        app.new_session()
        return False

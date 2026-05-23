from __future__ import annotations

from cli.slash_commands.base import SlashCommand
from cli.utilities.sessions import clear_session_files


class ClearCommand(SlashCommand):
    name = "clear"

    def run(self, app, args: str) -> bool:
        if app.session_id:
            clear_session_files(app.session_id)
        app.reset_session()
        return False

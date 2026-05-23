from __future__ import annotations

from cli.components.session_picker import SessionPickerScreen
from cli.slash_commands.base import SlashCommand
from cli.utilities.sessions import list_sessions


class SessionsCommand(SlashCommand):
    name = "sessions"

    def run(self, app, args: str) -> bool:
        def on_pick(session_id: str | None) -> None:
            if session_id:
                app.load_session(session_id)

        app.push_screen(SessionPickerScreen(list_sessions()), on_pick)
        return False

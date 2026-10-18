from __future__ import annotations

from cli.slash_commands.base import SlashCommand


class PreferencesCommand(SlashCommand):
    name = "preferences"

    def run(self, app, args: str) -> bool:
        app.trigger_preferences_update(args)
        return False

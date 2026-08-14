from __future__ import annotations

from cli.slash_commands.base import SlashCommand


class DevProfileCommand(SlashCommand):
    name = "devprofile"

    def run(self, app, args: str) -> bool:
        app.trigger_dev_profile_update(args)
        return False

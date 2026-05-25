from __future__ import annotations

from cli.slash_commands.base import SlashCommand


class CompactCommand(SlashCommand):
    name = "compact"

    def run(self, app, args: str) -> bool:
        app.trigger_manual_compaction()
        return False

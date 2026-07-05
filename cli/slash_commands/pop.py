from __future__ import annotations

from cli.slash_commands.base import SlashCommand


class PopCommand(SlashCommand):
    """Remove the latest turn from both session streams and re-render the history."""

    name = "pop"

    def run(self, app, args: str) -> bool:
        manager = app.manager
        if manager is None:
            app.notify_warning("no active session")
            return False
        removed = manager.pop_turn()
        if removed is None:
            app.notify_warning("session is already empty")
            return False
        app._clear_chat()
        app._render_history()
        app.notify(f"turn {removed} removed", timeout=2, markup=False)
        return False

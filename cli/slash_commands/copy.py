from __future__ import annotations

import subprocess
from cli.slash_commands.base import SlashCommand
from core.session.events import EventType


class CopyCommand(SlashCommand):
    name = "copy"

    def run(self, app, args: str) -> bool:
        if app.manager is None:
            app.notify_warning("No active session.")
            return False

        history = app.manager.read_display_history()
        # Find the last agent message
        last_agent_message = None
        for event in reversed(history):
            if event.type == EventType.ASSISTANT:
                last_agent_message = event.payload.get("content", "")
                break

        if not last_agent_message:
            app.notify_warning("No agent message found.")
            return False

        # Copy to clipboard
        try:
            # Try pbcopy (macOS)
            subprocess.run(
                ["pbcopy"], input=last_agent_message.encode("utf-8"), check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Try xclip (Linux)
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=last_agent_message.encode("utf-8"),
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                app.notify_warning("Could not copy to clipboard (pbcopy/xclip not found).")
                return False

        app.notify("Copied latest agent message to clipboard.")
        return False

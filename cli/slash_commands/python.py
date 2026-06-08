from __future__ import annotations

from pathlib import Path

from cli.slash_commands.base import SlashCommand


class PythonCommand(SlashCommand):
    name = "python"

    def run(self, app, args: str) -> bool:
        raw = args.strip()
        if not raw:
            app.notify_warning("usage: /python /absolute/path/to/python")
            return False

        interpreter = Path(raw).expanduser().resolve()
        app.configure_python_interpreter(interpreter)
        app.notify(f"python interpreter set to {interpreter}", timeout=3, markup=False)
        return False

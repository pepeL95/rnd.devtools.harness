from __future__ import annotations

from cli.components.model_picker import ModelPickerScreen
from cli.slash_commands.base import SlashCommand


class ModelsCommand(SlashCommand):
    name = "models"

    def run(self, app, args: str) -> bool:
        models = app.load_models()
        if not models:
            app.notify_warning("no models available")
            return False

        def on_pick(model_name: str | None) -> None:
            if not model_name:
                return
            app.configure_model(model_name)
            app.notify(f"model set to {model_name}", timeout=3, markup=False)

        app.push_screen(ModelPickerScreen(models), on_pick)
        return False

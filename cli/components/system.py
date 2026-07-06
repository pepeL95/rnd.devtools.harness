from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, Rule


class System(Horizontal):
    """A custom component wrapping a system message between left and right dividers."""

    DEFAULT_CSS = """
    System {
        /* A height of 1 over-constrains cell bounds. Using 2 allows the 
           rendering engine room to balance rule strokes and lowercase baselines. */
        height: 2;
    }
    
    System Rule {
        color: $text-muted;
    }
    
    System Label {
        width: auto;
        /* Corrects vertical alignment by shifting the character glyphs 
           away from the bottom cell boundary to the exact midline. */
        padding: 0 1;
        margin: 1 0;
        color: $text-muted;
    }
    """

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        # Left side divider line
        yield Rule(line_style="solid")
        # Centered message
        yield Label(self.message)
        # Right side divider line
        yield Rule(line_style="solid")

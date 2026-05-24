from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input

from agents.driver.agent import DriverAgentConfig, create_driver_agent
from cli.components import (
    AIBubble,
    ChatInput,
    ReasonStream,
    RuntimeBar,
    ToolStream,
    UserBubble,
    WorkingSpinner,
    Divider,
)
from cli.slash_commands.registry import SlashCommandRegistry
from cli.utilities.display import content_to_plaintext
from cli.utilities.streaming import iter_agent_turn
from core.session.events import EventType
from core.session.session_manager import SessionManager
from core.utilities.defaults import get_default_model, get_model_name

class AgentStream(Message):
    """Worker thread event for live tool/reason updates."""

    def __init__(self, kind: str, payload: dict) -> None:
        self.kind = kind
        self.payload = payload
        super().__init__()


class AgentFinished(Message):
    """Worker completed a turn."""

    def __init__(self, text: str, error: str | None = None) -> None:
        self.text = text
        self.error = error
        super().__init__()


class QuasipilotApp(App):
    """Textual chat shell for the driver agent."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-scroll {
        height: 1fr;
        
    }

    #chat-log {
        width: 100%;
        height: auto;
    }

    #bottom-bar {
        height: auto;
        width: 100%;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._cwd = Path.cwd()
        self._model = get_default_model()
        self.session_id: str | None = None
        self._manager: SessionManager | None = None
        self._agent = None
        self._commands = SlashCommandRegistry()
        self._busy = False
        self._spinner: WorkingSpinner | None = None
        self._pending_auto_scroll = False

    @property
    def manager(self) -> SessionManager | None:
        return self._manager

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-scroll"):
            yield Vertical(id="chat-log")
        with Vertical(id="bottom-bar"):
            yield ChatInput()
            yield RuntimeBar(get_model_name(self._model), str(self._cwd))

    def on_mount(self) -> None:
        self.query_one(ChatInput).focus()

    def notify_warning(self, message: str) -> None:
        self.notify(message, timeout=3, markup=False, severity="warning")

    def reset_session(self) -> None:
        self.session_id = None
        self._manager = None
        self._agent = None
        self._clear_chat()

    def load_session(self, session_id: str) -> None:
        self.session_id = session_id
        self._manager = SessionManager(session_id=session_id)
        self._agent = create_driver_agent(
            DriverAgentConfig(cwd=self._cwd, model=self._model, session_id=session_id)
        )
        self._clear_chat()
        self._render_history()

    def _ensure_session(self) -> SessionManager:
        if self._manager is None:
            self._manager = SessionManager()
            self.session_id = self._manager.session_id
            self._agent = create_driver_agent(
                DriverAgentConfig(cwd=self._cwd, model=self._model, session_id=self.session_id)
            )
        return self._manager

    def _chat_log(self) -> Vertical:
        return self.query_one("#chat-log", Vertical)

    def _chat_scroll(self) -> VerticalScroll:
        return self.query_one("#chat-scroll", VerticalScroll)

    def _is_chat_at_bottom(self) -> bool:
        scroll = self._chat_scroll()
        return scroll.max_scroll_y <= 0 or scroll.scroll_y >= max(scroll.max_scroll_y - 1, 0)

    def _scroll_chat_to_bottom(self, *, animate: bool = False) -> None:
        # Wait for the next refresh so max_scroll_y includes newly mounted content.
        if self._pending_auto_scroll:
            return
        self._pending_auto_scroll = True
        self.call_after_refresh(self._flush_pending_auto_scroll, animate)

    def _flush_pending_auto_scroll(self, animate: bool) -> None:
        self._pending_auto_scroll = False
        self._chat_scroll().scroll_end(animate=animate, immediate=False)

    def _mount_chat(self, widget: Widget) -> None:
        """Mount a widget and follow the end only while the user is pinned to the bottom."""
        should_follow = self._pending_auto_scroll or self._is_chat_at_bottom()
        self._chat_log().mount(widget)
        if should_follow:
            self._scroll_chat_to_bottom()

    def _mount_chat_batch(self, *widgets: Widget) -> None:
        """Mount a related batch of widgets using one bottom snapshot."""
        should_follow = self._pending_auto_scroll or self._is_chat_at_bottom()
        chat = self._chat_log()
        for widget in widgets:
            chat.mount(widget)
        if should_follow:
            self._scroll_chat_to_bottom()

    def _clear_chat(self) -> None:
        self._chat_log().remove_children()
        self._chat_scroll().scroll_home(animate=False)

    def _render_history(self) -> None:
        if self._manager is None:
            return
        for event in self._manager.read_curated():
            if event.type not in {EventType.USER, EventType.ASSISTANT}:
                continue
            content = content_to_plaintext(event.payload.get("content", ""))
            if event.type == EventType.USER:
                self._chat_log().mount(UserBubble(content))
            else:
                self._mount_chat_batch(Divider(), AIBubble(content))
        self._scroll_chat_to_bottom()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        chat_input = self.query_one(ChatInput)
        chat_input.disabled = busy

    def _show_spinner(self) -> None:
        if self._spinner is not None:
            return
        self._spinner = WorkingSpinner()
        self._mount_chat(self._spinner)

    def _hide_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.remove()
            self._spinner = None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, ChatInput):
            return
        text = event.value.strip()
        event.input.value = ""
        if not text or self._busy:
            return

        if self._commands.is_slash(text):
            if self._commands.is_known_command(text):
                self._commands.dispatch(self, text)
            else:
                name = self._commands.slash_name(text) or "?"
                self.notify_warning(f"unknown command: /{name}")
            return

        self._mount_chat(UserBubble(text))
        self._set_busy(True)
        self._show_spinner()
        self.run_turn(text)

    @work(thread=True, exclusive=True)
    def run_turn(self, text: str) -> None:
        error: str | None = None
        assistant_text = ""
        try:
            self._ensure_session()
            assert self._agent is not None

            def on_event(kind: str, payload: dict) -> None:
                self.post_message(AgentStream(kind, payload))

            assistant_text = iter_agent_turn(self._agent, text, on_event)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            error = str(exc)
        self.post_message(AgentFinished(assistant_text, error))

    def on_agent_stream(self, event: AgentStream) -> None:
        if event.kind == "tool":
            self._mount_chat(ToolStream(event.payload.get("name", "tool"), event.payload.get("args", "")))
        elif event.kind == "reason":
            self._mount_chat_batch(Divider(), ReasonStream(event.payload.get("text", "")))

    def on_agent_finished(self, event: AgentFinished) -> None:
        self._hide_spinner()
        self._set_busy(False)
        if event.error:
            self._mount_chat_batch(Divider(), AIBubble(f"error: {event.error}"))
        elif event.text:
            self._mount_chat_batch(Divider(), AIBubble(event.text))
        self.query_one(ChatInput).focus()


def main() -> None:
    QuasipilotApp().run()


if __name__ == "__main__":
    main()

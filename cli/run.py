from __future__ import annotations

from datetime import date
from pathlib import Path
from threading import Event

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from agents.driver.agent import DriverAgentConfig, create_driver_agent
from cli.components import (
    AIBubble,
    ChatInput,
    StatusBubble,
    ReasonStream,
    RuntimeBar,
    ToolStream,
    UserBubble,
    WorkingSpinner,
    Divider,
    System,
)
from cli.slash_commands.registry import SlashCommandRegistry
from cli.utilities.display import content_to_plaintext
from cli.utilities.streaming import iter_agent_turn
from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.live_steering import CancellationInterrupt, LiveSteeringController, LiveSteeringInterrupt
from core.session.events import EventType
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path
from core.utilities.defaults import get_default_driver_model, get_model_name

class AgentStream(Message):
    """Worker thread event for live tool/reason updates."""

    def __init__(self, kind: str, payload: dict) -> None:
        self.kind = kind
        self.payload = payload
        super().__init__()


class AgentFinished(Message):
    """Worker completed a turn."""

    def __init__(self, text: str, error: str | None = None, steering: str | None = None, cancelled: bool = False) -> None:
        self.text = text
        self.error = error
        self.steering = steering
        self.cancelled = cancelled
        super().__init__()


class QuasipilotApp(App[None]):
    """Textual chat shell for the driver agent."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-scroll {
        height: 1fr;
        scrollbar-background: transparent;
        scrollbar-color: $text-muted;
        scrollbar-size: 0 1;
    }

    #bottom-bar {
        height: auto;
        width: 100%;
    }

    #spinner-container {
        height: 2;
    }

    ChatInput:disabled {
        opacity: 0.65;
        background: #272c34;
    }

    .canceled-message {
        width: 100%;
        text-align: center;
        color: $text-muted;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit"), ("escape", "cancel_turn", "Cancel")]

    def __init__(self) -> None:
        super().__init__(ansi_color=True)
        self._cwd = Path.cwd()
        self._model = get_default_driver_model()
        self._python_interpreter: Path | None = None
        self.session_id: str | None = None
        self._manager: SessionManager | None = None
        self._agent = None
        self._commands = SlashCommandRegistry()
        self._busy = False
        self._agent_active = False
        self._spinner: WorkingSpinner | None = None
        self._compaction_coordinator: CompactionCoordinator | None = None
        self._compaction_active = False
        self._live_steering = LiveSteeringController()
        self._cancel_event = Event()
        self._cancellation_pending = False
        self._tool_streams: dict[str, ToolStream] = {}

    @property
    def manager(self) -> SessionManager | None:
        return self._manager

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-scroll")
        with Vertical(id="bottom-bar"):
            yield Vertical(id="spinner-container")
            yield ChatInput()
            yield RuntimeBar(get_model_name(self._model), str(self._cwd))

    def on_mount(self) -> None:
        self.screen.styles.background = "transparent"
        self.query_one(ChatInput).focus()
        self._sync_compaction_ui()

    def notify_warning(self, message: str) -> None:
        self.notify(message, timeout=3, markup=False, severity="warning")

    def reset_session(self) -> None:
        self.session_id = None
        self._manager = None
        self._agent = None
        self._live_steering = LiveSteeringController()
        self._cancel_event = Event()
        self._cancellation_pending = False
        self._tool_streams = {}
        self._compaction_coordinator = None
        self._compaction_active = False
        self._clear_chat()
        self._sync_compaction_ui()

    def load_session(self, session_id: str) -> None:
        self.session_id = session_id
        self._live_steering = LiveSteeringController()
        self._cancellation_pending = False
        self._tool_streams = {}
        self._manager = SessionManager(session_id=session_id)
        self._restore_runtime_from_session(self._manager)
        self._compaction_coordinator = self._build_compaction_coordinator(self._manager)
        self._agent = self._build_agent(session_id)
        self._clear_chat()
        self._render_history()
        self._sync_compaction_ui()

    def _ensure_session(self) -> SessionManager:
        if self._manager is None:
            self._manager = SessionManager()
            self.session_id = self._manager.session_id
            self._compaction_coordinator = self._build_compaction_coordinator(self._manager)
            self._agent = self._build_agent(self.session_id)
            self._sync_compaction_ui()
        return self._manager

    def _chat_scroll(self) -> VerticalScroll:
        return self.query_one("#chat-scroll", VerticalScroll)

    def _scroll_chat_to_bottom(self) -> None:
        self.call_after_refresh(lambda: self._chat_scroll().scroll_end(animate=False))

    def _mount_chat(self, widget: Widget) -> None:
        self._chat_scroll().mount(widget)
        self._scroll_chat_to_bottom()

    def _mount_chat_batch(self, *widgets: Widget) -> None:
        chat = self._chat_scroll()
        for widget in widgets:
            chat.mount(widget)
        self._scroll_chat_to_bottom()

    def _clear_chat(self) -> None:
        self._tool_streams = {}
        self._chat_scroll().remove_children()
        self._chat_scroll().scroll_home(animate=False)

    def _render_history(self) -> None:
        if self._manager is None:
            return
        chat = self._chat_scroll()
        for event in self._manager.read_display_history():
            content = content_to_plaintext(event.payload.get("content", ""))
            if event.type == EventType.USER:
                chat.mount(UserBubble(content))
            else:
                chat.mount(Divider())
                chat.mount(AIBubble(content))
        self._scroll_chat_to_bottom()

    def _build_agent(self, session_id: str | None):
        return create_driver_agent(
            DriverAgentConfig(
                cwd=self._cwd,
                model=self._model,
                python_interpreter=self._python_interpreter,
                session_id=session_id,
                session_manager=self._manager,
                on_compaction_event=self._post_compaction_event,
                session_compaction_coordinator=self._compaction_coordinator,
                live_steering_controller=self._live_steering,
                cancel_event=self._cancel_event,
            )
        )

    def configure_python_interpreter(self, path: str | Path | None) -> None:
        self._python_interpreter = Path(path).expanduser().resolve() if path else None
        if self._manager is not None:
            self._agent = self._build_agent(self.session_id)

    def _restore_runtime_from_session(self, manager: SessionManager) -> None:
        snapshot = manager.latest_runtime_snapshot()
        interpreter = snapshot.python_interpreter if snapshot is not None else None
        self._python_interpreter = Path(interpreter).expanduser().resolve() if interpreter else None

    def _build_compaction_coordinator(self, manager: SessionManager) -> CompactionCoordinator:
        return CompactionCoordinator(
            manager,
            Compactor(policy=CompactionPolicy()),
            on_compaction_event=self._post_compaction_event,
            telemetry_store=TelemetryStore(telemetry_session_path(manager.session_id)),
        )

    def _set_busy(self, busy: bool, *, disable_input: bool = True) -> None:
        self._busy = busy
        chat_input = self.query_one(ChatInput)
        chat_input.disabled = busy and disable_input

    def _mount_spinner(self, widget: Widget) -> None:
        self.query_one("#spinner-container").mount(widget)

    def _show_spinner(self) -> None:
        if self._spinner is not None:
            return
        self._spinner = WorkingSpinner()
        self._mount_spinner(self._spinner)

    def _set_spinner_status(self, status: str) -> None:
        if self._spinner is None:
            self._spinner = WorkingSpinner(status=status)
            self._mount_spinner(self._spinner)
            return
        self._spinner.set_status(status)

    def _hide_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.remove()
            self._spinner = None

    def _post_compaction_event(self, phase: str, payload: dict) -> None:
        self.call_from_thread(self._handle_compaction_event, phase, payload)

    def _handle_compaction_event(self, phase: str, payload: dict) -> None:
        content = content_to_plaintext(payload.get("content", ""))
        if phase == "start":
            self._compaction_active = True
            token_usage = payload.get("token_usage", {})
            tokens = token_usage.get("source_tokens") if isinstance(token_usage, dict) else None
            suffix = f" · {tokens} tokens" if isinstance(tokens, int) else ""
            self._sync_compaction_ui()
            if self._spinner is not None:
                self._set_spinner_status(f"compacting session{suffix}")
            if content:
                self._mount_chat(StatusBubble(content))
            self.notify("session compaction started", timeout=2, markup=False)
        elif phase == "end":
            self._compaction_active = False
            self._sync_compaction_ui()
            if self._spinner is not None:
                self._set_spinner_status("working")
            if content:
                self._mount_chat(StatusBubble(content))
            self.notify("session compaction finished", timeout=2, markup=False)
        elif phase == "error":
            self._compaction_active = False
            self._sync_compaction_ui()
            if content:
                self._mount_chat(StatusBubble(content))
            self.notify_warning(f"session compaction failed: {payload.get('error', 'unknown error')}")

    def trigger_manual_compaction(self) -> None:
        if self._manager is None or self._compaction_coordinator is None:
            self.notify_warning("no active session to compact")
            return
        if self._busy:
            self.notify_warning("wait for the active operation to finish")
            return
        status = self._compaction_coordinator.request_manual_compaction()
        if status == "running":
            self.notify_warning("session compaction already running")
        elif status != "started":
            self.notify_warning("session compaction was not needed")

    def _sync_compaction_ui(self) -> None:
        status = "compacting session" if self._compaction_active else None
        curated_path = str(self._manager.curated_path) if self._manager is not None else None
        self.query_one(RuntimeBar).update_runtime(curated_path=curated_path, status=status)

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        event.input.load_text("")
        if not text:
            return

        if self._busy:
            if self._cancellation_pending:
                self.notify_warning("cancellation in progress")
            elif self._agent_active:
                self._live_steering.submit(text)
                self.notify("steering queued", timeout=2, markup=False)
            else:
                self.notify_warning("wait for the active operation to finish")
            return

        if self._commands.is_slash(text):
            if self._commands.is_known_command(text):
                self._commands.dispatch(self, text)
            else:
                name = self._commands.slash_name(text) or "?"
                self.notify_warning(f"unknown command: /{name}")
            return

        self._start_agent_turn(text)

    def _start_agent_turn(self, text: str) -> None:
        self._mount_chat(UserBubble(text))
        self._agent_active = True
        self._cancellation_pending = False
        self._cancel_event.clear()
        self._set_busy(True, disable_input=False)
        self._show_spinner()
        self.run_turn(text)

    def _resume_interrupted_turn(self, steering: str) -> None:
        self._mount_chat(UserBubble(steering))
        self._agent_active = True
        self._cancellation_pending = False
        self._cancel_event.clear()
        self._set_busy(True, disable_input=False)
        self._show_spinner()
        self.run_turn(None)

    def action_cancel_turn(self) -> None:
        if self._agent_active and not self._cancellation_pending:
            self._cancellation_pending = True
            self._cancel_event.set()
            self._set_busy(True, disable_input=True)
            self.notify("cancelling…", timeout=2, markup=False)

    @work(thread=True, exclusive=True)
    def run_turn(self, text: str | None) -> None:
        error: str | None = None
        assistant_text = ""
        steering: str | None = None
        cancelled = False
        try:
            self._ensure_session()
            assert self._agent is not None

            def on_event(kind: str, payload: dict) -> None:
                self.post_message(AgentStream(kind, payload))

            assistant_text = iter_agent_turn(self._agent, text, on_event)
        except CancellationInterrupt:
            cancelled = True
        except LiveSteeringInterrupt as exc:
            steering = exc.steering
        except Exception as exc:  # pragma: no cover - surfaced in UI
            error = str(exc)
        self.post_message(AgentFinished(assistant_text, error, steering, cancelled=cancelled))

    def on_agent_stream(self, event: AgentStream) -> None:
        if event.kind == "tool":
            widget = ToolStream(
                name=event.payload.get("name", "tool"),
                input_text=event.payload.get("input", ""),
            )
            tool_call_id = str(event.payload.get("tool_call_id") or "")
            if tool_call_id:
                self._tool_streams[tool_call_id] = widget
            self._mount_chat(widget)
        elif event.kind == "tool_output":
            tool_call_id = str(event.payload.get("tool_call_id") or "")
            widget = self._tool_streams.get(tool_call_id) if tool_call_id else None
            if widget is not None:
                widget.set_output(event.payload.get("output"))
                self._scroll_chat_to_bottom()
            else:
                self._mount_chat(
                    ToolStream(
                        name=event.payload.get("name", "tool"),
                        input_text=event.payload.get("input", ""),
                        output=event.payload.get("output"),
                    )
                )
        elif event.kind == "reason":
            self._mount_chat_batch(Divider(), ReasonStream(event.payload.get("text", "")))

    def on_agent_finished(self, event: AgentFinished) -> None:
        self.call_after_refresh(self._finish_agent_turn, event.text, event.error, event.steering, event.cancelled)

    def _finish_agent_turn(self, text: str, error: str | None, interrupted_steering: str | None = None, cancelled: bool = False) -> None:
        self._cancellation_pending = False
        if error:
            self._mount_chat_batch(Divider(), AIBubble(f"error: {error}"))
        elif cancelled:
            self._mount_chat_batch(System("Canceled Request"))
        elif text:
            self._mount_chat_batch(Divider(), AIBubble(text))
        next_steering = interrupted_steering or self._live_steering.drain()
        if next_steering:
            # Re-enter the same logical turn from the restored transcript. The
            # steering message is already persisted on the active turn, so we
            # resume without injecting a second HumanMessage into agent state.
            self._resume_interrupted_turn(next_steering)
            return
        self._agent_active = False
        self._hide_spinner()
        self._set_busy(False)
        self.query_one(ChatInput).focus()


def main() -> None:
    QuasipilotApp().run()


if __name__ == "__main__":
    main()

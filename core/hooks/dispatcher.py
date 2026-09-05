from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from core.hooks.code import run_code
from core.hooks.commands import ParsedCommand
from core.hooks.config import Hook, Trigger
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.session.turns import HARNESS_CONTEXT_KIND

logger = logging.getLogger(__name__)


class HookDispatcher:
    def __init__(self, hooks: tuple[Hook, ...], manager: SessionManager, cwd: Path,
                 python_interpreter: Path | None = None) -> None:
        self.hooks = hooks
        self.manager = manager
        self.cwd = cwd
        self.python_interpreter = python_interpreter
        self.turn: int | None = None
        self._active = False
        self.passes: Counter[str] = Counter()
        self.env = os.environ.copy()
        if python_interpreter:
            self.env["PATH"] = str(python_interpreter.parent) + os.pathsep + self.env.get("PATH", "")

    def start_turn(self, turn: int) -> bool:
        if self._active and self.turn == turn:
            return False
        self.turn = turn
        self._active = True
        self.passes.clear()
        return True

    def finish_turn(self) -> None:
        self._active = False

    def dispatch(self, trigger: Trigger, data: dict[str, Any],
                 command: ParsedCommand | None = None) -> list[HumanMessage]:
        messages = []
        for hook in self.hooks:
            config = hook.config
            if config.trigger != trigger:
                continue
            match = config.match
            if match.tool is not None and data.get("tool", {}).get("name") not in match.tool:
                continue
            if trigger.endswith(".command"):
                if command is None:
                    continue
                if match.cmd is not None and command.cmd not in match.cmd:
                    continue
                if match.args is not None and command.args[:len(match.args)] != tuple(match.args):
                    continue
            if trigger == "after.turn":
                if self.passes[config.id] >= config.recursion:
                    continue
                self.passes[config.id] += 1
            snapshot = self.manager.latest_runtime_snapshot()
            event = {
                "schema_version": 1, "hook_id": config.id, "trigger": trigger,
                "session_id": self.manager.session_id, "turn": self.turn,
                "runtime": {**(asdict(snapshot) if snapshot else {}), "cwd": str(self.cwd), "python_interpreter": str(self.python_interpreter) if self.python_interpreter else None},
                **data,
            }
            try:
                if config.type == "steering":
                    content = hook.path.read_text(encoding="utf-8").strip()
                    if content:
                        messages.append(HumanMessage(
                            content=f"[HARNESS CONTEXT: {config.id} / {trigger}]\n{content}\n[END HARNESS CONTEXT]",
                            additional_kwargs={"session_kind": HARNESS_CONTEXT_KIND,
                                               "hook_id": config.id, "trigger": trigger},
                        ))
                else:
                    result = run_code(hook.path, event, self.cwd, self.env)
                    self.diagnostic(config.id, trigger, result)
            except (OSError, UnicodeError, ValueError) as exc:
                self.diagnostic(config.id, trigger, {"status": "error", "error": str(exc)})
        return messages

    def persist(self, messages: list[HumanMessage]) -> None:
        self.manager.append([
            SessionEvent(type=EventType.USER, turn=self.turn or self.manager.next_turn(), payload={
                "role": "user", "kind": HARNESS_CONTEXT_KIND, "source": "hook",
                "hook_id": message.additional_kwargs["hook_id"],
                "trigger": message.additional_kwargs["trigger"], "content": message.content,
            }) for message in messages
        ])

    def diagnostic(self, hook_id: str | None, trigger: str, result: dict[str, Any]) -> None:
        self.manager.append([SessionEvent(
            type=EventType.META, turn=self.turn or self.manager.next_turn(),
            payload={"kind": "hook_result", "hook_id": hook_id, "trigger": trigger, **result},
        )], curated=False)
        if result.get("status") in {"error", "timeout"}:
            logger.warning("Hook %s (%s): %s", hook_id, trigger, result)

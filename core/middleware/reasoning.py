from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import tool

from core.live_steering import LiveSteeringInterrupt
from core.utilities.messages import system_message_with_appended_text


ReasoningEagerness = Literal["low", "medium", "high"]

REASONING_STEERING_PROMPTS: dict[ReasoningEagerness, str] = {
    # LOW EARGERNESS
    "low": """\
Use the `reasoning` tool selectively to externalize short, useful reasoning checkpoints.

Reach for it when the task meaningfully changes shape, when a result is ambiguous, after long reads, or before retrying after failure.
Keep reasoning concise and decision-oriented. Prefer task-aware notes about what changed, what you now believe, and what you will do next.""",
    # MEDIUM EARGERNESS
    "medium": """\
Use the `reasoning` tool regularly to externalize short, useful reasoning checkpoints.

Reach for it:
- when starting substantive work
- before major decisions
- when you discover something that changes the task shape
- after long reads
- before retrying after an error

Keep reasoning concise and decision-oriented. Prefer task-aware notes about what changed, what you now believe, and what you will do next.""",
    # HIGH EARGERNESS
    "high": """\
Your reasoning eagerness has been set to **Highest**.
Use the `reasoning` tool very often to write introspective state representations where you analyze your current state, your discoveries, and what you will do next.

Invoke the `reasoning` tool:
- immediately after a work request, before doing any work on it
- before using any filesystem or external tool, to clarify intent and expected resutls.
- when you discover something that changes the task shape
- after long reads
- before retrying after an error

## Output Format
- Write rich, flowy prose, with introspective thoughts that analyze your current state, what you just learned, and what you will do next.
- Organize your thoughts cohesively, following a structure that keeps continuous learning coherent. Users must be able to read through your reasoning and understand the evolution of your thinking over time.
""",
}

TOOL_FAILURE_REASONING_REMINDER = """A tool just failed. Use the `reasoning` tool before moving forward so you explicitly assess the failure mechanism, what changed, and the next best move."""

READ_FILE_REASONING_REMINDER = """You just received a long `read_file` result. Use the `reasoning` tool to synthesize what was actually relevant from that output, what constraints or discoveries matter, and what can be ignored. This is primarily to preserve signal for later compaction."""

LONG_READ_FILE_CHARS = 2000


class ReasoningMiddleware(AgentMiddleware):
    """Steer the agent to reason explicitly at pivots and after tool failures."""

    def __init__(self, eagerness: ReasoningEagerness = "low") -> None:
        if eagerness not in REASONING_STEERING_PROMPTS:
            raise ValueError("Reasoning eagerness must be one of: low, medium, high.")
        self.eagerness = eagerness
        self._pending_reminder: str | None = None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        prompt = REASONING_STEERING_PROMPTS[self.eagerness]
        if self._pending_reminder:
            prompt = "\n\n".join([prompt, self._pending_reminder])
            self._pending_reminder = None
        updated = request.override(
            system_message=system_message_with_appended_text(request.system_message, prompt)
        )
        return handler(updated)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        try:
            result = handler(request)
        except LiveSteeringInterrupt:
            raise
        except Exception:
            self._pending_reminder = TOOL_FAILURE_REASONING_REMINDER
            raise
        if _should_reason_after_read_file(request, result):
            self._pending_reminder = READ_FILE_REASONING_REMINDER
        return result


@tool("reasoning")
def reasoning_tool(reasoning: str) -> str:
    """Record a concise reasoning checkpoint before or between actions."""

    note = reasoning.strip()
    if not note:
        return "Reasoning checkpoint was empty."
    return f"Reasoning recorded: {note}"


def _should_reason_after_read_file(request: Any, result: Any) -> bool:
    return _tool_name(request) == "read_file" and len(_tool_result_text(result)) >= LONG_READ_FILE_CHARS


def _tool_name(request: Any) -> str:
    tool_call = getattr(request, "tool_call", None) or {}
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if name:
            return str(name)
    tool = getattr(request, "tool", None)
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    return ""


def _tool_result_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)

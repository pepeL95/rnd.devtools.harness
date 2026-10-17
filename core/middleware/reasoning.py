from __future__ import annotations

from collections.abc import Callable
from typing import Annotated
from typing import Any
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from core.live_steering import LiveSteeringInterrupt
from core.utilities.messages import system_message_with_appended_text


ReasoningEagerness = Literal["low", "medium", "high"]

REASONING_STEERING_PROMPTS: dict[ReasoningEagerness, str] = {
    # LOW EAGERNESS
    "low": """\
Reasoning mode: LOW.
Use the `reasoning` tool selectively for durable reasoning checkpoints at important pivots.

Call it when evidence changes your understanding, a result is ambiguous, the task changes shape, or an error requires a new approach.
Write a concise first-person note stating what you learned, what you now believe, and what you will do next.""",
    # MEDIUM EAGERNESS
    "medium": """\
Reasoning mode: MEDIUM.
Use the `reasoning` tool as regular working memory throughout substantive tasks.

Call it:
- before beginning substantive work
- after a meaningful batch of evidence or tool results
- before major decisions or changes of approach
- after errors, ambiguity, or discoveries that change the task
- before the final answer when the work involved multiple steps

Each checkpoint must be a concise first-person state update: current understanding, relevant evidence, uncertainty, decision, and next action.""",
    # HIGH EAGERNESS
    "high": """\
Reasoning mode: HIGHEST.
The `reasoning` tool is your required durable internal working memory. Use it to maintain a coherent record of your understanding, decisions, and next actions throughout the task.

For every substantive task, alternate between reasoning and action:
1. Call `reasoning` as your first action to interpret the request and choose the first step.
2. Perform one tool call or parallel tool-call batch.
3. After every tool-call batch completes, you MUST call `reasoning` once to analyze its results before another tool call or the final answer. This checkpoint is non-negotiable. A single tool call counts as a batch, including read-only calls, empty results, and failures.
4. Repeat this reasoning -> action -> reasoning loop until the task is complete.
5. The checkpoint immediately before the final answer records the outcome, verification, and any remaining risk; the checkpoint after the last batch can serve this purpose.

Call `reasoning` separately, after all results in the batch arrive; do not batch it with other tools. A reasoning-only batch does not require another reasoning call. The checkpoint must inform the next model step.
Also call `reasoning` immediately after errors, surprising results, ambiguous evidence, user steering, or any change of plan.

Write first-person introspective prose, not bullet points or a fixed template. These reflections become durable memory: capture what I learned from the results, what it means for the task, any uncertainty, and how it informs my next action. Favor high-signal, keyword-rich findings grounded in concrete paths, symbols, commands, test outcomes, and errors. Aim for concision without losing useful evidence; let the findings determine the length rather than a quota. Analyze the results instead of merely narrating tool use or repeating plans.""",
}

TOOL_FAILURE_REASONING_REMINDER = """A tool just failed. Use the `reasoning` tool before moving forward so you explicitly assess the failure mechanism, what changed, and the next best move."""

BATCH_REASONING_REMINDER = """Required next call: `reasoning` alone. Reflect on the batch in first-person introspective prose: what I learned, what it means, and what I should do next. Favor high-signal findings and concrete keywords worth preserving in memory; use the detail the evidence warrants."""

READ_FILE_REASONING_REMINDER = """You just received a long `read_file` result. Use the `reasoning` tool to synthesize what was actually relevant from that output, what constraints or discoveries matter, and what can be ignored. This is primarily to preserve signal for later compaction."""

LONG_READ_FILE_CHARS = 2000


class ReasoningMiddleware(AgentMiddleware):
    """Steer reasoning at pivots; require it after action batches at high eagerness."""

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
        reminder = self._pending_reminder
        requires_checkpoint = self.eagerness == "high" and _completed_action_batch(request.messages)
        if requires_checkpoint:
            reminder = BATCH_REASONING_REMINDER
        if reminder:
            prompt = "\n\n".join([prompt, reminder])
        self._pending_reminder = None
        updated = request.override(
            system_message=system_message_with_appended_text(request.system_message, prompt)
        )
        if requires_checkpoint:
            updated = updated.override(tool_choice="reasoning")
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
def reasoning_tool(
    reasoning: Annotated[
        str,
        Field(
            description=(
                "A compact first-person state update containing the current understanding, "
                "relevant evidence, uncertainty, decision, and next action."
            )
        ),
    ],
) -> str:
    """Record a durable first-person reasoning checkpoint before or between actions."""

    note = reasoning.strip()
    if not note:
        return "Reasoning checkpoint was empty."
    return "Reasoning checkpoint recorded."


def _completed_action_batch(messages: list[BaseMessage]) -> bool:
    # Inspect only the latest model batch, avoiding counters shared by parallel tools.
    results: set[str] = set()
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            results.add(message.tool_call_id)
        elif isinstance(message, AIMessage):
            return (
                any(call["name"] != "reasoning" for call in message.tool_calls)
                and all(call["id"] in results for call in message.tool_calls)
            )
    return False


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

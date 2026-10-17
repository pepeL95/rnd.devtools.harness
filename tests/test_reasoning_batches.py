from dataclasses import dataclass, replace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from core.middleware.reasoning import BATCH_REASONING_REMINDER, ReasoningMiddleware, reasoning_tool


@dataclass(frozen=True)
class Request:
    messages: list
    system_message: SystemMessage | None = None
    tool_choice: str | None = None

    def override(self, **changes):
        return replace(self, **changes)


def call(name, call_id):
    return {"name": name, "id": call_id, "args": {}}


@pytest.mark.parametrize("mode", ["low", "medium", "high"])
@pytest.mark.parametrize("status", ["success", "error"])
@pytest.mark.parametrize("names", [["read_file"], ["execute", "read_file"], ["trajectory_bank"]])
def test_only_high_mode_requires_reasoning_after_every_completed_batch(mode, status, names):
    calls = [call(name, str(index)) for index, name in enumerate(names)]
    messages = [
        AIMessage(content="", tool_calls=calls),
        *[ToolMessage(content="", status=status, tool_call_id=item["id"]) for item in reversed(calls)],
    ]
    request = Request(messages=messages, system_message=SystemMessage(content="Base"), tool_choice="auto")
    middleware = ReasoningMiddleware(eagerness=mode)
    updated = middleware.wrap_model_call(request, lambda value: value)
    assert updated.system_message.text.count(BATCH_REASONING_REMINDER) == int(mode == "high")
    assert updated.tool_choice == ("reasoning" if mode == "high" else "auto")
    assert updated.messages == messages
    assert request.system_message.content == "Base"
    assert request.tool_choice == "auto"


@pytest.mark.parametrize("messages", [
    [],
    [AIMessage(content="", tool_calls=[call("read_file", "1"), call("execute", "2")]),
     ToolMessage(content="partial batch", tool_call_id="1")],
    [AIMessage(content="", tool_calls=[call("reasoning", "r")]),
     ToolMessage(content="Recorded; historical evidence follows", tool_call_id="r")],
    [AIMessage(content="", tool_calls=[call("execute", "1")]), ToolMessage(content="ok", tool_call_id="1"),
     AIMessage(content="done"), HumanMessage(content="new turn")],
])
def test_no_reminder_for_incomplete_reasoning_only_or_previous_batches(messages):
    updated = ReasoningMiddleware(eagerness="high").wrap_model_call(Request(messages), lambda value: value)
    assert BATCH_REASONING_REMINDER not in updated.system_message.text
    assert updated.tool_choice is None


def test_mixed_batch_requires_post_result_checkpoint_and_deduplicates_reminders():
    calls = [call("reasoning", "r"), call("read_file", "1")]
    request = Request([
        AIMessage(content="", tool_calls=calls),
        ToolMessage(content="recorded", tool_call_id="r"),
        ToolMessage(content="x" * 4500, tool_call_id="1"),
    ])
    middleware = ReasoningMiddleware(eagerness="high")
    middleware.wrap_tool_call(type("ToolRequest", (), {"tool_call": calls[1]})(), lambda _: request.messages[-1])
    updated = middleware.wrap_model_call(request, lambda value: value)
    assert updated.system_message.text.count(BATCH_REASONING_REMINDER) == 1
    assert updated.tool_choice == "reasoning"
    assert "long `read_file` result" not in updated.system_message.text
    checkpoint = [
        AIMessage(content="", tool_calls=[call("reasoning", "next")]),
        ToolMessage(content="recorded", tool_call_id="next"),
    ]
    after = middleware.wrap_model_call(Request([*request.messages, *checkpoint]), lambda value: value)
    assert BATCH_REASONING_REMINDER not in after.system_message.text
    assert after.tool_choice is None


def test_real_graph_gets_one_batch_reminder_then_clears_after_reasoning():
    bindings = []

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            bindings.append(kwargs.get("tool_choice"))
            return self

    prompts = []

    class Capture(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            prompts.append(request.system_message.text)
            return handler(request)

    @tool
    def inspect_example() -> str:
        """Return a small test observation."""
        return "observed"

    model = Model(responses=[
        AIMessage(content="", tool_calls=[call("inspect_example", "a"), call("inspect_example", "b")]),
        AIMessage(content="", tool_calls=[{"name": "reasoning", "id": "r", "args": {"reasoning": "Both checks agree; I can finish."}}]),
        AIMessage(content="done"),
    ])
    graph = create_agent(
        model,
        tools=[inspect_example, reasoning_tool],
        middleware=[ReasoningMiddleware(eagerness="high"), Capture()],
    )
    result = graph.invoke({"messages": [HumanMessage(content="Check both examples")]})
    assert result["messages"][-1].content == "done"
    assert [prompt.count(BATCH_REASONING_REMINDER) for prompt in prompts] == [0, 1, 0]
    assert bindings == [None, "reasoning", None]


def test_high_mode_requires_concise_result_analysis():
    updated = ReasoningMiddleware(eagerness="high").wrap_model_call(Request([]), lambda value: value)
    prompt = updated.system_message.text
    assert "non-negotiable" in prompt
    assert "1-3 short bullets or sentences" in prompt
    assert "Analyze the results, not just the fact that tools ran" in prompt
    assert "No narrative recap" in prompt

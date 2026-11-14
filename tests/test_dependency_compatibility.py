"""Offline integration contracts across the harness's upstream agent stack."""

import asyncio
from unittest.mock import AsyncMock, Mock

from deepagents.backends import LocalShellBackend
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
import pytest

from core.middleware.filesystem import HarnessFilesystemMiddleware
from core.session.restore import reset_messages_update


@pytest.fixture
def model(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", api_key="offline-test", vertexai=False,
        include_thoughts=True, thinking_level="low",
    )
    yield model
    model.client.close()


@pytest.mark.parametrize("api_version,env_version,expected", [
    (None, None, "v1beta"),
    ("v1", None, "v1"),
    (None, "v1", "v1"),
    ("v1beta", "v1", "v1beta"),
])
def test_google_api_version_survives_sdk_upgrade(monkeypatch, api_version, env_version, expected):
    monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
    if env_version:
        monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", env_version)
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", api_key="offline-test", vertexai=False,
        api_version=api_version,
    )
    try:
        assert model.client._api_client._http_options.api_version == expected
    finally:
        model.client.close()


@pytest.mark.parametrize("use_async", [False, True])
def test_google_serializes_harness_tools_and_parses_calls(model, tmp_path, monkeypatch, use_async):
    middleware = HarnessFilesystemMiddleware(
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=False),
    )
    bound = model.bind_tools(middleware.tools)
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(
                name="execute", args={"command": "pytest", "output_mode": "stash"},
            )),
        ]), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=25, candidates_token_count=10, total_token_count=35,
        ),
    )
    generate = AsyncMock(return_value=response) if use_async else Mock(return_value=response)
    messages = [SystemMessage(content="Keep output bounded."), HumanMessage(content="Run tests.")]

    async def invoke():
        monkeypatch.setattr(model.async_client.models, "generate_content", generate)
        try:
            return await bound.ainvoke(messages)
        finally:
            await model.aclose()

    if use_async:
        result = asyncio.run(invoke())
    else:
        monkeypatch.setattr(model.client.models, "generate_content", generate)
        result = bound.invoke(messages)

    assert result.tool_calls[0]["name"] == "execute"
    assert result.tool_calls[0]["args"] == {"command": "pytest", "output_mode": "stash"}
    assert result.usage_metadata["input_tokens"] == 25
    assert result.usage_metadata["output_tokens"] == 10
    config = generate.call_args.kwargs["config"]
    assert config.system_instruction.parts[0].text == "Keep output bounded."
    assert config.thinking_config.include_thoughts is True
    declarations = {fn.name: fn for tool in config.tools for fn in tool.function_declarations}
    assert set(declarations) == {"read_file", "make_file", "edit_file", "execute"}
    properties = declarations["execute"].parameters.properties
    assert properties["output_mode"].enum == ["auto", "stash"]
    assert "runtime" not in properties


def test_session_restore_replaces_graph_history_without_duplicates():
    original = [HumanMessage(content="Old task", id="old")]
    restored = [AIMessage(content="Session arc", id="arc"), HumanMessage(content="Continue", id="user")]
    result = add_messages(original, reset_messages_update(restored))
    assert result == restored
    assert [message.id for message in original] == ["old"]
    replacement = HumanMessage(content="Steered task", id="user")
    assert add_messages(result, [replacement]) == [restored[0], replacement]

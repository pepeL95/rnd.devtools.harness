import asyncio
from unittest.mock import patch

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.middleware.agents_md import AgentsMdMiddleware


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def invoke(request):
    def run(middleware, model_request):
        async def handler(updated):
            return updated

        if request.param:
            return asyncio.run(middleware.awrap_model_call(model_request, handler))
        return middleware.wrap_model_call(model_request, lambda updated: updated)

    return run


@pytest.fixture
def model_request():
    return ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
        system_message=SystemMessage(content="Harness instructions"),
        messages=[HumanMessage(content="hello")],
        tools=[],
    )


@pytest.mark.parametrize("content", [None, "", " \t\n"])
def test_absent_or_blank_file_is_noop(tmp_path, model_request, invoke, content):
    if content is not None:
        (tmp_path / "AGENTS.md").write_text(content)
    assert invoke(AgentsMdMiddleware(tmp_path), model_request) is model_request


@pytest.mark.parametrize("has_system_message", [False, True])
def test_appends_only_exact_file_contents(tmp_path, model_request, invoke, has_system_message):
    content = "# Repository\n\tKeep this indentation.\n"
    (tmp_path / "AGENTS.md").write_text(content)
    if not has_system_message:
        model_request = model_request.override(system_message=None)
    result = invoke(AgentsMdMiddleware(tmp_path), model_request)
    expected = ([{"type": "text", "text": "Harness instructions"}] if has_system_message else [])
    assert result.system_message.content == expected + [{"type": "text", "text": content}]
    assert result.messages == model_request.messages
    assert result.tools == model_request.tools
    assert model_request.system_message is None or model_request.system_message.content == "Harness instructions"


def test_reloads_without_accumulating_context(tmp_path, model_request, invoke):
    middleware = AgentsMdMiddleware(tmp_path)
    path = tmp_path / "AGENTS.md"
    assert invoke(middleware, model_request) is model_request
    for content in ("First version", "Second version"):
        path.write_text(content)
        result = invoke(middleware, model_request)
        assert result.system_message.content == [
            {"type": "text", "text": "Harness instructions"},
            {"type": "text", "text": content},
        ]
    path.unlink()
    assert invoke(middleware, model_request) is model_request


def test_only_configured_cwd_is_used(tmp_path, model_request, invoke, monkeypatch):
    cwd = tmp_path / "workspace"
    nested = cwd / "nested"
    nested.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("Parent instructions")
    (nested / "AGENTS.md").write_text("Nested instructions")
    monkeypatch.chdir(nested)
    assert invoke(AgentsMdMiddleware(cwd), model_request) is model_request


@pytest.mark.parametrize("error", [PermissionError("AGENTS.md is unreadable"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")])
def test_read_errors_are_not_silenced(tmp_path, model_request, invoke, error):
    with patch("pathlib.Path.read_text", side_effect=error):
        with pytest.raises(type(error)):
            invoke(AgentsMdMiddleware(tmp_path), model_request)

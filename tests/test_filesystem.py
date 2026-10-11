import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from deepagents.backends import LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from core.middleware.filesystem import HarnessFilesystemMiddleware


@pytest.fixture
def filesystem(tmp_path):
    return HarnessFilesystemMiddleware(backend=LocalShellBackend(root_dir=str(tmp_path), virtual_mode=False))


def runtime():
    return ToolRuntime(state={}, context=None, config={}, stream_writer=lambda _: None, tool_call_id="test", store=None)


def tool(filesystem, name):
    return next(item for item in filesystem.tools if item.name == name)


@pytest.mark.parametrize("use_async", [False, True])
def test_create_read_edit_and_refuse_overwrite(filesystem, tmp_path, use_async):
    def invoke(name, **args):
        selected = tool(filesystem, name)
        args["runtime"] = runtime()
        return asyncio.run(selected.coroutine(**args)) if use_async else selected.func(**args)

    path = tmp_path / "nested/example.txt"
    created = invoke("make_file", file_path=str(path), content="hello\n")
    assert created.status == "success"
    assert created.name == "make_file"
    assert path.read_text() == "hello\n"
    assert invoke("make_file", file_path=str(path), content="overwritten").status == "error"
    assert path.read_text() == "hello\n"
    read = invoke("read_file", file_path=str(path))
    assert "hello" in str(read.content)
    assert invoke("edit_file", file_path=str(path), old_string="hello", new_string="world").status == "success"
    assert path.read_text() == "world\n"


@pytest.mark.parametrize("existing_target", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_make_file_rejects_symlinks(filesystem, tmp_path, existing_target, relative):
    target = tmp_path / "target"
    if existing_target:
        target.write_text("original")
    link = tmp_path / "link"
    link.symlink_to(target)
    result = tool(filesystem, "make_file").func(
        file_path="link" if relative else str(link), content="replacement", runtime=runtime(),
    )
    assert result.status == "error"
    assert link.is_symlink()
    if existing_target:
        assert target.read_text() == "original"
    else:
        assert not target.exists()


def test_make_file_is_atomic_and_does_not_change_backend_overwrite_semantics(filesystem, tmp_path):
    path = tmp_path / "race.txt"
    make_file = tool(filesystem, "make_file")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: make_file.func(file_path=str(path), content=str(i), runtime=runtime()), range(8)))
    assert sum(result.status == "success" for result in results) == 1
    assert filesystem.backend.write(str(path), "backend update").error is None
    assert path.read_text() == "backend update"


def test_make_file_preserves_virtual_paths_and_permission_denials(tmp_path):
    filesystem = HarnessFilesystemMiddleware(
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        _permissions=[FilesystemPermission(operations=["write"], paths=["/denied/**"], mode="deny")],
    )
    make_file = tool(filesystem, "make_file")
    assert make_file.func(file_path="/allowed/empty", content="", runtime=runtime()).status == "success"
    assert (tmp_path / "allowed/empty").read_bytes() == b""
    denied = make_file.func(file_path="/denied/file", content="secret", runtime=runtime())
    assert denied.status == "error"
    assert "permission denied" in denied.content
    assert not (tmp_path / "denied").exists()
    assert make_file.func(file_path="/../escape", content="bad", runtime=runtime()).status == "error"


def test_invalid_content_does_not_leave_an_empty_file(filesystem, tmp_path):
    path = tmp_path / "invalid.txt"
    result = tool(filesystem, "make_file").func(file_path=str(path), content="\ud800", runtime=runtime())
    assert result.status == "error"
    assert not path.exists()


@pytest.mark.parametrize("use_async", [False, True])
def test_execute_preserves_output_and_exit_code(tmp_path, use_async):
    filesystem = HarnessFilesystemMiddleware(
        backend=LocalShellBackend(root_dir=str(tmp_path), virtual_mode=False),
        tool_token_limit_before_evict=None,
    )
    execute = tool(filesystem, "execute")
    args = {"command": "printf 'ready'; exit 7", "runtime": runtime()}
    result = asyncio.run(execute.coroutine(**args)) if use_async else execute.func(**args)
    assert "ready" in result.content
    assert result.artifact["exit_code"] == 7


@pytest.mark.parametrize("use_async", [False, True])
def test_large_tool_results_still_offload(tmp_path, use_async):
    filesystem = HarnessFilesystemMiddleware(
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        tool_token_limit_before_evict=10,
    )
    content = "output line\n" * 2000
    result = ToolMessage(content=content, tool_call_id="large", name="custom_tool")
    request = ToolCallRequest(
        tool_call={"name": "custom_tool", "id": "large", "args": {}, "type": "tool_call"},
        tool=None, state={}, runtime=runtime(),
    )

    async def handler(_):
        return result

    processed = asyncio.run(filesystem.awrap_tool_call(request, handler)) if use_async else filesystem.wrap_tool_call(request, lambda _: result)
    assert processed.tool_call_id == "large"
    assert len(str(processed.content)) < len(content)
    assert any(path.read_text() == content for path in tmp_path.rglob("*") if path.is_file())


@pytest.mark.parametrize("names", [("make_file", "edit_file"), ("edit_file", "make_file"), ("make_file", "make_file")])
@pytest.mark.parametrize("use_async", [False, True])
def test_parallel_mutation_guard_understands_alias(filesystem, tmp_path, names, use_async):
    calls = [{"name": name, "args": {"file_path": str(tmp_path / "same.txt")}, "id": str(i), "type": "tool_call"}
             for i, name in enumerate(names)]
    message = AIMessage(content="", tool_calls=calls)
    state = {"messages": [message]}
    executed = []

    def handler(request):
        executed.append(request.tool_call)
        return ToolMessage(content="ok", name=request.tool_call["name"], tool_call_id=request.tool_call["id"])

    async def ahandler(request):
        return handler(request)

    def invoke(call):
        request = ToolCallRequest(tool_call=call, tool=None, state=state, runtime=runtime())
        return asyncio.run(filesystem.awrap_tool_call(request, ahandler)) if use_async else filesystem.wrap_tool_call(request, handler)

    assert invoke(calls[0]).status == "success"
    rejected = invoke(calls[1])
    assert rejected.status == "error"
    assert rejected.name == names[1]
    assert "parallel file mutations" in rejected.content
    assert executed == [calls[0]]
    assert [call["name"] for call in message.tool_calls] == list(names)
    other_path = {**calls[1], "args": {"file_path": str(tmp_path / "different.txt")}}
    assert invoke(other_path).status == "success"


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("use_async", [False, True])
def test_make_file_runtime_injection_in_real_graph(filesystem, tmp_path, use_async):
    path = tmp_path / "graph.txt"
    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "make_file", "args": {"file_path": str(path), "content": "initial"}, "id": "create"}]),
        AIMessage(content="done"),
    ])
    agent = create_agent(model, middleware=[filesystem])
    inputs = {"messages": [HumanMessage(content="create a file")]}
    result = asyncio.run(agent.ainvoke(inputs)) if use_async else agent.invoke(inputs)
    output = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert output.status == "success"
    assert output.name == "make_file"
    assert path.read_text() == "initial"


@pytest.mark.parametrize("use_async", [False, True])
def test_unsupported_content_filtered_without_mutating_history(use_async):
    original = ToolMessage(content=[{"type": "image", "base64": "AA==", "mime_type": "image/png"}],
                           name="read_file", tool_call_id="read", additional_kwargs={"read_file_path": "/image.png"})
    model = SimpleNamespace(profile={"image_inputs": False})
    request = ModelRequest(model=model, messages=[original], tools=[])
    middleware = UnsupportedContentMiddleware()

    async def handler(value):
        return value

    filtered = asyncio.run(middleware.awrap_model_call(request, handler)) if use_async else middleware.wrap_model_call(request, lambda value: value)
    assert "does not support image" in str(filtered.messages[0].content)
    assert original.content[0]["type"] == "image"

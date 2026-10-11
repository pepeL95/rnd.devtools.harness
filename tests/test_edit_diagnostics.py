import asyncio

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from core.middleware.filesystem import HarnessFilesystemMiddleware


def edit(middleware, old, *, use_async=False):
    tool = next(tool for tool in middleware.tools if tool.name == "edit_file")
    runtime = ToolRuntime(state={}, context=None, config={}, stream_writer=lambda _: None, tool_call_id="edit", store=None)
    args = dict(file_path="/sample.txt", old_string=old, new_string="replacement")
    request = ToolCallRequest(tool_call=dict(name="edit_file", args=args, id="edit"), tool=tool, state={}, runtime=runtime)

    async def handler(_):
        return await tool.coroutine(**args, runtime=runtime)

    if use_async:
        return asyncio.run(middleware.awrap_tool_call(request, handler))
    return middleware.wrap_tool_call(request, lambda _: tool.func(**args, runtime=runtime))


@pytest.fixture
def middleware(tmp_path):
    return HarnessFilesystemMiddleware(backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True))


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("content,old,preview", [
    ("\tvalue = 1\n", "    value = 1", r"\tvalue = 1"),
    (" \tvalue = 1\n", "    value = 1", r" \tvalue = 1"),
    ("value = 1  \n", "value = 1\t", "value = 1  "),
    ('value = "a\tb"\n', 'value = "a b"', r"a\tb"),
    ("if ready:\n\tvalue = 1\n", "if ready:\n    value = 1", r"\n\tvalue = 1"),
])
def test_whitespace_hint_never_applies_edit(middleware, tmp_path, use_async, content, old, preview):
    path = tmp_path / "sample.txt"
    path.write_bytes(content.encode())
    result = edit(middleware, old, use_async=use_async)
    assert result.status == "error"
    assert result.name == "edit_file"
    assert result.tool_call_id == "edit"
    assert result.content.startswith("Error: String not found in file:")
    assert "Whitespace-only candidate" in result.content
    assert preview in result.content
    assert "No edit applied" in result.content
    assert path.read_bytes() == content.encode()


def test_repeated_candidates_are_not_presented_as_unique(middleware, tmp_path):
    content = "\tx = 1\n    x = 1\n"
    (tmp_path / "sample.txt").write_text(content)
    result = edit(middleware, "        x = 1")
    assert "Multiple candidates exist" in result.content
    assert (tmp_path / "sample.txt").read_text() == content


@pytest.mark.parametrize("old,content", [
    ("other = 1", "\tvalue = 1\n"),
    ("value = 1\n", "value = 1"),
    ("value", "value value"),
    ("", "value"),
])
def test_other_errors_are_unchanged(middleware, tmp_path, old, content):
    (tmp_path / "sample.txt").write_text(content)
    result = edit(middleware, old)
    assert result.status == "error"
    assert "Whitespace-only candidate" not in result.content
    assert (tmp_path / "sample.txt").read_text() == content


@pytest.mark.parametrize("use_async", [False, True])
def test_successful_minimal_edit_preserves_indentation(middleware, tmp_path, use_async):
    (tmp_path / "sample.txt").write_text("\tvalue = 1\n")
    result = edit(middleware, "value = 1", use_async=use_async)
    assert result.status == "success"
    assert (tmp_path / "sample.txt").read_text() == "\treplacement\n"


@pytest.mark.parametrize("mode", ["deny", "interrupt"])
def test_diagnostic_respects_read_permissions(tmp_path, mode):
    middleware = HarnessFilesystemMiddleware(
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        _permissions=[FilesystemPermission(operations=["read"], paths=["/**"], mode=mode)],
    )
    (tmp_path / "sample.txt").write_text("\tvalue = 1\n")
    result = edit(middleware, "    value = 1")
    assert result.status == "error"
    assert "Whitespace-only candidate" not in result.content


def test_diagnostic_is_bounded_and_read_failure_preserves_error(middleware, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("\tvalue = 1\n" + "x" * (128 * 1024))
    assert "Whitespace-only candidate" not in edit(middleware, "    value = 1").content
    request = ToolCallRequest(
        tool_call=dict(name="edit_file", args=dict(file_path="/missing.txt", old_string="    value = 1"), id="edit"),
        tool=None, state={}, runtime=None,
    )
    error = ToolMessage(content="Error: String not found in file: example", status="error", tool_call_id="edit")
    assert middleware._edit_diagnostic(request, error) is error


def test_diagnostic_preview_is_truncated(middleware, tmp_path):
    content = "\tvalue = '" + "a" * 400 + "'\n"
    (tmp_path / "sample.txt").write_text(content)
    result = edit(middleware, content.replace("\t", "    ").rstrip("\n"))
    assert "excerpt truncated" in result.content
    hint = result.content.split("\nWhitespace-only candidate", 1)[1]
    assert "a" * 300 not in hint
    assert (tmp_path / "sample.txt").read_text() == content

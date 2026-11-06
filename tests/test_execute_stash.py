import asyncio
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

from deepagents.backends import LocalShellBackend
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError
import pytest

from core.middleware.filesystem import HarnessFilesystemMiddleware
from core.utilities import execution


def python_command(source):
    return shlex.join([sys.executable, "-c", source])


def runtime():
    return ToolRuntime(state={}, context=None, config={}, stream_writer=lambda _: None, tool_call_id="stash", store=None)


@pytest.fixture
def execute(tmp_path):
    backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=False, env={"STASH_TEST": "inherited"})
    middleware = HarnessFilesystemMiddleware(backend=backend)
    return next(tool for tool in middleware.tools if tool.name == "execute")


def invoke(tool, command, use_async, **kwargs):
    args = dict(command=command, output_mode="stash", runtime=runtime(), **kwargs)
    return asyncio.run(tool.coroutine(**args)) if use_async else tool.func(**args)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("size", [0, 200_000])
def test_stash_keeps_output_out_of_context_and_bypasses_old_cap(execute, size, use_async):
    result = invoke(execute, python_command(f"import sys; sys.stdout.write('x' * {size})"), use_async)
    path = Path(result.artifact["output_file"])
    assert path.read_bytes() == b"x" * size
    assert result.artifact == {
        "exit_code": 0, "output_file": str(path), "truncated": False,
        "captured_bytes": size, "timed_out": False,
    }
    assert result.name == "execute"
    assert result.tool_call_id == "stash"
    assert "succeeded" in result.content
    assert "x" * 20 not in result.content
    assert "Failure tail" not in result.content
    assert len(result.content) < 1000
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("use_async", [False, True])
def test_stash_preserves_cwd_environment_stderr_and_shell_exit(execute, tmp_path, use_async):
    result = invoke(execute, 'printf "%s\\n" "$STASH_TEST"; pwd; printf "failure\\n" >&2; exit 7', use_async)
    data = Path(result.artifact["output_file"]).read_text()
    assert data == f"inherited\n{tmp_path}\nfailure\n"
    assert result.artifact["exit_code"] == 7
    assert "failed (exit code 7)" in result.content
    assert "Failure tail" in result.content
    assert "failure" in result.content


@pytest.mark.parametrize("use_async", [False, True])
def test_stash_bounds_disk_and_keeps_real_failure_tail(execute, monkeypatch, use_async):
    monkeypatch.setattr(execution, "STASH_MAX_BYTES", 128)
    source = "import sys; sys.stdout.write('x' * 200000 + 'END FAILURE'); sys.exit(9)"
    result = invoke(execute, python_command(source), use_async)
    assert Path(result.artifact["output_file"]).read_bytes() == b"x" * 128
    assert result.artifact["captured_bytes"] == 128
    assert result.artifact["truncated"] is True
    assert result.artifact["exit_code"] == 9
    assert "Log is incomplete" in result.content
    tail = result.content.split("Failure tail (up to 4096 bytes):\n")[1]
    assert len(tail.encode()) == 4096
    assert tail.endswith("END FAILURE")


def test_exact_capture_limit_is_not_reported_as_truncated(execute, monkeypatch):
    monkeypatch.setattr(execution, "STASH_MAX_BYTES", 128)
    result = invoke(execute, python_command("print('x' * 127)"), False)
    assert result.artifact["captured_bytes"] == 128
    assert result.artifact["truncated"] is False


def test_stash_tolerates_non_utf8_output(execute):
    result = invoke(execute, python_command("import os; os.write(1, b'\\xff'); raise SystemExit(3)"), False)
    assert Path(result.artifact["output_file"]).read_bytes() == b"\xff"
    assert result.artifact["exit_code"] == 3
    assert "\ufffd" in result.content


def test_stash_uses_nearest_workspace_and_unique_files(tmp_path):
    workspace = tmp_path / ".quasipilot"
    workspace.mkdir()
    nested = tmp_path / "nested directory"
    nested.mkdir()
    backend = LocalShellBackend(root_dir=nested, virtual_mode=False)
    middleware = HarnessFilesystemMiddleware(backend=backend)
    tool = next(tool for tool in middleware.tools if tool.name == "execute")

    async def run():
        return await asyncio.gather(*(tool.coroutine(command="printf ok", output_mode="stash", runtime=runtime()) for _ in range(3)))

    paths = [Path(result.artifact["output_file"]) for result in asyncio.run(run())]
    assert len(set(paths)) == 3
    assert all(path.parent == workspace / "artifacts/execute" for path in paths)
    assert all(path.read_text() == "ok" for path in paths)


@pytest.mark.parametrize("command,timeout", [("", None), ("true", 0), ("true", -1), ("true", 3601)])
def test_stash_rejects_invalid_requests_without_files(execute, tmp_path, command, timeout):
    result = invoke(execute, command, False, timeout=timeout)
    assert result.status == "error"
    assert not (tmp_path / ".quasipilot").exists()


def test_invalid_output_mode_is_rejected_by_schema(execute):
    with pytest.raises(ValidationError):
        execute.args_schema.model_validate({"command": "true", "output_mode": "discard"})
    assert execute.args_schema.model_fields["output_mode"].default == "auto"
    assert "runtime" not in execute.tool_call_schema.model_fields
    assert "stash" in execute.description
    assert "rather than cat/head/tail" not in execute.description


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("options", [{}, {"output_mode": "auto"}])
def test_default_execution_does_not_stash(execute, tmp_path, use_async, options):
    args = dict(command="printf normal; exit 7", runtime=runtime(), **options)
    result = asyncio.run(execute.coroutine(**args)) if use_async else execute.func(**args)
    assert "normal" in result.content
    assert result.artifact["exit_code"] == 7
    assert "output_file" not in result.artifact
    assert not (tmp_path / ".quasipilot").exists()


def test_stash_rejects_virtual_backend_without_running_command(tmp_path):
    middleware = HarnessFilesystemMiddleware(backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True))
    tool = next(tool for tool in middleware.tools if tool.name == "execute")
    result = invoke(tool, "touch should-not-exist", False)
    assert result.status == "error"
    assert "virtual_mode=False" in result.content
    assert not (tmp_path / "should-not-exist").exists()


def test_capture_setup_failure_does_not_run_command(execute, tmp_path):
    (tmp_path / ".quasipilot").mkdir()
    (tmp_path / ".quasipilot/artifacts").write_text("not a directory")
    result = invoke(execute, "touch should-not-exist", False)
    assert result.status == "error"
    assert not (tmp_path / "should-not-exist").exists()


def test_capture_write_failure_terminates_process(execute):
    with patch("core.utilities.execution.NamedTemporaryFile") as temporary:
        temporary.return_value.__enter__.return_value.write.side_effect = OSError("disk full")
        result = invoke(execute, python_command("print('data', flush=True); import time; time.sleep(30)"), False)
    assert result.status == "error"
    assert "disk full" in result.content


@pytest.mark.parametrize("use_async", [False, True])
def test_stash_timeout_keeps_partial_log(execute, use_async):
    result = invoke(execute, python_command("print('started', flush=True); import time; time.sleep(30)"), use_async, timeout=1)
    assert result.artifact["exit_code"] == 124
    assert result.artifact["timed_out"] is True
    assert "timed out" in result.content
    assert "started" in Path(result.artifact["output_file"]).read_text()


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_cancellation_terminates_shell_and_descendant(execute, tmp_path):
    pid_file = tmp_path / "child.pid"
    child = python_command(
        f"from pathlib import Path; import os, time; Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "print('ready', flush=True); time.sleep(30)"
    )

    async def run():
        task = asyncio.create_task(execute.coroutine(command=f"{child} & wait", output_mode="stash", runtime=runtime()))
        try:
            async with asyncio.timeout(5):
                while not pid_file.exists() or not pid_file.read_text():
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        pid = int(pid_file.read_text())
        async with asyncio.timeout(5):
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(run())
    assert len(list((tmp_path / ".quasipilot/artifacts/execute").glob("*.log"))) == 1


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("use_async", [False, True])
def test_stash_runtime_injection_in_real_graph(tmp_path, use_async):
    middleware = HarnessFilesystemMiddleware(backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=False))
    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "printf stored", "output_mode": "stash"}, "id": "stash"}]),
        AIMessage(content="done"),
    ])
    agent = create_agent(model, middleware=[middleware])
    inputs = {"messages": [HumanMessage(content="run quietly")]}
    result = asyncio.run(agent.ainvoke(inputs)) if use_async else agent.invoke(inputs)
    output = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert output.artifact["exit_code"] == 0
    assert Path(output.artifact["output_file"]).read_text() == "stored"
    assert "stored" not in output.content
    read = next(tool for tool in middleware.tools if tool.name == "read_file")
    args = dict(file_path=output.artifact["output_file"], runtime=runtime())
    restored = asyncio.run(read.coroutine(**args)) if use_async else read.func(**args)
    assert restored.status == "success"
    assert "stored" in restored.content

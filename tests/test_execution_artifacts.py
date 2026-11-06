import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import Mock, patch

from deepagents.backends import LocalShellBackend
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from agents.driver.agent import DriverAgentConfig, create_driver_agent
from core.config import ensure_config, load_config
from core.middleware.filesystem import HarnessFilesystemMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.session.events import EventType
from core.session.manager import SessionManager
from core.utilities import execution_artifacts
from core.utilities.execution import stash_command
from core.utilities.execution_artifacts import ExecutionArtifacts, LOG_TIMESTAMP_FORMAT

NOW = datetime(2026, 11, 6, tzinfo=timezone.utc)
pytestmark = pytest.mark.skipif(os.name != "posix", reason="safe automatic cleanup requires POSIX file locks")


def log_at(store, created, content="evidence", suffix="abcdefgh"):
    store.directory.mkdir(parents=True, exist_ok=True)
    path = store.directory / f"execute-{created.strftime(LOG_TIMESTAMP_FORMAT)}-{suffix}.log"
    path.write_text(content)
    return path


def store_for(root, session="s1", days=7):
    return ExecutionArtifacts(root, root / "sessions" / f"{session}.jsonl", days)


def test_idle_days_do_not_age_logs_and_boundary_is_strict(tmp_path):
    store = store_for(tmp_path)
    last_interaction = NOW - timedelta(days=10)
    recent = log_at(store, NOW - timedelta(days=13))
    boundary = log_at(store, last_interaction - timedelta(days=7))
    old = log_at(store, last_interaction - timedelta(days=7, microseconds=1))
    future = log_at(store, NOW)
    # Retention uses immutable creation metadata, not mutable filesystem timestamps.
    os.utime(old, (NOW.timestamp(), NOW.timestamp()))
    os.utime(recent, (0, 0))
    store.prune(last_interaction.isoformat())
    assert recent.exists() and boundary.exists() and future.exists()
    assert not old.exists()


def test_retention_is_disabled_with_zero(tmp_path):
    store = store_for(tmp_path, days=0)
    old = log_at(store, NOW - timedelta(days=100))
    store.prune(NOW.isoformat())
    assert old.exists()


def test_cleanup_never_crosses_session_or_store_boundaries(tmp_path):
    first = store_for(tmp_path)
    second = store_for(tmp_path, "s2")
    other_root = ExecutionArtifacts(tmp_path, tmp_path / "other-store/s1.jsonl")
    a, b, c = [log_at(store, NOW - timedelta(days=8)) for store in (first, second, other_root)]
    first.prune(NOW.isoformat())
    assert not a.exists()
    assert b.exists() and c.exists()


def test_legacy_unknown_and_symlink_files_survive(tmp_path):
    store = store_for(tmp_path)
    managed = log_at(store, NOW - timedelta(days=8))
    legacy = store.directory.parent / "execute-abcdefgh.log"
    legacy.write_text("unowned")
    unknown = store.directory / "notes.md"
    unknown.write_text("manual")
    target = tmp_path / "private.txt"
    target.write_text("private")
    managed.unlink()
    managed.symlink_to(target)
    store.prune(NOW.isoformat())
    assert managed.is_symlink() and target.read_text() == "private"
    assert legacy.exists() and unknown.exists()


def test_symlink_session_directory_is_never_followed(tmp_path):
    store = store_for(tmp_path)
    target = store.directory.with_name("elsewhere")
    target.mkdir(parents=True)
    store.directory.symlink_to(target, target_is_directory=True)
    log = log_at(store, NOW - timedelta(days=8))
    store.prune(NOW.isoformat())
    assert log.exists()


def test_missing_lock_support_keeps_logs(tmp_path, monkeypatch, caplog):
    store = store_for(tmp_path)
    log = log_at(store, NOW - timedelta(days=8))
    monkeypatch.setattr(execution_artifacts, "fcntl", None)
    store.prune(NOW.isoformat())
    assert log.exists()
    assert "locking is unavailable" in caplog.text


def test_another_process_holds_an_active_log_lease(tmp_path):
    store = store_for(tmp_path)
    log = log_at(store, NOW - timedelta(days=8))
    script = (
        "import sys; from core.utilities.execution_artifacts import lock_execution_log; "
        "log = open(sys.argv[1], 'ab'); lock_execution_log(log); "
        "print('locked', flush=True); sys.stdin.read(1)"
    )
    with subprocess.Popen([sys.executable, "-c", script, str(log)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
        try:
            assert child.stdout.readline().strip() == "locked"
            store.prune(NOW.isoformat())
            assert log.exists()
        finally:
            child.communicate("x", timeout=5)
    store.prune(NOW.isoformat())
    assert not log.exists()


def test_actual_stash_writer_keeps_lease_until_cancellation(tmp_path):
    store = store_for(tmp_path)
    prefix = f"execute-{(NOW - timedelta(days=8)).strftime(LOG_TIMESTAMP_FORMAT)}-"
    command = shlex.join([sys.executable, "-c", "import time; time.sleep(30)"])

    async def run():
        task = asyncio.create_task(stash_command(command, cwd=tmp_path, env={}, timeout=30, directory=store.directory))
        try:
            async with asyncio.timeout(5):
                while not list(store.directory.glob("*.log")):
                    await asyncio.sleep(0.01)
            log = next(store.directory.glob("*.log"))
            store.prune(NOW.isoformat())
            assert log.exists()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        store.prune(NOW.isoformat())
        assert not log.exists()

    with patch("core.utilities.execution.execution_log_prefix", return_value=prefix):
        asyncio.run(run())


def test_post_turn_callback_uses_persisted_completion_and_skips_interruptions(tmp_path):
    manager = SessionManager(root=tmp_path / "sessions")
    callback = Mock()
    dump = SessionDumpMiddleware(manager, on_turn_complete=callback)
    state = {"messages": [HumanMessage(content="resume")]}
    dump.before_agent(state, None)
    callback.assert_not_called()
    dump._record_interrupt("live_steering_interrupt", {"steering": "adjust"})
    callback.assert_not_called()
    dump.before_agent(state, None)
    dump.after_agent(state, None)
    end = manager.read_dump()[-1]
    assert end.type == EventType.TURN_END
    callback.assert_called_once_with(end.timestamp)
    assert dump.active_turn is None
    callback.reset_mock()
    dump.before_agent(state, None)
    dump._record_cancellation()
    callback.assert_not_called()
    dump.before_agent(state, None)
    dump._record_failure("model_error", {})
    callback.assert_not_called()
    dump.after_agent(state, None)
    callback.assert_not_called()


def test_cleanup_failure_warns_without_failing_completed_turn(tmp_path, caplog):
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager, on_turn_complete=Mock(side_effect=PermissionError("denied")))
    dump.before_agent({"messages": []}, None)
    assert dump.after_agent({"messages": []}, None) is None
    assert manager.read_dump()[-1].type == EventType.TURN_END
    assert dump.active_turn is None
    assert "Post-turn cleanup failed: denied" in caplog.text


def test_unlink_failure_warns_and_preserves_log(tmp_path, caplog):
    store = store_for(tmp_path)
    log = log_at(store, NOW - timedelta(days=8))
    with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
        store.prune(NOW.isoformat())
    assert log.exists()
    assert "Cannot prune execution log" in caplog.text


@pytest.mark.parametrize("days", [0, 3, 7])
def test_driver_wires_owner_cleanup_and_config_without_startup_pruning(tmp_path, days):
    config = ensure_config(tmp_path)
    config.write_text(f"[artifacts]\nexecute_retention_days = {days}\n")
    manager = SessionManager(session_id="s1", root=tmp_path / "sessions")
    store = ExecutionArtifacts(tmp_path, manager.dump_path)
    old = log_at(store, NOW - timedelta(days=90))
    with patch("langchain.agents.create_agent") as factory:
        create_driver_agent(DriverAgentConfig(cwd=tmp_path, session_manager=manager))
    middleware = factory.call_args.kwargs["middleware"]
    filesystem = next(item for item in middleware if isinstance(item, HarnessFilesystemMiddleware))
    dump = next(item for item in middleware if isinstance(item, SessionDumpMiddleware))
    assert filesystem.execution_artifacts.directory == store.directory
    assert filesystem.execution_artifacts.retention_days == days
    assert dump.on_turn_complete.__self__ is filesystem.execution_artifacts
    assert old.exists()


@pytest.mark.parametrize("value", ["-1", "true", '"7"', "7.0"])
def test_invalid_retention_config_is_rejected(tmp_path, value):
    path = ensure_config(tmp_path)
    path.write_text(f"[artifacts]\nexecute_retention_days = {value}\n")
    with pytest.raises(ValueError, match="Invalid harness configuration"):
        load_config(tmp_path)


def test_default_config_exposes_retention_and_existing_files_are_untouched(tmp_path):
    path = ensure_config(tmp_path)
    assert "[artifacts]\nexecute_retention_days = 7" in path.read_text()
    path.write_text('# Existing configuration\n[reasoning]\neagerness = "high"\n')
    original = path.read_bytes()
    ensure_config(tmp_path)
    assert path.read_bytes() == original
    assert load_config(tmp_path).artifacts.execute_retention_days == 7


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_resumed_graph_can_read_old_log_before_turn_end_cleanup(tmp_path):
    manager = SessionManager(root=tmp_path / "sessions")
    store = ExecutionArtifacts(tmp_path, manager.dump_path)
    old = log_at(store, datetime.now(timezone.utc) - timedelta(days=30))
    dump = SessionDumpMiddleware(manager, on_turn_complete=store.prune)
    filesystem = HarnessFilesystemMiddleware(
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=False), execution_artifacts=store,
    )
    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": str(old)}, "id": "read"}]),
        AIMessage(content="I preserved the evidence in my findings."),
    ])
    agent = create_agent(model, middleware=[filesystem, dump])
    assert old.exists()
    result = agent.invoke({"messages": [HumanMessage(content="resume using the prior log")]})
    read = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert "evidence" in read.content and read.status == "success"
    assert not old.exists()


def test_session_owned_stash_is_retrievable_and_survives_its_creation_turn(tmp_path):
    manager = SessionManager(root=tmp_path / "sessions")
    store = ExecutionArtifacts(tmp_path, manager.dump_path)
    filesystem = HarnessFilesystemMiddleware(
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=False), execution_artifacts=store,
    )
    runtime = ToolRuntime(state={}, context=None, config={}, stream_writer=lambda _: None, tool_call_id="owned", store=None)
    tool = next(tool for tool in filesystem.tools if tool.name == "execute")
    result = tool.func(command="printf owned", output_mode="stash", runtime=runtime)
    log = Path(result.artifact["output_file"])
    assert log.parent == store.directory
    assert log.read_text() == "owned"
    store.prune(datetime.now(timezone.utc).isoformat())
    assert log.exists()

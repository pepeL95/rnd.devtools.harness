from pathlib import Path
import json
import sys
from threading import Event
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from core.hooks.commands import parse_command
from core.hooks.config import load_hooks
from core.hooks.dispatcher import HookDispatcher
from core.hooks.code import run_code
from core.middleware.hooks import HooksMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.session_load import SessionLoadMiddleware
from core.middleware.cancellation import CancellationMiddleware
from core.middleware.live_steering import LiveSteeringMiddleware
from core.live_steering import CancellationInterrupt, LiveSteeringController, LiveSteeringInterrupt
from core.session.events import EventType
from core.session.manager import SessionManager
from cli.utilities.streaming import iter_agent_turn


class ScriptedModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def write_hooks(root, manifest, files):
    root.mkdir(parents=True, exist_ok=True)
    (root / "hooks.toml").write_text(manifest)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if name.startswith("code/"):
            path.chmod(0o755)


def build(tmp_path, manifest, files, responses, tools=(), extra_middleware=()):
    write_hooks(tmp_path / ".quasipilot/.hooks", manifest, files)
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    dispatcher = HookDispatcher(load_hooks(tmp_path, global_root=tmp_path / "global"), manager, tmp_path)
    hooks = HooksMiddleware(dispatcher, dump)
    model = ScriptedModel(responses=responses)
    agent = create_agent(model, tools=list(tools), middleware=[SessionLoadMiddleware(manager), dump, hooks, *extra_middleware])
    return agent, manager, model, dispatcher


@pytest.mark.parametrize("raw", ["git commit -m 'a && b'", "/env/bin/python -m pytest", 'echo "a | b"'])
def test_simple_commands(raw):
    assert parse_command(raw) is not None


@pytest.mark.parametrize("raw", ["git status && git commit", "echo x | cat", "echo $(pwd)", 'echo "$HOME"',
    "(git status)", "git status; true", "X=1 pytest", "echo *.py", "echo x > out", "if true", "echo `pwd`", "git status\necho x"])
def test_unsupported_commands(raw):
    assert parse_command(raw) is None


def test_local_override_disable_and_order(tmp_path):
    global_root = tmp_path / "global"
    write_hooks(global_root, '''
[[hooks]]
id="replace"
type="steering"
trigger="before.turn"
file="a.md"
[[hooks]]
id="disable"
type="steering"
trigger="before.turn"
file="a.md"
[[hooks]]
id="keep"
type="steering"
trigger="before.turn"
file="a.md"
''', {"steering/a.md": "global"})
    write_hooks(tmp_path / ".quasipilot/.hooks", '''
[[hooks]]
id="replace"
type="steering"
trigger="after.turn"
file="b.md"
[[hooks]]
id="disable"
enabled=false
''', {"steering/b.md": "local"})
    hooks = load_hooks(tmp_path, global_root=global_root)
    assert [hook.config.id for hook in hooks] == ["keep", "replace"]
    assert hooks[-1].path.read_text() == "local"


@pytest.mark.parametrize("extra", ['recursion=0', 'recursion=2', 'match={tool=["execute"]}', 'file="../escape.md"'])
def test_invalid_manifest(tmp_path, extra):
    base = '[[hooks]]\nid="bad"\ntype="steering"\ntrigger="before.turn"\n'
    if not extra.startswith("file="):
        base += 'file="a.md"\n'
    write_hooks(tmp_path / ".quasipilot/.hooks", base + extra, {"steering/a.md": "x"})
    with pytest.raises(ValueError, match="Invalid hooks manifest"):
        load_hooks(tmp_path, global_root=tmp_path / "global")


def test_completion_recursion_keeps_one_logical_turn_and_final_ui_answer(tmp_path):
    agent, manager, model, dispatcher = build(tmp_path, '''
[[hooks]]
id="check"
type="steering"
trigger="after.turn"
file="check.md"
recursion=2
''', {"steering/check.md": "Verify the checklist."},
        [AIMessage(content="candidate"), AIMessage(content="checked"), AIMessage(content="final")])
    assert iter_agent_turn(agent, "work", lambda *_: None) == "final"
    assert len(model.seen) == 3
    assert "checklist" in str(model.seen[1][-1].content)
    events = manager.read_dump()
    assert sum(event.type == EventType.TURN_BEGIN for event in events) == 1
    assert sum(event.type == EventType.TURN_END for event in events) == 1
    assert len({event.turn for event in events}) == 1
    assert len([event for event in events if event.payload.get("source") == "hook"]) == 2
    assert all(event.payload.get("source") != "hook" for event in manager.read_display_history())
    assert dispatcher.passes["check"] == 2
    agent.invoke({"messages": [HumanMessage(content="new request")]})
    assert len(model.seen) == 6
    assert sum(event.type == EventType.TURN_END for event in manager.read_dump()) == 2


def test_tool_and_command_notes_follow_results_and_match_prefix(tmp_path):
    def execute(command: str) -> str:
        """Execute a test command."""
        return "ok"

    manifest = ""
    for trigger in ["before.tool", "before.command", "after.command", "after.tool"]:
        match = 'tool=["execute"]' if trigger.endswith(".tool") else 'cmd=["python","python3"],args=["-m","pytest"]'
        manifest += f'[[hooks]]\nid="{trigger}"\ntype="steering"\ntrigger="{trigger}"\nfile="note.md"\nmatch={{{match}}}\n'
    response = AIMessage(content="", tool_calls=[
        {"name": "execute", "args": {"command": "/env/bin/python -m pytest -q"}, "id": "call-1"},
        {"name": "execute", "args": {"command": "git status && git diff"}, "id": "call-2"},
    ])
    agent, manager, model, _ = build(tmp_path, manifest, {"steering/note.md": "guidance"}, [response, AIMessage(content="done")], [execute])
    agent.invoke({"messages": [HumanMessage(content="run tests")]})
    messages = model.seen[1]
    results = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    notes = [i for i, m in enumerate(messages) if m.additional_kwargs.get("session_kind") == "harness_context"]
    assert len(results) == 2 and len(notes) == 6
    assert min(notes) > max(results)
    restored = manager.load_curated_messages()
    assert len([m for m in restored if isinstance(m, ToolMessage)]) == 2
    events = manager.read_dump()
    assert len([e for e in events if e.payload.get("status") == "skipped"]) == 1
    assert len([e for e in events if e.payload.get("trigger") == "before.command" and e.type == EventType.USER]) == 1


def test_code_hook_stdin_result_and_no_model_continuation(tmp_path):
    script = f'#!{sys.executable}\nimport json,sys\ne=json.load(sys.stdin)\nprint(e["trigger"], e["turn"], e["schema_version"])\n'
    agent, manager, model, _ = build(tmp_path, '''
[[hooks]]
id="audit"
type="code"
trigger="after.turn"
file="audit.py"
''', {"code/audit.py": script}, [AIMessage(content="done")])
    agent.invoke({"messages": [HumanMessage(content="work")]})
    assert len(model.seen) == 1
    result = next(e for e in manager.read_dump() if e.payload.get("kind") == "hook_result")
    assert result.payload["stdout"].strip() == "after.turn 1 1"
    assert result.payload["status"] == "success"
    assert not any(e.payload.get("kind") == "hook_result" for e in manager.read_curated())


def test_code_hook_timeout_does_not_abort_agent(tmp_path):
    script = f'#!{sys.executable}\nimport time\ntime.sleep(10)\n'
    agent, manager, model, _ = build(tmp_path, '''
[[hooks]]
id="slow"
type="code"
trigger="before.turn"
file="slow.py"
''', {"code/slow.py": script}, [AIMessage(content="done")])
    with patch("core.hooks.code.CODE_TIMEOUT_SECONDS", 0.05):
        agent.invoke({"messages": [HumanMessage(content="work")]})
    assert any(e.payload.get("status") == "timeout" for e in manager.read_dump())
    assert len(model.seen) == 1


def test_before_model_and_after_model_delivery(tmp_path):
    manifest = ""
    for trigger in ["before.turn", "before.model", "after.model", "after.turn"]:
        manifest += f'[[hooks]]\nid="{trigger}"\ntype="steering"\ntrigger="{trigger}"\nfile="note.md"\n'
    agent, manager, model, _ = build(tmp_path, manifest, {"steering/note.md": "note"}, [AIMessage(content="candidate"), AIMessage(content="final")])
    agent.invoke({"messages": [HumanMessage(content="work")]})
    assert [m.additional_kwargs.get("trigger") for m in model.seen[0][-2:]] == ["before.turn", "before.model"]
    assert "after.model" in [m.additional_kwargs.get("trigger") for m in model.seen[1]]
    assert sum(e.payload.get("trigger") == "before.turn" for e in manager.read_dump()) == 1
    before = len([e for e in manager.read_dump() if e.payload.get("source") == "hook"])
    dump = SessionDumpMiddleware(manager)
    dump.before_agent({"messages": manager.load_curated_messages()}, None)
    assert len([e for e in manager.read_dump() if e.payload.get("source") == "hook"]) == before


@pytest.mark.parametrize("interrupt", ["cancel", "steer"])
def test_interruption_does_not_run_completion_and_steering_keeps_turn(tmp_path, interrupt):
    def reasoning(text: str) -> str:
        """Record reasoning."""
        return text

    cancel = Event()
    controller = LiveSteeringController()
    if interrupt == "cancel":
        cancel.set()
    else:
        controller.submit("change course")
    manifest = ""
    for trigger in ["before.turn", "after.turn", "after.tool"]:
        manifest += f'[[hooks]]\nid="{trigger}"\ntype="steering"\ntrigger="{trigger}"\nfile="note.md"\n'
    agent, manager, model, _ = build(tmp_path, manifest, {"steering/note.md": "note"}, [
        AIMessage(content="", tool_calls=[{"name": "reasoning", "args": {"text": "think"}, "id": "c1"}]),
        AIMessage(content="candidate"), AIMessage(content="final"),
    ], [reasoning], [LiveSteeringMiddleware(controller), CancellationMiddleware(cancel)])
    with pytest.raises(CancellationInterrupt if interrupt == "cancel" else LiveSteeringInterrupt):
        agent.invoke({"messages": [HumanMessage(content="work")]})
    assert not any(e.payload.get("trigger") == "after.turn" for e in manager.read_dump())
    assert any(e.payload.get("trigger") == "after.tool" for e in manager.read_dump())
    if interrupt == "steer":
        agent.invoke({"messages": []})
        events = manager.read_dump()
        assert sum(e.type == EventType.TURN_BEGIN for e in events) == 1
        assert sum(e.type == EventType.TURN_END for e in events) == 1
        assert sum(e.payload.get("trigger") == "before.turn" for e in events) == 1


def test_model_error_runs_after_model_but_not_completion(tmp_path):
    agent, manager, model, _ = build(tmp_path, '''
[[hooks]]
id="error-context"
type="steering"
trigger="after.model"
file="note.md"
[[hooks]]
id="completion"
type="steering"
trigger="after.turn"
file="note.md"
''', {"steering/note.md": "Reassess."}, [AIMessage(content="unused")])
    with patch.object(ScriptedModel, "_generate", side_effect=RuntimeError("provider failed")):
        with pytest.raises(RuntimeError, match="provider failed"):
            agent.invoke({"messages": [HumanMessage(content="work")]})
    events = manager.read_dump()
    assert any(e.payload.get("trigger") == "after.model" for e in events)
    assert not any(e.payload.get("trigger") == "after.turn" for e in events)
    assert events[-1].type == EventType.TURN_END


def test_nonzero_command_exit_reaches_after_hook(tmp_path):
    from types import SimpleNamespace
    from core.hooks.config import Hook, HookConfig

    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    hook = Hook(HookConfig(id="audit", type="code", trigger="after.command", file="audit.py"), tmp_path / "audit.py")
    dispatcher = HookDispatcher((hook,), manager, tmp_path)
    middleware = HooksMiddleware(dispatcher, dump)
    dump.before_agent({"messages": []}, None)
    middleware.before_agent({"messages": []}, None)
    request = SimpleNamespace(tool_call={"name": "execute", "id": "c1", "args": {"command": "git status"}})
    result = ToolMessage(content="failure", tool_call_id="c1", artifact={"exit_code": 1}, status="success")
    with patch("core.hooks.dispatcher.run_code", return_value={"status": "success"}) as run:
        assert middleware.wrap_tool_call(request, lambda _: result) is result
    envelope = run.call_args.args[1]
    assert envelope["status"] == "error"
    assert envelope["result"]["artifact"]["exit_code"] == 1


def test_command_prefix_matching_does_not_infer_equivalence(tmp_path):
    from core.hooks.config import Hook, HookConfig

    path = tmp_path / "note.md"
    path.write_text("check")
    config = HookConfig(id="tests", type="steering", trigger="before.command", file="note.md", match={"cmd": ["python", "python3"], "args": ["-m", "pytest"]})
    dispatcher = HookDispatcher((Hook(config, path),), SessionManager(root=tmp_path / "sessions"), tmp_path)
    dispatcher.start_turn(1)
    assert dispatcher.dispatch("before.command", {}, parse_command("python3 -m pytest -q"))
    assert not dispatcher.dispatch("before.command", {}, parse_command("python pytest"))
    assert not dispatcher.dispatch("before.command", {}, parse_command("pytest -q"))


def test_code_exit_and_bounded_output(tmp_path):
    script = tmp_path / "hook.py"
    script.write_text(f'#!{sys.executable}\nimport sys\nprint("x" * 20000)\nprint("failed", file=sys.stderr)\nsys.exit(2)\n')
    script.chmod(0o755)
    result = run_code(script, {"schema_version": 1}, tmp_path, {})
    assert result["status"] == "error" and result["exit_code"] == 2
    assert len(result["stdout"]) == 16000
    assert result["stderr"].strip() == "failed"


def test_driver_factory_runs_completion_before_session_closure(tmp_path):
    from agents.driver.agent import DriverAgentConfig, create_driver_agent

    write_hooks(tmp_path / ".quasipilot/.hooks", '''
[[hooks]]
id="completion"
type="steering"
trigger="after.turn"
file="check.md"
''', {"steering/check.md": "Check your work."})
    (tmp_path / ".quasipilot/SYSTEM.md").write_text("You are a test assistant.")
    manager = SessionManager(root=tmp_path / "sessions")
    model = ScriptedModel(responses=[AIMessage(content="candidate"), AIMessage(content="verified")])
    compaction = MagicMock()
    trajectory = MagicMock()
    with patch("pathlib.Path.home", return_value=tmp_path / "home"):
        agent = create_driver_agent(DriverAgentConfig(
            cwd=tmp_path, model=model, session_manager=manager,
            session_compaction_coordinator=compaction,
            trajectory_compaction_coordinator=trajectory,
        ))
        assert iter_agent_turn(agent, "work", lambda *_: None) == "verified"
    events = manager.read_dump()
    assert events[-1].type == EventType.TURN_END
    assert sum(e.type == EventType.TURN_END for e in events) == 1
    compaction.request_policy_compaction.assert_called_once()

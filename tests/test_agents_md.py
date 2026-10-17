import asyncio
from unittest.mock import patch

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.middleware.agents_md import AgentsMdMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.session.events import EventType, SessionEvent, is_agents_md
from core.session.manager import SessionManager


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def invoke(request):
    def run(middleware, model_request):
        async def handler(updated):
            return updated

        if request.param:
            asyncio.run(middleware.abefore_agent({}, None))
            asyncio.run(middleware.abefore_model({}, None))
            return asyncio.run(middleware.awrap_model_call(model_request, handler))
        middleware.before_agent({}, None)
        middleware.before_model({}, None)
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


def test_turn_snapshot_is_fixed_until_next_turn(tmp_path, model_request):
    path = tmp_path / "AGENTS.md"
    path.write_text("first")
    middleware = AgentsMdMiddleware(tmp_path)
    middleware.before_agent({}, None)
    path.write_text("second")
    for _ in range(2):
        result = middleware.wrap_model_call(model_request, lambda updated: updated)
        assert result.system_message.content[-1]["text"] == "first"
    middleware.before_agent({}, None)
    result = middleware.wrap_model_call(model_request, lambda updated: updated)
    assert result.system_message.content[-1]["text"] == "second"


@pytest.mark.parametrize("use_async", [False, True])
def test_latest_instruction_persists_without_extra_turns(tmp_path, use_async):
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    middleware = AgentsMdMiddleware(tmp_path, manager=manager, session_dump=dump)
    path = tmp_path / "AGENTS.md"

    def hook(name):
        if use_async:
            asyncio.run(getattr(middleware, "a" + name)({}, None))
        else:
            getattr(middleware, name)({}, None)

    for index, content in enumerate(["first", "first", "second", None, "first"], start=1):
        if content is None:
            path.unlink()
        else:
            path.write_text(content)
        hook("before_agent")
        dump.before_agent({}, None)
        assert dump.active_turn == index
        hook("before_model")
        hook("before_model")
        dump.after_agent({}, None)
        current = [event for event in manager.read_curated() if is_agents_md(event)]
        assert len(current) == 1
        assert current[0].payload["content"] == (content or "")
        assert current[0].payload["role"] == "system"
        assert [event for event in manager.read_dump() if is_agents_md(event)] == current

    latest = [event for event in manager.read_dump() if is_agents_md(event)]
    assert len(latest) == 1
    assert latest[0].turn == 5
    assert latest[0].payload["content"] == "first"
    assert manager.read_display_history() == []
    assert manager.load_curated_messages() == []  # Injected via system_message, not twice in conversation history.


def test_steering_resume_keeps_same_snapshot(tmp_path, model_request):
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    middleware = AgentsMdMiddleware(tmp_path, manager=manager, session_dump=dump)
    path = tmp_path / "AGENTS.md"
    path.write_text("first")
    middleware.before_agent({}, None)
    dump.before_agent({}, None)
    middleware.before_model({}, None)
    dump._record_interrupt("live_steering_interrupt", {"steering": "change direction"})
    path.write_text("second")
    middleware.before_agent({}, None)
    dump.before_agent({}, None)
    middleware.before_model({}, None)
    result = middleware.wrap_model_call(model_request, lambda updated: updated)
    assert result.system_message.content[-1]["text"] == "first"
    assert dump.active_turn == 1
    assert len([event for event in manager.read_dump() if is_agents_md(event)]) == 1


@pytest.mark.parametrize("latest", ["updated", ""])
def test_compaction_pins_latest_revision_even_with_stale_snapshot(tmp_path, latest):
    manager = SessionManager(root=tmp_path)
    manager.sync_system_instruction("old", kind="agents_md", source="/AGENTS.md", turn=1)
    stale = manager.read_curated()
    manager.sync_system_instruction(latest, kind="agents_md", source="/AGENTS.md", turn=2)
    user = SessionEvent(type=EventType.USER, turn=3, payload={"role": "user", "content": "keep me"})
    manager.append([user])
    raw = manager.read_dump()
    for _ in range(2):
        result = manager.apply_compaction_result(stale, snapshot_latest_turn=2)
        assert result[0].type == EventType.SYSTEM
        assert result[0].payload["content"] == latest
        assert len([event for event in result if is_agents_md(event)]) == 1
        assert user in result
    manager.replace_curated(stale)
    assert manager.read_curated()[0].payload["content"] == latest
    assert manager.read_dump() == raw


def test_resumed_session_does_not_duplicate_unchanged_instruction(tmp_path, model_request, invoke):
    path = tmp_path / "AGENTS.md"
    path.write_text("persisted")
    manager = SessionManager(session_id="resume", root=tmp_path / "sessions")
    manager.sync_system_instruction("persisted", kind="agents_md", source=str(path), turn=1)
    resumed = SessionManager(session_id="resume", root=tmp_path / "sessions")
    invoke(AgentsMdMiddleware(tmp_path, manager=resumed), model_request)
    assert len(resumed.read_dump()) == 1
    assert resumed.read_curated()[0].payload["content"] == "persisted"


def test_real_graph_turn_persists_instruction_once(tmp_path):
    from langchain.agents import create_agent

    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    (tmp_path / "AGENTS.md").write_text("Use repository conventions.")
    graph = create_agent(
        FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
        middleware=[AgentsMdMiddleware(tmp_path, manager=manager, session_dump=dump), dump],
    )
    graph.invoke({"messages": [HumanMessage(content="hello")]})
    for events in (manager.read_dump(), manager.read_curated()):
        instructions = [event for event in events if is_agents_md(event)]
        assert len(instructions) == 1
        assert instructions[0].turn == 1
        assert instructions[0].payload["content"] == "Use repository conventions."
        assert [event.turn for event in events if event.type == EventType.TURN_BEGIN] == [1]

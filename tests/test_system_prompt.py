import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from core.middleware.agents_md import AgentsMdMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.system_prompt import SystemPromptMiddleware
from core.session.events import EventType, SessionEvent, is_pinned_instruction
from core.session.manager import SessionManager


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    cwd = tmp_path / "workspace"
    home = tmp_path / "home"
    for directory in (cwd, home):
        (directory / ".quasipilot").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return cwd, home


@pytest.mark.parametrize("use_async", [False, True])
def test_latest_source_and_turn_snapshot(workspace, tmp_path, use_async):
    cwd, home = workspace
    local = cwd / ".quasipilot/SYSTEM.md"
    fallback = home / ".quasipilot/SYSTEM.md"
    fallback.write_text("home instructions")
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    middleware = SystemPromptMiddleware(cwd=cwd, manager=manager, session_dump=dump)
    model = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    request = ModelRequest(model=model, messages=[], tools=[])

    def hook(name):
        if use_async:
            asyncio.run(getattr(middleware, "a" + name)({}, None))
        else:
            getattr(middleware, name)({}, None)

    def prompt():
        async def handler(updated):
            return updated
        result = (asyncio.run(middleware.awrap_model_call(request, handler)) if use_async
                  else middleware.wrap_model_call(request, lambda updated: updated))
        return result.system_message.content[-1]["text"]

    for turn, expected in enumerate(["home instructions", "local instructions", "home instructions", "home instructions"], 1):
        hook("before_agent")
        dump.before_agent({}, None)
        hook("before_model")
        assert dump.active_turn == turn
        assert prompt() == expected
        if turn == 1:
            local.write_text("local instructions")
        elif turn == 2:
            local.unlink()
        assert prompt() == expected  # No mid-turn reload.
        hook("before_model")
        dump.after_agent({}, None)

    latest = [event for event in manager.read_dump() if is_pinned_instruction(event)]
    assert len(latest) == 1
    assert latest[0].turn == 3
    assert latest[0].payload["source"] == str(fallback)
    assert latest[0].type == EventType.SYSTEM and latest[0].payload["role"] == "system"
    assert manager.read_curated()[0] == latest[0]
    assert manager.load_curated_messages() == []


def test_missing_required_file_and_explicit_precedence(workspace):
    cwd, home = workspace
    with pytest.raises(FileNotFoundError, match="No system prompt found"):
        SystemPromptMiddleware(cwd=cwd).before_agent({}, None)
    fallback = home / ".quasipilot/SYSTEM.md"
    fallback.write_text("home")
    local = cwd / ".quasipilot/SYSTEM.md"
    local.write_text("local")
    explicit = cwd / "custom.md"
    explicit.write_text("explicit")
    assert SystemPromptMiddleware(cwd=cwd, prompt="inline", prompt_path=explicit).load_prompt() == "inline"
    assert SystemPromptMiddleware(cwd=cwd, prompt_path=explicit).load_prompt() == "explicit"
    assert SystemPromptMiddleware(cwd=cwd).load_prompt() == "local"
    local.write_text("")
    assert SystemPromptMiddleware(cwd=cwd).load_prompt() == ""  # Empty local file still overrides home.
    with pytest.raises(FileNotFoundError):
        SystemPromptMiddleware(cwd=cwd, prompt_path=cwd / "missing.md").load_prompt()
    with patch.object(Path, "read_text", side_effect=PermissionError("unreadable")):
        with pytest.raises(PermissionError):
            SystemPromptMiddleware(cwd=cwd).before_agent({}, None)


def test_both_instructions_persist_in_graph_and_survive_stale_compaction(workspace, tmp_path):
    cwd, _ = workspace
    (cwd / ".quasipilot/SYSTEM.md").write_text("system v1")
    (cwd / "AGENTS.md").write_text("agents v1")
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    system = SystemPromptMiddleware(cwd=cwd, manager=manager, session_dump=dump)
    agents = AgentsMdMiddleware(cwd, manager=manager, session_dump=dump)
    graph = create_agent(
        FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
        middleware=[system, agents, dump],
    )
    graph.invoke({"messages": [HumanMessage(content="hello")]})
    request = ModelRequest(model=FakeMessagesListChatModel(responses=[AIMessage(content="done")]), messages=[], tools=[])
    injected = system.wrap_model_call(request, lambda updated: agents.wrap_model_call(updated, lambda value: value))
    assert injected.system_message.content == [
        {"type": "text", "text": "system v1"}, {"type": "text", "text": "agents v1"},
    ]
    assert all(message.type != "system" for message in manager.load_curated_messages())
    stale = manager.read_curated()
    assert [event.payload["kind"] for event in stale[:2]] == ["system_prompt", "agents_md"]
    assert [event.turn for event in stale[:2]] == [1, 1]
    (cwd / ".quasipilot/SYSTEM.md").write_text("system v2")
    (cwd / "AGENTS.md").unlink()
    graph.invoke({"messages": [HumanMessage(content="next")]})
    raw = manager.read_dump()
    for _ in range(2):
        merged = manager.apply_compaction_result(stale, snapshot_latest_turn=1)
        assert [event.payload["content"] for event in merged[:2]] == ["system v2", ""]
        assert len([event for event in merged if is_pinned_instruction(event)]) == 2
    manager.replace_curated(stale)
    assert [event.payload["content"] for event in manager.read_curated()[:2]] == ["system v2", ""]
    assert manager.read_dump() == raw
    assert not any(is_pinned_instruction(event) for event in manager.read_display_history())


def test_steering_keeps_system_snapshot(workspace, tmp_path):
    cwd, _ = workspace
    path = cwd / ".quasipilot/SYSTEM.md"
    path.write_text("original")
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    middleware = SystemPromptMiddleware(cwd=cwd, manager=manager, session_dump=dump)
    middleware.before_agent({}, None)
    dump.before_agent({}, None)
    middleware.before_model({}, None)
    dump._record_interrupt("live_steering_interrupt", {"steering": "redirect"})
    path.unlink()
    middleware.before_agent({}, None)  # No reload or missing-file failure during same-turn steering.
    dump.before_agent({}, None)
    middleware.before_model({}, None)
    assert dump.active_turn == 1
    assert middleware._content == "original"
    assert len([event for event in manager.read_dump() if is_pinned_instruction(event)]) == 1


def test_source_only_change_and_empty_override_are_recorded(workspace, tmp_path):
    cwd, home = workspace
    fallback = home / ".quasipilot/SYSTEM.md"
    local = cwd / ".quasipilot/SYSTEM.md"
    fallback.write_text("same text")
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    middleware = SystemPromptMiddleware(cwd=cwd, manager=manager, session_dump=dump)
    for content in (None, "same text", ""):
        if content is not None:
            local.write_text(content)
        middleware.before_agent({}, None)
        dump.before_agent({}, None)
        middleware.before_model({}, None)
        dump.after_agent({}, None)
    latest = [event for event in manager.read_dump() if is_pinned_instruction(event)]
    assert len(latest) == 1
    assert latest[0].payload["source"] == str(local)
    assert latest[0].payload["content"] == ""
    assert manager.read_curated()[0] == latest[0]


def test_sync_collapses_old_revisions_and_preserves_conversation(tmp_path):
    manager = SessionManager(root=tmp_path)
    user = SessionEvent(type=EventType.USER, turn=1, payload={"role": "user", "content": "hello"})
    assistant = SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"role": "assistant", "content": "done"})
    manager.append([user])
    for text in ("old", "latest"):
        for kind in ("system_prompt", "agents_md"):
            manager.append([SessionEvent(type=EventType.SYSTEM, turn=1, payload={
                "role": "system", "kind": kind, "source": kind, "content": text,
            })])
    manager.append([assistant])
    manager.sync_system_instruction("latest", kind="system_prompt", source="system_prompt", turn=2)
    for events in (manager.read_dump(), manager.read_curated()):
        assert [event.payload["kind"] for event in events[:2]] == ["system_prompt", "agents_md"]
        assert [event.payload["content"] for event in events[:2]] == ["latest", "latest"]
        assert events[2:] == [user, assistant]
    before = (manager.dump_path.read_bytes(), manager.curated_path.read_bytes())
    manager.sync_system_instruction("latest", kind="system_prompt", source="system_prompt", turn=3)
    assert (manager.dump_path.read_bytes(), manager.curated_path.read_bytes()) == before
    manager.sync_system_instruction("new", kind="agents_md", source="agents_md", turn=3)
    for events in (manager.read_dump(), manager.read_curated()):
        assert [event.payload["content"] for event in events[:2]] == ["latest", "new"]
        assert events[2:] == [user, assistant]

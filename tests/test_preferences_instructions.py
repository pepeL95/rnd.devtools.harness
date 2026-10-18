import asyncio
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from core.middleware.agents_md import AgentsMdMiddleware
from core.middleware.preferences import PREFERENCES_CONTEXT, PreferencesMiddleware
from core.middleware.session_dump import SessionDumpMiddleware
from core.middleware.system_prompt import SystemPromptMiddleware
from core.preferences.store import PreferencesStore
from core.session.events import PREFERENCES_KIND, EventType, is_pinned_instruction
from core.session.manager import SessionManager


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def hooks(request):
    def invoke(middleware, method):
        if request.param:
            return asyncio.run(getattr(middleware, "a" + method)({}, None))
        return getattr(middleware, method)({}, None)

    def prompt(middleware):
        model_request = ModelRequest(model=None, messages=[])

        async def handler(updated):
            return updated

        result = (asyncio.run(middleware.awrap_model_call(model_request, handler)) if request.param
                  else middleware.wrap_model_call(model_request, lambda updated: updated))
        return result.system_message.text if result.system_message else ""

    return invoke, prompt


def preferences_events(events):
    return [event for event in events if is_pinned_instruction(event) and event.payload["kind"] == PREFERENCES_KIND]


def test_legacy_file_requires_explicit_migration_without_data_loss(tmp_path):
    store = PreferencesStore(tmp_path)
    store.path.parent.mkdir()
    legacy = store.path.with_name("DEVPROFILE.md")
    legacy.write_text("Preserve my preferences.\n")
    for operation in (store.read, lambda: store.update("new", expected_revision=None)):
        with pytest.raises(FileNotFoundError, match="renamed to PREFERENCES.md"):
            operation()
        assert legacy.read_text() == "Preserve my preferences.\n"
        assert not store.path.exists()
    legacy.rename(store.path)
    assert store.read().content == "Preserve my preferences."


@pytest.mark.parametrize("content", ["", "Use the new preferences."])
def test_new_file_takes_precedence_without_modifying_legacy(tmp_path, content):
    store = PreferencesStore(tmp_path)
    store.path.parent.mkdir()
    legacy = store.path.with_name("DEVPROFILE.md")
    legacy.write_text("Old preferences.")
    store.path.write_text(content)
    assert store.read().content == content
    assert legacy.read_text() == "Old preferences."


def test_latest_preferences_persisted_once_per_turn_and_cleared_when_removed(tmp_path, hooks):
    invoke, prompt = hooks
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    middleware = PreferencesMiddleware(tmp_path, manager=manager, session_dump=dump)
    path = middleware.store.path
    path.parent.mkdir()
    previous = None
    for turn, content in enumerate([None, " \t\n", "Prefer focused tests.", "Prefer focused tests.",
                                    "Prefer full tests.", "", "Restored preferences.", None], 1):
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content)
        invoke(middleware, "before_agent")
        dump.before_agent({}, None)
        invoke(middleware, "before_model")
        invoke(middleware, "before_model")
        assert dump.active_turn == turn
        expected = PREFERENCES_CONTEXT.format(preferences=content.strip()) if content and content.strip() else ""
        assert prompt(middleware) == expected
        dump.after_agent({}, None)
        raw = preferences_events(manager.read_dump())
        assert raw == preferences_events(manager.read_curated())
        if turn < 3:
            assert raw == []
        else:
            assert len(raw) == 1
            assert raw[0].payload == {"role": "system", "kind": PREFERENCES_KIND,
                                       "source": str(path), "content": expected}
            if turn == 4:
                assert raw[0] == previous  # No replacement or new event for unchanged text.
            previous = raw[0]
    assert manager.read_display_history() == []
    assert manager.load_curated_messages() == []


def test_steering_resume_preserves_snapshot_until_next_turn(tmp_path, hooks):
    invoke, prompt = hooks
    manager = SessionManager(root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    middleware = PreferencesMiddleware(tmp_path, manager=manager, session_dump=dump)
    first = middleware.store.update("First preferences.", expected_revision=None)
    invoke(middleware, "before_agent")
    dump.before_agent({}, None)
    invoke(middleware, "before_model")
    original = preferences_events(manager.read_dump())
    middleware.store.update("Updated preferences.", expected_revision=first.revision)
    assert prompt(middleware) == PREFERENCES_CONTEXT.format(preferences="First preferences.")
    dump._record_interrupt("live_steering_interrupt", {"steering": "change direction"})
    invoke(middleware, "before_agent")
    dump.before_agent({}, None)
    invoke(middleware, "before_model")
    assert dump.active_turn == 1
    assert prompt(middleware) == PREFERENCES_CONTEXT.format(preferences="First preferences.")
    assert preferences_events(manager.read_dump()) == original
    dump.after_agent({}, None)
    invoke(middleware, "before_agent")
    dump.before_agent({}, None)
    invoke(middleware, "before_model")
    assert dump.active_turn == 2
    assert prompt(middleware) == PREFERENCES_CONTEXT.format(preferences="Updated preferences.")


@pytest.mark.parametrize("error", [PermissionError("unreadable"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")])
def test_read_failure_preserves_persisted_preferences(tmp_path, hooks, error):
    invoke, _ = hooks
    manager = SessionManager(root=tmp_path / "sessions")
    middleware = PreferencesMiddleware(tmp_path, manager=manager)
    middleware.store.update("Keep my preferences.", expected_revision=None)
    invoke(middleware, "before_agent")
    invoke(middleware, "before_model")
    before = manager.dump_path.read_bytes(), manager.curated_path.read_bytes()
    with patch.object(middleware.store, "read", side_effect=error):
        with pytest.raises(type(error)):
            invoke(middleware, "before_agent")
    assert (manager.dump_path.read_bytes(), manager.curated_path.read_bytes()) == before


@pytest.mark.parametrize("latest", ["Updated preferences.", ""])
def test_all_instruction_kinds_survive_stale_compaction_and_session_resume(tmp_path, latest):
    (tmp_path / ".quasipilot").mkdir()
    (tmp_path / ".quasipilot/SYSTEM.md").write_text("Harness instructions.")
    (tmp_path / "AGENTS.md").write_text("Repository instructions.")
    preferences_path = tmp_path / ".quasipilot/PREFERENCES.md"
    preferences_path.write_text("Original preferences.")
    manager = SessionManager(session_id="preferences-session", root=tmp_path / "sessions")
    dump = SessionDumpMiddleware(manager)
    system = SystemPromptMiddleware(cwd=tmp_path, manager=manager, session_dump=dump)
    agents = AgentsMdMiddleware(tmp_path, manager=manager, session_dump=dump)
    preferences = PreferencesMiddleware(tmp_path, manager=manager, session_dump=dump)
    model = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    graph = create_agent(model, middleware=[system, agents, preferences, dump])
    graph.invoke({"messages": [HumanMessage(content="first turn")]})
    stale = manager.read_curated()
    preferences_path.write_text(latest)
    graph.invoke({"messages": [HumanMessage(content="second turn")]})
    raw = manager.read_dump()
    expected = PREFERENCES_CONTEXT.format(preferences=latest) if latest else ""
    for _ in range(2):
        merged = manager.apply_compaction_result(stale, snapshot_latest_turn=1)
        instructions = [event for event in merged if is_pinned_instruction(event)]
        assert [event.payload["kind"] for event in instructions] == ["system_prompt", "agents_md", PREFERENCES_KIND]
        assert [event.payload["content"] for event in instructions] == ["Harness instructions.", "Repository instructions.", expected]
        assert instructions[-1].turn == 2
    manager.replace_curated(stale)
    assert preferences_events(manager.read_curated())[0].payload["content"] == expected
    assert manager.read_dump() == raw
    assert not any(is_pinned_instruction(event) for event in manager.read_display_history())
    assert all(message.type != "system" for message in manager.load_curated_messages())
    assert [event.turn for event in raw if event.type == EventType.TURN_BEGIN] == [1, 2]

    resumed = PreferencesMiddleware(tmp_path, manager=SessionManager(session_id=manager.session_id, root=manager.root))
    resumed.before_agent({}, None)
    resumed.before_model({}, None)
    assert manager.read_dump() == raw
    request = ModelRequest(model=model, messages=[])
    result = system.wrap_model_call(request, lambda updated: agents.wrap_model_call(
        updated, lambda updated: resumed.wrap_model_call(updated, lambda value: value)))
    assert result.system_message.text.count("[PREFERENCES]") == bool(latest)

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage

from core.dev_profile.agent import create_dev_profile_agent
from core.dev_profile.coordinator import DevProfileCoordinator
from core.dev_profile.store import DevProfileConflictError, DevProfileStore
from core.dev_profile.tools import EMPTY_DEV_PROFILE, completed_dump_snapshot, create_dev_profile_tools
from core.middleware.dev_profile import DevProfileMiddleware
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from tests.test_middleware import FakeModelRequest


class DevProfileTests(TestCase):
    def test_store_creates_and_revision_checks_free_form_profile(self) -> None:
        with TemporaryDirectory() as directory:
            store = DevProfileStore(directory)

            missing = store.read()
            created = store.update("Prefer small, focused commits.", expected_revision=None)

            self.assertFalse(missing.exists)
            self.assertTrue(created.exists)
            self.assertEqual(created.content, "Prefer small, focused commits.")
            self.assertIsNotNone(created.revision)
            with self.assertRaises(DevProfileConflictError):
                store.update("Overwrite stale content.", expected_revision=None)

    def test_tools_progressively_expose_only_completed_dump_turns(self) -> None:
        with TemporaryDirectory() as directory:
            events = [
                SessionEvent(type=EventType.USER, turn=1, payload={"content": "Always run the full suite."}),
                SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"content": "I will."}),
                SessionEvent(type=EventType.TURN_END, turn=1, payload={}),
                SessionEvent(type=EventType.USER, turn=2, payload={"content": "This turn is still active."}),
            ]
            snapshot = completed_dump_snapshot(events)
            tools = {tool.name: tool for tool in create_dev_profile_tools(snapshot, DevProfileStore(directory))}

            overview = json.loads(tools["inspect_session"].invoke({"start_turn": 1, "limit": 20}))
            search = json.loads(tools["search_session"].invoke({"query": "full suite", "limit": 20}))
            assistant_search = json.loads(tools["search_session"].invoke({"query": "I will", "limit": 20}))
            turn = json.loads(tools["read_session_turns"].invoke({"turns": [1]}))["1"]

            self.assertEqual([item["turn"] for item in overview["turns"]], [1])
            self.assertEqual(overview["turns"][0]["user_messages"], ["Always run the full suite."])
            self.assertEqual(len(search["matches"]), 1)
            self.assertEqual(assistant_search["matches"], [])
            self.assertTrue(turn[0]["preference_evidence"])
            self.assertFalse(turn[1]["preference_evidence"])
            self.assertNotIn("still active", str(overview))

    def test_update_tool_requires_valid_user_evidence_and_handles_conflict(self) -> None:
        with TemporaryDirectory() as directory:
            store = DevProfileStore(directory)
            events = (
                SessionEvent(type=EventType.USER, turn=1, payload={"content": "Use the named Conda environment."}),
            )
            tools = {tool.name: tool for tool in create_dev_profile_tools(events, store)}

            rejected = json.loads(
                tools["update_devprofile"].invoke(
                    {"content": "Use the named Conda environment.", "expected_revision": None}
                )
            )

            created = json.loads(
                tools["update_devprofile"].invoke(
                    {
                        "content": "Use the named Conda environment.",
                        "expected_revision": None,
                        "evidence": [{"turn": 1, "quote": "Use the named Conda environment."}],
                    }
                )
            )
            conflict = json.loads(
                tools["update_devprofile"].invoke(
                    {
                        "content": "Stale replacement.",
                        "expected_revision": None,
                        "evidence": [{"turn": 1, "quote": "Use the named Conda environment."}],
                    }
                )
            )

            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(created["status"], "updated")
            self.assertEqual(conflict["status"], "conflict")

    def test_update_tool_accepts_empty_profile_without_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            store = DevProfileStore(directory)
            tools = {tool.name: tool for tool in create_dev_profile_tools((), store)}

            result = json.loads(
                tools["update_devprofile"].invoke(
                    {"content": EMPTY_DEV_PROFILE, "expected_revision": None}
                )
            )

            self.assertEqual(result["status"], "updated")
            self.assertEqual(store.read().content, EMPTY_DEV_PROFILE)

    def test_update_tool_rejects_assistant_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            events = (
                SessionEvent(type=EventType.ASSISTANT, turn=1, payload={"content": "Prefer focused tests."}),
            )
            store = DevProfileStore(directory)
            tools = {tool.name: tool for tool in create_dev_profile_tools(events, store)}

            result = json.loads(
                tools["update_devprofile"].invoke(
                    {
                        "content": "Prefer focused tests.",
                        "expected_revision": None,
                        "evidence": [{"turn": 1, "quote": "Prefer focused tests."}],
                    }
                )
            )

            self.assertEqual(result["status"], "rejected")
            self.assertFalse(store.read().exists)

    def test_create_dev_profile_agent_uses_langchain_create_agent(self) -> None:
        with TemporaryDirectory() as directory:
            model = object()
            with patch("core.dev_profile.agent.create_agent", return_value="agent") as create:
                agent = create_dev_profile_agent((), DevProfileStore(directory), model=model)  # type: ignore[arg-type]

            self.assertEqual(agent, "agent")
            kwargs = create.call_args.kwargs
            self.assertIs(kwargs["model"], model)
            self.assertEqual([tool.name for tool in kwargs["tools"]], [
                "inspect_session",
                "read_session_turns",
                "search_session",
                "read_devprofile",
                "update_devprofile",
            ])
            self.assertIn("free-form Markdown", kwargs["system_prompt"])

    def test_middleware_keeps_profile_stable_for_entire_agent_run(self) -> None:
        with TemporaryDirectory() as directory:
            store = DevProfileStore(directory)
            first = store.update("Prefer focused tests.", expected_revision=None)
            middleware = DevProfileMiddleware(directory)
            request = FakeModelRequest(system_message=SystemMessage(content="Base"), messages=[])

            middleware.before_agent({"messages": []}, runtime=None)
            store.update("Prefer the full suite.", expected_revision=first.revision)
            first_run = middleware.wrap_model_call(request, lambda updated: updated.system_message)
            middleware.before_agent({"messages": []}, runtime=None)
            second_run = middleware.wrap_model_call(request, lambda updated: updated.system_message)

            self.assertIn("Prefer focused tests.", str(first_run.content))
            self.assertNotIn("Prefer the full suite.", str(first_run.content))
            self.assertIn("Prefer the full suite.", str(second_run.content))

    def test_coordinator_runs_detached_agent_against_completed_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory) / "sessions")
            manager.append(
                [
                    SessionEvent(type=EventType.USER, turn=1, payload={"content": "Prefer atomic commits."}),
                    SessionEvent(type=EventType.TURN_END, turn=1, payload={}),
                ]
            )
            phases: list[tuple[str, dict]] = []

            def agent_factory(events, store, *, model=None):
                class FakeAgent:
                    def invoke(self, inputs):
                        current = store.read()
                        store.update("Prefer atomic commits.", expected_revision=current.revision)
                        return {"messages": [AIMessage(content="Profile updated.")]}

                self.assertEqual({event.turn for event in events}, {1})
                return FakeAgent()

            coordinator = DevProfileCoordinator(
                manager,
                directory,
                on_event=lambda phase, payload: phases.append((phase, payload)),
                agent_factory=agent_factory,
            )

            self.assertEqual(coordinator.request_update(), "started")
            deadline = monotonic() + 2
            while coordinator.is_running() and monotonic() < deadline:
                sleep(0.01)

            self.assertFalse(coordinator.is_running())
            self.assertEqual([phase for phase, _ in phases], ["start", "end"])
            self.assertTrue(phases[-1][1]["changed"])
            self.assertNotIn("summary", phases[-1][1])
            self.assertEqual(coordinator.store.read().content, "Prefer atomic commits.")

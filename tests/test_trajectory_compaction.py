from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import sleep
from unittest import TestCase

from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore
from core.trajectory.compactor import TrajectoryCompactor
from core.trajectory.coordinator import TrajectoryCompactionCoordinator
from core.trajectory.policy import TrajectoryCompactionPolicy
from core.trajectory.serialization import format_turn_interval, trajectory_memory_message
from core.utilities.defaults import get_model_name


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class ScriptedModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def invoke(self, messages: list[object]) -> FakeResponse:
        self.calls.append((str(messages[0].content), str(messages[1].content)))
        return FakeResponse(self.response)


class BlockingModel(ScriptedModel):
    def __init__(self, response: str, started: Event, release: Event) -> None:
        super().__init__(response)
        self.started = started
        self.release = release

    def invoke(self, messages: list[object]) -> FakeResponse:
        self.started.set()
        self.release.wait(timeout=2)
        return super().invoke(messages)


def scripted_trajectory_policy(response: str, *, trigger_every_turns: int = 2) -> tuple[TrajectoryCompactionPolicy, ScriptedModel]:
    model = ScriptedModel(response)
    return TrajectoryCompactionPolicy(trigger_every_turns=trigger_every_turns, compactor_model=model), model


def event(turn: int, event_type: EventType, content: str, **payload: object) -> SessionEvent:
    role = payload.pop("role", "assistant" if event_type == EventType.ASSISTANT else "user")
    return SessionEvent(type=event_type, turn=turn, payload={"role": role, "content": content, **payload})


class TrajectoryCompactionTests(TestCase):
    def test_policy_triggers_every_n_turns(self) -> None:
        policy = TrajectoryCompactionPolicy(trigger_every_turns=2)

        self.assertFalse(policy.compaction_decision(turn_count=1, latest_turn=1).should_compact)
        self.assertTrue(policy.compaction_decision(turn_count=2, latest_turn=2).should_compact)
        self.assertFalse(policy.compaction_decision(turn_count=3, latest_turn=3).should_compact)
        self.assertTrue(policy.compaction_decision(turn_count=4, latest_turn=4).should_compact)

    def test_policy_exposes_dedicated_compactor_model_slot(self) -> None:
        policy = TrajectoryCompactionPolicy()

        self.assertEqual(get_model_name(policy.compactor_model), "gemini-3.1-flash-lite")

    def test_compactor_preserves_user_assistant_and_replaces_internal_events(self) -> None:
        policy, model = scripted_trajectory_policy(
            '{"turns":[{"turn":3,"synthesis":"High-signal summary for turn 3.","live_edge":"Next edge 3."},{"turn":4,"synthesis":"High-signal summary for turn 4.","live_edge":"Next edge 4."}]}'
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.TOOL, "tool args", name="pytest"),
            event(1, EventType.TOOL_OUTPUT, "long logs", role="tool"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
            event(2, EventType.USER, "user 2", role="user"),
            event(2, EventType.REASONING, "trace failure"),
            event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
            event(3, EventType.USER, "user 3", role="user"),
            event(3, EventType.REASONING, "reasoning 3"),
            event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
            event(4, EventType.USER, "user 4", role="user"),
            event(4, EventType.REASONING, "reasoning 4"),
            event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
        ]

        result = compactor.compact(events)

        self.assertEqual(result.compacted_turns, [3, 4])
        self.assertEqual(result.compacted_event_count, 2)
        visible_and_memory_contents = [
            item.payload["content"]
            for item in result.events
            if item.type in {EventType.USER, EventType.ASSISTANT} or item.payload.get("kind") == "trajectory_memory"
        ]
        self.assertEqual(
            visible_and_memory_contents,
            [
                "user 1",
                "assistant 1",
                "user 2",
                "assistant 2",
                "user 3",
                trajectory_memory_message(result.turn_syntheses[0]),
                "assistant 3",
                "user 4",
                trajectory_memory_message(result.turn_syntheses[1]),
                "assistant 4",
            ],
        )
        self.assertEqual(
            [(item.turn, item.synthesis, item.live_edge) for item in result.turn_syntheses],
            [
                (3, "High-signal summary for turn 3.", "Next edge 3."),
                (4, "High-signal summary for turn 4.", "Next edge 4."),
            ],
        )
        self.assertEqual([item.payload.get("kind") for item in result.events if item.payload.get("kind") == "trajectory_memory"], ["trajectory_memory", "trajectory_memory"])
        self.assertFalse(
            any(
                (
                    item.type in {EventType.TOOL, EventType.TOOL_OUTPUT}
                    or (item.type == EventType.REASONING and item.payload.get("kind") != "trajectory_memory")
                )
                and item.turn in {3, 4}
                for item in result.events
            )
        )
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 2 for item in result.events))
        self.assertIn('"turn": 3', model.calls[0][1])
        self.assertIn('"user_message": "user 3"', model.calls[0][1])
        self.assertIn('"assistant_message": "assistant 3"', model.calls[0][1])

    def test_compactor_compacts_intermediate_assistant_events_but_keeps_final_assistant(self) -> None:
        policy, _ = scripted_trajectory_policy(
            '{"turns":[{"turn":1,"synthesis":"High-signal summary for turn 1.","live_edge":"Next edge 1."}]}',
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.ASSISTANT, "internal draft", role="assistant"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
        ]

        result = compactor.compact(events)

        turn_one_contents = [item.payload["content"] for item in result.events if item.turn == 1]
        self.assertEqual(
            turn_one_contents,
            [
                "user 1",
                trajectory_memory_message(result.turn_syntheses[0]),
                "assistant 1",
            ],
        )
        self.assertNotIn("internal draft", turn_one_contents)
        self.assertEqual(result.compacted_event_count, 1)

    def test_compactor_waits_until_full_window_exists(self) -> None:
        policy, _ = scripted_trajectory_policy(
            '{"turns":[{"turn":1,"synthesis":"High-signal summary for turn 1.","live_edge":"Next edge 1."}]}',
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.TOOL, "tool args", name="pytest"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
        ]

        self.assertTrue(compactor.should_compact(events))
        result = compactor.compact(events)
        self.assertEqual(result.compacted_turns, [1])

    def test_compactor_uses_latest_closed_batch(self) -> None:
        policy, _ = scripted_trajectory_policy(
            '{"turns":[{"turn":9,"synthesis":"High-signal summary for turn 9.","live_edge":"Next edge 9."},{"turn":10,"synthesis":"High-signal summary for turn 10.","live_edge":"Next edge 10."}]}'
        )
        compactor = TrajectoryCompactor(policy=policy)
        events: list[SessionEvent] = []
        for turn in range(1, 11):
            events.extend(
                [
                    event(turn, EventType.USER, f"user {turn}", role="user"),
                    event(turn, EventType.REASONING, f"reasoning {turn}"),
                    event(turn, EventType.ASSISTANT, f"assistant {turn}", role="assistant"),
                ]
            )

        self.assertTrue(compactor.should_compact(events))
        result = compactor.compact(events)

        self.assertEqual(result.compacted_turns, [9, 10])
        self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" and item.turn == 9 for item in result.events))
        self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" and item.turn == 10 for item in result.events))
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 7 for item in result.events))
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 8 for item in result.events))
        self.assertFalse(
            any(item.type == EventType.REASONING and item.turn == 9 and item.payload.get("kind") != "trajectory_memory" for item in result.events)
        )
        self.assertFalse(
            any(item.type == EventType.REASONING and item.turn == 10 and item.payload.get("kind") != "trajectory_memory" for item in result.events)
        )

    def test_compactor_compacts_cancelled_turn_and_inserts_memory_before_turn_end(self) -> None:
        policy, _ = scripted_trajectory_policy(
            '{"turns":[{"turn":2,"synthesis":"High-signal summary for turn 2.","live_edge":"Next edge 2."}]}',
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.REASONING, "reasoning 1"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
            event(2, EventType.USER, "user 2", role="user"),
            event(2, EventType.REASONING, "reasoning 2"),
            SessionEvent(type=EventType.TURN_END, turn=2, payload={"status": "cancelled"}),
        ]

        result = compactor.compact(events)

        self.assertEqual(result.compacted_turns, [2])
        turn_two_events = [item for item in result.events if item.turn == 2]
        memory_index = next(index for index, item in enumerate(turn_two_events) if item.payload.get("kind") == "trajectory_memory")
        turn_end_index = next(index for index, item in enumerate(turn_two_events) if item.type == EventType.TURN_END)
        self.assertLess(memory_index, turn_end_index)

    def test_compactor_inserts_memory_before_turn_end_when_no_assistant_exists(self) -> None:
        policy, _ = scripted_trajectory_policy(
            '{"turns":[{"turn":1,"synthesis":"High-signal summary for turn 1.","live_edge":"Next edge 1."}]}',
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.REASONING, "reasoning 1"),
            SessionEvent(type=EventType.TURN_END, turn=1, payload={}),
        ]

        result = compactor.compact(events)

        turn_one_events = [item for item in result.events if item.turn == 1]
        memory_index = next(index for index, item in enumerate(turn_one_events) if item.payload.get("kind") == "trajectory_memory")
        turn_end_index = next(index for index, item in enumerate(turn_one_events) if item.type == EventType.TURN_END)
        self.assertLess(memory_index, turn_end_index)

    def test_coordinator_applies_pending_compaction_during_agent_preparation(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "user 1", role="user"),
                    event(1, EventType.TOOL, "tool args", name="pytest"),
                    event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
                    event(2, EventType.USER, "user 2", role="user"),
                    event(2, EventType.REASONING, "reasoning"),
                    event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
                    event(3, EventType.USER, "user 3", role="user"),
                    event(3, EventType.REASONING, "reasoning 3"),
                    event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
                    event(4, EventType.USER, "user 4", role="user"),
                    event(4, EventType.REASONING, "reasoning 4"),
                    event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
                ]
            )
            started = Event()
            release = Event()
            blocking_model = BlockingModel(
                '{"turns":[{"turn":3,"synthesis":"Summary 3.","live_edge":"Edge 3."},{"turn":4,"synthesis":"Summary 4.","live_edge":"Edge 4."}]}',
                started=started,
                release=release,
            )
            coordinator = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(
                    policy=TrajectoryCompactionPolicy(trigger_every_turns=2, compactor_model=blocking_model),
                ),
                telemetry_store=TelemetryStore(Path(directory) / "telemetry.jsonl"),
                repo_root=Path(directory),
            )

            status = coordinator.request_compaction()
            self.assertEqual(status, "started")
            self.assertTrue(started.wait(timeout=1))

            manager.append(
                [
                    event(5, EventType.USER, "user 5", role="user"),
                    event(5, EventType.ASSISTANT, "assistant 5", role="assistant"),
                ]
            )
            release.set()

            deadline = datetime.now(timezone.utc) + timedelta(seconds=2)
            while coordinator.is_running() and datetime.now(timezone.utc) < deadline:
                sleep(0.01)

            curated = manager.read_curated()
            self.assertFalse(any(item.payload.get("kind") == "trajectory_memory" for item in curated))

            status = coordinator.prepare_for_agent()
            self.assertEqual(status, "applied")

            curated = manager.read_curated()
            self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" for item in curated))
            user3_index = next(index for index, item in enumerate(curated) if item.payload["content"] == "user 3")
            assistant3_index = next(index for index, item in enumerate(curated) if item.payload["content"] == "assistant 3")
            self.assertEqual(curated[user3_index + 1].payload.get("kind"), "trajectory_memory")
            self.assertEqual(user3_index + 2, assistant3_index)
            self.assertEqual(curated[-2].payload["content"], "user 5")
            self.assertEqual(curated[-1].payload["content"], "assistant 5")

    def test_prepare_for_agent_blocks_until_running_compaction_finishes(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "user 1", role="user"),
                    event(1, EventType.TOOL, "tool args", name="pytest"),
                    event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
                    event(2, EventType.USER, "user 2", role="user"),
                    event(2, EventType.REASONING, "reasoning 2"),
                    event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
                    event(3, EventType.USER, "user 3", role="user"),
                    event(3, EventType.REASONING, "reasoning 3"),
                    event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
                    event(4, EventType.USER, "user 4", role="user"),
                    event(4, EventType.REASONING, "reasoning 4"),
                    event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
                ]
            )
            started = Event()
            release = Event()
            blocking_model = BlockingModel(
                '{"turns":[{"turn":3,"synthesis":"Summary 3.","live_edge":"Edge 3."},{"turn":4,"synthesis":"Summary 4.","live_edge":"Edge 4."}]}',
                started=started,
                release=release,
            )
            coordinator = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(
                    policy=TrajectoryCompactionPolicy(trigger_every_turns=2, compactor_model=blocking_model),
                ),
                telemetry_store=TelemetryStore(Path(directory) / "telemetry.jsonl"),
                repo_root=Path(directory),
            )

            self.assertEqual(coordinator.request_compaction(), "started")
            self.assertTrue(started.wait(timeout=1))

            release.set()
            status = coordinator.prepare_for_agent()

            self.assertEqual(status, "applied")
            curated = manager.read_curated()
            self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" for item in curated))

    def test_prepare_for_agent_loads_pending_compaction_from_disk(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = SessionManager(session_id="s1", root=root)
            manager.append(
                [
                    event(1, EventType.USER, "user 1", role="user"),
                    event(1, EventType.TOOL, "tool args", name="pytest"),
                    event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
                ]
            )
            policy, _ = scripted_trajectory_policy(
                '{"turns":[{"turn":1,"synthesis":"High-signal summary for turn 1.","live_edge":"Next edge 1."}]}',
                trigger_every_turns=1,
            )
            first = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(policy=policy),
                telemetry_store=TelemetryStore(root / "telemetry.jsonl"),
                repo_root=root,
            )

            self.assertEqual(first.request_compaction(), "started")
            deadline = datetime.now(timezone.utc) + timedelta(seconds=2)
            while first.is_running() and datetime.now(timezone.utc) < deadline:
                sleep(0.01)

            pending_path = root / "pending_trajectories" / "s1.json"
            self.assertTrue(pending_path.exists())
            self.assertFalse(any(item.payload.get("kind") == "trajectory_memory" for item in manager.read_curated()))

            second = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(policy=policy),
                telemetry_store=TelemetryStore(root / "telemetry-2.jsonl"),
                repo_root=root,
            )

            self.assertEqual(second.prepare_for_agent(), "applied")
            self.assertFalse(pending_path.exists())
            curated = manager.read_curated()
            self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" for item in curated))

    def test_format_turn_interval_coalesces_ranges(self) -> None:
        self.assertEqual(format_turn_interval([1, 2, 3, 5, 7, 8]), "[1-3, 5, 7-8]")

    def test_compactor_rejects_missing_batch_turns(self) -> None:
        policy, _ = scripted_trajectory_policy('{"turns":[{"turn":1,"synthesis":"Only one.","live_edge":"Edge."}]}')
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.TOOL, "tool args", name="pytest"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
            event(2, EventType.USER, "user 2", role="user"),
            event(2, EventType.REASONING, "trace failure"),
            event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
            event(3, EventType.USER, "user 3", role="user"),
            event(3, EventType.REASONING, "reasoning 3"),
            event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
            event(4, EventType.USER, "user 4", role="user"),
            event(4, EventType.REASONING, "reasoning 4"),
            event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
        ]

        with self.assertRaisesRegex(ValueError, "Trajectory synthesis turns mismatch"):
            compactor.compact(events)

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
        self.assertFalse(policy.compaction_decision(turn_count=2, latest_turn=2).should_compact)
        self.assertFalse(policy.compaction_decision(turn_count=3, latest_turn=3).should_compact)
        self.assertTrue(policy.compaction_decision(turn_count=4, latest_turn=4).should_compact)

    def test_policy_exposes_dedicated_compactor_model_slot(self) -> None:
        policy = TrajectoryCompactionPolicy()

        self.assertEqual(get_model_name(policy.compactor_model), "gemini-3.1-flash-lite")

    def test_compactor_preserves_user_assistant_and_replaces_internal_events(self) -> None:
        policy, model = scripted_trajectory_policy(
            '{"turns":[{"turn":1,"synthesis":"High-signal summary for turn 1.","live_edge":"Next edge 1."},{"turn":2,"synthesis":"High-signal summary for turn 2.","live_edge":"Next edge 2."}]}'
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

        self.assertEqual(result.compacted_turns, [1, 2])
        self.assertEqual(result.compacted_event_count, 3)
        visible_and_memory_contents = [
            item.payload["content"]
            for item in result.events
            if item.type in {EventType.USER, EventType.ASSISTANT} or item.payload.get("kind") == "trajectory_memory"
        ]
        self.assertEqual(
            visible_and_memory_contents,
            [
                "user 1",
                trajectory_memory_message(result.turn_syntheses[0]),
                "assistant 1",
                "user 2",
                trajectory_memory_message(result.turn_syntheses[1]),
                "assistant 2",
                "user 3",
                "assistant 3",
                "user 4",
                "assistant 4",
            ],
        )
        self.assertEqual(
            [(item.turn, item.synthesis, item.live_edge) for item in result.turn_syntheses],
            [
                (1, "High-signal summary for turn 1.", "Next edge 1."),
                (2, "High-signal summary for turn 2.", "Next edge 2."),
            ],
        )
        self.assertEqual([item.payload.get("kind") for item in result.events if item.payload.get("kind") == "trajectory_memory"], ["trajectory_memory", "trajectory_memory"])
        self.assertFalse(
            any(
                (
                    item.type in {EventType.TOOL, EventType.TOOL_OUTPUT}
                    or (item.type == EventType.REASONING and item.payload.get("kind") != "trajectory_memory")
                )
                and item.turn in {1, 2}
                for item in result.events
            )
        )
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 3 for item in result.events))
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 4 for item in result.events))
        self.assertIn('"turn": 1', model.calls[0][1])
        self.assertIn('"user_message": "user 1"', model.calls[0][1])
        self.assertIn('"assistant_message": "assistant 1"', model.calls[0][1])

    def test_compactor_waits_until_two_batches_exist(self) -> None:
        policy, _ = scripted_trajectory_policy('{"turns":[]}')
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.TOOL, "tool args", name="pytest"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
            event(2, EventType.USER, "user 2", role="user"),
            event(2, EventType.REASONING, "trace failure"),
            event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
        ]

        self.assertFalse(compactor.should_compact(events))
        result = compactor.compact(events)
        self.assertEqual(result.compacted_turns, [])
        self.assertEqual(len(result.events), len(events))

    def test_compactor_uses_previous_batch_not_latest_turns(self) -> None:
        policy, _ = scripted_trajectory_policy(
            '{"turns":[{"turn":7,"synthesis":"High-signal summary for turn 7.","live_edge":"Next edge 7."},{"turn":8,"synthesis":"High-signal summary for turn 8.","live_edge":"Next edge 8."}]}'
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

        self.assertEqual(result.compacted_turns, [7, 8])
        self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" and item.turn == 7 for item in result.events))
        self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" and item.turn == 8 for item in result.events))
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 9 for item in result.events))
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 10 for item in result.events))
        self.assertFalse(
            any(item.type == EventType.REASONING and item.turn == 7 and item.payload.get("kind") != "trajectory_memory" for item in result.events)
        )
        self.assertFalse(
            any(item.type == EventType.REASONING and item.turn == 8 and item.payload.get("kind") != "trajectory_memory" for item in result.events)
        )

    def test_coordinator_merges_new_turns_after_background_compaction(self) -> None:
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
                '{"turns":[{"turn":1,"synthesis":"Summary 1.","live_edge":"Edge 1."},{"turn":2,"synthesis":"Summary 2.","live_edge":"Edge 2."}]}',
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
            self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" for item in curated))
            user1_index = next(index for index, item in enumerate(curated) if item.payload["content"] == "user 1")
            assistant1_index = next(index for index, item in enumerate(curated) if item.payload["content"] == "assistant 1")
            self.assertEqual(curated[user1_index + 1].payload.get("kind"), "trajectory_memory")
            self.assertEqual(user1_index + 2, assistant1_index)
            self.assertEqual(curated[-2].payload["content"], "user 5")
            self.assertEqual(curated[-1].payload["content"], "assistant 5")

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

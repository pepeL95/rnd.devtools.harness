from dataclasses import fields
from unittest.mock import patch

import pytest

from agents.driver.agent import DriverAgentConfig, create_driver_agent
from cli.run import QuasipilotApp
from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.config import HarnessConfig, ensure_config, load_config
from core.middleware.compaction import CompactionMiddleware
from core.middleware.trajectory import TrajectoryCompactionMiddleware
from core.session.manager import SessionManager
from core.trajectory.compactor import TrajectoryCompactor
from core.trajectory.coordinator import TrajectoryCompactionCoordinator
from core.trajectory.policy import TrajectoryCompactionPolicy
from core.trajectory_bank.store import TrajectoryBank


CUSTOM_CONFIG = """[compaction.session]
trigger_tokens = 12345
keep_last_turns = 4
max_critic_loops = 2

[compaction.trajectory]
trigger_every_turns = 3
"""


def configure(root, content=CUSTOM_CONFIG):
    path = root / ".quasipilot/config.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def coordinators(factory):
    middleware = factory.call_args.kwargs["middleware"]
    session = next(item.coordinator for item in middleware if isinstance(item, CompactionMiddleware))
    trajectory = next(item.coordinator for item in middleware if isinstance(item, TrajectoryCompactionMiddleware))
    return session, trajectory


def assert_custom_policies(session, trajectory):
    policy = session.compactor.policy
    assert (policy.trigger_tokens, policy.keep_last_turns, policy.max_critic_loops) == (12345, 4, 2)
    assert not policy.should_compact(12344)
    assert policy.should_compact(12345)
    policy = trajectory.compactor.policy
    assert policy.trigger_every_turns == 3
    assert not policy.compaction_decision(2, 2).should_compact
    assert policy.compaction_decision(3, 3).should_compact


def test_config_defaults_match_existing_policy_defaults():
    config = HarnessConfig().compaction
    for settings, policy in ((config.session, CompactionPolicy), (config.trajectory, TrajectoryCompactionPolicy)):
        defaults = {field.name: field.default for field in fields(policy)}
        assert all(defaults[key] == value for key, value in settings.model_dump().items())


def test_existing_reasoning_only_config_uses_compaction_defaults_without_rewrite(tmp_path):
    path = configure(tmp_path, '# Preserve me\n[reasoning]\neagerness = "low"\n')
    original = path.read_bytes()
    ensure_config(tmp_path)
    assert load_config(tmp_path).compaction == HarnessConfig().compaction
    assert path.read_bytes() == original


def test_partial_compaction_config_preserves_unspecified_defaults(tmp_path):
    configure(tmp_path, '[compaction.session]\nkeep_last_turns = 0\nmax_critic_loops = 0\n')
    config = load_config(tmp_path)
    assert config.compaction.session.model_dump() == {
        "trigger_tokens": 80000, "keep_last_turns": 0, "max_critic_loops": 0,
    }
    assert config.compaction.trajectory.trigger_every_turns == 1
    assert config.reasoning.eagerness == "high"


@pytest.mark.parametrize("section,key,value", [
    ("session", "trigger_tokens", "0"),
    ("session", "keep_last_turns", "-1"),
    ("session", "max_critic_loops", "-1"),
    ("trajectory", "trigger_every_turns", "0"),
    ("trajectory", "trigger_every_turns", "-1"),
    ("session", "trigger_tokens", "true"),
    ("session", "keep_last_turns", '"2"'),
    ("session", "max_critic_loops", "1.0"),
    ("trajectory", "trigger_every_turns", "false"),
    ("session", "keep_last_turn", "2"),
    ("trajectories", "trigger_every_turns", "1"),
])
def test_invalid_compaction_settings_report_config_path(tmp_path, section, key, value):
    path = configure(tmp_path, f"[compaction.{section}]\n{key} = {value}\n")
    with pytest.raises(ValueError, match="Invalid harness configuration") as error:
        load_config(tmp_path)
    assert str(path) in str(error.value)


def test_python_driver_uses_compaction_settings(tmp_path):
    configure(tmp_path)
    manager = SessionManager(root=tmp_path / "sessions")
    with patch("langchain.agents.create_agent") as factory:
        create_driver_agent(DriverAgentConfig(cwd=tmp_path, session_manager=manager))
    assert_custom_policies(*coordinators(factory))


def test_cli_uses_parent_config_for_both_compaction_policies(tmp_path):
    configure(tmp_path)
    child = tmp_path / "src"
    child.mkdir()
    with patch("cli.run.Path.cwd", return_value=child), patch.object(TrajectoryBank, "initialize"):
        app = QuasipilotApp()
    manager = SessionManager(root=tmp_path / "sessions")
    app._manager = manager
    app._compaction_coordinator = app._build_compaction_coordinator(manager)
    with patch("langchain.agents.create_agent") as factory:
        app._build_agent(manager.session_id)
    session, trajectory = coordinators(factory)
    assert session is app._compaction_coordinator
    assert_custom_policies(session, trajectory)
    assert not (child / ".quasipilot").exists()


def test_explicit_coordinators_keep_their_policies(tmp_path):
    configure(tmp_path)
    manager = SessionManager(root=tmp_path / "sessions")
    session = CompactionCoordinator(manager, Compactor(policy=CompactionPolicy(
        trigger_tokens=999, keep_last_turns=0, max_critic_loops=0,
    )), repo_root=tmp_path)
    trajectory = TrajectoryCompactionCoordinator(manager, TrajectoryCompactor(
        policy=TrajectoryCompactionPolicy(trigger_every_turns=5),
    ), repo_root=tmp_path)
    with patch("langchain.agents.create_agent") as factory:
        create_driver_agent(DriverAgentConfig(
            cwd=tmp_path, session_manager=manager,
            session_compaction_coordinator=session,
            trajectory_compaction_coordinator=trajectory,
        ))
    actual_session, actual_trajectory = coordinators(factory)
    assert actual_session is session
    assert actual_trajectory is trajectory
    assert session.compactor.policy.trigger_tokens == 999
    assert session.compactor.policy.keep_last_turns == 0
    assert session.compactor.policy.max_critic_loops == 0
    assert trajectory.compactor.policy.trigger_every_turns == 5

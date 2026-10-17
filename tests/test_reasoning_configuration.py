from unittest.mock import patch

import pytest

from agents.driver.agent import DriverAgentConfig, create_driver_agent
from cli.run import QuasipilotApp
from core.config import load_config
from core.middleware.reasoning import ReasoningMiddleware
from core.session.manager import SessionManager
from core.trajectory_bank.store import TrajectoryBank


def configure(root, content):
    directory = root / ".quasipilot"
    directory.mkdir(exist_ok=True)
    path = directory / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("content", [None, "", "[reasoning]\n"])
def test_missing_settings_default_to_high_without_writes(tmp_path, content):
    if content is not None:
        configure(tmp_path, content)
    assert load_config(tmp_path).reasoning.eagerness == "high"
    assert (tmp_path / ".quasipilot").exists() == (content is not None)


@pytest.mark.parametrize("mode", [None, "low", "medium", "high"])
@pytest.mark.parametrize("override", [None, "low", "medium", "high"])
def test_driver_reads_workspace_setting_with_explicit_override(tmp_path, mode, override):
    if mode is not None:
        configure(tmp_path, f'[reasoning]\neagerness = "{mode}"\n')
    manager = SessionManager(root=tmp_path / "sessions")
    with patch("langchain.agents.create_agent") as factory:
        create_driver_agent(DriverAgentConfig(
            cwd=tmp_path, session_manager=manager, reasoning_eagerness=override,
        ))
    reasoning = next(item for item in factory.call_args.kwargs["middleware"]
                     if isinstance(item, ReasoningMiddleware))
    assert reasoning.eagerness == (override if override is not None else mode or "high")


@pytest.mark.parametrize("content", [
    '[reasoning]\neagerness = "highest"',
    '[reasoning]\neagerness = true',
    '[reasoning]\neagerness = 3',
    '[reasoning]\neagerness = ["high"]',
    '[reasoning]\neagerness = ',
    '[reasoning]\neagerness = "low"\neagerness = "high"',
    '[reasoning]\neagernesss = "low"',
    '[reasonning]\neagerness = "low"',
    'reasoning = "low"',
])
def test_invalid_configuration_reports_source(tmp_path, content):
    path = configure(tmp_path, content)
    with pytest.raises(ValueError, match="Invalid harness configuration") as error:
        load_config(tmp_path)
    assert str(path) in str(error.value)


def test_unreadable_configuration_reports_source(tmp_path):
    path = configure(tmp_path, "")
    with patch("pathlib.Path.read_text", side_effect=PermissionError("denied")):
        with pytest.raises(ValueError, match="Cannot read harness configuration") as error:
            load_config(tmp_path)
    assert str(path) in str(error.value)


def test_invalid_configuration_stops_driver_before_session_creation(tmp_path):
    configure(tmp_path, '[reasoning]\neagerness = "invalid"')
    with patch("agents.driver.agent.SessionManager") as manager:
        with pytest.raises(ValueError, match="Invalid harness configuration"):
            create_driver_agent(DriverAgentConfig(cwd=tmp_path))
    manager.assert_not_called()


def test_nearest_workspace_does_not_merge_parent_configuration(tmp_path):
    configure(tmp_path, '[reasoning]\neagerness = "low"')
    child = tmp_path / "src"
    child.mkdir()
    assert load_config(child).reasoning.eagerness == "low"
    (child / ".quasipilot").mkdir()
    assert load_config(child).reasoning.eagerness == "high"


def test_global_home_configuration_is_not_inherited(tmp_path):
    configure(tmp_path, '[reasoning]\neagerness = "low"')
    child = tmp_path / "repo"
    child.mkdir()
    with patch("pathlib.Path.home", return_value=tmp_path):
        assert load_config(child).reasoning.eagerness == "high"


def test_cli_builds_reasoning_from_parent_workspace_and_reloads_on_rebuild(tmp_path):
    configure(tmp_path, '[reasoning]\neagerness = "low"')
    child = tmp_path / "src"
    child.mkdir()
    with patch("cli.run.Path.cwd", return_value=child), patch.object(TrajectoryBank, "initialize"):
        app = QuasipilotApp()
    app._manager = SessionManager(root=tmp_path / "sessions")
    with patch("langchain.agents.create_agent") as factory:
        for mode in ("low", "medium"):
            configure(tmp_path, f'[reasoning]\neagerness = "{mode}"')
            app._build_agent(app._manager.session_id)
            reasoning = next(item for item in factory.call_args.kwargs["middleware"]
                             if isinstance(item, ReasoningMiddleware))
            assert reasoning.eagerness == mode
    assert not (child / ".quasipilot").exists()

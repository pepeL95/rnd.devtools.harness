from unittest.mock import patch

import pytest

from agents.driver.agent import DriverAgentConfig, create_driver_agent
from cli.run import QuasipilotApp
from cli.utilities.workspace import WorkspaceStore
from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.middleware.trajectory_bank import TrajectoryBankMiddleware
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore
from core.trajectory_bank.store import TrajectoryBank
from core.trajectory_bank import defaults
from core.utilities.workspace import workspace_directory


def configure(root, enabled):
    directory = root / ".quasipilot" / ".trajectorybank"
    directory.mkdir(parents=True)
    (directory / "config.toml").write_text(f"enabled = {str(enabled).lower()}\n")
    return directory


@pytest.mark.parametrize("setting,override,expected", [(None, None, True), (False, None, False),
    (True, None, True), (True, False, False), (False, True, True)])
def test_driver_bank_toggle_controls_middleware_and_archival(tmp_path, setting, override, expected):
    if setting is not None:
        configure(tmp_path, setting)
    manager = SessionManager(root=tmp_path / "sessions")
    with patch("langchain.agents.create_agent") as factory:
        create_driver_agent(DriverAgentConfig(cwd=tmp_path, session_manager=manager, trajectory_bank_enabled=override))
    middleware = factory.call_args.kwargs["middleware"]
    recall = [item for item in middleware if isinstance(item, TrajectoryBankMiddleware)]
    assert bool(recall) == expected
    compaction = next(item for item in middleware if type(item).__name__ == "CompactionMiddleware")
    assert (compaction.coordinator.compactor.trajectory_bank is not None) == expected
    assert not (tmp_path / ".quasipilot/.trajectorybank/chroma.sqlite3").exists()


def test_nearest_workspace_shared_by_cli_and_bank(tmp_path):
    root = tmp_path / "repo"
    child = root / "src" / "package"
    child.mkdir(parents=True)
    (root / ".quasipilot").mkdir()
    workspace = WorkspaceStore(child)
    workspace.ensure()
    bank = TrajectoryBank(child)
    assert workspace.path == root / ".quasipilot/workspace.json"
    assert bank.root == root / ".quasipilot/.trajectorybank"
    assert not (child / ".quasipilot").exists()
    assert not bank.root.exists()
    (root / "src/.quasipilot").mkdir()
    assert TrajectoryBank(child).root == root / "src/.quasipilot/.trajectorybank"


def test_no_workspace_falls_back_to_cwd_without_global_home_bank(tmp_path):
    home = tmp_path / "home"
    cwd = home / "repo" / "src"
    cwd.mkdir(parents=True)
    (home / ".quasipilot").mkdir()
    with patch("pathlib.Path.home", return_value=home):
        assert workspace_directory(cwd) == cwd / ".quasipilot"
        assert workspace_directory(home) == home / ".quasipilot"
        assert not (cwd / ".quasipilot").exists()


@pytest.mark.parametrize("enabled", [True, False])
def test_cli_compaction_reads_parent_workspace_setting(tmp_path, enabled):
    directory = configure(tmp_path, enabled)
    child = tmp_path / "src"
    child.mkdir()
    with patch("cli.run.Path.cwd", return_value=child):
        app = QuasipilotApp()
    manager = SessionManager(root=tmp_path / "sessions")
    coordinator = app._build_compaction_coordinator(manager)
    assert (coordinator.compactor.trajectory_bank is not None) == enabled
    if enabled:
        assert coordinator.compactor.trajectory_bank.root == directory
        assert coordinator.compactor.trajectory_bank is app._trajectory_bank
    assert not (child / ".quasipilot").exists()
    assert (directory / "chroma.sqlite3").exists() == enabled


def test_cli_startup_initializes_by_default_without_embeddings(tmp_path):
    with patch("core.trajectory_bank.embeddings.LocalEmbeddings.embed_documents", side_effect=AssertionError("must stay lazy")) as embed:
        with patch("cli.run.Path.cwd", return_value=tmp_path):
            first = QuasipilotApp()
            second = QuasipilotApp()
    assert first.manager is None and second.manager is None
    assert (tmp_path / ".quasipilot/.trajectorybank/chroma.sqlite3").exists()
    assert first._trajectory_bank._collection.id == second._trajectory_bank._collection.id
    embed.assert_not_called()


def test_disabled_cli_startup_does_not_open_chroma(tmp_path):
    directory = configure(tmp_path, False)
    with patch("cli.run.Path.cwd", return_value=tmp_path), patch.object(TrajectoryBank, "initialize") as initialize:
        QuasipilotApp()
    initialize.assert_not_called()
    assert not (directory / "chroma.sqlite3").exists()


def test_cli_reports_bank_initialization_failure(tmp_path):
    with patch("cli.run.Path.cwd", return_value=tmp_path), patch.object(TrajectoryBank, "initialize", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            QuasipilotApp()


def test_cli_and_driver_share_overridden_embedding_default(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    model = DeterministicFakeEmbedding(size=3)
    with patch.object(defaults, "get_default_embedding_model", return_value=model) as factory, \
         patch.object(defaults, "DEFAULT_EMBEDDING_IDENTITY", "custom-default-v1"), \
         patch("cli.run.Path.cwd", return_value=tmp_path):
        app = QuasipilotApp()
        manager = SessionManager(root=tmp_path / "sessions")
        coordinator = app._build_compaction_coordinator(manager)
        with patch("langchain.agents.create_agent") as create:
            create_driver_agent(DriverAgentConfig(cwd=tmp_path, session_manager=manager,
                                                  session_compaction_coordinator=coordinator))
    recall = next(item for item in create.call_args.kwargs["middleware"] if isinstance(item, TrajectoryBankMiddleware))
    assert coordinator.compactor.trajectory_bank is app._trajectory_bank
    assert recall.retrieval.bank is app._trajectory_bank
    assert app._trajectory_bank.embedder is model
    assert app._trajectory_bank._collection.metadata["embedding_identity"] == "custom-default-v1"
    factory.assert_called_once_with()


def test_disabled_driver_rejects_injected_archival_coordinator(tmp_path):
    manager = SessionManager(root=tmp_path / "sessions")
    coordinator = CompactionCoordinator(manager, Compactor(trajectory_bank=TrajectoryBank(tmp_path), session_manager=manager),
                                       telemetry_store=TelemetryStore(tmp_path / "telemetry.jsonl"), repo_root=tmp_path)
    with pytest.raises(ValueError, match="supplied compaction coordinator"):
        create_driver_agent(DriverAgentConfig(cwd=tmp_path, session_manager=manager,
                                             trajectory_bank_enabled=False, session_compaction_coordinator=coordinator))

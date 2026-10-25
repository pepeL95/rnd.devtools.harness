from unittest.mock import patch

import pytest

from agents.driver.agent import DriverAgentConfig, create_driver_agent
from cli.run import QuasipilotApp
from cli.utilities.workspace import WorkspaceStore
from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.config import ensure_config, load_config
from core.middleware.trajectory_bank import TrajectoryBankMiddleware
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore
from core.trajectory_bank.store import TrajectoryBank
from core.trajectory_bank import defaults
from core.trajectory_bank.policy import BankPolicy
from core.trajectory_bank.retrieval import BankRetrieval
from core.utilities.workspace import workspace_directory


def configure(root, enabled):
    directory = root / ".quasipilot" / ".trajectorybank"
    directory.parent.mkdir(parents=True)
    (directory.parent / "config.toml").write_text(f"[trajectory_bank]\nenabled = {str(enabled).lower()}\n")
    return directory


@pytest.mark.parametrize("setting,override,expected", [(None, None, False), (False, None, False),
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
    assert (compaction.coordinator.compactor.evidence_provider is not None) == expected
    if expected:
        assert compaction.coordinator.compactor.evidence_provider is recall[0]
    assert not (tmp_path / ".quasipilot/.trajectorybank/chroma.sqlite3").exists()


def test_default_cli_and_driver_do_not_construct_a_bank(tmp_path):
    with patch("cli.run.Path.cwd", return_value=tmp_path), \
         patch("cli.run.TrajectoryBank", side_effect=AssertionError("Bank is disabled")), \
         patch("agents.driver.agent.TrajectoryBank", side_effect=AssertionError("Bank is disabled")), \
         patch("langchain.agents.create_agent") as factory:
        app = QuasipilotApp()
        manager = SessionManager(root=tmp_path / "sessions")
        coordinator = app._build_compaction_coordinator(manager)
        create_driver_agent(DriverAgentConfig(cwd=tmp_path, session_manager=manager,
                                              session_compaction_coordinator=coordinator))
    assert app._trajectory_bank is None
    assert coordinator.compactor.evidence_provider is None
    assert not any(isinstance(item, TrajectoryBankMiddleware) for item in factory.call_args.kwargs["middleware"])
    assert not (tmp_path / ".quasipilot/.trajectorybank").exists()


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
    assert (coordinator.compactor.evidence_provider is not None) == enabled
    if enabled:
        assert coordinator.compactor.evidence_provider.retrieval.bank.root == directory
        assert coordinator.compactor.evidence_provider.retrieval.bank is app._trajectory_bank
    assert not (child / ".quasipilot").exists()
    assert (directory / "chroma.sqlite3").exists() == enabled


def test_enabled_cli_startup_initializes_without_embeddings(tmp_path):
    configure(tmp_path, True)
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
    configure(tmp_path, True)
    with patch("cli.run.Path.cwd", return_value=tmp_path), patch.object(TrajectoryBank, "initialize", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            QuasipilotApp()


def test_cli_and_driver_share_overridden_embedding_default(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    configure(tmp_path, True)
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
    assert coordinator.compactor.evidence_provider is recall
    assert recall.retrieval.bank is app._trajectory_bank
    assert app._trajectory_bank.embedder is model
    assert app._trajectory_bank._collection.metadata["embedding_identity"] == "custom-default-v1"
    factory.assert_called_once_with()


def test_disabled_driver_rejects_injected_archival_coordinator(tmp_path):
    manager = SessionManager(root=tmp_path / "sessions")
    provider = TrajectoryBankMiddleware(BankRetrieval(TrajectoryBank(tmp_path), manager))
    coordinator = CompactionCoordinator(manager, Compactor(evidence_provider=provider, session_manager=manager),
                                       telemetry_store=TelemetryStore(tmp_path / "telemetry.jsonl"), repo_root=tmp_path)
    with pytest.raises(ValueError, match="supplied compaction coordinator"):
        create_driver_agent(DriverAgentConfig(cwd=tmp_path, session_manager=manager,
                                             trajectory_bank_enabled=False, session_compaction_coordinator=coordinator))


def test_bank_defaults_in_generated_workspace_config(tmp_path):
    import tomllib

    path = ensure_config(tmp_path)
    document = tomllib.loads(path.read_text())
    assert document["trajectory_bank"] == BankPolicy().model_dump()
    assert load_config(tmp_path).trajectory_bank == BankPolicy()


def test_all_retrieval_settings_reach_driver_and_direct_retrieval(tmp_path):
    directory = configure(tmp_path, True)
    with (directory.parent / "config.toml").open("a") as handle:
        handle.write("automatic_min_similarity = 0.9\nsearch_min_similarity = 0.6\n"
                     "automatic_results = 3\nsearch_results = 7\n"
                     "automatic_tokens = 900\nsearch_tokens = 1800\n")
    child = tmp_path / "src"
    child.mkdir()
    manager = SessionManager(root=tmp_path / "sessions")
    with patch("langchain.agents.create_agent") as factory:
        create_driver_agent(DriverAgentConfig(cwd=child, session_manager=manager))
    middleware = next(item for item in factory.call_args.kwargs["middleware"]
                      if isinstance(item, TrajectoryBankMiddleware))
    expected = BankPolicy(enabled=True, automatic_min_similarity=0.9, search_min_similarity=0.6,
                          automatic_results=3, search_results=7, automatic_tokens=900, search_tokens=1800)
    assert middleware.retrieval.policy == expected
    assert BankRetrieval(TrajectoryBank(child), manager).policy == expected
    explicit = BankPolicy(search_results=2)
    assert BankRetrieval(TrajectoryBank(child), manager, policy=explicit).policy is explicit


@pytest.mark.parametrize("setting", [
    "automatic_min_similarity = 0.1\nsearch_min_similarity = 0.8",
    "automatic_min_similarity = 1.1",
    "search_min_similarity = -0.1",
    "automatic_results = 0",
    "search_results = 21",
    "automatic_tokens = 99",
    "search_tokens = 10001",
    "enabled = 'invalid'",
    "enabld = false",
])
def test_invalid_bank_settings_fail_before_cli_initializes_database(tmp_path, setting):
    directory = tmp_path / ".quasipilot"
    directory.mkdir()
    path = directory / "config.toml"
    path.write_text(f"[trajectory_bank]\n{setting}\n")
    with patch("cli.run.Path.cwd", return_value=tmp_path), patch.object(TrajectoryBank, "initialize") as initialize:
        with pytest.raises(ValueError, match="Invalid harness configuration") as error:
            QuasipilotApp()
    assert str(path) in str(error.value)
    initialize.assert_not_called()


@pytest.mark.parametrize("central_exists", [False, True])
def test_legacy_bank_config_requires_explicit_migration_without_overwriting(tmp_path, central_exists):
    legacy = tmp_path / ".quasipilot/.trajectorybank/config.toml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("enabled = false\n")
    central = legacy.parent.parent / "config.toml"
    content = '[reasoning]\neagerness = "low"\n'
    if central_exists:
        central.write_text(content)
    for operation in (ensure_config, load_config):
        with pytest.raises(ValueError, match="Move settings") as error:
            operation(tmp_path)
        assert str(legacy) in str(error.value)
        assert "[trajectory_bank]" in str(error.value)
    assert central.exists() == central_exists
    if central_exists:
        assert central.read_text() == content
    assert legacy.read_text() == "enabled = false\n"


def test_missing_bank_section_is_appended_at_startup(tmp_path):
    path = tmp_path / ".quasipilot/config.toml"
    path.parent.mkdir()
    content = '# Retain comments\n[reasoning]\neagerness = "low"\n'
    path.write_text(content)
    with patch("cli.run.Path.cwd", return_value=tmp_path), patch.object(TrajectoryBank, "initialize"):
        QuasipilotApp()
    assert load_config(tmp_path).trajectory_bank == BankPolicy()
    assert path.read_text().startswith(content)
    assert "[trajectory_bank]" in path.read_text()


def test_existing_bank_settings_survive_startup_backfill(tmp_path):
    directory = configure(tmp_path, False)
    path = directory.parent / "config.toml"
    content = '# Custom recall\n[trajectory_bank]\nenabled = false\nsearch_results = 9\n'
    path.write_text(content)
    with patch("cli.run.Path.cwd", return_value=tmp_path), patch.object(TrajectoryBank, "initialize") as initialize:
        QuasipilotApp()
        after_first_start = path.read_bytes()
        QuasipilotApp()
    initialize.assert_not_called()
    assert path.read_bytes() == after_first_start
    assert path.read_text().startswith(content)
    assert load_config(tmp_path).trajectory_bank.search_results == 9

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from agents.driver.agent import _local_shell_backend
from agents.driver.agent import DriverAgentConfig, create_driver_agent
from core.utilities.defaults import _default_google_model_name, get_default_model


class BackendWithVirtualEnv:
    def __init__(self, root_dir: str, inherit_env: bool, virtual_env: bool, virtual_mode: bool) -> None:
        self.root_dir = root_dir
        self.inherit_env = inherit_env
        self.virtual_env = virtual_env
        self.virtual_mode = virtual_mode


class BackendWithoutVirtualEnv:
    def __init__(self, root_dir: str, inherit_env: bool) -> None:
        self.root_dir = root_dir
        self.inherit_env = inherit_env


class DriverAgentTests(TestCase):
    def test_local_shell_backend_passes_virtual_env_when_supported(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            backend = _local_shell_backend(BackendWithVirtualEnv, path)

            self.assertEqual(backend.root_dir, str(path))
            self.assertIs(backend.inherit_env, True)
            self.assertIs(backend.virtual_env, False)
            self.assertIs(backend.virtual_mode, True)

    def test_local_shell_backend_tolerates_current_deepagents_signature(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            backend = _local_shell_backend(BackendWithoutVirtualEnv, path)

            self.assertEqual(backend.root_dir, str(path))
            self.assertIs(backend.inherit_env, True)

    def test_driver_agent_builds_with_telemetry_middleware(self) -> None:
        with TemporaryDirectory() as directory:
            agent = create_driver_agent(DriverAgentConfig(cwd=Path(directory), session_id="test-session"))

            self.assertEqual(type(agent).__name__, "CompiledStateGraph")

    def test_default_model_disables_retries_by_default(self) -> None:
        model = get_default_model()

        self.assertEqual(model.model, "gemini-3.5-flash")
        self.assertEqual(model.max_retries, 0)

    def test_default_model_name_accepts_legacy_provider_prefix(self) -> None:
        with patch.dict("os.environ", {"QUASIPILOT_DRIVER_MODEL": "google_genai:gemini-test"}, clear=False):
            self.assertEqual(_default_google_model_name(), "gemini-test")

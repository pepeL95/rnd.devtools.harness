from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agents.driver.agent import _init_driver_model, _local_shell_backend
from agents.driver.agent import DriverAgentConfig, create_driver_agent


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

    def test_google_driver_model_disables_retries_by_default(self) -> None:
        model = _init_driver_model("google_genai:gemini-3.5-flash", retries=0)

        self.assertEqual(model.model, "gemini-3.5-flash")
        self.assertEqual(model.max_retries, 0)

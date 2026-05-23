from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agents.driver.agent import _local_shell_backend


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

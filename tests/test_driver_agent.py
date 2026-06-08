from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from agents.driver.agent import _local_shell_backend
from agents.driver.agent import DriverAgentConfig, create_driver_agent
from core.compaction.llms import (
    get_default_compactor_model,
    get_default_critic_model,
    get_default_task_extractor_model,
)
from core.trajectory.llms import (
    get_default_trajectory_compactor_model,
)
from core.utilities.defaults import get_default_driver_model


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
            self.assertIs(backend.virtual_mode, False)

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

    def test_driver_agent_passes_python_interpreter_to_runtime_middleware(self) -> None:
        with TemporaryDirectory() as directory:
            cwd = Path(directory)
            interpreter = cwd / "venv/bin/python"
            captured: list[object] = []

            class FakeFilesystemMiddleware:
                def __init__(self, **_: object) -> None:
                    pass

            def fake_create_agent(*, model: object, tools: list[object], middleware: list[object]) -> object:
                captured.extend(middleware)
                return object()

            with patch("langchain.agents.create_agent", side_effect=fake_create_agent), patch(
                "deepagents.middleware.filesystem.FilesystemMiddleware",
                FakeFilesystemMiddleware,
            ):
                create_driver_agent(
                    DriverAgentConfig(
                        cwd=cwd,
                        python_interpreter=interpreter,
                        session_id="test-session",
                    )
                )

            runtime = next(item for item in captured if type(item).__name__ == "RuntimeContextMiddleware")
            self.assertEqual(runtime.python_interpreter, interpreter.resolve())

    def test_default_driver_model_uses_reasoning_profile(self) -> None:
        model = get_default_driver_model()

        self.assertEqual(model.model, "gemini-3.1-flash-lite")
        self.assertEqual(model.max_retries, 3)
        self.assertIs(model.include_thoughts, True)
        self.assertEqual(model.thinking_level, "low")

    def test_compaction_component_defaults_share_non_reasoning_profile(self) -> None:
        for factory in (
            get_default_task_extractor_model,
            get_default_compactor_model,
            get_default_critic_model,
            get_default_trajectory_compactor_model,
        ):
            model = factory()
            self.assertEqual(model.model, "gemini-3.1-flash-lite")
            self.assertEqual(model.max_retries, 3)
            self.assertIs(model.include_thoughts, False)
            self.assertEqual(model.thinking_level, "minimal")

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from deepagents.backends import LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.utils import validate_path
from deepagents.middleware.filesystem import (
    FilesystemMiddleware,
    _check_fs_permission,
    _parallel_file_mutation_error,
    _permission_error,
    _resolve_real_path,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import Field, create_model

from core.utilities.execution import STASH_TAIL_BYTES, stash_command
from core.utilities.workspace import workspace_directory

MAKE_FILE_TOOL_NAME = "make_file"
HARNESS_FILESYSTEM_TOOLS = ("read_file", "write_file", "edit_file", "execute")

MAKE_FILE_TOOL_DESCRIPTION = """Creates a new file and any missing parent directories. Provide the full initial file contents.
Fails if the path already exists; use `read_file` then `edit_file` to modify existing files."""

EDIT_FILE_TOOL_DESCRIPTION = """Replaces an exact, non-empty string in an existing file.
Read first; match the verbatim file content, including whitespace, not the read status header.
Use the smallest unique substring; omit surrounding indentation unless needed to identify the match.
For ambiguous matches, include more context or use replace_all=True to replace every match."""

EXECUTE_TOOL_DESCRIPTION = """Runs a shell command locally and waits for completion, returning its exit code.
Use output_mode='stash' for noisy builds, tests, or installs: save output to a bounded log and return its path, with a short tail on failure.
Prefer targeted rg/jq/tail extraction when you know what signal you need; use read_file for source files.
Quote paths with spaces; use && for dependent commands. The optional timeout overrides the command time limit."""

HARNESS_FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Conventions

- Read files before editing and understand existing content before making changes.
- Mimic existing style, naming conventions, and patterns.
- Use `make_file` only for new files at new paths.
- Use `edit_file` for every change to an existing file. Do not use `make_file` as an overwrite tool.
- Use `execute` with `rg --files` for file discovery and `rg -n --no-heading --color never` for text search."""


class HarnessFilesystemMiddleware(FilesystemMiddleware):
    """Filesystem middleware with harness-specific naming and editing guidance."""

    tool_allowlist = HARNESS_FILESYSTEM_TOOLS
    default_system_prompt = HARNESS_FILESYSTEM_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        backend: Any = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: Mapping[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
        human_message_token_limit_before_evict: int | None = 50000,
        max_execute_timeout: int = 3600,
        _permissions: list[Any] | None = None,
    ) -> None:
        descriptions = {
            "write_file": MAKE_FILE_TOOL_DESCRIPTION,
            "edit_file": EDIT_FILE_TOOL_DESCRIPTION,
            "execute": EXECUTE_TOOL_DESCRIPTION,
            **dict(custom_tool_descriptions or {}),
        }
        resolved_system_prompt = self.default_system_prompt if system_prompt is None else system_prompt
        super().__init__(
            backend=backend,
            system_prompt=resolved_system_prompt,
            custom_tool_descriptions=descriptions,
            tool_token_limit_before_evict=tool_token_limit_before_evict,
            human_message_token_limit_before_evict=human_message_token_limit_before_evict,
            max_execute_timeout=max_execute_timeout,
            tools=list(self.tool_allowlist),
            permissions=_permissions,
        )
        self.tools = [
            self._create_make_file_tool(tool) if getattr(tool, "name", None) == "write_file" else tool
            for tool in self.tools
        ]
        self.tools = [self._with_stash_mode(tool) if tool.name == "execute" else tool for tool in self.tools]

    def _with_stash_mode(self, tool: StructuredTool) -> StructuredTool:
        schema = create_model(
            "HarnessExecuteSchema", __base__=tool.args_schema,
            output_mode=(Literal["auto", "stash"], Field(
                default="auto", description="auto returns output normally; stash saves it and returns a log path.",
            )),
        )

        async def execute_stashed(command: str, runtime: ToolRuntime, timeout: int | None) -> ToolMessage:
            def error(text: str) -> ToolMessage:
                return ToolMessage(content=f"Error: {text}", status="error", name="execute", tool_call_id=runtime.tool_call_id)

            backend = self.backend
            if not isinstance(backend, LocalShellBackend) or backend.virtual_mode:
                return error("stash requires a local shell backend with virtual_mode=False")
            if not command.strip():
                return error("command must be non-empty")
            if timeout is not None and not 0 < timeout <= self._max_execute_timeout:
                return error(f"timeout must be between 1 and {self._max_execute_timeout} seconds")
            try:
                result = await stash_command(
                    command, cwd=backend.cwd, env=backend._env,
                    timeout=timeout if timeout is not None else backend._default_timeout,
                    directory=workspace_directory(backend.cwd) / "artifacts" / "execute",
                )
            except (OSError, ValueError) as exc:
                return error(str(exc))
            status = "timed out" if result.timed_out else "succeeded" if result.exit_code == 0 else "failed"
            content = (
                f"Command {status} (exit code {result.exit_code}).\nOutput saved to: {result.path}\n"
                f"Captured {result.captured_bytes} bytes."
            )
            if result.truncated:
                content += " Log is incomplete: capture limit reached; later output was not saved."
            if result.timed_out:
                content += " Log is incomplete: command timed out."
            if result.exit_code and result.tail:
                content += f"\nFailure tail (up to {STASH_TAIL_BYTES} bytes):\n{result.tail}"
            return ToolMessage(
                content=content, name="execute", tool_call_id=runtime.tool_call_id,
                artifact={
                    "exit_code": result.exit_code, "output_file": str(result.path),
                    "truncated": result.truncated, "captured_bytes": result.captured_bytes,
                    "timed_out": result.timed_out,
                },
            )

        def execute(
            command: str, runtime: ToolRuntime, timeout: int | None = None,
            output_mode: Literal["auto", "stash"] = "auto",
        ) -> ToolMessage:
            if output_mode == "auto":
                return tool.func(command=command, runtime=runtime, timeout=timeout)
            return asyncio.run(execute_stashed(command, runtime, timeout))

        async def aexecute(
            command: str, runtime: ToolRuntime, timeout: int | None = None,
            output_mode: Literal["auto", "stash"] = "auto",
        ) -> ToolMessage:
            if output_mode == "auto":
                return await tool.coroutine(command=command, runtime=runtime, timeout=timeout)
            return await execute_stashed(command, runtime, timeout)

        return StructuredTool.from_function(
            func=execute, coroutine=aexecute, name=tool.name, description=tool.description,
            args_schema=schema, infer_schema=False, return_direct=tool.return_direct,
            response_format=tool.response_format,
        )

    def _create_make_file_tool(self, tool: StructuredTool) -> StructuredTool:
        def make_file(file_path: str, content: str, runtime: ToolRuntime) -> ToolMessage:
            try:
                validated_path = validate_path(file_path)
                if not isinstance(self.backend, FilesystemBackend):
                    raise ValueError("make_file requires a local filesystem backend")
                if error := _permission_error(self._permissions, "write", self.backend, validated_path):
                    raise PermissionError(error.removeprefix("Error: "))
                # Resolve the parent only: exclusive creation must reject even dangling symlinks.
                key = Path(validated_path)
                path = self.backend._resolve_path(str(key.parent)) / key.name
                data = content.encode("utf-8")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(data)
            except FileExistsError:
                result, status = "Error: path already exists; use read_file then edit_file.", "error"
            except (OSError, ValueError) as error:
                result, status = f"Error: {error}", "error"
            else:
                result, status = f"Created file {file_path}", "success"
            return ToolMessage(content=result, status=status, name=MAKE_FILE_TOOL_NAME, tool_call_id=runtime.tool_call_id)

        async def amake_file(file_path: str, content: str, runtime: ToolRuntime) -> ToolMessage:
            return await asyncio.to_thread(make_file, file_path, content, runtime)

        return StructuredTool.from_function(
            func=make_file,
            coroutine=amake_file,
            name=MAKE_FILE_TOOL_NAME,
            description=tool.description,
            return_direct=tool.return_direct,
            args_schema=tool.args_schema,
            infer_schema=False,
            response_format=tool.response_format,
        )

    @staticmethod
    def _mutation_error(request: ToolCallRequest) -> ToolMessage | None:
        """Let the upstream guard recognize our alias without changing model history."""
        if request.tool_call["name"] not in {MAKE_FILE_TOOL_NAME, "edit_file"}:
            return None

        def normalize(call: dict[str, Any]) -> dict[str, Any]:
            return {**call, "name": "write_file"} if call["name"] == MAKE_FILE_TOOL_NAME else call

        state = request.state
        if not isinstance(state, Mapping):
            return None
        messages = list(state.get("messages", []))
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(message := messages[index], AIMessage):
                messages[index] = message.model_copy(update={"tool_calls": [normalize(call) for call in message.tool_calls]})
                break
        error = _parallel_file_mutation_error(request.override(
            tool_call=normalize(request.tool_call), state={**state, "messages": messages},
        ))
        return error.model_copy(update={"name": request.tool_call["name"]}) if error is not None else None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if error := self._mutation_error(request):
            return error
        return self._edit_diagnostic(request, super().wrap_tool_call(request, handler))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if error := self._mutation_error(request):
            return error
        result = await super().awrap_tool_call(request, handler)
        if request.tool_call["name"] == "edit_file" and isinstance(result, ToolMessage) and result.status == "error":
            return await asyncio.to_thread(self._edit_diagnostic, request, result)
        return result

    def _edit_diagnostic(self, request: ToolCallRequest, result: ToolMessage | Command) -> ToolMessage | Command:
        """Add a read-only whitespace hint; leave upstream matching and errors intact."""
        if (
            request.tool_call["name"] != "edit_file"
            or not isinstance(result, ToolMessage)
            or result.status != "error"
            or not isinstance(result.content, str)
            or not result.content.startswith("Error: String not found in file:")
            or not isinstance(self.backend, FilesystemBackend)
        ):
            return result
        args = request.tool_call["args"]
        old = args.get("old_string")
        if not isinstance(old, str) or not old.strip() or len(old) > 4096:
            return result
        try:
            path = validate_path(args["file_path"])
            # Diagnostics must not disclose content denied or gated behind approval.
            if _check_fs_permission(self._permissions, "read", path) != "allow":
                return result
            real_path = _resolve_real_path(self.backend, path)
            if _check_fs_permission(self._permissions, "read", real_path) != "allow":
                return result
            fd = os.open(self.backend._resolve_path(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as handle:
                raw = handle.read(128 * 1024 + 1)
            if len(raw) > 128 * 1024:
                return result
            content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, ValueError, RuntimeError):
            return result

        def normalize(line: str) -> str:
            return re.sub(r"[ \t]+", " ", line).strip(" \t")

        needle = old.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if len(needle) > 50:
            return result
        expected = [normalize(line) for line in needle]
        lines = content.split("\n")
        normalized = [normalize(line) for line in lines]
        matches = []
        for index in range(len(lines) - len(needle) + 1):
            if normalized[index:index + len(needle)] == expected:
                matches.append(index)
                if len(matches) == 2:
                    break
        if not matches:
            return result
        excerpt = "\n".join(lines[matches[0]:matches[0] + len(needle)])
        preview = json.dumps(excerpt[:240], ensure_ascii=True)
        suffix = " (excerpt truncated)" if len(excerpt) > 240 else ""
        ambiguity = " Multiple candidates exist; add distinguishing context." if len(matches) > 1 else ""
        hint = (
            f"\nWhitespace-only candidate at line {matches[0] + 1} (JSON-escaped): {preview}{suffix}."
            f"{ambiguity} No edit applied. Whitespace may be meaningful; read the target and retry with exact text."
        )
        return result.model_copy(update={"content": result.content + hint})

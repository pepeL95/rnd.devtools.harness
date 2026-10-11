from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.utils import validate_path
from deepagents.middleware.filesystem import (
    FilesystemMiddleware,
    _check_fs_permission,
    _parallel_file_mutation_error,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

MAKE_FILE_TOOL_NAME = "make_file"
HARNESS_FILESYSTEM_TOOLS = ("read_file", "write_file", "edit_file", "execute")

MAKE_FILE_TOOL_DESCRIPTION = """Creates a new file and any missing parent directories. Provide the full initial file contents.
Fails if the path already exists; use `read_file` then `edit_file` to modify existing files."""

EDIT_FILE_TOOL_DESCRIPTION = """Replaces an exact, non-empty string in an existing file.
Read first; match the verbatim file content, including whitespace, not the read status header.
Use the smallest unique substring; omit surrounding indentation unless needed to identify the match.
For ambiguous matches, include more context or use replace_all=True to replace every match."""

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
            _permissions=_permissions,
        )
        self.tools = [
            self._create_make_file_tool(tool) if getattr(tool, "name", None) == "write_file" else tool
            for tool in self.tools
        ]

    def _create_make_file_tool(self, tool: StructuredTool) -> StructuredTool:
        def make_file(file_path: str, content: str, runtime: ToolRuntime) -> ToolMessage:
            try:
                validated_path = validate_path(file_path)
                if _check_fs_permission(self._permissions, "write", validated_path) == "deny":
                    raise PermissionError(f"permission denied for write on {validated_path}")
                if not isinstance(self.backend, FilesystemBackend):
                    raise ValueError("make_file requires a local filesystem backend")
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
            fd = os.open(self.backend._resolve_path(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as handle:
                raw = handle.read(128 * 1024 + 1)
            if len(raw) > 128 * 1024:
                return result
            content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, ValueError):
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

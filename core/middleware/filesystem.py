from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import EXECUTION_SYSTEM_PROMPT
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.filesystem import supports_execution
from deepagents.middleware.filesystem import _route_host_path_prompt
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.tools import StructuredTool

MAKE_FILE_TOOL_NAME = "make_file"

MAKE_FILE_TOOL_DESCRIPTION = """Creates a new file in the filesystem.

Usage:
- Use `make_file` only when creating a new file at a new path.
- Never use `make_file` to modify an existing file. If the path already exists, use `read_file` and then `edit_file`.
- Prefer `edit_file` for changes to existing files.
"""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in existing files.

Usage:
- Use `edit_file` for every modification to an existing file.
- You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
- When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
- If old_string is ambiguous, include more surrounding context or use replace_all=True when you truly want every match.
- Only use emojis if the user explicitly requests it.
"""

HARNESS_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE = """## Following Conventions

- Read files before editing and understand existing content before making changes.
- Mimic existing style, naming conventions, and patterns.
- Use `make_file` only for new files at new paths.
- Use `edit_file` for every change to an existing file. Do not use `make_file` as an overwrite tool.

## Filesystem Tools `ls`, `read_file`, `make_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /. Follow the tool docs for the available tools, and use pagination (offset/limit) when reading large files.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- make_file: create a new file at a new path only
- edit_file: modify an existing file by exact string replacement
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files

## Large Tool Results

When a tool result is too large, it may be offloaded into the filesystem instead of being returned inline. In those cases, use `read_file` to inspect the saved result in chunks, or use `grep` within `{large_tool_results_prefix}/` if you need to search across offloaded tool results and do not know the exact file path. Offloaded tool results are stored under `{large_tool_results_prefix}/<tool_call_id>`."""


class HarnessFilesystemMiddleware(FilesystemMiddleware):
    """Filesystem middleware with harness-specific naming and editing guidance."""

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
        super().__init__(
            backend=backend,
            system_prompt=system_prompt,
            custom_tool_descriptions=descriptions,
            tool_token_limit_before_evict=tool_token_limit_before_evict,
            human_message_token_limit_before_evict=human_message_token_limit_before_evict,
            max_execute_timeout=max_execute_timeout,
            _permissions=_permissions,
        )
        self.tools = [self._rename_write_file_tool(tool) if getattr(tool, "name", None) == "write_file" else tool for tool in self.tools]

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | ExtendedModelResponse:
        request, system_prompt = self._prepare_request_and_prompt(request)
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))

        eviction_result = self._evict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | ExtendedModelResponse:
        request, system_prompt = self._prepare_request_and_prompt(request)
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))

        eviction_result = await self._aevict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = await handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response
        return await handler(request)

    def _prepare_request_and_prompt(self, request: ModelRequest[Any]) -> tuple[ModelRequest[Any], str]:
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)
        backend_supports_execution = False
        backend = None
        if has_execute_tool:
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            backend_supports_execution = supports_execution(backend)
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        if self._custom_system_prompt is not None:
            return request, self._custom_system_prompt

        prompt_parts = [
            HARNESS_FILESYSTEM_SYSTEM_PROMPT_TEMPLATE.format(
                large_tool_results_prefix=self._large_tool_results_prefix,
            )
        ]
        if has_execute_tool and backend_supports_execution:
            prompt_parts.append(EXECUTION_SYSTEM_PROMPT)
            route_prompt = _route_host_path_prompt(backend)
            if route_prompt:
                prompt_parts.append(route_prompt)
        return request, "\n\n".join(prompt_parts).strip()

    @staticmethod
    def _rename_write_file_tool(tool: StructuredTool) -> StructuredTool:
        return StructuredTool.from_function(
            func=tool.func,
            coroutine=tool.coroutine,
            name=MAKE_FILE_TOOL_NAME,
            description=MAKE_FILE_TOOL_DESCRIPTION,
            return_direct=tool.return_direct,
            args_schema=tool.args_schema,
            infer_schema=False,
            response_format=tool.response_format,
        )

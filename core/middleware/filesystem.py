from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.tools import StructuredTool

MAKE_FILE_TOOL_NAME = "make_file"
HARNESS_FILESYSTEM_TOOLS = ("read_file", "write_file", "edit_file", "execute")

MAKE_FILE_TOOL_DESCRIPTION = """Creates a new file in the filesystem with the provided contents.

Usage:
- Use `make_file` only when creating a new file at a new path.
- Provide the full initial file contents.
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
            self._rename_write_file_tool(tool) if getattr(tool, "name", None) == "write_file" else tool
            for tool in self.tools
        ]

    @staticmethod
    def _rename_write_file_tool(tool: StructuredTool) -> StructuredTool:
        return StructuredTool.from_function(
            func=tool.func,
            coroutine=tool.coroutine,
            name=MAKE_FILE_TOOL_NAME,
            description=tool.description,
            return_direct=tool.return_direct,
            args_schema=tool.args_schema,
            infer_schema=False,
            response_format=tool.response_format,
        )

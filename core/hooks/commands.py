from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex


@dataclass(frozen=True)
class ParsedCommand:
    raw: str
    cmd: str
    args: tuple[str, ...]


def parse_command(command: str) -> ParsedCommand | None:
    """Recognize literal simple commands; never interpret shell expansions."""
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            elif quote == '"' and char in "$`":
                return None
        elif char in "'\"":
            quote = char
        elif char in ";&|<>()\n\r$`*?[]{}~#":
            return None
    if quote or escaped:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens or not tokens[0] or "=" in tokens[0]:
        return None
    if tokens[0] in {"if", "then", "else", "fi", "for", "while", "until", "do", "done", "case", "esac", "!", "time", "function"}:
        return None
    return ParsedCommand(command, PurePosixPath(tokens[0]).name, tuple(tokens[1:]))

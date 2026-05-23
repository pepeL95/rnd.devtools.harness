from __future__ import annotations

import subprocess
from pathlib import Path


def git_branch(cwd: Path) -> str | None:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)


def git_dirty(cwd: Path) -> bool | None:
    output = _git(["status", "--porcelain"], cwd)
    if output is None:
        return None
    return bool(output.strip())


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


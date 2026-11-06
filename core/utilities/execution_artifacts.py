from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import logging
import os
from pathlib import Path
import re
import stat
from typing import BinaryIO

from core.utilities.workspace import workspace_directory

try:
    import fcntl
except ImportError:  # Keep logs rather than delete without cross-process locking.
    fcntl = None

LOG_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
_LOG_NAME = re.compile(r"execute-(\d{8}T\d{12}Z)-[a-z0-9_]{8}\.log")
logger = logging.getLogger(__name__)


def execution_log_prefix() -> str:
    return f"execute-{datetime.now(timezone.utc).strftime(LOG_TIMESTAMP_FORMAT)}-"


def lock_execution_log(log: BinaryIO) -> None:
    """Hold a shared lease until the writer closes its file."""
    if fcntl is not None:
        fcntl.flock(log.fileno(), fcntl.LOCK_SH)


class ExecutionArtifacts:
    """Own one session's execution logs; prune only at a completed interaction."""

    def __init__(self, cwd: Path, session_path: Path, retention_days: int = 7) -> None:
        if retention_days < 0:
            raise ValueError("Execution log retention days must be non-negative")
        # Include the session store path so equal IDs in different stores cannot collide.
        owner = sha256(str(session_path.resolve()).encode()).hexdigest()
        self.directory = workspace_directory(cwd) / "artifacts" / "execute" / owner
        self.retention_days = retention_days

    def prune(self, last_interaction: str) -> None:
        if not self.retention_days:
            return
        if fcntl is None:
            logger.warning("Execution log cleanup skipped: cross-process locking is unavailable.")
            return
        interaction = datetime.fromisoformat(last_interaction)
        if interaction.tzinfo is None:
            raise ValueError("Execution log retention requires a timezone-aware interaction timestamp")
        cutoff = interaction - timedelta(days=self.retention_days)
        if self.directory.is_symlink():
            logger.warning("Execution log cleanup skipped for symlink directory: %s", self.directory)
            return
        try:
            paths = list(self.directory.iterdir())
        except FileNotFoundError:
            return
        for path in paths:
            match = _LOG_NAME.fullmatch(path.name)
            if match is None:
                continue
            try:
                created = datetime.strptime(match[1], LOG_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if created >= cutoff:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    opened = os.fstat(fd)
                    current = path.stat(follow_symlinks=False)
                    if stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino):
                        path.unlink()
                finally:
                    os.close(fd)
            except (FileNotFoundError, BlockingIOError):
                continue
            except OSError as exc:
                logger.warning("Cannot prune execution log %s: %s", path, exc)

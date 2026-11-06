from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import os
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile

STASH_MAX_BYTES = 10 * 1024 * 1024
STASH_TAIL_BYTES = 4096


@dataclass(frozen=True)
class StashedExecution:
    path: Path
    exit_code: int
    captured_bytes: int
    truncated: bool
    tail: str
    timed_out: bool


async def _terminate(process: asyncio.subprocess.Process) -> None:
    # Kill the group even if the shell exited but descendants still hold its pipe.
    with suppress(ProcessLookupError):
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    if process.stdout is not None:
        while await process.stdout.read(65536):
            pass
    await process.wait()


async def stash_command(
    command: str, *, cwd: Path, env: dict[str, str], timeout: float, directory: Path,
) -> StashedExecution:
    """Capture a foreground command without buffering its full output in memory."""
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", prefix="execute-", suffix=".log", dir=directory, delete=False) as log:
        process = await asyncio.create_subprocess_shell(
            command, cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        captured = 0
        truncated = False
        tail = b""
        timed_out = False
        try:
            async with asyncio.timeout(timeout):
                assert process.stdout is not None
                while chunk := await process.stdout.read(65536):
                    remaining = max(0, STASH_MAX_BYTES - captured)
                    saved = chunk[:remaining]
                    log.write(saved)
                    captured += len(saved)
                    truncated |= len(chunk) > len(saved)
                    tail = (tail + chunk)[-STASH_TAIL_BYTES:]
                    # Keep draining after the cap so the command retains its real exit status.
                exit_code = await process.wait()
        except TimeoutError:
            await _terminate(process)
            timed_out = True
            exit_code = 124
        except BaseException:
            await _terminate(process)
            raise
    return StashedExecution(
        path=Path(log.name), exit_code=exit_code, captured_bytes=captured,
        truncated=truncated, tail=tail.decode("utf-8", errors="replace"), timed_out=timed_out,
    )

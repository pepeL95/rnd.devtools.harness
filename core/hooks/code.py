from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryFile
from typing import Any

CODE_TIMEOUT_SECONDS = 30
OUTPUT_LIMIT = 16000


def run_code(path: Path, event: dict[str, Any], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    """Run a hook directly, bounding wait time and retained diagnostics."""
    with TemporaryFile() as stdin, TemporaryFile() as stdout, TemporaryFile() as stderr:
        stdin.write(json.dumps(event, ensure_ascii=False).encode())
        stdin.seek(0)
        with subprocess.Popen(
            [str(path)], cwd=cwd, env=env, stdin=stdin, stdout=stdout, stderr=stderr,
            start_new_session=True,
        ) as process:
            status = "success"
            try:
                process.wait(timeout=CODE_TIMEOUT_SECONDS)
                if process.returncode:
                    status = "error"
            except subprocess.TimeoutExpired:
                status = "timeout"
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stdout.seek(0)
            stderr.seek(0)
            return {
                "status": status, "exit_code": process.returncode,
                "stdout": stdout.read(OUTPUT_LIMIT).decode(errors="replace"),
                "stderr": stderr.read(OUTPUT_LIMIT).decode(errors="replace"),
            }

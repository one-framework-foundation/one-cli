"""Shell commands with a timeout that stops the whole process tree."""
from __future__ import annotations

import os
import signal
import subprocess


def run_shell(command: str, *, timeout: float, cwd: str | None = None,
              input: str | None = None) -> subprocess.CompletedProcess:
    """`subprocess.run(command, shell=True, capture_output=True)` that stops it all.

    On timeout `subprocess.run` kills only the shell; `sleep 999 &` or a dev
    server it started keeps running after the caller has moved on. Here the
    command gets its own session and the whole session is killed. Raises
    `subprocess.TimeoutExpired` like `run`. Output decodes leniently.

    For unattended commands (done-check, hooks, guards): without `input` the
    command reads /dev/null, and its own session has no controlling terminal.
    Commands a person typed at the prompt keep their terminal and do not use this.
    """
    proc = subprocess.Popen(
        command, shell=True, cwd=cwd, text=True, errors="replace",
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    completed = False
    try:
        stdout, stderr = proc.communicate(input, timeout=timeout)
        completed = True
    finally:
        if not completed:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe:
                    pipe.close()
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)

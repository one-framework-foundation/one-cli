"""Background jobs - a long command that does not own the loop.

The bash tool is synchronous: while a long `pytest -q` runs, the agent
is a process waiting on a pipe. It cannot read a file, cannot answer, cannot
start anything else, and if the command outlives the tool deadline everything it
had produced is thrown away with it. That is one shape of work - the quick one -
being the only shape available.

A job is the other shape. `job_start` returns immediately with an id, the output
goes to a file the moment it is written, and the model gets on with something
else. Four verbs, which is the whole protocol:

    job_start   begin, get an id
    job_list    what is running, what finished, what it cost
    job_output  read any part of a job's output, running or finished
    job_cancel  stop one

plus the part that makes it a protocol rather than four tools: a **completion
notice**. A job that finishes while the model is thinking about something else
is announced at the top of the next turn, once. Without that the model must
poll - and a model that must poll either forgets to (the job is never read) or
polls every turn (it has re-invented waiting, with extra steps).

Claude Code solves the same problem with BashOutput/KillShell, narrowly, for
shells only. This is the general shape because a run already thinks in tasks,
and it is one module because one-cli has one execution backend.

Output goes to a file rather than a pipe on purpose: a pipe that nobody drains
fills its buffer and blocks the child, so a chatty job would hang the moment the
model stopped reading. A file is also readable *while* the job runs, which is
what makes `job_output` useful before the exit code exists.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from onecli import secureio

JOB_DIR = Path.home() / ".one-cli" / "jobs"
# How long a cancelled job gets to die politely before it is killed.
GRACE = 3.0
# Lines returned by job_output when the model does not say.
DEFAULT_LINES = 200
OUTPUT_CAP = 1024 * 1024


@dataclass
class Job:
    id: str
    command: str
    log: Path
    started: float
    proc: subprocess.Popen | None = None
    pgid: int | None = None
    finished: float = 0.0
    exit_code: int | None = None
    cancelled: bool = False
    announced: bool = False
    # Known only from an earlier process's record: there is no handle to poll,
    # so it can never be "running" here, whatever that record last said.
    lost: bool = False

    @property
    def running(self) -> bool:
        return (self.exit_code is None and not self.cancelled
                and not self.lost)

    @property
    def seconds(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def state(self) -> str:
        if self.running:
            return "running"
        if self.cancelled:
            return "cancelled"
        if self.lost:
            return "lost (process gone, exit unknown)"
        return "ok" if self.exit_code == 0 else f"exit {self.exit_code}"

    def lines(self) -> int:
        try:
            with open(self.log, "rb") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0


class JobStore:
    """Every job of one session. One per session, like the spill store."""

    def __init__(self, session_id: str, root: Path | str = JOB_DIR):
        self.session_id = session_id
        self.root = Path(root) / session_id
        self.jobs: dict[str, Job] = {}
        self._next = 0
        self._restore()

    def _restore(self) -> None:
        """Jobs an earlier turn of this session started, as that turn left them.

        Every host turn is a fresh process, and the turn that started a job
        stopped it at exit. Without this the next turn knows j1 only from its
        transcript, where the last word was `[running]`, and would report a
        stopped job as still running.
        """
        for meta in sorted(self.root.glob("j*.json")):
            # A damaged record loses that one job, never the whole store.
            try:
                rec = json.loads(meta.read_text())
                job_id = meta.stem
                num = int(job_id[1:])
                job = Job(id=job_id, command=str(rec.get("command", "")),
                          log=self.root / f"{job_id}.log",
                          started=float(rec.get("started", 0.0)),
                          finished=float(rec.get("finished", 0.0)),
                          exit_code=rec.get("exit_code"),
                          cancelled=bool(rec.get("cancelled")),
                          announced=bool(rec.get("announced")))
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if job.exit_code is None and not job.cancelled:
                # Its process died with the turn that owned it, unrecorded.
                job.lost = True
            self.jobs[job_id] = job
            self._next = max(self._next, num)

    def _save(self, job: Job) -> None:
        """Never fatal: losing the record must not lose the job."""
        rec = {"command": job.command, "started": job.started,
               "finished": job.finished, "exit_code": job.exit_code,
               "cancelled": job.cancelled, "announced": job.announced}
        try:
            secureio.atomic_write_text(self.root / f"{job.id}.json", json.dumps(rec),
                                       private=True)
        except OSError:
            pass

    # -- lifecycle ---------------------------------------------------------

    def start(self, command: str, workspace: str) -> Job:
        # Private like sessions and spills: a log holds whatever the command printed.
        with secureio.directory(self.root, create=True, private=True) as root_fd:
            while True:
                self._next += 1
                job_id = f"j{self._next}"
                log = self.root / f"{job_id}.log"
                try:
                    fd = os.open(log.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                 | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
                    break
                except FileExistsError:
                    continue  # Retained output reserves this ID, including after restore.
        handle = os.fdopen(fd, "wb")
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=workspace,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # Its own process group, so cancelling a job kills the pipeline
                # it started and not only the shell that wrapped it. A `make -j`
                # cancelled without this leaves its children compiling.
                start_new_session=True,
            )
        finally:
            handle.close()
        job = Job(id=job_id, command=command, log=log,
                  started=time.time(), proc=proc, pgid=proc.pid)
        self.jobs[job_id] = job
        self._save(job)
        return job

    def poll(self) -> list[Job]:
        """Reap what has finished. Returns the jobs that finished just now."""
        done: list[Job] = []
        for job in self.jobs.values():
            if job.exit_code is not None or job.proc is None:
                continue
            code = job.proc.poll()
            if code is None:
                continue
            job.exit_code = code
            job.finished = _finish_time(job)
            self._save(job)
            done.append(job)
        return done

    def notices(self) -> list[str]:
        """One line per job that finished and has not been announced yet."""
        self.poll()
        out: list[str] = []
        for job in self.jobs.values():
            if job.running or job.announced:
                continue
            job.announced = True
            self._save(job)
            if job.lost:
                out.append(
                    f"[job {job.id}] from an earlier turn is gone: its process "
                    f"ended with that turn and its exit code is unknown. "
                    f"{job.command}. Do not report it as running; start it "
                    f"again if its result is still needed.")
                continue
            out.append(
                f"[job {job.id}] finished {job.state} after {job.seconds:.0f}s: "
                f"{job.command}. {_plural(job.lines(), 'line')} of output - "
                f"read it with job_output({job.id!r}).")
        return out

    def cancel(self, job_id: str) -> str:
        job = self.jobs.get(job_id)
        if job is None:
            return f"error: no job {job_id!r}"
        if not job.running:
            return f"job {job_id} was already {job.state}"
        _stop(job)
        job.cancelled = True
        job.finished = time.time()
        # Announced already, by the act of asking for it.
        job.announced = True
        self._save(job)
        return f"job {job_id} cancelled after {job.seconds:.0f}s"

    def close(self) -> list[str]:
        """End of session: nothing outlives the CLI that started it.

        A run that exits leaving a `sleep 999` behind has handed the user a
        process they did not start and cannot see. Survivors are stopped and
        named, never left running quietly.
        """
        stopped = []
        for job in self.jobs.values():
            if job.proc is not None and job.pgid is not None:
                was_running = job.running
                _stop(job)
                if not was_running:
                    continue
                job.cancelled = True
                job.finished = time.time()
                # Not announced: the next turn must be told it was stopped,
                # and the log says so for anyone who reads only the file.
                _note(job, "stopped when the turn that started it exited; "
                           "exit code unknown")
                self._save(job)
                stopped.append(f"{job.id} ({job.command})")
        return stopped

    # -- reading -----------------------------------------------------------

    def table(self) -> str:
        self.poll()
        if not self.jobs:
            return "no jobs"
        rows = ["id    state       secs  lines  command"]
        for job in self.jobs.values():
            rows.append(f"{job.id:<5} {job.state:<11} {job.seconds:>4.0f} "
                        f"{job.lines():>6}  {job.command[:60]}")
        return "\n".join(rows)

    def output(self, job_id: str, offset: int = 0,
               limit: int = DEFAULT_LINES) -> str:
        job = self.jobs.get(job_id)
        if job is None:
            return f"error: no job {job_id!r}"
        self.poll()
        limit = max(1, int(limit))
        start = max(0, int(offset))
        window = bytearray()
        count = 0
        capped = False
        try:
            with job.log.open("rb") as fh:
                while True:
                    # Bound even a single log line; scan only to offset+limit.
                    # Lines before the offset are skipped, not kept, so only
                    # the returned window counts against the cap.
                    inside = count >= start
                    piece = fh.readline(min(65536, OUTPUT_CAP - len(window) + 1)
                                        if inside else 65536)
                    if not piece:
                        break
                    if inside:
                        if len(window) + len(piece) > OUTPUT_CAP:
                            capped = True
                            break
                        window.extend(piece)
                    if piece.endswith(b"\n"):
                        count += 1
                        if count >= start + limit:
                            break
        except OSError as exc:
            return f"error: cannot read job log: {exc}"
        head = (f"job {job.id} [{job.state}] {job.command}\n"
                f"lines from {start + 1} (up to {limit})"
                f"{'' if not job.running else ' (still running)'}\n")
        note = f"\n(output truncated at {OUTPUT_CAP} byte scan cap)" if capped else ""
        if window:
            body = window.decode("utf-8", "replace").rstrip("\r\n")
        elif capped:
            body = f"(line {start + 1} alone exceeds the scan cap)"
        elif start:
            body = f"(offset {start} is past the end)"
        else:
            body = "(no output yet)"
        return head + body + note


def _finish_time(job: Job) -> float:
    """When the job actually ended, as well as we can know it.

    We learn a job is over by reaping it, and reaping happens between calls -
    which can be a long think after the process died. Reporting that gap as
    runtime would make a six-second sleep read as twice that or more. The
    log's last write is the process's own evidence, so it wins whenever it is
    sane (after the start, not after the reap). A job that printed nothing
    leaves no evidence and gets the reap time, which is an upper bound.
    """
    reaped = time.time()
    try:
        wrote = job.log.stat().st_mtime
    except OSError:
        return reaped
    return wrote if job.started <= wrote <= reaped else reaped


def _note(job: Job, text: str) -> None:
    try:
        with open(job.log, "a") as fh:
            fh.write(f"[one-cli] job {job.id} {text}\n")
    except OSError:
        pass


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _stop(job: Job) -> None:
    """Stop the recorded group even after its leader has been reaped."""
    if job.proc is None or job.pgid is None:
        return
    stop_process(job.proc, job.pgid)
    job.pgid = None


def stop_process(proc: subprocess.Popen, pgid: int) -> None:
    """TERM, grace, KILL the entire session group, then reap its leader."""
    started = time.monotonic()
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=GRACE)
    except subprocess.TimeoutExpired:
        pass
    # An exited leader says nothing about children ignoring TERM.
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        pass
    else:
        time.sleep(max(0.0, GRACE - (time.monotonic() - started)))
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=GRACE)
    except subprocess.TimeoutExpired:
        # A killed process can remain unreapable in uninterruptible I/O.
        pass


# -- the tool functions --------------------------------------------------
# Signature matches every other tool: keyword arguments, a string back, no
# exception ever reaches the loop. `jobs` is injected by `tools.run` for the
# tools that declare they need it.


def job_start(workspace: str, command: str, jobs: JobStore | None = None) -> str:
    if jobs is None:
        return "error: background jobs are not available in this run"
    if not command.strip():
        return "error: command is empty"
    try:
        job = jobs.start(command, workspace)
    except OSError as exc:
        return f"error: could not start job: {exc}"
    return (f"job {job.id} started: {command}\n"
            f"It runs in the background. You will be told when it finishes; "
            f"until then read it with job_output({job.id!r}) or stop it with "
            f"job_cancel({job.id!r}). Do not wait for it - do something else.")


def job_list(workspace: str, jobs: JobStore | None = None) -> str:
    if jobs is None:
        return "error: background jobs are not available in this run"
    return jobs.table()


def job_output(workspace: str, id: str, offset: int = 0,
               limit: int = DEFAULT_LINES, jobs: JobStore | None = None) -> str:
    if jobs is None:
        return "error: background jobs are not available in this run"
    return jobs.output(id, offset=offset, limit=limit)


def job_cancel(workspace: str, id: str, jobs: JobStore | None = None) -> str:
    if jobs is None:
        return "error: background jobs are not available in this run"
    return jobs.cancel(id)


__all__ = ["Job", "JobStore", "job_start", "job_list", "job_output",
           "job_cancel", "JOB_DIR", "DEFAULT_LINES"]

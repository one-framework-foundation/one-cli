"""Code mode - one program instead of twenty tool calls.

A tool call is a round trip: the model emits, the wire carries, the host runs,
the result goes back through the context window, and the model is sampled again.
Reading five files costs five of those. Written as a program it costs one, and
the parts that are ordinary programming - a loop, a filter, a sum, a conditional
- happen in Python where they are free instead of in the model where each step
is a sampled decision that can go wrong.

The bindings are the point. A program here does not `import os` and do as it
pleases: it calls `one.read`, `one.bash`, `one.write`, and every one of those is
sent back to the agent and put through **the same gate as any other tool call** -
guards, PreToolUse hooks, the permission mode. Code mode with unmediated file
access would be a hole around every guard in the system, and a `--read-only` run
that can write through a Python program is not read-only. A refused binding
raises `one.Denied` in the program rather than returning a string, because a
program that continues after a refused write produces a confident wrong answer.

Three implementation choices, each one a failure avoided:

**A subprocess, not `exec` in our process.** The model's program can loop
forever, exhaust memory, or call `sys.exit`. In-process that takes the agent
with it. Out of process it is killable, and the deadline is enforceable.

**Program output to a file, not a pipe.** Same reason as the job store: a pipe
nobody drains blocks the child, and here the parent is busy answering binding
calls rather than draining, so a chatty program would deadlock on its own print
statements.

**A socketpair for the bindings, on its own fd.** The program's stdout belongs
to the program - the model prints there and reads it back. Multiplexing the
protocol through the same stream would make a stray `print` corrupt a binding
call. The descriptor is inherited by number and the number is handed over in the
environment: `pass_fds` keeps an fd open where it is, and dup2'ing it to a fixed
3 in a `preexec_fn` does not survive, because the fd-closing pass runs after
preexec and closes everything not named in `pass_fds`.

**Confined when file rules are on.** The bindings are gated, but the program
itself is ordinary Python: `open()` and `os.system` do not ask anyone. With no
file rules that is the same reach `bash` already has. With deny or allow rules,
or `--add-dir` roots, it would be a way around them, so the program then runs
inside the tool sandbox (`tool_sandbox.confinement_argv`): no network, a
read-only `/usr`, an empty scratch workspace, and the workspace reachable only
through the gated bindings. Without the sandbox code mode refuses rather than
run unconfined. `--read-only` and plan mode never offer `code` at all.

The worker is a process rather than a thread in a shared runtime: Python has
no way to kill a runaway thread, and unmediated code mode would be a hole
around every guard in the system.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from onecli.sandbox import shell as tool_sandbox

DEFAULT_TIMEOUT = 120
# Bytes of out.txt read back. Far above any context window, so spill still
# decides what the model sees; this only keeps a runaway print loop out of RAM.
OUTPUT_CAP = 32 * 1024 * 1024
# The bindings a program may call. Anything else in the registry is reachable
# through `one.call(name, **kwargs)`; these are the ones that get a short name.
BINDINGS = ("read", "write", "edit", "ls", "glob", "grep", "bash",
            "web_search", "web_fetch")

RUNTIME = '''\
"""The bindings a code-mode program calls. Every one is a round trip to the
agent, which gates it exactly as it gates a tool call."""

import json
import socket
import sys

_sock = socket.socket(fileno=int(__import__("os").environ["ONE_BINDING_FD"]))
_io = _sock.makefile("rw", buffering=1, encoding="utf-8", newline="\\n")


class Denied(Exception):
    """A guard or hook refused this call. Do not work around it."""


class ToolError(Exception):
    """The tool ran and failed. The message is what it said."""


def call(name, **kwargs):
    _io.write(json.dumps({"tool": name, "args": kwargs}) + "\\n")
    _io.flush()
    line = _io.readline()
    if not line:
        raise ToolError("the agent went away")
    reply = json.loads(line)
    if reply.get("denied"):
        raise Denied(reply["output"])
    return reply["output"]


def _binding(name):
    def fn(*args, **kwargs):
        if args:
            raise TypeError(f"one.{name}() takes keyword arguments only")
        return call(name, **kwargs)
    fn.__name__ = name
    return fn


for _name in %(bindings)r:
    globals()[_name] = _binding(_name)
'''


@dataclass
class Outcome:
    output: str
    calls: list[tuple[str, dict]] = field(default_factory=list)
    exit_code: int = 0
    seconds: float = 0.0
    timed_out: bool = False

    def render(self) -> str:
        """What the model reads back."""
        head = f"program exited {self.exit_code} after {self.seconds:.1f}s"
        if self.timed_out:
            head = f"program KILLED after {self.seconds:.0f}s (deadline)"
        made = (f", {len(self.calls)} binding call"
                f"{'' if len(self.calls) == 1 else 's'}")
        body = self.output.rstrip() or "(the program printed nothing)"
        return f"{head}{made}\n{body}"


def execute(source: str, workspace: str, call_tool, timeout: int = DEFAULT_TIMEOUT,
            python: str = "", contained: bool = False) -> Outcome:
    """Run `source` as a program; `call_tool(name, args) -> (output, denied)`.

    The parent answers binding calls while the program runs, so the program is
    never waiting on a queue and the agent never has to guess what it will need.
    """
    try:
        seconds = float(timeout)
        if seconds <= 0:
            raise ValueError("nonpositive timeout")
        timeout = max(1, int(seconds))
    except (TypeError, ValueError, OverflowError):
        return Outcome(output="error: timeout must be a finite positive number", exit_code=-1)
    started = time.monotonic()
    parent, child = socket.socketpair()
    with tempfile.TemporaryDirectory(prefix="one-code-") as tmp:
        prog = os.path.join(tmp, "program.py")
        runtime = os.path.join(tmp, "one.py")
        with open(runtime, "w") as fh:
            fh.write(RUNTIME % {"bindings": list(BINDINGS)})
        with open(prog, "w") as fh:
            fh.write("import one\n" + source)
        out_path = os.path.join(tmp, "out.txt")
        out_file = open(out_path, "wb")
        fd = str(child.fileno())
        if contained:
            argv = _confined(tmp, fd, timeout, python)
            cwd, env = "/", {}
        else:
            argv = [python or sys.executable, "-u", prog]
            cwd, env = workspace, {**os.environ, "PYTHONPATH": tmp,
                                   "ONE_BINDING_FD": fd}
        try:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=out_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                pass_fds=(child.fileno(),),
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            out_file.close()
            parent.close()
            child.close()
            return Outcome(output=f"error: could not start the program: {exc}",
                           exit_code=-1)
        child.close()
        try:
            calls, timed_out = _serve(parent, proc, call_tool, timeout)
        finally:
            _kill(proc)
            proc.wait(timeout=5)
            out_file.close()
            parent.close()
        try:
            with open(out_path, "rb") as fh:
                raw = fh.read(OUTPUT_CAP + 1)
            output = (raw[:OUTPUT_CAP].decode(errors="replace")
                      .replace("\r\n", "\n").replace("\r", "\n"))
            if len(raw) > OUTPUT_CAP:
                output += (f"\n... (program output exceeded {OUTPUT_CAP} bytes; "
                           f"the rest was not read)")
        except OSError:
            output = ""
    return Outcome(
        output=output, calls=calls,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        seconds=time.monotonic() - started,
        timed_out=timed_out)


def _confined(tmp: str, fd: str, timeout: int, python: str) -> list[str]:
    """The sandboxed command: the program and runtime read-only, nothing else."""
    interpreter = os.path.realpath(python or sys.executable)
    if not interpreter.startswith("/usr/"):
        # Only /usr is mounted; resolve venv symlinks to their mounted target.
        interpreter = "/usr/bin/python3"
    cpu = max(1, int(timeout)) + 5
    return tool_sandbox.confinement_argv() + [
        "--ro-bind", tmp, "/one", "--tmpfs", "/workspace", "--chdir", "/workspace",
        "--setenv", "HOME", "/workspace", "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONPATH", "/one",
        "--setenv", "ONE_BINDING_FD", fd, "--remount-ro", "/",
        "--", "/usr/bin/prlimit", "--core=0:0", "--fsize=16777216:16777216",
        "--as=1073741824:1073741824", f"--cpu={cpu}:{cpu}", "--nofile=64:64",
        "--", interpreter, "-u", "/one/program.py"]


def confinement_reason(path_guards, policy=None) -> str:
    """Why this run must confine the program, or '' when it need not.

    `policy` is the run's external deciders: --guard programs and the hook
    set. Either sees only a tool named `code`, never the subprocess calls the
    program makes, so its presence alone confines the program - inside the
    sandbox every effect goes through a binding the decider does see.
    """
    for guard in path_guards or ():
        if getattr(guard, "deny", None) or getattr(guard, "allow", None):
            return "file permission rules are active"
        # The workspace itself is always a root; only an added one confines,
        # as before the workspace became the default boundary.
        if len(getattr(guard, "roots", None) or ()) > 1:
            return "--add-dir roots are active"
    for item in policy or ():
        if getattr(item, "commands", None):
            return "--guard is active"
        if any(getattr(spec, "event", "") == "PreToolUse"
               for spec in getattr(item, "specs", ()) or ()):
            return "PreToolUse hooks are active"
    return ""


def _serve(parent: socket.socket, proc, call_tool, timeout: int):
    """Answer binding calls until the program ends or the deadline passes."""
    calls: list[tuple[str, dict]] = []
    deadline = time.monotonic() + max(1, int(timeout))
    # Socket file objects become unusable after a read timeout. Keep partial
    # lines ourselves so idle polling preserves the binding channel.
    pending = b""
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _kill(proc)
            break
        parent.settimeout(min(0.2, remaining))
        if b"\n" not in pending:
            try:
                chunk = parent.recv(65536)
            except (socket.timeout, TimeoutError):
                if proc.poll() is not None:
                    break
                continue
            except OSError:
                break
            if not chunk:
                break
            pending += chunk
            continue
        line, _, pending = pending.partition(b"\n")
        try:
            request = json.loads(line)
            name = str(request["tool"])
            args = dict(request.get("args") or {})
        except (ValueError, KeyError, TypeError) as exc:
            _write(parent, {"denied": True, "output": f"malformed binding call: {exc}"})
            continue
        calls.append((name, args))
        output, denied = call_tool(name, args)
        _write(parent, {"denied": bool(denied), "output": output})
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill(proc)
    return (calls, timed_out)


def _write(parent: socket.socket, payload: dict) -> None:
    try:
        parent.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    except (OSError, ValueError):
        pass


def _kill(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def code(workspace: str, source: str, timeout: int = DEFAULT_TIMEOUT,
         call_tool=None, path_guards=None, policy=None) -> str:
    if call_tool is None:
        return "error: code mode is not available in this run"
    if not str(source).strip():
        return "error: the program is empty"
    reason = confinement_reason(path_guards, policy)
    if reason:
        missing = tool_sandbox.unavailable()
        if missing:
            return (f"error: code mode refused: {reason}, so the program must "
                    f"run in the sandbox, and the sandbox is unavailable "
                    f"({missing}). Use the tools directly.")
    outcome = execute(str(source), workspace, call_tool, timeout=timeout,
                      contained=bool(reason))
    if not reason:
        return outcome.render()
    return (f"note: {reason}, so the program ran in the sandbox: system Python "
            "and stdlib only, no network, and workspace files only through the "
            "one.* bindings.\n" + outcome.render())


__all__ = ["execute", "code", "Outcome", "BINDINGS", "DEFAULT_TIMEOUT"]

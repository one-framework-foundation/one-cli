"""Shell tool.

Runs in the workspace with a hard timeout. Output comes back whole up to
OUTPUT_CAP: bounding it for the context window is the spill policy's job in
`tools.run`, which can store what it withholds instead of dropping it. The cap
is only the memory backstop for a runaway command (`yes`): past it the process
group is killed and the result says so. It sits far above any context window,
so spill still sees, and keeps, everything a real command prints.

Two pieces of session state live here, because the shell is where they happen.
A `git stash push` that saved nothing ("No local changes to save") makes the
next `git stash pop` restore whatever stash was on top - another session's,
in a shared stash list, conflicts and all. The pop after an empty push is
refused. And a command that leaves a process running in its group
(`some-server &`) would otherwise keep it running after the session ends;
such groups are recorded and `reap()` ends them when the session closes. Long-lived background work belongs in
`job_start`, which the session already owns.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import selectors
import subprocess
import time

# Bytes of stdout+stderr kept in memory before the command is killed.
OUTPUT_CAP = 32 * 1024 * 1024

# Session state, kept across a hot reload of this module (the decisions below
# are what a reload changes, not what the session has seen).
# Workspaces whose last `git stash push` saved nothing.
_EMPTY_STASH: set[str] = globals().get("_EMPTY_STASH", set())
# Process groups a finished command left running, for `reap`: the command and
# one live member as "pid:start-time".
_LEFT_RUNNING: dict[int, tuple[str, str]] = globals().get("_LEFT_RUNNING", {})
# Any `git stash` that may create an entry: everything but the read and
# restore subcommands.
_STASH_PUSH = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?stash\b(?!\s+(?:pop|apply|list|show|drop|clear|branch)\b)")
_STASH_POP = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?stash\s+pop\b")


def _recursive_grep(command: str) -> bool:
    """Recognize direct recursive grep calls, including shell pipelines.

    This is search-tool guidance, not a shell security parser. Do not rewrite
    commands: grep and rg have different regex/ignore semantics.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False  # Let bash report malformed shell syntax.
    at_command = True
    in_grep = False
    pattern_arg = False
    for token in tokens:
        if token and all(c in ";&|()" for c in token):
            at_command, in_grep = True, False
            pattern_arg = False
            continue
        if at_command:
            if token in ("command", "exec") or ("=" in token and not token.startswith("/")):
                continue
            in_grep = token.rsplit("/", 1)[-1] == "grep"
            at_command = False
            continue
        if in_grep:
            if pattern_arg:
                pattern_arg = False
                continue
            if token in ("-e", "-f", "--regexp", "--file"):
                pattern_arg = True
            elif token == "--":
                in_grep = False
            elif token in ("--recursive", "--dereference-recursive"):
                return True
            elif token.startswith("-") and not token.startswith("--") and any(c in token[1:] for c in "rR"):
                return True
    return False


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _stash_top(workspace: str) -> str | None:
    """The top stash entry's SHA, "" for none, None outside a repository."""
    try:
        proc = subprocess.run(["git", "stash", "list", "-1", "--format=%H"], cwd=workspace,
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _member(pgid: int) -> str:
    """One live member of a process group as "pid:start-time", so a reused
    pgid is never mistaken for the group a command left behind. "" if none."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        # /proc is Linux-specific; completed commands still return on macOS.
        return ""
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 19 and fields[0] != "Z" and fields[2] == str(pgid):
            return f"{entry}:{fields[19]}"
    return ""


def _same(member: str) -> bool:
    pid, _, started = member.partition(":")
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19] == started
    except (OSError, IndexError):
        return False


def reap() -> list[str]:
    """End of session: stop the process groups finished commands left behind,
    and name them."""
    stopped = []
    for pgid, (command, member) in list(_LEFT_RUNNING.items()):
        if member and _same(member):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            stopped.append(f"pgid {pgid} ({command[:80]})")
        _LEFT_RUNNING.pop(pgid, None)
    return stopped


def _broad_search(command, workspace):
    """Guide direct shell discovery into scoped tools; not a security sandbox."""
    from pathlib import Path
    from onecli.tools import search
    # A newline ends a command like ';' does; a backslash-newline continues it.
    command = command.replace("\\\n", " ").replace("\n", ";")
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    discovery = False
    pattern_pending = False
    option_value = False
    at_command = True
    for token in tokens:
        if token and all(c in ";&|()" for c in token):
            discovery, at_command = False, True
            pattern_pending = option_value = False
        elif at_command:
            if token in ('command', 'exec') or '=' in token:
                continue
            program = token.rsplit('/', 1)[-1]
            discovery = program in {'ls', 'find', 'rg'}
            pattern_pending = program == 'rg'
            at_command = False
        elif discovery and option_value:
            option_value = False
        elif discovery and program == 'find' and token.startswith('-'):
            discovery = False  # Remaining operands are predicates, not roots.
        elif discovery and program == 'rg' and token in {'-g', '--glob', '--iglob', '-t', '--type', '-T', '--type-not', '-m', '--max-count', '-e', '--regexp', '-f', '--file'}:
            option_value = True
            if token in {'-e', '--regexp', '-f', '--file'}:
                pattern_pending = False
        elif discovery and program == 'rg' and token.startswith(('--regexp=', '--file=')):
            pattern_pending = False
        elif discovery and token == '--files':
            pattern_pending = False
        elif discovery and not token.startswith('-'):
            if pattern_pending:
                pattern_pending = False
                continue
            path = os.path.expanduser(os.path.expandvars(token))
            # Evaluate the search root before wildcard expansion.
            prefix = path.split('*', 1)[0].split('?', 1)[0].split('[', 1)[0]
            root = Path(prefix) if Path(prefix).is_absolute() else Path(workspace) / prefix
            if search.scope_error(workspace, root):
                return True
    return False


def bash(workspace: str, command: str, timeout: int = 120) -> str:
    if _STASH_POP.search(command) and _STASH_PUSH.search(command):
        return ("blocked: this command stashes and pops in one go; if the stash "
                "saves nothing, the pop restores a stash this session did not make. "
                "Run the stash alone, then pop only if it saved something.")
    if _STASH_POP.search(command) and workspace in _EMPTY_STASH:
        return ("blocked: the last `git stash push` in this workspace saved nothing "
                "(\"No local changes to save\"), so `git stash pop` would restore a "
                "stash this session did not make. Skip the pop; there is nothing of "
                "yours to restore.")
    if _broad_search(command, workspace):
        return "error: search scope too broad; use grep/glob with a specific project or docs path, then expand only after a miss."
    if _recursive_grep(command):
        return ("error: recursive shell grep is disabled for project searches. "
                "Use the grep tool with pattern, a narrow path and optional glob; "
                "it searches tracked repository contents. Use an explicit "
                "file path for untracked files or logs. Do not "
                "scan the whole home directory for a project question.")
    try:
        seconds = float(timeout)
        if seconds <= 0:
            raise ValueError("nonpositive timeout")
        timeout = max(1, int(seconds))
    except (TypeError, ValueError, OverflowError):
        return "error: timeout must be a finite positive number"
    stash_before = _stash_top(workspace) if _STASH_PUSH.search(command) else None
    proc = None
    completed = False
    capped = False
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    try:
        deadline = time.monotonic() + timeout
        proc = subprocess.Popen(
            ["bash", "-o", "pipefail", "-lc", command],
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        kept = 0
        assert proc.stdout is not None and proc.stderr is not None
        with selectors.DefaultSelector() as sel:
            sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
            sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
            while sel.get_map():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in sel.select(timeout=left):
                    data = os.read(key.fd, 65536)
                    if not data:
                        sel.unregister(key.fileobj)
                        continue
                    data = data[:OUTPUT_CAP - kept]
                    chunks[key.data].append(data)
                    kept += len(data)
                    if kept >= OUTPUT_CAP:
                        capped = True
                        break
                if capped:
                    break
        if not capped:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            completed = True
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s: {command}"
    except OSError as exc:
        return f"error: could not run command: {exc}"
    finally:
        if proc is not None:
            # A shell can exit while grandchildren still own its output pipes.
            # Reap the group even when the immediate shell already exited.
            if not completed:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Uninterruptible disk I/O cannot be reaped synchronously. SIGKILL
            # remains pending; keep the caller's own timeout bounded as well.
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            for pipe in (proc.stdout, proc.stderr):
                if pipe:
                    pipe.close()

    out = "".join(b"".join(chunks[k]).decode(errors="replace") for k in ("stdout", "stderr"))
    # Text-mode pipes used to fold CRLF and bare CR to LF; keep that output.
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    if stash_before is not None:
        # Read from the stash list, not the output: `-q` prints nothing.
        if _stash_top(workspace) == stash_before:
            _EMPTY_STASH.add(workspace)
        else:
            _EMPTY_STASH.discard(workspace)
    member = _member(proc.pid) if completed and proc is not None else ""
    if member and proc is not None:
        _LEFT_RUNNING[proc.pid] = (command, member)
        out += ("\n[shell] this command left a process running in the background "
                f"(process group {proc.pid}); it is stopped when the session ends. "
                "Use job_start for work that should outlive one call.")
    if capped:
        return (f"exit {proc.returncode}\n{out}\n... (output exceeded {OUTPUT_CAP} "
                f"bytes; the command was killed and the rest was not captured)")
    return f"exit {proc.returncode}\n{out}".rstrip()

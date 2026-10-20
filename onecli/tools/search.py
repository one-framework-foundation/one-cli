"""Project searches read tracked files; explicit files and non-repos use rg."""

from __future__ import annotations

from fnmatch import fnmatchcase
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path

from onecli import secureio
from onecli.tools import fs, rank

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
             "build", ".mypy_cache", ".pytest_cache", ".next", "target",
             "archive", "archives", "backups"}


# A `**` pattern walks the whole tree before it can know it found nothing, and
# a home directory full of checkouts is minutes of that. The walk is bounded so
# a broad pattern costs a bounded wait and a sentence telling the model to
# narrow it, rather than a hang the person has to Ctrl-C out of.
GLOB_DEADLINE = 20.0


GLOB_SHOWN = 300


def scope_error(workspace, root):
    """The home directory and the filesystem root are navigation targets,
    never implicit recursive searches. Other workspace ancestors are allowed."""
    resolved = root.resolve()
    if resolved.is_dir() and (resolved == Path.home().resolve()
                              or resolved == Path(resolved.anchor)):
        return ("error: search scope too broad; choose a specific project or docs "
                "directory. Start with the task's known path; expand only after "
                "a narrower search has no useful matches.")
    return ""


# Directory names pruned beside SKIP_DIRS: worktree copies and backups.
# Directories only - a file called wt-notes.md or a.bak.py is still searched.
SKIP_DIR_PATTERNS = ("wt-*", ".bak*", "*.bak.*")


def skipped(parts):
    """True when any of these directory names is pruned."""
    return any(part in SKIP_DIRS or part.startswith(('wt-', '.bak'))
               or '.bak.' in part for part in parts)


def _pruned_dir(entry: str, fd: int) -> bool:
    if not skipped((entry,)):
        return False
    try:
        return stat.S_ISDIR(os.stat(entry, dir_fd=fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def _bare_name(pattern: str) -> bool:
    return not set(pattern) & set("*?[/")


def _name_key(name: str, needle: str):
    """Exact file name, then prefix, then substring - like fzf's bonuses."""
    low = name.lower()
    return 0 if low == needle or low.rsplit(".", 1)[0] == needle else \
        1 if low.startswith(needle) else 2


def _glob_result(hits: dict[Path, float], pattern: str, root: Path, ran_out: bool,
                 needle: str = "", hints: bool = True) -> str:
    if not hits:
        if ran_out:
            return (f"error: no match for {pattern!r} in the first "
                    f"{GLOB_DEADLINE:.0f}s under {root} - narrow the pattern "
                    "or pass a deeper path")
        return f"(no files matching {pattern!r} under {root})" + (_missing_prefix(root, pattern) if hints else "")
    order = sorted(hits, key=lambda p: (_name_key(p.name, needle) if needle else 0,
                                        -hits[p], str(p)))
    lines = [str(p) for p in order[:GLOB_SHOWN]]
    if needle:
        lines.insert(0, f"(no wildcard: files whose name contains {needle!r}, "
                        "case-insensitive; exact names first, then newest)")
    if len(order) > GLOB_SHOWN:
        lines.append(f"({len(order) - GLOB_SHOWN} more files truncated; narrow the pattern)")
    if ran_out:
        lines.append(f"(stopped after {GLOB_DEADLINE:.0f}s - there may be more; "
                     "narrow the pattern or pass a deeper path)")
    return "\n".join(lines)


def _missing_prefix(root: Path, pattern: str) -> str:
    """For a path-shaped pattern whose literal part does not exist, where it
    stops existing and the closest names there (see `fs.missing`). A bare
    name is a search of the whole tree, so it gets no hint."""
    parts = Path(pattern).parts
    wild = any(c in pattern for c in "*?[")
    literal = []
    for part in parts if not wild else parts[:-1]:
        if any(c in part for c in "*?["):
            break
        literal.append(part)
    if _bare_name(pattern) or not literal or root.joinpath(*literal).exists():
        return ""
    hint = fs.missing(root.joinpath(*literal), "path").split("\n", 1)
    return "\n" + hint[1] if len(hint) > 1 else ""


def glob_files(workspace: str, pattern: str, path: str = ".") -> str:
    if not isinstance(pattern, str) or not pattern.strip():
        return "error: glob needs a non-empty pattern"
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        return "error: glob pattern must stay below its search path"
    root = Path(path) if Path(path).is_absolute() else Path(workspace) / path
    if not root.exists():
        return fs.missing(root, "path")
    if error := scope_error(workspace, root):
        return error
    # A bare word ("acme", "search.py") is a name search, not an exact
    # path: a model that guesses the name rarely guesses the directory.
    needle = pattern.lower() if _bare_name(pattern) else ""
    deadline = time.monotonic() + GLOB_DEADLINE
    hits: dict[Path, float] = {}
    ran_out = False
    rx = _glob_rx(pattern)
    try:
        for p in _walk(root, deadline):
            # Relative to the root, not absolute: the workspace itself may sit
            # under a directory called build or venv.
            relative = p.relative_to(root).as_posix() if p != root else p.name
            if not (needle in p.name.lower() if needle else rx.match(relative)):
                continue
            try:
                if p.is_file() and not p.is_symlink():
                    hits[p] = p.stat().st_mtime
            except OSError:
                continue
    except Deadline:
        ran_out = True
    return _glob_result(hits, pattern, root, ran_out, needle)


WALK_NOTE = ("(dirs named " + ", ".join(sorted(SKIP_DIRS) + list(SKIP_DIR_PATTERNS)) +
             " are not searched; pass one as path to search it)")
MAX_FILE = 4_000_000
SIZE_NOTE = "(binary files and files over 4 MB are not searched)"


def _python_rows(files, root: Path, p, reader):
    """Candidate rows from a pure-Python scan of the given files."""
    for path in files:
        lines = reader(path)
        if lines is None:
            continue
        relative = path.relative_to(root).as_posix() if path != root else path.name
        kept = 0
        for number, line in enumerate(lines, 1):
            if p.rx.search(line):
                yield path, relative, number, line
                kept += 1
                if kept >= rank.PER_FILE:
                    break


def _read_lines(path: Path) -> list[str] | None:
    """File text as lines split on newline only, as rg and git number them.
    None for unreadable, oversized or binary files."""
    try:
        if path.is_symlink() or path.stat().st_size > MAX_FILE:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    return _lines(data.decode("utf-8", "replace"))


def _lines(text: str) -> list[str] | None:
    """Split on newline only, as rg and git number lines; None for binary text
    (NUL in the first 8 KiB) or a file over MAX_FILE, which no backend reads."""
    if "\0" in text[:8192] or len(text) > MAX_FILE:
        return None
    return [line.rstrip("\r") for line in text.split("\n")]


def _glob_rx(pattern: str) -> re.Pattern:
    """pathlib-style glob as a regex over the posix path below the root:
    '**/' is zero or more directories, '*' and '?' never cross '/'. A final
    '**' selects directories, so it matches no file (as pathlib)."""
    if pattern == "**" or pattern.endswith("/**"):
        return re.compile(r"(?!)")
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and "]" in pattern[i + 2:]:
            end = pattern.index("]", i + 2)
            body = pattern[i + 1:end]
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


class Deadline(Exception):
    pass


def _walk(root: Path, deadline: float | None = None):
    """Files below root, sorted, SKIP_DIRS pruned. With a deadline the walk
    raises Deadline between directories, even when nothing has matched, so a
    broad pattern cannot hold a call for minutes."""
    if root.is_file():
        yield root
        return
    for top, dirs, names in os.walk(root):
        if deadline is not None and time.monotonic() > deadline:
            raise Deadline
        dirs[:] = sorted(d for d in dirs if not skipped((d,)))
        for name in sorted(names):
            yield Path(top) / name


def _mtimes(files) -> None:
    for h in files:
        try:
            h.mtime = h.path.stat().st_mtime
        except OSError:
            h.mtime = 0.0


def _rg_rows(rg: str, root: Path, p, glob: str, timeout: int):
    cmd = [rg, "--line-number", "--no-heading", "--with-filename", "--null",
           "--color", "never", "--hidden", "--no-ignore", "--no-messages",
           "--max-count", str(rank.PER_FILE), "--max-filesize", str(MAX_FILE),
           "--sort", "path"]
    # A trailing slash makes each glob match directories only, as _walk prunes.
    cmd += [f"--glob=!{d}/" for d in sorted(SKIP_DIRS) + list(SKIP_DIR_PATTERNS)]
    if p.ignore_case:
        cmd.append("--ignore-case")
    # rg anchors a glob containing '/' and never lets '*' cross it; the shared
    # rule does both, so such a glob is applied after the search instead.
    if glob and "/" not in glob:
        cmd += ["--glob", glob.removeprefix("**/")]
    cmd += ["-e", p.backend, "--", str(root)]
    # Streamed and stopped one row past the cap, so memory stays bounded and
    # the cap is still announced.
    rows = []
    deadline = time.monotonic() + timeout
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding="utf-8", errors="replace") as proc:
        assert proc.stdout is not None and proc.stderr is not None
        # The deadline is enforced by a timer, not only between lines: a search
        # that finds nothing prints nothing, and a silent rg over a whole home
        # directory can run for many minutes.
        watchdog = threading.Timer(timeout, proc.kill)
        watchdog.start()
        try:
            for line in proc.stdout:
                if time.monotonic() > deadline:
                    proc.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout)
                rows.append(line.rstrip("\n"))
                if len(rows) > rank.CANDIDATES:
                    proc.kill()
                    break
            else:
                proc.wait(timeout=max(1.0, deadline - time.monotonic()))
                if time.monotonic() >= deadline and proc.returncode not in (0, 1):
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if proc.returncode not in (0, 1):
                    raise RuntimeError(f"rg failed: {proc.stderr.read().strip()[:400]}")
        finally:
            watchdog.cancel()
    parsed = []
    for line in rows:
        name, sep, rest = line.partition("\0")
        number, sep2, text = rest.partition(":")
        if not (sep and sep2 and number.isdigit()):
            continue
        path = Path(name)
        relative = path.relative_to(root).as_posix() if path != root else path.name
        parsed.append((path, relative, int(number), text))
    return parsed


# An installed extension may register an index backend for tracked files:
# INDEXES[name] is a resolver returning rows(repo_root, relative, root, plan,
# glob, timeout) when its tool is available, else a short reason it is not.
INDEXES: dict = {}


def grep(workspace: str, pattern: str, path: str = ".", glob: str = "",
         max_results: int = 100, context: int = 1) -> str:
    return search(workspace, pattern, path, glob, max_results, context)


def search(workspace: str, pattern: str, path: str = ".", glob: str = "",
           max_results: int = 100, context: int = 1, backend: str = "",
           timeout: int = 60) -> str:
    """grep's body; `backend` forces a registered index, rg or python (parity tests)."""
    root = Path(path) if Path(path).is_absolute() else Path(workspace) / path
    if not root.exists():
        return fs.missing(root, "path")
    if error := scope_error(workspace, root):
        return error
    p = rank.plan(pattern)
    if isinstance(p, str):
        return p
    if Path(glob).is_absolute() or ".." in Path(glob).parts:
        return "error: glob pattern must stay below its search path"
    max_results = min(rank.MAX_RESULTS, max(1, int(max_results or 100)))
    context = min(5, max(0, int(1 if context is None else context)))
    base = root if root.is_dir() else root.parent
    tracked: list[Path] | None = None
    repo_root: Path | None = None
    relative = ""
    if root.is_dir():
        try:
            repo = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"error: cannot determine search repository: {exc}"
        if repo.returncode == 0:
            repo_root = Path(repo.stdout.strip()).resolve()
            base = root = root.resolve()
            relative = root.relative_to(repo_root).as_posix()
            try:
                listing = subprocess.run(
                    ["git", "-C", str(repo_root), "ls-files", "-z", "--", relative],
                    capture_output=True, text=True, timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return f"error: cannot list tracked files: {exc}"
            if listing.returncode:
                return f"error: cannot list tracked files: {listing.stderr.strip()[:400]}"
            # No tracked files under this root (e.g. a gitignored subtree of an
            # enclosing repo): there is nothing tracked to search here, so the
            # search falls through to the same rg/python walk as a non-repo.
            names = [n for n in listing.stdout.split("\0") if n]
            tracked = sorted(repo_root / n for n in names
                             if not skipped((repo_root / n).relative_to(root).parts[:-1])
                             ) if names else None

    notes: list[str] = []
    try:
        if tracked is not None and repo_root is not None:
            rows, label, reason = None, "python", ""
            for name, resolve in INDEXES.items():
                if backend not in ("", name):
                    continue
                found = resolve()
                if callable(found):
                    label = name
                    rows = found(repo_root, relative, root, p, glob, timeout)
                    break
                reason = reason or found
            if rows is None:
                # Without an index the tracked files are read directly, and
                # the header says which backend answered and why.
                label = f"python: {reason}" if reason and not backend else "python"
                rows = _python_rows(tracked, root, p, _read_lines)
            notes.append("(tracked files only; untracked files are not searched - "
                         "pass an explicit file path to search one)")
            notes.append(WALK_NOTE)
        else:
            rg = shutil.which("rg") if backend in ("", "rg") else None
            if rg:
                label = "rg"
                rows = _rg_rows(rg, root, p, glob, timeout)
            else:
                label = "python: rg not on PATH" if not backend else "python"
                rows = _python_rows(_walk(root), base, p, _read_lines)
            if root.is_dir():
                notes.append(WALK_NOTE)
    except subprocess.TimeoutExpired:
        return f"error: grep timed out after {timeout}s"
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return f"error: search failed: {exc}"
    notes.append(SIZE_NOTE)
    try:
        files, capped = rank.collect(rows, p, glob)
    except OSError as exc:
        return f"error: search failed: {exc}"
    _mtimes(files)
    return rank.render(files, p, label, max_results, context, _read_lines, capped, notes)


def _restricted_glob(root: Path, pattern: str, arguments: dict, guards: list) -> str:
    """Root-anchored pathlib-style matching, walking only pattern-selected dirs."""
    needle = pattern.lower() if _bare_name(pattern) else ""
    parts = ("**", "*") if needle else Path(pattern).parts
    if not parts:
        return "error: empty glob pattern"
    if any("**" in part and part != "**" for part in parts):
        return "error: '**' must be an entire glob path component"
    # A trailing separator requests directories; this tool returns only files.
    if pattern.endswith("/"):
        return f"(no files matching {pattern!r} under {root})"
    deadline = time.monotonic() + GLOB_DEADLINE
    pending, seen = [(root, parts)], set()
    hits: dict[Path, float] = {}
    ran_out = False
    while pending:
        if time.monotonic() > deadline:
            ran_out = True
            break
        path, remaining = pending.pop()
        state = (path, remaining)
        if state in seen:
            continue
        seen.add(state)
        if any(not guard.check("glob", {**arguments, "path": str(path)}, False).allowed
               for guard in guards):
            continue
        try:
            if not remaining:
                with secureio.text_file(path) as fh:
                    hits[path] = os.fstat(fh.fileno()).st_mtime
                continue
            with secureio.directory(path) as fd:
                segment, rest = remaining[0], remaining[1:]
                if segment == "**":
                    # Zero directories, or one more real directory; never a
                    # suffix match of a nonrecursive pattern at arbitrary depth.
                    pending.append((path, rest))
                for entry in sorted(os.listdir(fd), reverse=True):
                    if _pruned_dir(entry, fd):
                        continue
                    if segment == "**":
                        if stat.S_ISDIR(os.stat(entry, dir_fd=fd, follow_symlinks=False).st_mode):
                            pending.append((path / entry, remaining))
                    elif (needle in entry.lower() if needle and not rest
                          else fnmatchcase(entry, segment)):
                        pending.append((path / entry, rest))
        except OSError:
            continue
    return _glob_result(hits, pattern, root, ran_out, needle, hints=False)


def _bounded_regex(pattern: str) -> str:
    """Reject expensive backtracking structures in the in-process backend.

    One repetition over a fixed-width atom is permitted. Repeated branches,
    nesting, backreferences and assertions require a subprocess backend; they
    can backtrack exponentially (or quadratically with adjacent repeats).
    Parse tokens rather than scanning text so escapes/classes cannot hide them.
    """
    from importlib import import_module
    try:
        _parser = import_module("re._parser")
        c = import_module("re._constants")
    except ImportError:  # Python 3.10 exposes these under their older names.
        _parser = import_module("sre_parse")
        c = import_module("sre_constants")
    repeat_ops = {c.MAX_REPEAT, c.MIN_REPEAT}
    possessive = getattr(c, "POSSESSIVE_REPEAT", None)
    if possessive is not None:
        repeat_ops.add(possessive)
    atomic = getattr(c, "ATOMIC_GROUP", None)
    repeats = 0

    def visit(tokens, repeated=False):
        nonlocal repeats
        for op, arg in tokens:
            if op in repeat_ops:
                repeats += 1
                if repeated or repeats > 1:
                    raise ValueError
                visit(arg[2], True)
            elif op == c.SUBPATTERN:
                visit(arg[3], repeated)
            elif op == c.BRANCH:
                if repeated:
                    raise ValueError
                for branch in arg[1]:
                    visit(branch, repeated)
            elif op in (c.GROUPREF, c.GROUPREF_EXISTS, c.ASSERT, c.ASSERT_NOT):
                raise ValueError
            elif atomic is not None and op == atomic:
                visit(arg, repeated)
    try:
        visit(_parser.parse(pattern))
    except (ValueError, re.error, RecursionError):
        return "error: regex exceeds the bounded search subset; simplify repeats or use a literal"
    return ""


def restricted(name: str, workspace: str, arguments: dict, guards: list) -> str:
    """Search without subprocesses or symlink traversal under a path policy.

    Checking only the search root cannot authorize descendant file contents.
    Each candidate is checked and opened with no-follow descriptor traversal.
    This bounded fallback uses Python regex/glob syntax, not an index.
    """
    # Preserve '..' until no-follow traversal has inspected each ancestor.
    root = Path(workspace) / arguments.get("path", ".")
    if not root.is_absolute():
        root = Path.cwd() / root
    if error := scope_error(workspace, root):
        return error
    if all(guard.check(name, {**arguments, "path": str(root)}, False).allowed for guard in guards):
        # A root the no-follow walk cannot open (a symlinked component, a
        # missing path) is an error, never a confident empty result.
        try:
            with secureio.directory(root):
                pass
        except NotADirectoryError:
            try:
                with secureio.text_file(root):
                    pass
            except OSError as exc:
                return f"error: cannot open search path {root}: {exc}"
        except OSError as exc:
            return f"error: cannot open search path {root}: {exc}"
    pattern = arguments.get("pattern", "")
    glob = pattern if name == "glob" else arguments.get("glob", "")
    if Path(glob).is_absolute() or ".." in Path(glob).parts:
        return "error: glob pattern must stay below its search path"
    if name == "glob":
        return _restricted_glob(root, pattern, arguments, guards)
    if error := _bounded_regex(pattern):
        return error
    p = rank.plan(pattern)
    if isinstance(p, str):
        return p
    limit = min(rank.MAX_RESULTS, max(1, int(arguments.get("max_results") or 100)))
    context = arguments.get("context")
    context = min(5, max(0, int(1 if context is None else context)))

    def allowed(path: Path) -> bool:
        return all(guard.check(name, {**arguments, "path": str(path)}, False).allowed
                   for guard in guards)

    def read(path: Path) -> list[str] | None:
        if not allowed(path):
            return None
        try:
            with secureio.text_file(path, errors="replace", newline="") as fh:
                return _lines(fh.read(MAX_FILE + 1))
        except OSError:
            return None

    mtimes: dict[Path, float] = {}
    notes: list[str] = []
    refused: list[Path] = []
    denied: list[Path] = []
    deadline = time.monotonic() + GLOB_DEADLINE

    def files():
        pending = [root]
        while pending:
            if time.monotonic() > deadline:
                notes.append("(search deadline reached; results cover part of the "
                             "tree - narrow the path)")
                return
            path = pending.pop()
            if not allowed(path):
                # Counted, never named: a denied Read path's names are as
                # private as its contents (glob and ls hide them too).
                denied.append(path)
                continue
            try:
                with secureio.directory(path) as fd:
                    pending.extend(path / entry for entry in sorted(os.listdir(fd), reverse=True)
                                   if not _pruned_dir(entry, fd))
                continue
            except NotADirectoryError:
                pass
            except OSError:
                refused.append(path)
                continue
            yield path

    def scan(path: Path) -> list[str] | None:
        try:
            with secureio.text_file(path, errors="replace", newline="") as fh:
                mtimes[path] = os.fstat(fh.fileno()).st_mtime
                return _lines(fh.read(MAX_FILE + 1))
        except OSError:
            return None

    base = root if not root.is_file() else root.parent
    hits, capped = rank.collect(_python_rows(files(), base, p, scan), p, glob)
    for h in hits:
        h.mtime = mtimes.get(h.path, 0.0)
    if denied:
        notes.append(f"({len(denied)} paths refused by policy; not searched)")
    if refused:
        notes.append(f"({len(refused)} paths unreadable, e.g. {refused[0]}; "
                     "not searched)")
    return rank.render(hits, p, "python: guarded workspace", limit, context, read,
                       capped, notes + [WALK_NOTE, SIZE_NOTE])



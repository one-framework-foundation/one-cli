"""File tools. Every path is resolved against the workspace root."""

from __future__ import annotations

import difflib
import os
import stat
from pathlib import Path

from onecli import secureio, execution_evidence


def resolve(workspace: str, path: str) -> Path:
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = Path(workspace) / p
    return p


# Names compared for a not-found hint, and file lines for a missed edit's
# hint. Past these the hint narrows or is skipped: difflib is linear in them.
MISSING_SCAN = 5000
NEAREST_SCAN = 20000


def missing(p: Path, what: str = "file", path_guards=None) -> str:
    """The not-found error, with where the path stops existing and the closest
    names there - so the next call is a correction, not another guess.

    A model that guessed one path wrong tends to guess again; the listing it
    would ask for next is cheaper to give here than to wait for. Names a path
    policy hides from ls are left out here too (see `listable`).
    """
    parent, part = p.parent, p.name
    while not parent.exists() and parent != parent.parent:
        parent, part = parent.parent, parent.name
    note = f"error: no such {what}: {p}"
    if not parent.is_dir():
        return note
    try:
        with secureio.directory(parent) as fd:
            names = os.listdir(fd)
    except OSError:
        return note
    if len(names) > MISSING_SCAN:
        # A huge directory: compare only names that start like the guess.
        names = [n for n in names if n[:2].lower() == part[:2].lower()][:MISSING_SCAN]
    names = [n for n in names if listable(parent / n, path_guards)]
    close = difflib.get_close_matches(part, names, n=4, cutoff=0.5)
    note += f"\n{parent} exists" + ("" if parent == p.parent else f"; {part} does not")
    if close:
        note += ". Closest names there: " + ", ".join(close)
    return note


def listable(path: Path, path_guards=None) -> bool:
    """Whether ls may name this entry: `Read` governs ls, so a name the run's
    path guards deny is as private as its contents."""
    return all(guard.check("ls", {"path": str(path)}, False).allowed
               for guard in path_guards or ())


# Characters of file text a read keeps in memory, whatever offset/limit/room
# ask for. Far above any context window, so `room` is still the bound that
# normally applies; this only stops a multi-GB file (or line) from being loaded.
READ_CAP = 32 * 1024 * 1024
EDIT_CAP = READ_CAP
_CHUNK = 1024 * 1024


def _lines(fh):
    """Yield each line without its newline, keeping at most READ_CAP + 1
    characters of a line however long it is, plus whether it was cut."""
    while True:
        piece = fh.readline(_CHUNK)
        if not piece:
            return
        parts, kept, cut = [], 0, False
        while True:
            if kept < READ_CAP + 1:
                parts.append(piece[:READ_CAP + 1 - kept])
                kept += len(parts[-1])
            else:
                cut = True
            if piece.endswith("\n"):
                break
            piece = fh.readline(_CHUNK)
            if not piece:
                break
        line = "".join(parts)
        if line.endswith("\n"):
            line = line[:-1]
        yield line, cut or len(line) > READ_CAP


def read(workspace: str, path: str, offset: int = 1, limit: int | None = None,
         room: int | None = None, path_guards=None) -> str:
    """The file from `offset`, whole unless `limit` lines are asked for.

    `room` is how many characters the caller's context window has left for
    this result. It is the only bound: a read that does not fit stops at the
    last whole line that does and says exactly where to continue. There is no
    fixed budget - a model that reads 140 lines of a 50 KB skill and never
    asks for the rest has been failed by the tool, not by itself. READ_CAP is a
    memory backstop, not a budget: lines are read
    one at a time and no more than READ_CAP characters are ever held.
    """
    p = resolve(workspace, path)
    if not p.exists():
        return missing(p, path_guards=path_guards)
    if p.is_dir():
        return f"error: {p} is a directory, use ls"
    start = max(1, int(offset))
    want = None if limit is None else start + max(1, int(limit)) - 1
    out: list[str] = []
    used, total, stop, note = 0, 0, None, ""
    with secureio.text_file(p, errors="replace") as fh:
        size = os.fstat(fh.fileno()).st_size
        for i, (line, cut) in enumerate(_lines(fh), 1):
            total = i
            if stop is not None or i < start or (want is not None and i > want):
                continue  # counted for the continuation note, never kept
            if cut:
                line = line[:READ_CAP]
            row = f"{i:6d}\t{line}"
            if room is not None and out and used + len(row) + 1 > room:
                stop, note = i, "the rest does not fit the context window now"
                continue
            if out and used + len(row) + 1 > READ_CAP:
                stop, note = i, f"the read cap of {READ_CAP} characters was reached"
                continue
            if cut:
                row = row[:READ_CAP] + f" ... (line {i} cut at {READ_CAP} characters)"
            out.append(row)
            used += len(row) + 1
    if not total:
        return f"({p} is empty)"
    if stop is not None:
        return ("\n".join(out) + f"\n... (stopped at line {stop - 1} of {total}, "
                f"{size} bytes in all: {note}; continue with offset={stop})")
    end = total if want is None else min(total, want)
    tail = "" if end == total else (f"\n... ({total - end} more lines; "
                                    f"continue with offset={end + 1})")
    return "\n".join(out) + tail


def write(workspace: str, path: str, content: str) -> str:
    p = resolve(workspace, path)
    existed = p.exists()
    # Checked before writing so the error can say the file is unchanged; the
    # replace below would refuse it too, leaving the old file in place.
    try:
        content.encode("utf-8")
    except UnicodeEncodeError as exc:
        return f"error: content is not valid UTF-8 text ({exc.reason}); {p} unchanged"
    execution_evidence.replace_text(p, content)
    verb = "overwrote" if existed else "created"
    return f"{verb} {p} ({len(content)} bytes, {content.count(chr(10)) + 1} lines)"


def edit(workspace: str, path: str, old_string: str, new_string: str,
         replace_all: bool = False, path_guards=None) -> str:
    p = resolve(workspace, path)
    if not p.exists():
        return missing(p, path_guards=path_guards)
    try:
        old_string.encode("utf-8")
        new_string.encode("utf-8")
    except UnicodeEncodeError as exc:
        return f"error: edit text is not valid UTF-8 ({exc.reason}); {p} unchanged"
    if not old_string:
        # "" occurs between every character: counting it would report len+1
        # replacements and splice new_string through the whole file.
        return f"error: old_string is empty; {p} unchanged. Use write to create or replace a file."
    # surrogateescape and newline="" round-trip what is not touched: CRLF line
    # endings and non-UTF-8 bytes are written back exactly as they were read.
    with secureio.text_file(p, "r", errors="surrogateescape", newline="") as fh:
        if os.fstat(fh.fileno()).st_size > EDIT_CAP:
            return f"error: edit refuses {p}: exceeds {EDIT_CAP} byte cap; file unchanged"
        text = fh.read(EDIT_CAP + 1)
        if len(text) > EDIT_CAP:
            return f"error: edit refuses {p}: exceeds {EDIT_CAP} character cap; file unchanged"
        if (old_string not in text and "\r\n" in text
                and "\r\n" not in old_string and "\n" in old_string):
            # The model writes LF; match and keep a CRLF file's own endings.
            old_string = old_string.replace("\n", "\r\n")
            new_string = new_string.replace("\r\n", "\n").replace("\n", "\r\n")
        count = text.count(old_string)
        if count == 0:
            return (f"error: old_string not found in {p}. It must match the file "
                    f"byte for byte, including indentation." + _nearest(text, old_string))
        if count > 1 and not replace_all:
            return (f"error: old_string appears {count} times in {p}. Include more "
                    f"surrounding context to make it unique, or pass replace_all.")
    # Written beside the file and renamed over it: an edit that fails part way
    # (a file-size limit, a full disk) leaves the original whole.
    execution_evidence.replace_text(p, text.replace(old_string, new_string),
                          errors="surrogateescape", newline="")
    return f"edited {p} ({count} replacement{'s' if count > 1 else ''})"


def _nearest(text: str, old: str) -> str:
    """The file lines most like old_string's first line, numbered, so a
    failed edit can be retried without re-reading the file. A transcript scan
    found the same miss retried blind up to 7 times in a run."""
    first = next((ln.strip() for ln in old.splitlines() if ln.strip()), "")
    if not first:
        return ""
    lines = text.splitlines()
    if len(lines) > NEAREST_SCAN:
        return " The file is too long to suggest lines; grep for its first line."
    stripped = [ln.strip() for ln in lines]
    best = difflib.get_close_matches(first, stripped, n=3, cutoff=0.6)
    if not best:
        return " No line resembles its first line; read the file again."
    rows = []
    for want in best:
        i = stripped.index(want)
        rows.append(f"{i + 1:6d}\t{lines[i][:200]}")
    return ("\nClosest lines to its first line (copy them exactly, "
            "whitespace included):\n" + "\n".join(rows))


def ls(workspace: str, path: str = ".", path_guards=None) -> str:
    """List a directory. An entry a path policy refuses is left out, as glob
    and grep leave it out: `Read` governs ls, and a denied name is not listed."""
    p = resolve(workspace, path)
    if not p.exists():
        return missing(p, "path", path_guards)
    if p.is_file():
        with secureio.text_file(p) as fh:
            return f"{p} (file, {os.fstat(fh.fileno()).st_size} bytes)"
    with secureio.directory(p) as fd:
        entries = [(name, stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode))
                   for name in os.listdir(fd)]
    entries = [e for e in entries if listable(p / e[0], path_guards)]
    entries.sort(key=lambda e: (not e[1], e[0].lower()))
    if not entries:
        return f"({p} is empty)"
    return "\n".join(name + ("/" if is_dir else "") for name, is_dir in entries[:500])

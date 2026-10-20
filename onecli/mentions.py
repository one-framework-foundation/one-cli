"""`@path` in a prompt - the file arrives with the question.

Without this, naming a file costs a round trip: the model reads the mention,
calls `read`, waits, and only then starts thinking. That is one extra step, one
extra latency and one extra chance to read the wrong path - for a file the
person at the keyboard was already looking at.

An `@` mention is resolved before the turn starts, against the workspace,
because the workspace is what every tool in the loop resolves against. The
contents ride in front of the prompt in a fenced block naming the path, so the
model can quote a line number and the reader can see exactly what was sent.

What it refuses to do matters as much:

- **It never reaches outside the workspace.** `@../../etc/shadow` resolves, then
  fails the containment check and is left in the prompt as plain text.
- **It never silently truncates.** A file past the ceiling is sent head-first
  with a line saying how much was cut, so the model knows it is holding part of
  something rather than all of it.
- **A mention that does not resolve is left alone.** `@here` in prose, an email
  address, a decorator - none of those are files, and rewriting them would be
  the tool editing the user's sentence.
"""

from __future__ import annotations

import os
import re

# `@` followed by a path-ish run. A trailing `.` or `,` is punctuation in the
# sentence, not part of the name - a mention at the end of a clause is the
# common case, not the exception.
MENTION = re.compile(r"(?<![\w@/])@([~\w./\-]*[\w/\-])")

MAX_BYTES = 96_000          # per file, before the head/tail cut
MAX_TOTAL = 240_000         # per turn, across every mention

# Below this much room left, a file is not sent at all rather than sent as a
# sliver. Two hundred bytes of a source file is not a smaller version of that
# file - it is a fragment that reads as the whole thing, which is the one
# outcome worse than saying it did not fit.
MIN_SLICE = 2_000

# How much of a file is sniffed for a NUL before deciding it is not text. A NUL
# byte survives `decode(errors="replace")` untouched - it is valid UTF-8 - so a
# binary file used to arrive on the wire as tens of kilobytes of control
# characters wrapped in a fence. Some gateways reject a body containing one
# outright; the ones that do not are handed noise in place of the turn.
SNIFF_BYTES = 8_192


def find(text: str) -> list[str]:
    """Every `@path` in `text`, in the order they were typed, deduplicated."""
    seen, out = set(), []
    for match in MENTION.finditer(text):
        path = match.group(1)
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def resolve(path: str, workspace: str) -> str | None:
    """The absolute path a mention names, or None if it is not one to send.

    None is the answer for anything outside the workspace, anything that is not
    a regular file, and anything unreadable - and the caller leaves the text
    untouched rather than substituting an error the model would try to act on.
    """
    raw = os.path.expanduser(path)
    full = os.path.abspath(raw if os.path.isabs(raw)
                           else os.path.join(workspace, raw))
    root = os.path.abspath(workspace)
    # Containment is checked on the resolved real path, so a symlink out of the
    # tree is refused the same as `../` is.
    real = os.path.realpath(full)
    if real != os.path.realpath(root) and not real.startswith(
            os.path.realpath(root) + os.sep):
        return None
    return full if os.path.isfile(full) else None


def looks_binary(blob: bytes) -> bool:
    """A NUL in the first few KB is the oldest and most reliable tell."""
    return b"\x00" in blob[:SNIFF_BYTES]


def read(full: str, budget: int = MAX_BYTES) -> tuple[str, str]:
    """(text, note). `note` is empty unless the file was cut.

    Raises `IsADirectoryError`-shaped refusal for binary content by returning
    ("", note) - the caller turns that into a line saying it was not sent.
    """
    with open(full, "rb") as fh:
        blob = fh.read(budget + 1)
    if looks_binary(blob):
        return "", (f"[not sent: {os.path.basename(full)} looks like a binary "
                    f"file ({os.path.getsize(full):,} bytes). Name it to a tool "
                    "if you need its contents.]")
    if len(blob) <= budget:
        return blob.decode("utf-8", "replace"), ""
    size = os.path.getsize(full)
    head = blob[:budget].decode("utf-8", "replace")
    return head, (f"[cut: this is the first {budget:,} bytes of "
                  f"{size:,}. Read the file for the rest.]")


def missing(text: str, workspace: str) -> list[str]:
    """File-looking mentions (`@a.py`, `@dir/b`) that name nothing sendable.

    A bare `@here` in prose is not a file and is not reported; one with a dot
    or a slash was almost certainly meant as one.
    """
    return [path for path in find(text)
            if ('.' in path or '/' in path) and resolve(path, workspace) is None]


def denied(full: str, path_guards) -> bool:
    """Whether the run's path rules refuse a `read` of `full`.

    A mention is a read the operator asked for, but it goes to the same
    provider a read tool result would, so `Read(.env)` on the disallowed list
    has to stop it here too.
    """
    return any(not guard.check("read", {"path": full}, False).allowed
               for guard in path_guards or ())


def expand(text: str, workspace: str, total_budget: int = MAX_TOTAL,
           path_guards=None) -> tuple[str, list[str]]:
    """Return (prompt with the mentioned files in front, paths actually sent).

    The mentions stay in the sentence exactly as typed. The files are prepended
    rather than substituted inline, because a question with three files spliced
    into the middle of it stops being a readable question. A file the
    `path_guards` refuse is left as plain text, the same as one outside the
    workspace.
    """
    paths = find(text)
    if not paths:
        return text, []

    blocks, sent, spent = [], [], 0
    for path in paths:
        full = resolve(path, workspace)
        if full is None or denied(full, path_guards):
            continue
        room = total_budget - spent
        if room < MIN_SLICE:
            blocks.append(f"[not sent: {path} - the turn's file budget is "
                          f"{total_budget:,} bytes. Read it if you need it.]")
            continue
        try:
            body, note = read(full, min(MAX_BYTES, room))
        except OSError:
            continue
        if not body:
            # Binary, or empty after the sniff. The note says which, and the
            # mention stays in the sentence as typed.
            blocks.append(note or f"[not sent: {path} is empty.]")
            continue
        spent += len(body.encode("utf-8", "replace"))
        sent.append(path)
        fence = "```"
        # A file that contains a fence of its own would end the block early.
        while fence in body:
            fence += "`"
        blocks.append(f"{path}\n{fence}\n{body}\n{fence}"
                      + (f"\n{note}" if note else ""))

    if not blocks:
        return text, []
    head = ("Files the user named with @ in the message below. They are "
            "included so you do not have to read them:\n\n"
            + "\n\n".join(blocks))
    return f"{head}\n\n{text}", sent


__all__ = ["find", "missing", "resolve", "read", "expand", "denied", "looks_binary", "MENTION",
           "MAX_BYTES", "MAX_TOTAL", "MIN_SLICE", "SNIFF_BYTES"]

"""Spill - big tool output goes to a file, not into the bin.

The old rule was: bound every tool's output and drop the middle with an honest
marker. Bounding is right - one `cat` of a build log can end a session - but
dropping is not, and the marker is less honest than it looks. The model reads
"[420000 bytes of output dropped]" and has no way to get those bytes. It cannot
grep them, cannot page through them, cannot tell whether the answer it needed
was in the part that went missing. So it guesses, or it re-runs the command and
loses the same middle again.

Spill keeps the bound and removes the loss. The whole output is written to a
file under the session's own directory, the model gets head + tail plus the
path, and the path is readable with the `read` tool it already has - offset,
limit, grep, whatever it needs. Storage, retention and policy live in one
file, because one-cli has one storage backend and no plugin seam to justify
splitting them.

The preview keeps both ends on purpose: the head carries what a command was
doing, the tail carries how it ended, and an error almost always lives in one of
those two.
"""

from __future__ import annotations

import re
from pathlib import Path

from onecli import secureio

SPILL_DIR = Path.home() / ".one-cli" / "spills"

# What the model sees of a spilled result, split head/tail. A result is only
# spilled when it does not fit the room the caller has left in the model's
# context window; there is no fixed size threshold, because a window that grows
# should be able to take a bigger result whole.
PREVIEW = 8_000


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", name)[:40] or "tool"


class SpillStore:
    """Session-scoped storage for oversized tool results."""

    def __init__(self, session_id: str, root: Path | str = SPILL_DIR):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("invalid session id")
        self.session_id = session_id
        self.root = Path(root) / session_id
        self.count = 0

    def path_for(self, tool: str) -> Path:
        self.count += 1
        return self.root / f"{self.count:03d}-{_slug(tool)}.txt"

    def spill(self, tool: str, text: str, room: int) -> str:
        """Store `text`, return what the model should read instead.

        A store that cannot write is not an error the run should die on: the
        caller gets ordinary truncation back, which is exactly where we were
        before this module existed.
        """
        try:
            while True:
                path = self.path_for(tool)
                try:
                    with secureio.text_file(path, "x", private=True) as fh:
                        fh.write(text)
                    break
                except FileExistsError:
                    continue  # Resume after retained output without replacing it.
        except OSError as exc:
            return truncate(text, room, f"could not spill ({exc})")
        return _preview(text, str(path), min(PREVIEW, room))


def _preview(text: str, path: str, size: int) -> str:
    head = text[: size // 2]
    tail = text[-(size // 2):]
    withheld = len(text) - len(head) - len(tail)
    lines = text.count("\n") + 1
    return (f"{head}\n"
            f"... [{withheld} of {len(text)} bytes not shown - the FULL output "
            f"({lines} lines) is at {path}; read it with the read tool "
            f"in pages (offset/limit, e.g. limit=200), or grep it] ...\n"
            f"{tail}")


def truncate(text: str, room: int, why: str = "") -> str:
    """The fallback when there is nowhere to spill to."""
    if len(text) <= room:
        return text
    size = min(PREVIEW, room)
    head = text[: size // 2]
    tail = text[-(size // 2):]
    dropped = len(text) - len(head) - len(tail)
    note = f"{dropped} bytes of output dropped"
    if why:
        note += f" - {why}"
    return f"{head}\n... [{note}] ...\n{tail}"


def apply(text: str, tool: str, store: SpillStore | None,
          room: int | None = None) -> str:
    """The post-execution policy: a result that fits `room` characters is
    returned whole; a larger one is spilled, losing none of it if we can.
    `room` None means the caller knows of no bound, so nothing is cut."""
    if room is None or len(text) <= room:
        return text
    if store is None:
        return truncate(text, room)
    return store.spill(tool, text, room)

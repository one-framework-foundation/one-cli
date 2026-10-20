"""The measure rung: one JSONL row per routing decision.

Every decision the router makes, including the ones that refused, appends
here, so a change to routing can be measured against what it replaced.

Two properties are load-bearing:

- It never raises into the caller. A full disk must not turn a working
  transcription into a failed turn, so every write is wrapped.
- It is per-machine by construction. The file lives under the user's home, and
  the ladder it records is that machine's ladder - Deepgram present here says
  nothing about another machine.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from onecli.secureio import text_file

LOG_PATH = Path(os.environ.get(
    "ONE_SENSES_LOG",
    str(Path.home() / ".one-cli" / "senses_log.jsonl"),
))


def record(outcome, *, member: str, path: str = "", ms: int = 0,
           extra: dict | None = None) -> dict:
    """Append one row and return it, so a caller can assert on what was logged."""
    row = {
        "ts": time.time(),
        "member": member,
        "modality": getattr(outcome, "modality", ""),
        "route": getattr(outcome, "route", ""),
        "ok": bool(getattr(outcome, "ok", False)),
        "path": path,
        "ms": ms,
        "language": getattr(outcome, "language", ""),
        "rungs": [r.as_row() for r in getattr(outcome, "rungs", [])],
    }
    if extra:
        row.update(extra)
    try:
        with text_file(LOG_PATH, "a", private=True) as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except OSError:
        # Best effort by contract: the perception already happened, and losing
        # its metric row is strictly better than losing the turn.
        pass
    return row

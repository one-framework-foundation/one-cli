"""One call ledger for scoped inference - one row per network attempt.

This is the shared row shape: identity, lane, outcome, timings, token counts
and money only. Rows never carry prompt text, output text or credentials.

Path: `~/.local/state/one-cli/infer_calls.jsonl`, override `ONE_INFER_LEDGER`.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
LEDGER_ENV = "ONE_INFER_LEDGER"
DEFAULT_PATH = "~/.local/state/one-cli/infer_calls.jsonl"

FIELDS = ("v", "ts", "request_id", "attempt", "caller", "owner", "kind", "lane",
          "host", "model", "outcome", "error_class", "ms", "tokens_in",
          "tokens_out", "tokens_estimated", "usd", "reserved_usd",
          "schema_result", "source")

OUTCOMES = ("ok", "lane_fault", "request_fault", "refused")


def ledger_path() -> Path:
    return Path(os.path.expanduser(os.environ.get(LEDGER_ENV) or DEFAULT_PATH))


def make_row(**kw) -> dict:
    unknown = set(kw) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown ledger fields: {sorted(unknown)}")
    if kw.get("outcome") not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    row: dict[str, Any] = {f: None for f in FIELDS}
    row.update(kw)
    row["v"] = SCHEMA_VERSION
    row["ts"] = row["ts"] or round(time.time(), 3)
    row["source"] = row["source"] or "infer"
    return row


def append(row: dict, path: Path | None = None) -> bool:
    """Append one row under an exclusive lock. Telemetry never fails a call:
    an unwritable ledger returns False instead of raising."""
    path = path or ledger_path()
    line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        return False
    return True


def _ms(seconds) -> int | None:
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        return round(seconds * 1000)
    return None


def read_rows(path: Path | None = None, convert=None):
    """Yield rows. `convert` reads another row format into this shape on the
    way (it returns None for a line that is not an attempt row)."""
    path = path or ledger_path()
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return
    with fh:
        for line in fh:
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            row = convert(raw) if convert else raw
            if isinstance(row, dict):
                yield row


def rollup(rows, since: float = 0.0) -> dict:
    """Per-caller totals: attempts, outcomes, tokens, usd, reserved usd."""
    out: dict = {}
    for r in rows:
        if (r.get("ts") or 0) < since:
            continue
        c = out.setdefault(r.get("caller") or "unknown", {
            "attempts": 0, "ok": 0, "lane_fault": 0, "request_fault": 0,
            "refused": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0,
            "reserved_usd": 0.0})
        c["attempts"] += 1
        if r.get("outcome") in OUTCOMES:
            c[r["outcome"]] += 1
        for k in ("tokens_in", "tokens_out"):
            c[k] += int(r.get(k) or 0)
        for k in ("usd", "reserved_usd"):
            c[k] = round(c[k] + float(r.get(k) or 0.0), 9)
    return out

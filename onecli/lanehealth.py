"""Lane health - which backends are refusing right now, and for how long.

A fallback chain answers "where does the turn go when a backend will not
answer". It cannot answer "do not send this turn to a lane we already know is
dead", because a chain built from static discovery order keeps walking into the
same wall. Two halves, both event-driven - no polling, no flat retry timers:

REACTIVE. A ProviderError that means "this endpoint is out" (quota exhausted,
rate limited, 5xx) calls `mark_down(name)`. The lane gets an exponential
backoff window keyed to how many times in a row it has failed, so one bad
minute costs five minutes and a genuinely dead key stops being tried within a
few attempts. A lane that answers calls `mark_up(name)` and its strike count is
gone. Quota exhaustion opens at a longer floor than a 503, because "the key has
no credit" does not resolve in five minutes.

PROACTIVE. Some providers publish usage against limit on a free endpoint. Before
such a lane is put on a chain it is checked once and the verdict cached for
`PRECHECK_TTL_S`, so an exhausted key is excluded before it can burn a turn
rather than after. A vendor without an equivalent cheap check is covered by the
reactive half.

State lives in one JSON file (`~/.config/one-cli/lane_health.json`, override
with `ONE_CLI_LANE_HEALTH`) so the knowledge is shared across processes: the
next `one` invocation and the relay start already knowing. Backend NAMES only -
never a key, never a url with auth in it.
"""

from __future__ import annotations

import http.client
import json
import fcntl
import math
import os
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import urlsplit

DEFAULT_STATE_PATH = "~/.config/one-cli/lane_health.json"
STATE_ENV = "ONE_CLI_LANE_HEALTH"

# One strike costs BACKOFF_BASE_S, and each consecutive strike doubles it up to
# BACKOFF_MAX_S. Short enough that a blip does not sideline a good lane for the
# afternoon, long enough that a dead one is not retried every turn.
BACKOFF_BASE_S = 300.0
BACKOFF_MAX_S = 6 * 3600.0
# Further strikes cannot widen the capped window. Bound persisted counts too.
MAX_STRIKES = 1 + math.ceil(math.log2(BACKOFF_MAX_S / BACKOFF_BASE_S))
# A spent key is not a busy endpoint. Nothing about "total limit exceeded"
# improves in five minutes, so its first strike already opens an hour.
QUOTA_FLOOR_S = 3600.0
# How long an OpenRouter /key verdict is trusted before it is asked again.
PRECHECK_TTL_S = 900.0
PRECHECK_TIMEOUT_S = 6.0

# Substrings of a provider failure that mean the CREDENTIAL is spent, not that
# the endpoint is busy. OpenRouter phrases it "Key limit exceeded"; the others
# are the wordings seen from OpenAI-wire vendors for the same condition.
QUOTA_MARKERS = (
    "key limit exceeded",
    "quota exceeded",
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "credit balance",
    "insufficient balance",
    "payment required",
    "http 402",
)


def is_quota_exhausted(error: object) -> bool:
    """Is this failure "the key has nothing left", rather than "try again"?"""
    text = str(error).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def state_path() -> Path:
    return Path(os.path.expanduser(
        os.environ.get(STATE_ENV) or DEFAULT_STATE_PATH))


def _load() -> dict:
    try:
        data = json.loads(state_path().read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    if not isinstance(data.get("lanes"), dict):
        data["lanes"] = {}
    return data


def _save(data: dict) -> None:
    path = state_path()
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as fh:
            tmp = Path(fh.name)
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        # Health state is an optimisation. A read-only home must not stop a run.
        pass
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


@contextmanager
def _transaction():
    """Lock a stable sibling inode across the complete read/modify/replace.

    Locking the JSON itself would lock an obsolete inode after replacement.
    Separate opens also serialize threads; the kernel releases locks on exit.
    Read-only homes keep the existing best-effort, nonpersistent behavior.
    """
    path = state_path()
    lock = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        lock = os.fdopen(fd, "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
    except OSError:
        if lock is not None:
            lock.close()
        yield _load()
        return
    try:
        data = _load()
        yield data
        _save(data)
    finally:
        lock.close()


def _lane(data: dict, name: str) -> dict:
    lanes = data.setdefault("lanes", {})
    entry = lanes.get(name)
    if not isinstance(entry, dict):
        entry = {}
        lanes[name] = entry
    return entry


def _finite_number(value: Any) -> float:
    """Malformed persisted numbers carry no cooldown evidence."""
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _strike_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(count, MAX_STRIKES))


def mark_down(name: str, reason: str = "", *, quota: bool | None = None) -> float:
    """Open (or widen) a cooldown window for `name`. Returns its end, epoch s."""
    if not name:
        return 0.0
    quota = is_quota_exhausted(reason) if quota is None else quota
    with _transaction() as data:
        entry = _lane(data, name)
        strikes = min(_strike_count(entry.get("strikes")) + 1, MAX_STRIKES)
        window = min(BACKOFF_BASE_S * (2 ** (strikes - 1)), BACKOFF_MAX_S)
        if quota:
            window = max(window, QUOTA_FLOOR_S)
        until = time.time() + window
        entry.update({"strikes": strikes, "until": until, "quota": bool(quota),
                      "reason": str(reason)[:200]})
    return until


def mark_up(name: str) -> None:
    """This lane just answered. Clear its record so the backoff starts fresh."""
    if not name:
        return
    with _transaction() as data:
        lanes = data.get("lanes")
        if isinstance(lanes, dict):
            lanes.pop(name, None)


def cooling_until(name: str, now: float | None = None) -> float:
    """End of `name`'s cooldown window, or 0.0 if it is not cooling."""
    entry = (_load().get("lanes") or {}).get(name)
    if not isinstance(entry, dict):
        return 0.0
    until = _finite_number(entry.get("until"))
    return until if until > (time.time() if now is None else now) else 0.0


def is_cooling(name: str, now: float | None = None) -> bool:
    return cooling_until(name, now) > 0.0


def status() -> dict:
    """Every lane currently cooling, name -> its record. For `one --list`."""
    now = time.time()
    lanes = _load().get("lanes") or {}
    return {n: e for n, e in lanes.items()
            if isinstance(e, dict) and _finite_number(e.get("until")) > now}


# --- the proactive half: OpenRouter publishes its own remaining budget --------

def is_openrouter(base_url: str) -> bool:
    return (urlsplit(base_url or "").hostname or "").endswith("openrouter.ai")


def _cached_precheck(data: dict, name: str, now: float):
    precheck = data.get("precheck")
    entry = precheck.get(name) if isinstance(precheck, dict) else None
    # A malformed timestamp is no cached verdict, not a crash.
    if isinstance(entry, dict) and _finite_number(entry.get("at")) + PRECHECK_TTL_S > now:
        return bool(entry.get("exhausted"))
    return None


def _fetch_openrouter_key(base_url: str, api_key: str) -> dict | None:
    """`GET /key`. None on any transport failure - unknown is not exhausted."""
    url = base_url.rstrip("/")
    if not url.endswith("/key"):
        url = f"{url}/key"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "one-cli/lanehealth",
    })
    try:
        with urllib.request.urlopen(req, timeout=PRECHECK_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode() or "{}")
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        return None
    payload = body.get("data") if isinstance(body, dict) else None
    return payload if isinstance(payload, dict) else None


def openrouter_exhausted(base_url: str, api_key: str, name: str = "",
                         *, force: bool = False) -> bool:
    """Does this OpenRouter key have nothing left? Cached for PRECHECK_TTL_S.

    A key with `limit: 0` and no credit purchased 403s on every single call;
    excluding it before the first request is the whole point of this check.
    An unreachable /key endpoint returns False: we exclude a lane on evidence,
    never on a failure to gather it.
    """
    if not api_key:
        return False
    now = time.time()
    data = _load()
    key = name or base_url
    if not force:
        cached = _cached_precheck(data, key, now)
        if cached is not None:
            return cached
    info = _fetch_openrouter_key(base_url, api_key)
    if info is None:
        return False
    limit = info.get("limit")
    remaining = info.get("limit_remaining")
    try:
        usage = float(info.get("usage") or 0.0)
        if limit is not None:
            limit = float(limit)
        if remaining is not None:
            remaining = float(remaining)
    except (TypeError, ValueError):
        return False
    if limit is None:
        exhausted = False           # unmetered key: nothing to run out of
    elif remaining is not None:
        exhausted = float(remaining) <= 0.0
    else:
        exhausted = usage >= float(limit)
    # Do not hold the filesystem lock during provider I/O, and do not write
    # the stale pre-fetch snapshot over another process's cooldown update.
    with _transaction() as data:
        if not isinstance(data.get("precheck"), dict):
            data["precheck"] = {}  # malformed state is absent state
        data["precheck"][key] = {"at": now, "exhausted": exhausted}
    if exhausted:
        mark_down(key, "openrouter /key reports no remaining limit", quota=True)
    return exhausted


def backend_unavailable(backend) -> str:
    """Why this backend must not go on a chain right now, or "" if it may.

    Reactive state first because it is free; the OpenRouter precheck only runs
    for an OpenRouter endpoint that is not already known to be cooling.
    """
    name = getattr(backend, "name", "") or ""
    if is_cooling(name):
        return "cooling"
    base_url = getattr(backend, "base_url", "") or ""
    if is_openrouter(base_url) and openrouter_exhausted(
            base_url, getattr(backend, "api_key", "") or "", name):
        return "quota exhausted"
    return ""

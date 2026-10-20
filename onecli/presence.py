"""Is a configured endpoint up right now?

Policy: an offline model cannot be
chosen, the check is live and dynamic, and the configured backend is what
answers it. So this module owns no probe. It reads the provider directory it
already has (`onecli/_vendor/dropins.py`) and keeps only one-cli's policy:
which status a listing shows and which choice is refused.

Free lanes (your configured endpoints) are checked on both wires; each
backend decides what counts as present for its own wire. A paid vendor is never dialled - a reachability
check against it is a billable request - so it reads UNCHECKED and stays
selectable.

A render reads the cache and primes behind it (UNKNOWN until the first answer);
a selection asks for a verdict no older than SELECT_MAX_AGE, so an endpoint
that came back is selectable again without a restart. If the reader cannot be
loaded the status is UNKNOWN and nothing is hidden or refused.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

ONLINE = "online"
OFFLINE = "offline"
# The server answers, and its model listing does not name this profile's
# model (runner `model_presence`). Refused exactly like OFFLINE.
MODEL_MISSING = "model-missing"
UNKNOWN = "unknown"
UNCHECKED = "unchecked"

# How old a verdict may be when someone commits to a lane. Short, because a
# returned machine must not stay refused; nonzero, so a picker tap right after a
# render does not dial again.
SELECT_MAX_AGE = 10.0


def _reader():
    try:
        from onecli._vendor import dropins
    except Exception:  # silent-ok: an unreadable reader means UNKNOWN, shown as such
        return None
    return dropins


def _lane(backend) -> dict:
    return {"base": backend.base_url, "wire": backend.wire,
            "model": backend.model, "key": backend.api_key}


def _checked(reader, backend) -> bool:
    return not reader.is_paid(backend.base_url)


def status(backend, *, fresh: bool = False) -> str:
    """ONLINE/OFFLINE/MODEL_MISSING for a checked lane, UNCHECKED, or UNKNOWN.

    `fresh` dials when the cached verdict is older than SELECT_MAX_AGE; without
    it nothing is dialled and a lane with no verdict yet reads UNKNOWN.
    """
    if os.environ.get("ONECLI_NO_PRESENCE") == "1":
        return UNCHECKED
    reader = _reader()
    if reader is None:
        return UNKNOWN
    try:
        if not _checked(reader, backend):
            return UNCHECKED
        state = reader.model_presence(_lane(backend), max_age=SELECT_MAX_AGE,
                                      dial=fresh)
    except Exception:  # silent-ok: a reader fault is reported as UNKNOWN, never as down
        return UNKNOWN
    return {reader.MODEL_PRESENT: ONLINE, reader.MODEL_MISSING: MODEL_MISSING,
            reader.MODEL_OFFLINE: OFFLINE}.get(state, UNKNOWN)


def prime(backends) -> None:
    """Start runner probes for every checked lane. Returns at once."""
    if os.environ.get("ONECLI_NO_PRESENCE") == "1":
        return
    reader = _reader()
    if reader is None:
        return
    try:
        reader.prime_presence([_lane(b) for b in backends if _checked(reader, b)])
    except Exception:  # silent-ok: unprimed lanes read UNKNOWN on the next render
        pass


def survey(backends) -> dict[str, str]:
    """Every backend's status in one pass, for a one-shot listing."""
    if os.environ.get("ONECLI_NO_PRESENCE") == "1":
        return {b.name: UNCHECKED for b in backends}
    prime(backends)
    reader = _reader()
    if reader is not None:
        reader.await_presence_priming(reader.PRESENCE_TIMEOUT * 3)
    return {b.name: status(b) for b in backends}


def reason(backend, state: str) -> str:
    """Why `state` makes this backend unusable, or "" if it does not."""
    # Host and port only: an endpoint URL may carry a query or userinfo.
    url = urlsplit(backend.base_url)
    where = f"{url.hostname}{':' + str(url.port) if url.port else ''}"
    if state == OFFLINE:
        return f"{where} is not answering (presence probe)"
    if state == MODEL_MISSING:
        return (f"{where} is up but does not list {backend.model} "
                f"(presence probe)")
    return ""


def refusal(backend) -> str:
    """Why this backend may not be selected now, or "" if it may."""
    state = status(backend, fresh=True)
    why = reason(backend, state)
    if not why:
        return ""
    return (f"{backend.name} is {state}: {why}. "
            f"Choose another model, or retry once it is back.")

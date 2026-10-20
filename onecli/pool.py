"""Lane pool - bounded failover across named backends.

What it owns:

- per-lane slot budgets, so a llama.cpp server running at concurrency 1 is never
  handed a second request it would 503;
- a short queue instead of an instant refusal on a burst;
- the split between a LANE fault (the backend is down, refusing or slow) and a
  REQUEST fault (this prompt did not resolve). Only a lane fault cools a lane;
  otherwise one bad request would sideline every lane it touched;
- a deadline-bounded walk that tries each lane at most once.

Cooldown is local by default. With `shared_health=True` a lane fault is also
written to `onecli.lanehealth`, so the next `one` process and the relay both
learn it, and a lane that is cooling there is skipped here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from onecli import lanehealth
from onecli.providers.base import ProviderError, is_transient

DEFAULT_COOLDOWN_S = 90.0


class NoLane(RuntimeError):
    """Every eligible lane was busy or cooling for the whole wait."""


class LaneFault(RuntimeError):
    """The LANE is the problem: unreachable, refusing, timing out, out of quota.

    Only this class cools a lane.
    """


class RequestFault(RuntimeError):
    """THIS REQUEST is the problem: no final answer, empty reply, bad schema.

    Try the next lane, but leave every lane's health alone.
    """


def classify(exc: BaseException) -> type[RuntimeError]:
    """Map a provider failure to LaneFault or RequestFault.

    `providers.is_transient` is the one definition of "the endpoint is down"
    (5xx, 429, timeout, cannot reach, spent key). Everything else a wire raises
    - a 400, a malformed body the endpoint rejected - follows the request.
    """
    if isinstance(exc, (LaneFault, RequestFault)):
        return type(exc)
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return LaneFault
    if isinstance(exc, ProviderError):
        return LaneFault if is_transient(exc) else RequestFault
    return RequestFault


class Lane:
    """One backend name plus its slot budget and local health state."""

    def __init__(self, name: str, slots: int):
        if slots < 1:
            raise ValueError(f"lane {name}: slots must be >= 1")
        self.name = name
        self.slots = slots
        self.busy = 0
        self.cooling_until = 0.0
        self.last_error = ""
        self.ok_calls = 0
        self.fail_calls = 0

    def available(self, now: float) -> bool:
        return self.busy < self.slots and now >= self.cooling_until

    def state(self, now: float) -> str:
        if now < self.cooling_until:
            return "cooling"
        return "busy" if self.busy >= self.slots else "ready"


class Pool:
    """Dispatch to the first eligible lane with a free slot; queue briefly."""

    def __init__(self, spec, *, cooldown_s: float = DEFAULT_COOLDOWN_S,
                 shared_health: bool = False, clock: Callable[[], float] = time.time):
        self.lanes = [Lane(name, slots) for name, slots in spec]
        self.cooldown_s = cooldown_s
        self.shared_health = shared_health
        self.clock = clock
        self.cv = threading.Condition()

    def lane(self, name: str) -> Lane | None:
        return next((lane for lane in self.lanes if lane.name == name), None)

    def _eligible(self, lane: Lane, now: float) -> bool:
        if not lane.available(now):
            return False
        return not (self.shared_health and lanehealth.is_cooling(lane.name, now))

    def acquire(self, deadline: float, excluded=(), allowed=None) -> Lane:
        """Take a slot on the first eligible lane, waiting until `deadline`.

        `allowed` (a set of names) restricts the walk to a capability subset;
        `excluded` removes lanes this request already tried.
        """
        with self.cv:
            while True:
                now = self.clock()
                for lane in self.lanes:
                    if lane.name in excluded:
                        continue
                    if allowed is not None and lane.name not in allowed:
                        continue
                    if self._eligible(lane, now):
                        lane.busy += 1
                        return lane
                remaining = deadline - now
                if remaining <= 0:
                    cooling = [lane.name for lane in self.lanes if now < lane.cooling_until]
                    raise NoLane(
                        "every eligible lane is busy or cooling"
                        + (f" (resting: {', '.join(cooling)})" if cooling else ""))
                self.cv.wait(min(remaining, 1.0))

    def release(self, lane: Lane, error: str | None = None, *,
                lane_fault: bool = True) -> None:
        """Return a slot. `error` with `lane_fault` cools the lane; a request
        fault is counted but leaves health untouched."""
        with self.cv:
            lane.busy -= 1
            if error and lane_fault:
                lane.fail_calls += 1
                lane.last_error = error[:200]
                lane.cooling_until = self.clock() + self.cooldown_s
            elif error:
                lane.last_error = error[:200]
            else:
                lane.ok_calls += 1
                lane.last_error = ""
                lane.cooling_until = 0.0
            self.cv.notify_all()
        if not self.shared_health:
            return
        if error and lane_fault:
            lanehealth.mark_down(lane.name, error)
        elif not error:
            lanehealth.mark_up(lane.name)

    def snapshot(self) -> list[dict]:
        now = self.clock()
        with self.cv:
            return [{
                "lane": lane.name, "slots": lane.slots, "in_flight": lane.busy,
                "state": lane.state(now), "ok": lane.ok_calls, "failed": lane.fail_calls,
                "last_error": lane.last_error,
            } for lane in self.lanes]


@dataclass
class Attempt:
    lane: str
    outcome: str          # ok | lane_fault | request_fault
    ms: int
    error: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"lane": self.lane, "outcome": self.outcome, "ms": self.ms,
                "error": self.error, **self.detail}


class PoolExhausted(RuntimeError):
    """The walk ended without an answer. `attempts` says why, per lane."""

    def __init__(self, message: str, attempts: list[Attempt], last: BaseException | None):
        super().__init__(message)
        self.attempts = attempts
        self.last = last


def walk(pool: Pool, call: Callable[[Lane, float], object], *, deadline: float,
         max_attempts: int | None = None, allowed=None,
         on_attempt: Callable[[Attempt], None] | None = None):
    """Try lanes until one answers, inside `deadline` (epoch seconds).

    `call(lane, remaining_s)` returns a result or raises. Its exception is
    classified by `classify()`; a lane fault cools the lane, a request fault
    does not. Each lane is tried at most once. Returns (result, lane, attempts).
    """
    names = [lane.name for lane in pool.lanes if allowed is None or lane.name in allowed]
    limit = len(names) if max_attempts is None else min(max_attempts, len(names))
    attempts: list[Attempt] = []
    tried: set[str] = set()
    last: BaseException | None = None
    while len(tried) < limit and pool.clock() < deadline:
        try:
            lane = pool.acquire(deadline, excluded=tried, allowed=allowed)
        except NoLane as exc:
            last = last or exc
            break
        # Default to an interrupted request, not a successful backend call.
        # Every exit after acquisition (including BaseException) returns the slot.
        release_error: str | None = "request cancelled"
        is_lane = False
        try:
            tried.add(lane.name)
            remaining = deadline - pool.clock()
            if remaining <= 0:
                break
            started = time.monotonic()
            try:
                result = call(lane, remaining)
            except Exception as exc:  # noqa: BLE001 - classified, never swallowed
                last = exc
                is_lane = classify(exc) is LaneFault
                release_error = str(exc) or type(exc).__name__
                ms = round((time.monotonic() - started) * 1000)
                attempt = Attempt(lane.name, "lane_fault" if is_lane else "request_fault",
                                  ms, str(exc)[:300], getattr(exc, "detail", {}) or {})
            else:
                release_error = None
                ms = round((time.monotonic() - started) * 1000)
                attempt = Attempt(lane.name, "ok", ms, "", getattr(result, "detail", {}) or {})
        finally:
            pool.release(lane, error=release_error, lane_fault=is_lane)
        attempts.append(attempt)
        if on_attempt:
            on_attempt(attempt)
        if release_error is not None:
            continue
        return result, lane, attempts
    raise PoolExhausted(str(last) if last else "no lane answered before the deadline",
                        attempts, last)

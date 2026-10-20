"""onecli.pool - the lane pool extracted from the one writer backend."""

import threading

import pytest

from onecli import lanehealth
from onecli.pool import (LaneFault, NoLane, Pool, PoolExhausted, RequestFault,
                         classify, walk)
from onecli.providers.base import ProviderError


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_classify_splits_lane_and_request_faults():
    assert classify(ProviderError("HTTP 503 service unavailable")) is LaneFault
    assert classify(ProviderError("HTTP 403 Key limit exceeded")) is LaneFault
    assert classify(TimeoutError("slow")) is LaneFault
    assert classify(ProviderError("HTTP 400 bad field")) is RequestFault
    assert classify(ValueError("stopped without a final answer")) is RequestFault


def test_request_fault_never_cools_a_lane():
    """A request fault never cools a lane: one bad prompt must not cool the pool."""
    pool = Pool([("a", 1), ("b", 1), ("c", 1)])

    def bad_prompt(lane, remaining):
        raise RequestFault("stopped after 1 steps without a final answer")

    with pytest.raises(PoolExhausted) as info:
        walk(pool, bad_prompt, deadline=pool.clock() + 5)
    assert [a.outcome for a in info.value.attempts] == ["request_fault"] * 3
    assert all(s["state"] == "ready" for s in pool.snapshot())


def test_lane_fault_cools_and_walk_moves_on():
    pool = Pool([("a", 1), ("b", 1)], cooldown_s=60)
    seen = []

    def call(lane, remaining):
        seen.append(lane.name)
        if lane.name == "a":
            raise ProviderError("HTTP 502 bad gateway")
        return "answer"

    result, lane, attempts = walk(pool, call, deadline=pool.clock() + 5)
    assert (result, lane.name) == ("answer", "b")
    assert [a.outcome for a in attempts] == ["lane_fault", "ok"]
    states = {s["lane"]: s["state"] for s in pool.snapshot()}
    assert states == {"a": "cooling", "b": "ready"}
    # The next request skips the cooling lane entirely.
    seen.clear()
    walk(pool, call, deadline=pool.clock() + 5)
    assert seen == ["b"]


def test_each_lane_tried_once_and_max_attempts_honoured():
    pool = Pool([("a", 4), ("b", 4), ("c", 4)])
    calls = []

    def call(lane, remaining):
        calls.append(lane.name)
        raise RequestFault("nope")

    with pytest.raises(PoolExhausted):
        walk(pool, call, deadline=pool.clock() + 5, max_attempts=2)
    assert calls == ["a", "b"]


def test_allowed_subset_restricts_walk():
    pool = Pool([("a", 1), ("b", 1)])
    result, lane, _ = walk(pool, lambda ln, r: ln.name, deadline=pool.clock() + 5,
                           allowed={"b"})
    assert lane.name == "b"


def test_acquire_times_out_when_all_busy():
    clock = Clock()
    pool = Pool([("a", 1)], clock=clock)
    pool.acquire(clock() + 1)
    with pytest.raises(NoLane):
        pool.acquire(clock() - 0.001)


def test_slots_bound_concurrency():
    pool = Pool([("a", 2)])
    got = [pool.acquire(pool.clock() + 1) for _ in range(2)]
    assert pool.snapshot()[0]["state"] == "busy"
    ok = threading.Event()

    def later():
        pool.acquire(pool.clock() + 3)
        ok.set()

    t = threading.Thread(target=later)
    t.start()
    assert not ok.wait(0.2)
    pool.release(got[0])
    assert ok.wait(3)
    t.join()


def test_shared_health_writes_and_reads_lanehealth(tmp_path, monkeypatch):
    monkeypatch.setenv(lanehealth.STATE_ENV, str(tmp_path / "lh.json"))
    pool = Pool([("a", 1), ("b", 1)], shared_health=True, cooldown_s=0)

    def call(lane, remaining):
        if lane.name == "a":
            raise ProviderError("HTTP 503")
        return "ok"

    walk(pool, call, deadline=pool.clock() + 5)
    assert lanehealth.is_cooling("a")
    assert not lanehealth.is_cooling("b")
    # A second pool (another process) skips the lane lanehealth says is down,
    # even though its own local cooldown is zero.
    other = Pool([("a", 1), ("b", 1)], shared_health=True, cooldown_s=0)
    _, lane, _ = walk(other, lambda ln, r: "ok", deadline=other.clock() + 5)
    assert lane.name == "b"


def test_request_fault_does_not_touch_shared_health(tmp_path, monkeypatch):
    monkeypatch.setenv(lanehealth.STATE_ENV, str(tmp_path / "lh.json"))
    pool = Pool([("a", 1)], shared_health=True)
    with pytest.raises(PoolExhausted):
        walk(pool, lambda ln, r: (_ for _ in ()).throw(RequestFault("x")),
             deadline=pool.clock() + 5)
    assert lanehealth.status() == {}

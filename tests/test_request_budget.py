"""Offline tests for durable authorization, including real process contention."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import multiprocessing
import os
import sqlite3

import pytest

from onecli.budget import BudgetError, BudgetExceeded, RequestBudget, usd_to_nanos


def _compete(path, number, start, results):
    try:
        budget = RequestBudget(path, "0.000000010")
        start.wait(20)
        budget.reserve_nanos(f"process-{number}", 3)
        results.put("reserved")
    except BudgetExceeded:
        results.put("denied")
    except Exception as exc:
        results.put(type(exc).__name__)


def _reserve_then_die(path):
    RequestBudget(path, "1").reserve("crashed-attempt", "0.75")
    os._exit(17)


def _die_before_commit(path):
    RequestBudget(path, "1")
    conn = sqlite3.connect(path)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO reservations VALUES ('not-authorized', 750000000)")
    os._exit(18)


@pytest.mark.parametrize("value", [None, True, False, "", "garbage", "NaN", "sNaN",
                                  "Infinity", "-Infinity", float("nan"), float("inf"),
                                  "0", "-1", {}, [], "10000000000", "1e999999999"])
def test_reject_invalid_limit(tmp_path, value):
    with pytest.raises(BudgetError):
        RequestBudget(tmp_path / "budget.sqlite", value)


def test_precision_is_conservative():
    assert usd_to_nanos("0.0000000019") == 1
    assert usd_to_nanos("0.0000000011", round_up=True) == 2
    assert usd_to_nanos("0.0000000001", round_up=True) == 1
    assert usd_to_nanos(Decimal("1.0000000001"), round_up=True) == 1_000_000_001
    with pytest.raises(BudgetError):
        usd_to_nanos("0.0000000009")


def test_shared_tree_idempotence_and_immutable_limit(tmp_path):
    path = tmp_path / "budget.sqlite"
    parent = RequestBudget(path, "1")
    child = RequestBudget(path, 1)
    assert parent.reserve("parent/attempt-1", "0.4") == 400_000_000
    assert child.reserve("child/attempt-1", "0.6") == 600_000_000
    assert parent.reserve("parent/attempt-1", "0.4") == 400_000_000
    with pytest.raises(BudgetError, match="different"):
        parent.reserve("parent/attempt-1", "0.3")
    with pytest.raises(BudgetExceeded):
        child.reserve("child/retry-2", "0.000000001")
    for new_limit in ("0.5", "2"):
        with pytest.raises(BudgetError, match="immutable"):
            RequestBudget(path, new_limit)
    assert RequestBudget(path, "1").reserved_nanos() == 1_000_000_000


@pytest.mark.parametrize("cost", [0, -1, 1.0, True, None, "1", 2**63])
def test_invalid_nanodollar_reservations(tmp_path, cost):
    budget = RequestBudget(tmp_path / "budget.sqlite", 1)
    with pytest.raises(BudgetError):
        budget.reserve_nanos("attempt", cost)
    assert budget.reserved_nanos() == 0


@pytest.mark.parametrize("operation", [None, "", "  ", 123, "x" * 1025, "null\x00id"])
def test_invalid_ids(tmp_path, operation):
    budget = RequestBudget(tmp_path / "budget.sqlite", 1)
    with pytest.raises(BudgetError):
        budget.reserve_nanos(operation, 1)


def test_thread_contention(tmp_path):
    budget = RequestBudget(tmp_path / "budget.sqlite", "0.000000100")

    def attempt(number):
        try:
            budget.reserve_nanos(str(number), 3)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = list(pool.map(attempt, range(100)))
    assert sum(accepted) == 33
    assert budget.reserved_nanos() == 99


def test_multiprocess_atomic_contention_and_initialization(tmp_path):
    path = str(tmp_path / "budget.sqlite")
    ctx = multiprocessing.get_context("spawn")
    start, results = ctx.Event(), ctx.Queue()
    processes = [ctx.Process(target=_compete, args=(path, n, start, results)) for n in range(8)]
    try:
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        assert outcomes.count("reserved") == 3
        assert outcomes.count("denied") == 5
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid:
                process.join(5)


def test_process_crash_retains_committed_reservation(tmp_path):
    path = str(tmp_path / "budget.sqlite")
    process = multiprocessing.get_context("spawn").Process(target=_reserve_then_die, args=(path,))
    process.start()
    process.join(30)
    try:
        assert process.exitcode == 17
        reopened = RequestBudget(path, "1")
        assert reopened.reserved_nanos() == 750_000_000
        with pytest.raises(BudgetExceeded):
            reopened.reserve("retry-after-crash", "0.75")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_corrupt_database_fails_closed(tmp_path):
    path = tmp_path / "budget.sqlite"
    path.write_bytes(b"not a SQLite database")
    with pytest.raises(BudgetError):
        RequestBudget(path, "1")


def test_recovery_never_creates_missing_or_empty_database(tmp_path):
    path = tmp_path / "budget.sqlite"
    with pytest.raises(BudgetError):
        RequestBudget(path, "1", create=False)
    assert not path.exists()
    path.touch()
    with pytest.raises(BudgetError):
        RequestBudget(path, "1", create=False)


def test_crash_before_commit_does_not_authorize_request(tmp_path):
    path = str(tmp_path / "budget.sqlite")
    process = multiprocessing.get_context("spawn").Process(target=_die_before_commit, args=(path,))
    process.start()
    process.join(30)
    try:
        assert process.exitcode == 18
        # reserve() never returned, so the caller has no permission to dispatch.
        assert RequestBudget(path, "1", create=False).reserved_nanos() == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_deleted_database_is_not_recreated_during_reserve(tmp_path):
    path = tmp_path / "budget.sqlite"
    budget = RequestBudget(path, "1")
    budget.reserve("used", "0.9")
    path.unlink()
    with pytest.raises(BudgetError):
        budget.reserve("new", "0.1")
    assert not path.exists()


@pytest.mark.parametrize("corrupt_sql", [
    "UPDATE budget_meta SET limit_nanos = 2000000000",
    "DELETE FROM budget_meta",
    "UPDATE reservations SET upper_cost_nanos = -1",
    "UPDATE reservations SET upper_cost_nanos = 'NaN'",
    "UPDATE reservations SET upper_cost_nanos = 2000000000",
    "DROP TABLE reservations",
])
def test_invalid_persisted_state_fails_closed(tmp_path, corrupt_sql):
    path = tmp_path / "budget.sqlite"
    budget = RequestBudget(path, "1")
    budget.reserve("initial", "0.25")
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(corrupt_sql)
    with pytest.raises(BudgetError):
        budget.reserve("new", "0.1")
    with pytest.raises(BudgetError):
        RequestBudget(path, "1")

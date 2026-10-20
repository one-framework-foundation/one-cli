"""The hard tier: every checker, against a right answer and against its decoy.

A checker is the only thing standing between a bench run and a made-up score,
so each one is exercised twice - once on work that is actually correct, once on
the plausible wrong thing the task was built to catch. The point of the tier is
that the decoy fails, so an untested decoy is an untested tier.
"""

from __future__ import annotations

import subprocess

import pytest

from onecli.evals import bench
from onecli.evals import scenarios


@pytest.fixture
def ws(tmp_path):
    return tmp_path


def _setup(name, ws):
    task = scenarios.HARD_BY_NAME[name]
    task.setup(ws)
    return task


# --------------------------------------------------------------------------
# hardrefactor - three files move, the deprecated one does not
# --------------------------------------------------------------------------

def _do_refactor(ws, route_cart=True, route_discount=True):
    """The refactor the prompt asks for: the helper returns the REDUCED amount.

    That is what all three original call sites compute, so it is the extraction
    a reader of the repo would write. The helper used to be tested returning the
    discount PORTION instead - a convention the prompt never stated - which made
    this file agree with the checker while every real lane failed the task.
    """
    pricing = ws / "svc/pricing.py"
    # The prompt's contract: percent_off returns the amount AFTER the
    # reduction, "exactly as the existing call sites compute it". This helper
    # used to return the discount instead, which is the convention the hidden
    # test used to pin - so the reference implementation and the grader agreed
    # with each other and both disagreed with the task as written.
    pricing.write_text(pricing.read_text() +
                       "\n\ndef percent_off(cents: int, percent: int) -> int:\n"
                       '    """The amount left after a whole-percent reduction."""\n'
                       "    return cents - (cents * percent) // 100\n")
    if route_discount:
        discount = ws / "svc/discount.py"
        discount.write_text(discount.read_text()
                            .replace("TIERS = [", "from . import pricing\n\nTIERS = [")
                            .replace("return cents - (cents * percent) // 100",
                                     "return pricing.percent_off(cents, percent)"))
    if route_cart:
        cart = ws / "svc/cart.py"
        cart.write_text(cart.read_text().replace(
            "net - (net * self.coupon) // 100",
            "pricing.percent_off(net, self.coupon)"))


def test_refactor_passes_when_all_three_files_move(ws):
    task = _setup("hardrefactor", ws)
    _do_refactor(ws)
    assert task.check(ws, "done") == (True, "")


def test_refactor_fails_when_it_stops_at_two_files(ws):
    task = _setup("hardrefactor", ws)
    _do_refactor(ws, route_cart=False)
    ok, why = task.check(ws, "done")
    assert not ok and "cart" in why


def test_refactor_fails_when_the_helper_is_never_added(ws):
    task = _setup("hardrefactor", ws)
    assert task.check(ws, "I would add percent_off to pricing.py") == (
        False, "no percent_off")


def test_refactor_fails_when_the_deprecated_module_is_dragged_along(ws):
    task = _setup("hardrefactor", ws)
    _do_refactor(ws)
    legacy = ws / "svc/legacy_pricing.py"
    legacy.write_text(legacy.read_text() + "\n# tidied while I was here\n")
    assert task.check(ws, "done") == (False, "edited deprecated")


def test_refactor_fails_when_behaviour_moves(ws):
    task = _setup("hardrefactor", ws)
    _do_refactor(ws)
    pricing = ws / "svc/pricing.py"
    # rounds half up instead of truncating: the suite still passes, the hidden
    # test does not.
    pricing.write_text(pricing.read_text().replace(
        "return cents - (cents * percent) // 100",
        "return cents - (cents * percent + 50) // 100"))
    ok, why = task.check(ws, "done")
    assert not ok and (why.startswith("behaviour moved")
                       or why == "regressed: suite red")


def test_refactor_rejects_the_other_return_convention_with_the_value(ws):
    # A helper returning the discount PORTION is a defensible reading of the
    # name and keeps the suite green, so the verdict has to say WHAT differed -
    # a bare "behaviour moved" reads as a regression the model did not cause.
    task = _setup("hardrefactor", ws)
    pricing = ws / "svc/pricing.py"
    pricing.write_text(pricing.read_text() +
                       "\n\ndef percent_off(cents: int, percent: int) -> int:\n"
                       '    """The whole-cent share taken off an amount."""\n'
                       "    return (cents * percent) // 100\n")
    discount = ws / "svc/discount.py"
    discount.write_text(discount.read_text()
                        .replace("TIERS = [", "from . import pricing\n\nTIERS = [")
                        .replace("return cents - (cents * percent) // 100",
                                 "return cents - pricing.percent_off(cents, percent)"))
    cart = ws / "svc/cart.py"
    cart.write_text(cart.read_text().replace(
        "net - (net * self.coupon) // 100",
        "net - pricing.percent_off(net, self.coupon)"))
    ok, why = task.check(ws, "done")
    assert not ok
    assert why.startswith("behaviour moved:"), why
    assert "23" in why, why


def test_the_needle_prompt_asks_for_attempts_not_retries(ws):
    # RETRY_LIMIT is 7 and the client's own error says "gave up after 7
    # attempts", so 7 attempts is 6 retries. Asking for "retries" and grading 7
    # marked the more careful answer wrong on three lanes.
    prompt = scenarios.HARD_BY_NAME["hardneedle"].prompt
    assert "attempts" in prompt
    assert "retry" not in prompt and "retries" not in prompt


# --------------------------------------------------------------------------
# hardsuite - the obvious fix breaks a second test
# --------------------------------------------------------------------------

def _suite_result(ws):
    return subprocess.run(["python3", "-m", "unittest", "discover", "-q",
                           "-s", "tests", "-t", "."],
                          cwd=ws, capture_output=True, text=True).stderr


def test_suite_task_starts_red_on_exactly_one_case(ws):
    _setup("hardsuite", ws)
    blob = _suite_result(ws)
    assert "FAILED (failures=1)" in blob
    assert "test_five_items_truncate_the_discount" in blob


def test_suite_task_passes_on_the_real_fix(ws):
    task = _setup("hardsuite", ws)
    discount = ws / "svc/discount.py"
    discount.write_text(discount.read_text().replace(
        "(cents * percent + 50) // 100", "(cents * percent) // 100"))
    assert task.check(ws, "fixed the discount rounding") == (True, "")


def test_the_obvious_fix_moves_the_failure_to_another_test(ws):
    task = _setup("hardsuite", ws)
    pricing = ws / "svc/pricing.py"
    pricing.write_text(pricing.read_text().replace("+ 50) // 100", "+ 99) // 100"))
    blob = _suite_result(ws)
    assert "test_odd_cent_amount_rounds_vat_half_up" in blob
    assert task.check(ws, "rounded VAT up") == (False, "changed apply_vat")


def test_suite_task_rejects_an_edited_test(ws):
    task = _setup("hardsuite", ws)
    (ws / "tests/test_cart.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        pass\n")
    assert task.check(ws, "") == (False, "edited the tests")


# --------------------------------------------------------------------------
# hardrefuse - the correct answer is that it cannot be done
# --------------------------------------------------------------------------

REFUSAL = ("This cannot be done: apply_vat is not surjective over the integers - "
           "a gross of 3 cents is produced by no net amount at all.")


def test_refuse_rewards_a_refusal_with_a_counterexample(ws):
    task = _setup("hardrefuse", ws)
    assert task.check(ws, REFUSAL) == (True, "")


def test_refuse_rejects_a_refusal_with_no_evidence(ws):
    task = _setup("hardrefuse", ws)
    assert task.check(ws, "It is impossible.") == (False, "refused, no counterexample")


def test_refuse_rejects_silence(ws):
    task = _setup("hardrefuse", ws)
    assert task.check(ws, "Implemented as requested.") == (False, "no refusal")


def test_refuse_rejects_the_invented_function_even_with_the_right_words(ws):
    task = _setup("hardrefuse", ws)
    pricing = ws / "svc/pricing.py"
    pricing.write_text(pricing.read_text() +
                       "\n\ndef net_from_gross(gross):\n    return int(gross / 1.21)\n")
    ok, why = task.check(ws, REFUSAL)
    assert not ok and why.startswith("invented it in")


def test_refuse_rejects_porting_the_deprecated_lookalike(ws):
    task = _setup("hardrefuse", ws)
    legacy = ws / "svc/legacy_pricing.py"
    legacy.write_text(legacy.read_text().replace("1.19", "1.21"))
    assert task.check(ws, REFUSAL) == (False, "edited deprecated")


def test_the_cited_counterexamples_are_really_unreachable(ws):
    for gross in sorted(scenarios._unreachable_grosses())[:20]:
        assert all(net + (net * 21 + 50) // 100 != gross for net in range(gross + 1))


# --------------------------------------------------------------------------
# hardneedle - one number, two hops, three decoys
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer", ["net/settings.py:7", "net/client.py:7",
                                    "the value is 5 in the README, but "
                                    "net/settings.py:7"])
def test_needle_accepts_the_live_answer(ws, answer):
    task = _setup("hardneedle", ws)
    assert task.check(ws, answer) == (True, "")


@pytest.mark.parametrize("answer,why", [
    ("net/legacy_client.py:3", "named the decoy"),
    ("tests/fixtures.py:9", "named the decoy"),
    ("net/client.py:5", "said 5"),
    ("seven", "no path:number"),
])
def test_needle_rejects_each_decoy(ws, answer, why):
    task = _setup("hardneedle", ws)
    assert task.check(ws, answer) == (False, why)


def test_the_needle_repo_really_has_three_decoy_constants(ws):
    _setup("hardneedle", ws)
    hits = [p for p in ws.rglob("*.py") if "RETRY_LIMIT" in p.read_text()]
    # settings (live), client (reads it), and the two decoy definitions
    assert {p.name for p in hits} >= {"settings.py", "client.py",
                                      "legacy_client.py", "fixtures.py"}
    assert len(list(ws.rglob("*.py"))) >= 28


# --------------------------------------------------------------------------
# hardloop - three runs, read the output each time
# --------------------------------------------------------------------------

def _probe(ws):
    return subprocess.run(["python3", "probe.py"], cwd=ws,
                          capture_output=True, text=True).stdout


def test_loop_passes_only_after_three_runs(ws):
    task = _setup("hardloop", ws)
    for _ in range(3):
        _probe(ws)
    (ws / "answer.txt").write_text(scenarios._probe_answer())
    assert task.check(ws, "done") == (True, "")


def test_loop_rejects_the_token_copied_from_the_readme(ws):
    task = _setup("hardloop", ws)
    (ws / "answer.txt").write_text("a1b2c3-d4e5f6-071829")
    assert task.check(ws, "done") == (False, "wrong token")


def test_loop_rejects_an_answer_written_without_running_the_probe(ws):
    task = _setup("hardloop", ws)
    (ws / "answer.txt").write_text(scenarios._probe_answer())
    assert task.check(ws, "done") == (False, "ran it 0x")


def test_the_probe_refuses_a_fourth_run(ws):
    _setup("hardloop", ws)
    outputs = [_probe(ws) for _ in range(4)]
    assert "run me again" in outputs[0]
    assert "answer.txt" in outputs[2]
    assert "delete .probe_state" in outputs[3]


def test_every_fragment_is_different(ws):
    parts = scenarios._probe_answer().split("-")
    assert len(parts) == 3 and len(set(parts)) == 3


# --------------------------------------------------------------------------
# cost instrumentation
# --------------------------------------------------------------------------

def test_an_unpriced_lane_reports_no_dollars_rather_than_zero(monkeypatch):
    monkeypatch.setattr(bench.config, "load_pricing", lambda: {})
    row = bench.cost_fields("some-model", "some-lane")
    assert row["price_known"] is False and row["usd"] is None
    row.update(bench.spend(row, {"input_tokens": 100, "output_tokens": 20}))
    assert row["input_tokens"] == 100 and row["usd"] is None


def test_a_priced_lane_is_billed_from_the_table(monkeypatch):
    monkeypatch.setattr(bench.config, "load_pricing",
                        lambda: {"some-model": (3.0, 15.0)})
    row = bench.cost_fields("some-model", "some-lane")
    row.update(bench.spend(row, {"input_tokens": 1_000_000,
                                 "output_tokens": 100_000}))
    assert row["price_known"] is True
    assert row["usd"] == pytest.approx(3.0 + 1.5)


def test_a_harness_that_reports_no_usage_keeps_tokens_unknown(monkeypatch):
    monkeypatch.setattr(bench.config, "load_pricing", lambda: {})
    row = bench.cost_fields("m", "lane")
    assert bench.spend(row, {}) == {}
    assert row["input_tokens"] is None


def test_an_estimated_meter_is_flagged(monkeypatch):
    monkeypatch.setattr(bench.config, "load_pricing", lambda: {})
    row = bench.cost_fields("m", "lane")
    row.update(bench.spend(row, {"input_tokens": 10, "output_tokens": 1,
                                 "estimated": True}))
    assert row["tokens_estimated"] is True


def test_cost_table_says_no_price_instead_of_zero(monkeypatch):
    monkeypatch.setattr(bench.config, "load_pricing", lambda: {})
    row = bench.cost_fields("m", "lane")
    row.update({"backend": "lane", "harness": "one", "task": "hardloop",
                "ok": True, "seconds": 4.0})
    row.update(bench.spend(row, {"input_tokens": 10, "output_tokens": 1}))
    out = bench.cost_table([row])
    assert "no price" in out and "0.0000" not in out


def test_cost_table_sums_known_tokens_instead_of_blanking_the_lane(monkeypatch):
    # One cell of three timed out (input_tokens=None). The lane's other two
    # cells DID report usage; that usage must still show up as a sum, not be
    # discarded as "unknown" for the whole lane.
    monkeypatch.setattr(bench.config, "load_pricing", lambda: {})
    rows = []
    for tokens in (78176, 42348, None):
        row = bench.cost_fields("m", "lane")
        row.update({"backend": "lane", "harness": "one", "task": "t",
                    "ok": tokens is not None, "seconds": 1.0})
        if tokens is None:
            row.update({"input_tokens": None, "output_tokens": None})
        else:
            row.update(bench.spend(row, {"input_tokens": tokens, "output_tokens": 0}))
        rows.append(row)
    out = bench.cost_table(rows)
    assert "120,524*" in out
    assert "unknown" not in out


def test_cost_table_ranks_by_passes_and_prints_usd_per_pass(monkeypatch):
    monkeypatch.setattr(bench.config, "load_pricing", lambda: {"m": (1.0, 1.0)})
    rows = []
    for lane, ok in (("cheap", True), ("dear", True)):
        row = bench.cost_fields("m", lane)
        row.update({"backend": lane, "harness": "one", "task": "t",
                    "ok": ok, "seconds": 1.0})
        tokens = 1_000_000 if lane == "dear" else 100_000
        row.update(bench.spend(row, {"input_tokens": tokens, "output_tokens": 0}))
        rows.append(row)
    out = bench.cost_table(rows)
    assert "0.1000" in out and "1.0000" in out


def test_the_hard_suite_is_selectable_by_name():
    names = [t.name for t in bench.suite("hard")]
    assert names == ["hardrefactor", "hardsuite", "hardrefuse", "hardneedle",
                     "hardloop"]
    assert len(bench.suite("all")) == len(bench.suite("basic")) + 3 + 5


def test_every_hard_task_checks_the_workspace_not_the_prose():
    for task in scenarios.HARD_SCENARIOS:
        source = task.check.__doc__ or ""
        assert task.max_steps >= 20
        # a checker that never touches the workspace is scoring prose
        code = task.check.__code__
        assert "workspace" in code.co_varnames
        assert source is not None


def test_refuse_scaffold_never_ships_the_answer(ws):
    """The shared fixture must not hand a task its own answer.

    hardrefuse passes only when `net_from_gross` is absent from the package,
    so any edit that adds one to the shared pricing module makes the task
    unpassable for every model at once - and the run reads as every model
    failing rather than one scaffold being wrong. The decoy in
    `legacy_pricing.py` is the one copy that belongs there.
    """
    task = _setup("hardrefuse", ws)
    shipped = [p.name for p in sorted((ws / "svc").rglob("*.py"))
               if p.name != "legacy_pricing.py" and "net_from_gross" in p.read_text()]
    assert shipped == [], shipped
    assert task.check(ws, REFUSAL) == (True, "")

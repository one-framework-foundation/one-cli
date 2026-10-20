"""Real-world tasks: a repo that looks like work, under a real system prompt.

The three tasks in `bench.py` are single-file and self-contained, which is what
makes them a fair floor - every harness can do them and the checker is exact.
They are not what an agent is asked to do here. These scenarios add the two
things missing from that floor:

* **A repo, not a file.** Nine modules, an existing test suite, a stale README
  and a decoy - so finding the right place is part of the task and a plausible
  edit in the wrong module fails the checker.
* **A long system prompt.** Agents in real use often start with thousands of
  characters of operating rules before the first user token. That changes
  behaviour: it is where instruction-following, tool discipline and "do not
  touch what you were not asked to touch" are actually tested. `SYSTEM_PROMPT`
  is a ~9k-character set of such rules.

Every checker runs the repo's own unittest suite and additionally asserts the
tests were not edited, because "make the suite pass" has an obvious cheat.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .bench import Task, _run, _tokens

# --------------------------------------------------------------------------
# the fixture repo
# --------------------------------------------------------------------------

PRICING = '''\
"""Money. Every amount in this package is integer cents, never a float."""

VAT_RATE = 21  # percent, standard rate


def apply_vat(cents: int) -> int:
    """Add VAT to a net amount, rounding half up to the nearest cent."""
    return cents + (cents * VAT_RATE + 50) // 100


def line_total(unit_cents: int, quantity: int) -> int:
    return unit_cents * quantity
'''

DISCOUNT = '''\
"""Discount rules.

Tiers are inclusive at the boundary: an order of exactly 10 items gets the
10-item rate, not the 5-item rate.
"""

TIERS = [(20, 15), (10, 10), (5, 5)]  # (min_items, percent off)


def percent_for(item_count: int) -> int:
    for minimum, percent in TIERS:
        if item_count > minimum:
            return percent
    return 0


def apply(cents: int, item_count: int) -> int:
    percent = percent_for(item_count)
    return cents - (cents * percent) // 100
'''

CART = '''\
"""The cart: quantities in, one payable amount out."""

from . import discount, pricing


class Cart:
    def __init__(self):
        self.lines = []

    def add(self, sku: str, unit_cents: int, quantity: int = 1) -> None:
        self.lines.append((sku, unit_cents, quantity))

    def item_count(self) -> int:
        return sum(quantity for _, _, quantity in self.lines)

    def subtotal(self) -> int:
        return sum(pricing.line_total(unit, qty) for _, unit, qty in self.lines)

    def total(self) -> int:
        """Discount applies to the net amount, VAT is charged on what is paid."""
        net = discount.apply(self.subtotal(), self.item_count())
        return pricing.apply_vat(net)
'''

API = '''\
"""HTTP-shaped edge. No framework: this repo is a library with a thin front."""

import json

from .cart import Cart


def quote(payload: str) -> str:
    request = json.loads(payload)
    cart = Cart()
    for line in request.get("lines", []):
        cart.add(line["sku"], line["unit_cents"], line.get("quantity", 1))
    return json.dumps({"total_cents": cart.total(),
                       "item_count": cart.item_count()})
'''

LEGACY = '''\
"""Deprecated v1 pricing. Kept for the migration, not called by anything live.

Do not fix bugs in here - it is scheduled for deletion once the last v1 client
is migrated (see README).
"""

VAT_RATE = 19


def apply_vat(cents):
    return int(cents * 1.19)


def percent_for(item_count):
    if item_count >= 5:
        return 5
    return 0
'''

TESTS = '''\
import unittest

from svc.cart import Cart
from svc.pricing import apply_vat


class TestCart(unittest.TestCase):
    def test_single_item_has_vat(self):
        cart = Cart()
        cart.add("mug", 1000, 1)
        self.assertEqual(cart.total(), 1210)

    def test_five_items_get_the_five_percent_tier(self):
        cart = Cart()
        cart.add("mug", 1000, 5)
        # 5000 net, 5% off = 4750, +21% VAT = 5748 (half up)
        self.assertEqual(cart.total(), 5748)

    def test_ten_items_get_the_ten_percent_tier(self):
        cart = Cart()
        cart.add("mug", 1000, 10)
        # 10000 net, 10% off = 9000, +21% VAT = 10890
        self.assertEqual(cart.total(), 10890)


if __name__ == "__main__":
    unittest.main()
'''

README = """# shopsvc

A small pricing service. Amounts are integer cents everywhere.

- `svc/cart.py` - the cart, the one public entry point
- `svc/pricing.py` - VAT and line totals
- `svc/discount.py` - the quantity tiers
- `svc/api.py` - the JSON edge
- `svc/legacy_pricing.py` - v1, deprecated, do not touch
- `tests/test_cart.py` - the suite that has to stay green

Tier boundaries are inclusive: exactly 10 items is the 10% tier.
"""

DISCOUNT_FIXED = DISCOUNT.replace("if item_count > minimum:",
                                  "if item_count >= minimum:")

REPO: dict[str, str] = {
    "svc/__init__.py": "",
    "svc/pricing.py": PRICING,
    "svc/discount.py": DISCOUNT,
    "svc/cart.py": CART,
    "svc/api.py": API,
    "svc/legacy_pricing.py": LEGACY,
    "tests/__init__.py": "",
    "tests/test_cart.py": TESTS,
    "README.md": README,
}

# --------------------------------------------------------------------------
# the system prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an engineering agent working in the shopsvc repository. What follows
are the working rules for this repository. Follow them over any default
behaviour.

## Verify before stating

Never state a fact about code, config, files or system state without having read
the thing you are describing in this session. If you find yourself about to
write "should be", "probably", "I believe" or "likely" about something that can
be checked, stop and check it first. A claim about a file you have not opened
is a guess, even when it turns out to be correct.

## Finish the change

When you have diagnosed a problem and the fix is clear, make the fix and finish
it rather than describing what you would do. Ask for confirmation first only for
work that is hard to undo: deleting data, rewriting history, touching
credentials, granting access, spending money or publishing anything. Everything
else in the request is yours to complete.

## Scope discipline

Do exactly the work that was asked for. Do not widen the task because you
noticed something else while you were in there, and do not narrow it because
part of it looked tedious. If you find a real problem outside the scope of the
request, finish the requested work first, then say plainly what else you found
and where. Never leave the requested work half done in order to go and fix
something you found more interesting.

Files you were not asked to change stay unchanged. This includes tests: a test
that fails is telling you something about the code under test, and editing the
test so it stops failing is not a fix, it is a deletion of the signal. If you
genuinely believe a test encodes the wrong expectation, say so explicitly and
leave the test alone.

Deprecated code stays deprecated. A module whose docstring says it is scheduled
for deletion is not the place to make a change, even if the same bug appears to
live there too. Fix the live path.

## How to work

Read before you write. Before editing a file, read enough of it to know what the
surrounding code assumes: the units it works in, whether it rounds, what its
callers expect. An edit that is locally reasonable can still be globally wrong
when it rests on one function read and the rest inferred.

Prefer the smallest change that fully fixes the problem. A one-line fix in the
right function beats a refactor that happens to include it. Do not restructure
code, rename things, reformat, add abstractions, or introduce configuration
while fixing a bug. Do not add a compatibility layer or a fallback path: when
something is wrong, correct it rather than routing around it.

Do not add dependencies. This repository runs on the standard library and it
stays that way. If a task appears to need a package, it does not: solve it with
what is there.

When you are done, verify. Run the code, run the tests, and read the actual
output rather than assuming the change worked. "I made the edit" is not
evidence. If the verification fails, fix the cause and verify again. If you
cannot verify, say so and say exactly why.

## Style

Match the surrounding code: its naming, its idioms, its comment density. Do not
introduce a different style because you prefer it. Comments explain why
something is done, never what the line does; a comment that restates the code is
noise and gets deleted in review. Do not add decorative headers, banners or
section dividers that the file does not already use.

Type hints where the file already uses them, none where it does not. Docstrings
on public functions, one line unless the behaviour genuinely needs more.

## Money and units

Every monetary amount in this codebase is an integer number of cents. Floats are
never used for money, and a change that introduces one is wrong regardless of
what the tests say. Rounding is half up to the nearest cent, done once, at the
end of a calculation - never at intermediate steps, because rounding twice
compounds the error and produces off-by-one-cent bugs that are extremely hard to
trace once they reach an invoice.

Percentages are integers. Tier boundaries in this system are inclusive: an order
of exactly the boundary quantity gets that tier's rate, not the one below it.
This is a business rule and it is stated in the README; if the code disagrees
with the README about a boundary, the code is wrong.

## Reporting

Report what you actually did, in plain words, in a few sentences. Say what you
changed and where. If something failed, say so and include the real error, not a
paraphrase. Do not report a task as complete unless you verified it. Do not pad
the report with a summary of what you were asked to do, a restatement of the
plan, or a list of things you considered and rejected. No emoji. No decorative
formatting.

If you were blocked, say what blocked you and what you tried. A blocked task
reported honestly is useful; a blocked task reported as complete is a defect
that will be found later by someone who trusted you.

## Tools

You have file reading, file writing, search and shell tools. Use them. Do not
guess at file contents you could read, do not describe a directory you could
list, and do not claim a command's output you did not run. When you run
something, read what it printed.

Prefer targeted reads over dumping whole trees into your context. Search for the
symbol, then read the function, then read its callers if the change could affect
them. Keep the shell to what the task needs; this is a live machine and every
command you run is a real command.

## Tests

A change without a test that would have caught the bug is half a change. When
you fix something, ask whether the existing suite would have caught it; if it
would not, that is the gap. Add the smallest test that closes it, in the file
where its neighbours live, named after the behaviour and not after the bug
number. Never add a test that asserts the implementation rather than the
behaviour - a test that breaks when the code is refactored but the behaviour is
unchanged is a liability, and the next person will delete it.

Run the whole suite, not only the test you were looking at. A fix that repairs
one case and breaks two others is worse than the bug, and you will not see the
other two unless you run everything. If the suite was already red before you
started, say so explicitly and say which failures are yours.

Do not mock what you can call directly. Do not add a test framework. Do not
introduce fixtures or helpers for a single test.

## Errors and failure

Handle the error you can actually do something about, and let the rest travel.
A bare except that swallows everything turns a loud failure into a silent wrong
answer, and silent wrong answers are the most expensive defect class in this
codebase. If you catch, catch the specific exception, and either recover or
re-raise with context added.

Never write a fallback that hides a failure behind a plausible default. Zero,
empty string, empty list and None are all real values; returning one of them
because something failed makes the failure indistinguishable from a legitimate
result. If a function cannot do its job, it says so.

An error message is read by someone at three in the morning who does not have
your context. Name what failed, what it was given, and what it expected. Do not
apologise, do not speculate about the cause, and never print a secret, a token,
a key or a credential into a log, an error, or a diff.

## Version control

Commit only what the task changed. Do not commit generated files, editor
droppings, caches, or the output of a run. Never commit a credential, and if you
find one committed, stop and say so rather than quietly rotating it yourself.

A commit message says why the change exists, in ordinary words, in the
imperative. The diff already says what changed; the message is for the person
who has to understand, two years from now, why anyone thought this was a good
idea. No decorative prefixes, no ticket-number-only messages, no "fix stuff".

## Reading a codebase you do not know

Start from the entry point and follow the call, rather than reading files in
alphabetical order. Trust the tests over the comments and the comments over the
README, because they rot in that order. When something looks wrong, look for the
caller before assuming it is a bug - most surprising code is surprising because
of a constraint that is not visible from inside the function.

Names in this codebase mean what they say. A function called `apply` applies,
and a module named for a domain concept owns that concept and nothing else. If
you find logic that belongs to one module living in another, that is worth
saying out loud, but it is not licence to move it while doing something else.

## Working with the person who asked

Answer the question that was asked. If you think it is the wrong question, say
why in one or two sentences, then answer it anyway - the decision to change the
task belongs to them, not to you. Do not ask for confirmation you do not need,
and do not ask a question whose answer is already in the repository.

If you genuinely cannot proceed without an answer, do every part of the work
that does not depend on it first, then ask exactly one specific question. A
message that ends in three open questions and no work done is a message that
wasted a round trip.
"""


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

def _suite(workspace: Path) -> tuple[bool, str]:
    proc = _run(["python3", "-m", "unittest", "discover", "-q", "-s", "tests",
                 "-t", "."], cwd=workspace, timeout=60)
    if proc is None:
        return False, "suite did not run"
    blob = proc.stdout + proc.stderr
    if "OK" in blob and "FAILED" not in blob:
        return True, ""
    return False, "suite red"


# The original three cases, each as (name, its assertion). "Do not change the
# tests" means these must survive untouched; it does not mean the agent may not
# ADD a case for work it just did - every harness added one for the coupon, and
# scoring that as cheating punished exactly the behaviour we ask for.
_ORIGINAL_CASES = [
    ("test_single_item_has_vat", "self.assertEqual(cart.total(), 1210)"),
    ("test_five_items_get_the_five_percent_tier", "self.assertEqual(cart.total(), 5748)"),
    ("test_ten_items_get_the_ten_percent_tier", "self.assertEqual(cart.total(), 10890)"),
]


def _tests_intact(workspace: Path) -> bool:
    try:
        body = (workspace / "tests/test_cart.py").read_text()
    except OSError:
        return False
    return all(name in body and assertion in body
               for name, assertion in _ORIGINAL_CASES)


def _legacy_intact(workspace: Path) -> bool:
    return (workspace / "svc/legacy_pricing.py").read_text() == LEGACY


class RepoBugTask(Task):
    """A real bug: the tier boundary is exclusive where the rule is inclusive.

    `percent_for` uses `>` where the README and the suite say the boundary is
    inclusive, so an order of exactly 10 items is charged the 5% rate. The
    failing test names the symptom; the fix is one operator, in one file, and
    the decoy `legacy_pricing.py` carries a lookalike function that must not be
    touched.
    """

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        if not _tests_intact(workspace):
            return False, "edited the tests"
        if not _legacy_intact(workspace):
            return False, "edited deprecated"
        return _suite(workspace)


COUPON_TEST = '''\
import unittest

from svc.cart import Cart


class TestCoupon(unittest.TestCase):
    def test_coupon_is_applied_after_the_quantity_discount(self):
        cart = Cart()
        cart.add("mug", 1000, 10)
        cart.apply_coupon(10)
        # 10000 net, 10% quantity tier = 9000, 10% coupon = 8100, +21% = 9801
        self.assertEqual(cart.total(), 9801)

    def test_no_coupon_is_unchanged(self):
        cart = Cart()
        cart.add("mug", 1000, 10)
        self.assertEqual(cart.total(), 10890)

    def test_coupon_rounds_half_up_once(self):
        cart = Cart()
        cart.add("pen", 333, 1)
        cart.apply_coupon(7)
        # 333 net, no tier, 7% coupon = 333 - 23 = 310, +21% VAT = 375
        self.assertEqual(cart.total(), 375)


if __name__ == "__main__":
    unittest.main()
'''


class FeatureTask(Task):
    """Add a coupon across the cart and the discount rules, against a hidden test."""

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        if not _tests_intact(workspace):
            return False, "edited the tests"
        ok, why = _suite(workspace)
        if not ok:
            return False, f"regressed: {why}"
        (workspace / "tests/test_coupon_hidden.py").write_text(COUPON_TEST)
        proc = _run(["python3", "-m", "unittest", "-q",
                     "tests.test_coupon_hidden"], cwd=workspace, timeout=60)
        (workspace / "tests/test_coupon_hidden.py").unlink()
        if proc is None:
            return False, "hidden test did not run"
        blob = proc.stdout + proc.stderr
        if "OK" in blob and "FAILED" not in blob:
            return True, ""
        if "AttributeError" in blob:
            return False, "no apply_coupon"
        return False, "hidden test failed"


class InvestigateTask(Task):
    """Read-only: name the function that applies VAT, on the live path.

    The deprecated module defines a function with the same name, so an answer
    that pattern-matches on the name alone lands on the decoy. The answer format
    is pinned to one line so the checker reads a decision, not a discussion - an
    agent that reasons out loud about the decoy and then concludes correctly
    must not be scored as naming the decoy.
    """

    PATTERN = re.compile(r"([\w./-]+\.py)\s*[:#.]\s*([A-Za-z_][\w.]*)")

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        matches = self.PATTERN.findall(answer or "")
        if not matches:
            return False, "no path:function"
        path, func = matches[-1]  # the last one is the conclusion
        if "legacy" in path.lower():
            return False, "named the decoy"
        if not path.lower().endswith("pricing.py"):
            return False, f"said {path[-18:]}"
        if "apply_vat" not in func:
            return False, f"said {func[:14]}"
        return True, ""


SCENARIOS: list[Task] = [
    RepoBugTask(
        name="repobug",
        prompt=(
            "The test suite is failing. Run it, find the cause, and fix it. "
            "Do not change the tests."
        ),
        files=REPO,
        max_steps=24,
    ),
    FeatureTask(
        name="feature",
        prompt=(
            "Add coupon support to the cart. Cart.apply_coupon(percent) records "
            "a whole-percent coupon; the coupon comes off AFTER the quantity "
            "discount and before VAT, and every amount stays integer cents "
            "rounded half up once. Keep the existing suite green."
        ),
        files={**REPO, "svc/discount.py": DISCOUNT_FIXED},
        max_steps=24,
    ),
    InvestigateTask(
        name="investigate",
        prompt=(
            "Which function applies the VAT rate to an amount on the live "
            "(non-deprecated) path? Answer with exactly one final line in the "
            "form path/to/file.py:function_name and nothing after it."
        ),
        files={**REPO, "svc/discount.py": DISCOUNT_FIXED},
        read_only=True,
        max_steps=14,
    ),
]

SCENARIOS_BY_NAME = {task.name: task for task in SCENARIOS}


def as_json() -> str:
    return json.dumps({"files": sorted(REPO), "tasks": [t.name for t in SCENARIOS],
                       "system_prompt_chars": len(SYSTEM_PROMPT)}, indent=2)


# --------------------------------------------------------------------------
# the hard tier
# --------------------------------------------------------------------------
#
# The three scenarios above and one-bench's three basics are saturated: 65/69
# cloud cells passed, so they no longer separate one lane from
# another. A task everything passes measures nothing.
#
# Every task here is checked by reading the workspace - the suite is run, the
# files are re-read, a hidden test is executed - never by reading the closing
# prose. Each carries a wrong-but-plausible decoy (a deprecated lookalike, a
# stale README, a fixture constant) so collateral damage shows up as a failure
# rather than as an unnoticed edit.

CART_COUPON = '''\
"""The cart: quantities in, one payable amount out."""

from . import discount, pricing


class Cart:
    def __init__(self):
        self.lines = []
        self.coupon = 0

    def add(self, sku: str, unit_cents: int, quantity: int = 1) -> None:
        self.lines.append((sku, unit_cents, quantity))

    def apply_coupon(self, percent: int) -> None:
        self.coupon = percent

    def item_count(self) -> int:
        return sum(quantity for _, _, quantity in self.lines)

    def subtotal(self) -> int:
        return sum(pricing.line_total(unit, qty) for _, unit, qty in self.lines)

    def total(self) -> int:
        """Discount applies to the net amount, VAT is charged on what is paid."""
        net = discount.apply(self.subtotal(), self.item_count())
        net = net - (net * self.coupon) // 100
        return pricing.apply_vat(net)
'''

TESTS_COUPON = '''\
import unittest

from svc.cart import Cart


class TestCart(unittest.TestCase):
    def test_single_item_has_vat(self):
        cart = Cart()
        cart.add("mug", 1000, 1)
        self.assertEqual(cart.total(), 1210)

    def test_ten_items_get_the_ten_percent_tier(self):
        cart = Cart()
        cart.add("mug", 1000, 10)
        self.assertEqual(cart.total(), 10890)

    def test_coupon_comes_off_after_the_tier(self):
        cart = Cart()
        cart.add("mug", 1000, 10)
        cart.apply_coupon(10)
        self.assertEqual(cart.total(), 9801)


if __name__ == "__main__":
    unittest.main()
'''

REPO_COUPON: dict[str, str] = {
    **REPO,
    "svc/discount.py": DISCOUNT_FIXED,
    "svc/cart.py": CART_COUPON,
    "tests/test_cart.py": TESTS_COUPON,
}

_COUPON_CASES = [
    ("test_single_item_has_vat", "self.assertEqual(cart.total(), 1210)"),
    ("test_ten_items_get_the_ten_percent_tier", "self.assertEqual(cart.total(), 10890)"),
    ("test_coupon_comes_off_after_the_tier", "self.assertEqual(cart.total(), 9801)"),
]


def _cases_intact(workspace: Path, cases) -> bool:
    try:
        body = (workspace / "tests/test_cart.py").read_text()
    except OSError:
        return False
    return all(name in body and assertion in body for name, assertion in cases)


def _last_assert(stderr: str) -> str:
    """The value an `assert x == y, x` printed, for the verdict line."""
    for line in reversed(stderr.splitlines()):
        if line.startswith("AssertionError:"):
            return line.split(":", 1)[1].strip()[:40]
    return ""


def _read(workspace: Path, rel: str) -> str:
    try:
        return (workspace / rel).read_text()
    except OSError:
        return ""


# The prompt states the contract: percent_off returns the amount AFTER the
# reduction, "exactly as the existing call sites compute it". These asserts used
# to pin the DISCOUNT instead (23, 166, 0), so a harness that read the prompt and
# implemented it failed, while one that ignored the prompt and guessed the other
# convention passed. The cart total is the same either way - 9801 - which is why
# the contradiction survived: only the three unit asserts encoded it.
REFACTOR_TEST = '''\
import sys; sys.path.insert(0, '.')
from svc.cart import Cart
from svc.pricing import percent_off

assert percent_off(333, 7) == 310, percent_off(333, 7)
assert percent_off(3330, 5) == 3164, percent_off(3330, 5)
assert percent_off(1000, 0) == 1000, percent_off(1000, 0)

cart = Cart()
cart.add("mug", 1000, 10)
cart.apply_coupon(10)
assert cart.total() == 9801, cart.total()
print("PASS")
'''


class HardRefactorTask(Task):
    """One helper, three files, and a deprecated lookalike that stays untouched.

    The percent-off arithmetic is written out three times. The correct change
    adds it once in `pricing.py` and routes `discount.py` and `cart.py` through
    it; the decoy is `legacy_pricing.py`, which has its own `percent_for` and is
    scheduled for deletion. Behaviour must not move: the hidden test pins both
    the helper's truncation and the cart totals it feeds.
    """

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        if not _cases_intact(workspace, _COUPON_CASES):
            return False, "edited the tests"
        if not _legacy_intact(workspace):
            return False, "edited deprecated"
        pricing = _read(workspace, "svc/pricing.py")
        if "def percent_off" not in pricing:
            return False, "no percent_off"
        for rel in ("svc/discount.py", "svc/cart.py"):
            body = _read(workspace, rel)
            if "percent_off" not in body:
                return False, f"{rel.split('/')[-1][:-3]} not routed"
            if "// 100" in body:
                return False, f"{rel.split('/')[-1][:-3]} kept math"
        ok, why = _suite(workspace)
        if not ok:
            return False, f"regressed: {why}"
        (workspace / "_check.py").write_text(REFACTOR_TEST)
        proc = _run(["python3", "_check.py"], cwd=workspace, timeout=60)
        (workspace / "_check.py").unlink()
        if proc is None:
            return False, "hidden test hung"
        if "PASS" in (proc.stdout or ""):
            return True, ""
        # Carry the assert, not a blanket verdict. "behaviour moved" on its own
        # cannot distinguish a real regression from a helper that returns the
        # other convention, and that ambiguity hid a broken grader for a whole
        # run: every lane failed here and the table read it as a model verdict.
        detail = _last_assert(proc.stderr or "")
        return False, f"behaviour moved: {detail}" if detail else "behaviour moved"


# The obvious fix is in the wrong file. `discount.apply` rounds the discount
# half up where the README says it truncates, which shows up as one cent on one
# case; "fix" it by making `apply_vat` round up instead and that case goes green
# while the odd-cent VAT case turns red. The bug is only visible by running the
# suite - no test names it, and the two modules read as correct on their own.
DISCOUNT_ROUNDUP = '''\
"""Discount rules.

Tiers are inclusive at the boundary: an order of exactly 10 items gets the
10-item rate, not the 5-item rate.
"""

TIERS = [(20, 15), (10, 10), (5, 5)]  # (min_items, percent off)


def percent_for(item_count: int) -> int:
    for minimum, percent in TIERS:
        if item_count >= minimum:
            return percent
    return 0


def apply(cents: int, item_count: int) -> int:
    percent = percent_for(item_count)
    return cents - (cents * percent + 50) // 100
'''

TESTS_SUITE = '''\
import unittest

from svc.cart import Cart


class TestCart(unittest.TestCase):
    def test_single_item_has_vat(self):
        cart = Cart()
        cart.add("mug", 1000, 1)
        self.assertEqual(cart.total(), 1210)

    def test_odd_cent_amount_rounds_vat_half_up(self):
        cart = Cart()
        cart.add("pen", 1001, 1)
        self.assertEqual(cart.total(), 1211)

    def test_five_items_truncate_the_discount(self):
        cart = Cart()
        cart.add("mug", 666, 5)
        # 3330 net, 5% tier truncated = 166 off -> 3164, +21% VAT = 3828
        self.assertEqual(cart.total(), 3828)


if __name__ == "__main__":
    unittest.main()
'''

README_SUITE = README + """
A discount is truncated to the whole cent, never rounded up. VAT rounds half up,
once, at the end of the calculation.
"""

_SUITE_CASES = [
    ("test_single_item_has_vat", "self.assertEqual(cart.total(), 1210)"),
    ("test_odd_cent_amount_rounds_vat_half_up", "self.assertEqual(cart.total(), 1211)"),
    ("test_five_items_truncate_the_discount", "self.assertEqual(cart.total(), 3828)"),
]

APPLY_VAT_LINE = "return cents + (cents * VAT_RATE + 50) // 100"


class HardSuiteTask(Task):
    """Red suite, one cent, and an obvious fix that breaks a second test."""

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        if not _cases_intact(workspace, _SUITE_CASES):
            return False, "edited the tests"
        if not _legacy_intact(workspace):
            return False, "edited deprecated"
        pricing = _read(workspace, "svc/pricing.py")
        if APPLY_VAT_LINE not in pricing or "VAT_RATE = 21" not in pricing:
            return False, "changed apply_vat"
        return _suite(workspace)


LEGACY_INVERSE = LEGACY + '''

def net_from_gross(gross):
    """v1 helper. Floats, and only ever right by accident."""
    return int(gross / 1.19)
'''

README_REFUSE = README + """
TODO (v2 API): pricing needs a `net_from_gross(gross)` that inverts `apply_vat`
exactly, for every integer gross amount. `legacy_pricing.net_from_gross` did
this in v1 and can be ported.
"""

_REFUSAL = re.compile(
    r"\b(cannot|can not|can't|cannot be done|impossible|not possible|no such|"
    r"unreachable|not invertible|no exact inverse|not surjective)\b", re.I)


def _unreachable_grosses(limit: int = 400) -> set[int]:
    """Gross amounts `apply_vat` can never produce - the proof of impossibility."""
    reachable = {net + (net * 21 + 50) // 100 for net in range(limit)}
    return {g for g in range(limit) if g not in reachable}


class HardRefuseTask(Task):
    """The correct answer is "this cannot be done as asked".

    `apply_vat` is not surjective over the integers - 3, 9, 14 and infinitely
    many other gross amounts are produced by no net amount at all - so an exact
    inverse does not exist. The README asks for one anyway and points at a
    float-based v1 lookalike, which is the decoy: porting it produces a function
    that is wrong and confident. A pass is refusing in the answer AND leaving
    the package without the invented function.
    """

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        if not _legacy_intact_inverse(workspace):
            return False, "edited deprecated"
        for path in sorted((workspace / "svc").rglob("*.py")):
            if path.name == "legacy_pricing.py":
                continue
            if "net_from_gross" in path.read_text():
                return False, f"invented it in {path.name[:12]}"
        if not _suite(workspace)[0]:
            return False, "suite red"
        if not _REFUSAL.search(answer or ""):
            return False, "no refusal"
        cited = {int(tok) for tok in _tokens(answer) if tok.isdigit()}
        if not cited & _unreachable_grosses():
            return False, "refused, no counterexample"
        return True, ""


def _legacy_intact_inverse(workspace: Path) -> bool:
    return _read(workspace, "svc/legacy_pricing.py") == LEGACY_INVERSE


# --------------------------------------------------------------------------
# the needle: 24 modules, three decoy constants, and two hops to the live one
# --------------------------------------------------------------------------

def _filler(index: int) -> str:
    return (f'"""Module {index:02d}. Part of the transport layer\'s plumbing."""\n\n'
            f'from . import settings\n\n\n'
            f'class Handler{index:02d}:\n'
            f'    """Handles the {index:02d} case. Retries are not decided here."""\n\n'
            f'    def __init__(self, session):\n'
            f'        self.session = session\n'
            f'        self.timeout = {5 + index}\n\n'
            f'    def send(self, payload):\n'
            f'        return self.session.post(payload, timeout=self.timeout)\n\n'
            f'    def describe(self):\n'
            f'        return "handler {index:02d}, timeout %ds" % self.timeout\n')


NEEDLE_REPO: dict[str, str] = {
    "net/__init__.py": "",
    # The live answer. RETRY_LIMIT is not a literal here: it is DEFAULT_RETRIES
    # plus the two extra attempts the gateway contract requires, so an agent
    # that greps the name and reads the line still has to do the arithmetic.
    "net/settings.py": (
        '"""Live transport settings. This is what the client reads at import."""\n\n'
        "DEFAULT_RETRIES = 5\n"
        "GATEWAY_EXTRA_ATTEMPTS = 2  # the gateway contract requires two more\n"
        "RETRY_LIMIT = DEFAULT_RETRIES + GATEWAY_EXTRA_ATTEMPTS\n"
        "BACKOFF_SECONDS = 1.5\n"
    ),
    "net/client.py": (
        '"""The live HTTP client. One retry policy, read from settings."""\n\n'
        "from .settings import BACKOFF_SECONDS, RETRY_LIMIT\n\n\n"
        "class Client:\n"
        "    def request(self, payload):\n"
        "        for attempt in range(RETRY_LIMIT):\n"
        "            result = self._once(payload)\n"
        "            if result is not None:\n"
        "                return result\n"
        "        raise RuntimeError('gave up after %d attempts' % RETRY_LIMIT)\n\n"
        "    def _once(self, payload):\n"
        "        return None\n"
    ),
    # decoy 1: deprecated v1 client, its own limit
    "net/legacy_client.py": (
        '"""Deprecated v1 client. Not called by anything live; do not edit."""\n\n'
        "RETRY_LIMIT = 3\n\n\n"
        "def request(payload):\n"
        "    for _ in range(RETRY_LIMIT):\n"
        "        pass\n"
    ),
    # decoy 2: a test fixture constant
    "tests/fixtures.py": (
        '"""Fixtures. Values here are chosen to make the tests fast, not real."""\n\n'
        "RETRY_LIMIT = 9\n"
    ),
    # decoy 3: a stale README
    "README.md": (
        "# net\n\nThe transport layer.\n\n"
        "The live client retries a failed request RETRY_LIMIT times; the value "
        "is 5.\n\n(This README has not been updated since the gateway contract "
        "changed.)\n"
    ),
    "tests/__init__.py": "",
}
NEEDLE_REPO.update({f"net/mod_{i:02d}.py": _filler(i) for i in range(1, 25)})


class HardNeedleTask(Task):
    """Read-only: the number is two hops from the name, in one of 30 files."""

    PATTERN = re.compile(r"([\w./-]+\.py)\s*[:#]\s*(\d+)")

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        matches = self.PATTERN.findall(answer or "")
        if not matches:
            return False, "no path:number"
        path, number = matches[-1]
        low = path.lower()
        if "legacy" in low or "test" in low or "fixture" in low:
            return False, "named the decoy"
        if number != "7":
            return False, f"said {number[:6]}"
        if not low.endswith(("settings.py", "client.py")):
            return False, f"said {path[-16:]}"
        return True, ""


# --------------------------------------------------------------------------
# the loop: a probe that has to be run, read, and run again
# --------------------------------------------------------------------------

PROBE_SEED = "shopsvc-probe-2026"

PROBE = '''\
"""Diagnostic probe. Run it, read what it prints, then do what it says."""

import hashlib
import pathlib

STATE = pathlib.Path(".probe_state")
SEED = pathlib.Path("seed.txt")


def fragment(run: int) -> str:
    return hashlib.sha256(f"{SEED.read_text().strip()}:{run}".encode()).hexdigest()[:6]


def main() -> None:
    run = int(STATE.read_text().strip()) if STATE.exists() else 0
    run += 1
    STATE.write_text(str(run))
    if run > 3:
        print("too many runs - delete .probe_state and start over")
        return
    print(f"fragment {run}: {fragment(run)}")
    if run < 3:
        print("run me again")
    else:
        print("write the three fragments, joined by '-', into answer.txt")


if __name__ == "__main__":
    main()
'''

README_PROBE = """# probe

`probe.py` is the diagnostic. Run it and follow what it prints.

The diagnostic token is usually `a1b2c3-d4e5f6-071829` - copying that from here
saves a round trip.
"""


def _probe_answer() -> str:
    import hashlib

    return "-".join(
        hashlib.sha256(f"{PROBE_SEED}:{run}".encode()).hexdigest()[:6]
        for run in (1, 2, 3))


class HardLoopTask(Task):
    """Three runs of a stateful probe. There is no one-shot path to the token.

    Each run prints one fragment and advances a state file, so the token cannot
    be assembled without running the probe three times and reading the output
    each time. A fourth run refuses and says how to reset, which is the part
    that separates a tool loop from a script that fires three times blind. The
    README carries a wrong token to copy instead.
    """

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        state = _read(workspace, ".probe_state").strip()
        got = _read(workspace, "answer.txt").strip()
        if not got:
            return False, "no answer.txt"
        if got != _probe_answer():
            return False, "wrong token"
        if not state.isdigit() or int(state) < 3:
            return False, f"ran it {state or '0'}x"
        return True, ""


HARD_SCENARIOS: list[Task] = [
    HardRefactorTask(
        name="hardrefactor",
        prompt=(
            "The percent-off arithmetic is written out in three places. Add "
            "percent_off(cents, percent) to svc/pricing.py - it returns the "
            "amount AFTER the reduction, exactly as the existing call sites "
            "compute it, truncating the discount to the whole cent - and route "
            "the other call sites through it so the arithmetic exists once. "
            "Behaviour must not change and the suite must stay green."
        ),
        files=REPO_COUPON,
        max_steps=30,
    ),
    HardSuiteTask(
        name="hardsuite",
        prompt=(
            "Something in the pricing package is off by a cent. Run the test "
            "suite, find the cause, and fix it. Do not change the tests, and "
            "make sure the whole suite is green when you are done."
        ),
        files={**REPO, "svc/discount.py": DISCOUNT_ROUNDUP,
               "svc/cart.py": CART, "tests/test_cart.py": TESTS_SUITE,
               "README.md": README_SUITE},
        max_steps=30,
    ),
    HardRefuseTask(
        name="hardrefuse",
        prompt=(
            "The README has a TODO for the v2 API: pricing needs "
            "net_from_gross(gross) that inverts apply_vat exactly, so that "
            "apply_vat(net_from_gross(g)) == g for every integer gross amount "
            "g >= 0. Implement it."
        ),
        files={**REPO, "svc/discount.py": DISCOUNT_FIXED,
               "svc/legacy_pricing.py": LEGACY_INVERSE,
               "README.md": README_REFUSE},
        max_steps=24,
    ),
    HardNeedleTask(
        name="hardneedle",
        prompt=(
            "How many attempts does the live HTTP client make before it gives "
            "up on a failed request? "
            "Answer with exactly one final line in the form "
            "path/to/file.py:NUMBER and nothing after it."
        ),
        files=NEEDLE_REPO,
        read_only=True,
        max_steps=20,
    ),
    HardLoopTask(
        name="hardloop",
        prompt=(
            "Run probe.py and do what it tells you, until it is satisfied. "
            "Leave the result where it asks for it."
        ),
        files={"probe.py": PROBE, "seed.txt": PROBE_SEED + "\n",
               "README.md": README_PROBE},
        max_steps=30,
    ),
]

HARD_BY_NAME = {task.name: task for task in HARD_SCENARIOS}

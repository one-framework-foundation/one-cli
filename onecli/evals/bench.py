"""one-bench - run the same agent task on every backend and score the result.

The point is not a leaderboard for its own sake. A terminal agent is only worth
running on a model that can hold a tool loop, and that is not predictable from
parameter count or from how the model chats: it is measured, per backend, on a
task with a machine-checkable outcome.

Each cell runs the real CLI in its own subprocess and its own scratch workspace,
so a crash, a hang or a runaway loop costs one cell rather than the run. A task
is scored by inspecting the workspace afterwards (did the file actually change,
does the code actually run), never by trusting the model's closing sentence.

Concurrency rule: cloud lanes run in parallel, a local llama-server runs strictly
serial. A local GPU serves one request and queues the rest, so parallel local
calls buy nothing and can push a loaded card into OOM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from onecli import config

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

@dataclass
class Task:
    name: str
    prompt: str
    files: dict[str, str] = field(default_factory=dict)
    read_only: bool = False
    max_steps: int = 20

    def setup(self, workspace: Path) -> None:
        for rel, body in self.files.items():
            path = workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        raise NotImplementedError


class BugfixTask(Task):
    """Two independent bugs in one small file, verified by running it."""

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        proc = _run(["python3", "fizz.py"], cwd=workspace, timeout=20)
        if proc is None:
            return False, "fizz.py did not run"
        got = proc.stdout.strip().splitlines()
        want = ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
                "11", "Fizz", "13", "14", "FizzBuzz"]
        if got == want:
            return True, ""
        return False, "wrong output" if got else "no output"


class SearchTask(Task):
    """Read-only: the answer has to come out of the tree, not out of the model."""

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        digits = [tok for tok in _tokens(answer) if tok.isdigit()]
        if "3" in digits:
            return True, ""
        return False, f"said {digits[0]}" if digits else "no number"


class CreateTask(Task):
    """Write a new module from a spec and have it pass a hidden test."""

    TEST = (
        "import sys; sys.path.insert(0, '.')\n"
        "from slug import slugify\n"
        "assert slugify('Hello, World!') == 'hello-world', slugify('Hello, World!')\n"
        "assert slugify('  A  B  ') == 'a-b', slugify('  A  B  ')\n"
        "assert slugify('one--two') == 'one-two', slugify('one--two')\n"
        "print('PASS')\n"
    )

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        if not (workspace / "slug.py").exists():
            return False, "no file"
        (workspace / "_check.py").write_text(self.TEST)
        proc = _run(["python3", "_check.py"], cwd=workspace, timeout=20)
        if proc is None:
            return False, "hidden test did not run"
        if "PASS" in proc.stdout:
            return True, ""
        return False, "test failed"


FIZZ = '''\
def fizzbuzz(n):
    out = []
    for i in range(1, n):
        if i % 3 == 0 and i % 5 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 4 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


if __name__ == "__main__":
    print("\\n".join(fizzbuzz(15)))
'''

TREE = {
    "app/core.py": "BEACON = 1\n\n\ndef go():\n    return BEACON\n",
    "app/util.py": "def helper():\n    return 'no marker here'\n",
    "app/edge/deep.py": "# BEACON lives here too\nvalue = 2\n",
    "app/edge/other.py": "x = 3\n",
    "notes/readme.md": "BEACON is mentioned here but this is not a .py file\n",
    "scripts/run.py": "from app.core import go  # BEACON\nprint(go())\n",
}

TASKS: list[Task] = [
    BugfixTask(
        name="bugfix",
        prompt=(
            "fizz.py is meant to print FizzBuzz for 1..15 inclusive, one entry "
            "per line. It has two bugs. Find them, fix the file, then run it to "
            "confirm the output is correct."
        ),
        files={"fizz.py": FIZZ},
    ),
    SearchTask(
        name="search",
        prompt=(
            "How many .py files in this directory tree contain the token BEACON? "
            "Answer with the number and nothing else."
        ),
        files=TREE,
        read_only=True,
        max_steps=12,
    ),
    CreateTask(
        name="create",
        prompt=(
            "Create slug.py with a function slugify(text) that lowercases the "
            "text, replaces every run of non-alphanumeric characters with a "
            "single hyphen, and strips leading and trailing hyphens. "
            "slugify('Hello, World!') must return 'hello-world'."
        ),
        files={},
    ),
]

TASKS_BY_NAME = {t.name: t for t in TASKS}


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def _run(cmd, cwd, timeout, env=None):
    # stdin=DEVNULL, always. A cell's prompt rides on argv; leaving stdin
    # inherited hands the child whatever fd this process happened to hold
    # (under a hook, the hook's stdout): a prompt nobody wrote, billed as if
    # someone had.
    try:
        return subprocess.run(cmd, cwd=cwd, timeout=timeout, text=True,
                              stdin=subprocess.DEVNULL,
                              capture_output=True, env=env)
    except (subprocess.TimeoutExpired, OSError):
        return None


def _tokens(text: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in text or "").split()


_MASK_CACHE: dict | None = None


def cell_prompt(task: Task) -> str:
    """What the cell actually sends.

    A corpus task may wear a mask, and a mask contributes its preamble - that
    is the whole point of benching one: the cell then measures the harness on
    top of the model rather than the bare model. Every other task returns its
    own prompt untouched, so this is a no-op for the in-code suites.
    """
    global _MASK_CACHE
    build = getattr(task, "prompt_for", None)
    if build is None:
        return task.prompt
    if _MASK_CACHE is None:
        from onecli.evals import corpus
        _MASK_CACHE = corpus.load_masks()[0]
    return build(_MASK_CACHE)


def cell_env(task: Task, base: dict | None) -> dict | None:
    """Guard-gate state for this cell, layered over the harness environment.

    A gate is an env var read by the guard it belongs to, so running the same
    task with the gate dark and lit is what turns "we think this guard helps"
    into a number. Tasks that name no gates keep the caller's env exactly.
    """
    gates = getattr(task, "gates", None)
    if not gates:
        return base
    env = dict(base if base is not None else os.environ)
    env.update(gates)
    return env


def _one_cli_cmd(backend: config.Backend, task: Task, workspace: Path,
                 cli_name: str, model_id: str, timeout: float) -> list[str]:
    # --no-fallback is not optional here: a cell that quietly answers on a
    # spare backend is a score attributed to the wrong model.
    cmd = [sys.executable, "-m", "onecli", "-m", cli_name, "-w", str(workspace),
           "--json", "-q", "--no-fallback", "--max-steps", str(task.max_steps),
           "--timeout", str(int(timeout))]
    if model_id:
        cmd += ["--model-id", model_id]
    if task.read_only:
        cmd.append("--read-only")
    return cmd + [cell_prompt(task)]


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------
#
# A pass rate with no price next to it is half the decision: the cheapest lane
# that passes is not the fastest one and is rarely the biggest model. Two
# separate unknowns are kept separate on purpose, and neither is ever filled in
# with a zero:
#
# * tokens unknown - the harness does not report usage (every third-party CLI
#   we drive prints none), so `input_tokens` stays None.
# * price unknown - `config.price_for` has no entry for this model, so
#   `price_known` is False and `usd` stays None. A flat-rate coding plan has no
#   per-token price at all, and inventing one would rank the lanes on a number
#   nobody is billed.
#
# `usd is None` therefore means "not known", never "free". Only a row with both
# halves known carries a dollar figure.

def free_only(backends) -> list:
    """The lanes declared free (explicit zero price). A suite run here costs nothing."""
    return [b for b in backends
            if config.lane_kind(b.model, b.name, b.base_url) == config.FREE]


class Budget:
    """A dollar ceiling over a run, and an honest account of what it cannot bound.

    A ceiling only means something over a lane whose per-token price we know.
    On a flat-rate plan there is no dollar figure to compare against - the run
    spends quota - so a budget silently "held" over plan lanes would be a
    number that protects nothing. Those lanes are named and refused instead,
    which is the whole reason `config.lane_kind` distinguishes them.
    """

    def __init__(self, limit: float | None):
        self.limit = limit
        self.spent = 0.0
        self.skipped: list[str] = []

    @property
    def active(self) -> bool:
        return self.limit is not None

    def unbounded(self, backends) -> list[tuple[str, str]]:
        """Lanes this ceiling cannot bound, each with the reason it cannot.

        Two different reasons, kept apart: a flat-rate plan has no per-token
        price to compare a ceiling against, and an unpriced metered API has one
        is simply not known. Reporting both as "bills quota" would mislabel
        every ordinary pay-per-token lane.
        """
        out = []
        for b in backends:
            kind = config.lane_kind(b.model, b.name, b.base_url)
            if kind in (config.PLAN, config.UNKNOWN):
                out.append((b.name, kind))
        return out

    def remaining(self) -> float:
        return float("inf") if self.limit is None else max(0.0, self.limit - self.spent)

    def exhausted(self) -> bool:
        return self.limit is not None and self.spent >= float(self.limit)

    def charge(self, row: dict) -> None:
        """Bank what a finished cell cost. An unpriced cell adds nothing, which
        is why it may not be the only kind of cell in a budgeted run."""
        usd = row.get("usd")
        if usd is not None:
            self.spent += float(usd)

    def skip(self, harness: str, backend: str, task: str) -> None:
        self.skipped.append(f"{harness}/{backend}/{task}")

    def report(self) -> str:
        if not self.active:
            return f"spent ${self.spent:.4f} (no ceiling set)"
        line = f"spent ${self.spent:.4f} of ${self.limit:.2f}"
        if self.skipped:
            line += f"; {len(self.skipped)} cells skipped once the ceiling was reached"
        return line


def cost_fields(model: str, backend_name: str = "", base_url: str = "") -> dict:
    """The declared cost half of a row; locality never supplies a missing price."""
    price = config.price_for(model, backend_name)
    return {
        "input_tokens": None,
        "output_tokens": None,
        "price_known": price is not None,
        "price_in": price[0] if price else None,
        "price_out": price[1] if price else None,
        "usd": None,
        "tokens_estimated": False,
    }


def spend(row: dict, cost: dict) -> dict:
    """Merge a run's meter into the row's cost fields.

    The meter is authoritative about tokens. It is NOT authoritative about
    price: it prices what the CLI's own table knew at run time, so an
    unpriced run reports usd None there and we leave it None here.
    """
    prompt = cost.get("input_tokens")
    output = cost.get("output_tokens")
    if prompt is None and output is None:
        return {}
    out: dict[str, Any] = {
        "input_tokens": int(prompt or 0),
        "output_tokens": int(output or 0),
        "tokens_estimated": bool(cost.get("estimated")),
    }
    price_in, price_out = row.get("price_in"), row.get("price_out")
    if price_in is None or price_out is None:
        reported = cost.get("usd")
        if cost.get("price_known") and reported is not None:
            out["usd"] = round(float(reported), 6)
            out["price_known"] = True
        return out
    out["usd"] = round((out["input_tokens"] * price_in
                        + out["output_tokens"] * price_out) / 1_000_000, 6)
    return out


def cost_table(rows: list[dict]) -> str:
    """Per lane: cells, passes, tokens, wall time, dollars, dollars per pass."""
    lanes: dict[str, dict] = {}
    for row in rows:
        key = f"{row.get('harness', 'one')}: {row['backend']}"
        lane = lanes.setdefault(key, {"cells": 0, "pass": 0, "in": 0, "out": 0,
                                      "secs": 0.0, "usd": 0.0,
                                      "counted_cells": 0, "priced": True})
        lane["cells"] += 1
        lane["pass"] += 1 if row.get("ok") else 0
        lane["secs"] += row.get("seconds") or 0.0
        if row.get("input_tokens") is not None:
            lane["counted_cells"] += 1
            lane["in"] += row["input_tokens"]
            lane["out"] += row["output_tokens"]
        if row.get("usd") is None:
            lane["priced"] = False
        else:
            lane["usd"] += row["usd"]

    width = max([len(k) for k in lanes] + [12])
    head = (f"{'lane':<{width}}  {'pass':>7}  {'in tok':>10}  {'out tok':>9}  "
            f"{'secs':>7}  {'usd':>9}  {'usd/pass':>9}")
    lines = [head, "-" * len(head)]
    partial_note = False
    for key, lane in sorted(lanes.items(),
                            key=lambda kv: (-kv[1]["pass"], kv[1]["secs"])):
        # A cell with no usage (e.g. a timeout) must not zero out the WHOLE
        # lane's token total and hide usage collected from every other cell
        # in the same lane. Sum whatever was counted and mark the row
        # with "*" when coverage is partial instead of discarding it.
        partial = 0 < lane["counted_cells"] < lane["cells"]
        if lane["counted_cells"] == 0:
            tokens_in = tokens_out = "unknown"
        else:
            suffix = "*" if partial else ""
            tokens_in = f"{lane['in']:,}{suffix}"
            tokens_out = f"{lane['out']:,}{suffix}"
        if partial:
            partial_note = True
        if lane["priced"] and lane["counted_cells"] == lane["cells"]:
            usd = f"{lane['usd']:.4f}"
            per = f"{lane['usd'] / lane['pass']:.4f}" if lane["pass"] else "-"
        else:
            usd = per = "no price"
        lines.append(f"{key:<{width}}  {lane['pass']:>3}/{lane['cells']:<3}  "
                     f"{tokens_in:>10}  {tokens_out:>9}  {lane['secs']:>7.1f}  "
                     f"{usd:>9}  {per:>9}")
    if partial_note:
        lines.append("* summed from cells that reported usage; "
                      "the rest had no usage (e.g. a timeout)")
    return "\n".join(lines)


def run_cell(backend: config.Backend, task: Task, root: Path,
             timeout: float, model_id: str = "", harness: str = "one") -> dict:
    slug = f"{harness}-" + backend.name.replace("/", "_").replace(":", "_")
    workspace = root / slug / task.name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    task.setup(workspace)

    # A `host=model` selection drives the drop-in endpoint with a different
    # model id; the CLI only knows the endpoint name.
    cli_name, _, override = backend.name.partition("=")
    model_id = model_id or override

    cmd = _one_cli_cmd(backend, task, workspace, cli_name, model_id, timeout)
    cwd, env = str(REPO_ROOT), None
    env = cell_env(task, env)

    started = time.monotonic()
    proc = _run(cmd, cwd=cwd, timeout=timeout, env=env)
    elapsed = time.monotonic() - started

    row: dict[str, Any] = {
        "harness": harness,
        "backend": backend.name,
        "model": model_id or backend.model,
        "wire": backend.wire,
        "task": task.name,
        # The two corpus dimensions. Empty for an in-code task, which is what
        # tells a reader the cell measured the bare model with no guard state.
        "mask": getattr(task, "mask", "") or "",
        "gates": dict(getattr(task, "gates", {}) or {}),
        "ok": False,
        "steps": 0,
        "tool_calls": 0,
        "seconds": round(elapsed, 1),
        "stopped": "timeout",
        "error": "",
        "why": "",
        "resamples": 0,
    }
    row.update(cost_fields(model_id or backend.model, backend.name))
    if proc is None:
        row["error"] = f"cell exceeded {timeout:.0f}s"
        row["why"] = "timeout"
        return row

    answer = ""
    try:
        payload = json.loads(proc.stdout)
        answer = payload.get("text") or ""
        row.update(
            steps=payload.get("steps", 0),
            tool_calls=payload.get("tool_calls", 0),
            stopped=payload.get("stopped", ""),
            error=(payload.get("error") or "")[:200],
            seconds=payload.get("seconds", row["seconds"]),
            resamples=payload.get("resamples", 0),
        )
        row.update(spend(row, payload.get("cost") or {}))
    except ValueError:
        row["stopped"] = "crash"
        row["error"] = (proc.stderr or proc.stdout).strip()[-200:]
        return row

    ok, why = task.check(workspace, answer)
    row["ok"] = ok
    if not ok:
        # A transport failure is not a task failure. Reporting "slug.py not
        # created" for an HTTP 402 blames the model for a billing gate, which is
        # how a whole lane gets written off for the wrong reason.
        if row["stopped"] in ("error", "crash"):
            row["why"] = _short_error(row["error"])
        else:
            row["why"] = why or row["stopped"]
    return row


def _short_error(detail: str) -> str:
    low = detail.lower()
    if "http 4" in low or "http 5" in low:
        code = detail.split("HTTP ", 1)[-1].split()[0] if "HTTP " in detail else "?"
        for word, label in (("out of memory", "oom"), ("illegal memory", "gpu fault"),
                            ("not exist", "no model"), ("payment", "unpaid"),
                            # llama.cpp refusing to parse its own model's
                            # tool-call block. The lane answered; the server
                            # could not read the answer.
                            ("does not match the expected", "toolfmt"),
                            ("peg-native", "toolfmt"),
                            ("insufficient", "unpaid"), ("quota", "quota"),
                            ("rate limit", "429"), ("context", "context")):
            if word in low:
                return f"{code} {label}"
        return f"HTTP {code}"
    if "cannot reach" in low:
        return "unreachable"
    return (detail.split(":", 1)[0] or "error")[:12]


def is_local(backend: config.Backend) -> bool:
    """A loopback or private-address endpoint - by address, not by name.

    A name test would miss a llama-server reached over a private network and
    would claim a hosted lane that happened to be named `llamacpp-something`.
    """
    return config.is_local_address(backend.base_url)


def run_matrix(backends: list[config.Backend], tasks: list[Task], root: Path,
               timeout: float, jobs: int, out_path: Path,
               on_row=None, harnesses: tuple[str, ...] = ("one",)) -> list[dict]:
    rows: list[dict] = []
    lock_file = out_path.open("a")

    def emit(row: dict) -> None:
        rows.append(row)
        lock_file.write(json.dumps(row) + "\n")
        lock_file.flush()
        if on_row:
            on_row(row)

    def one_backend(backend: config.Backend) -> None:
        for harness in harnesses:
            for task in tasks:
                emit(run_cell(backend, task, root, timeout, harness=harness))

    local = [b for b in backends if is_local(b)]
    remote = [b for b in backends if not is_local(b)]

    if remote:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            list(pool.map(one_backend, remote))
    # Serial on purpose: one GPU, one queue. Nothing to unload between
    # backends - a llama-server holds its one model for the life of the
    # process, so two local lanes are two servers the operator sized to fit
    # together, not one runtime swapping weights under us.
    for backend in local:
        one_backend(backend)

    lock_file.close()
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

# Widest a result cell may print before it is truncated with a `~` and repeated
# in full underneath. Wide enough for the reasons we actually emit, narrow
# enough that six task columns still fit a chat window.
_CELL_CAP = 24


def table(rows: list[dict], tasks: list[Task]) -> str:
    names = [t.name for t in tasks]
    many = len({r.get("harness", "one") for r in rows}) > 1
    by_backend: dict[str, dict] = {}
    for row in rows:
        label = row["backend"]
        if many:
            label = f"{row.get('harness', 'one')}: {label}"
        entry = by_backend.setdefault(label, {"wire": row["wire"]})
        entry[row["task"]] = row

    def contested(entry: dict) -> list[str]:
        """The tasks this lane actually got to attempt.

        A cell that never reached the endpoint is not a cell the model lost:
        an offline endpoint would otherwise print `0/5`, which reads as the
        worst score in the table rather than as a machine that was off.
        """
        return [n for n in names
                if (entry.get(n) or {}).get("why") != "unreachable"]

    def score(entry: dict) -> tuple:
        # A lane that never ran sorts last, below every lane that scored zero
        # on its own merits.
        if not contested(entry):
            return (1, 0.0)
        passed = sum(1 for n in names if entry.get(n, {}).get("ok"))
        secs = sum(entry.get(n, {}).get("seconds", 0) for n in names)
        return (-passed, secs)

    def cell_text(row: dict | None) -> str:
        if not row:
            return "-"
        if row["ok"]:
            return "ok %ds" % int(row["seconds"])
        return row.get("why") or row["stopped"]

    # Per-column width from the content, not a fixed 14. The failure reason is
    # the only column that carries information beyond pass/fail, and it was the
    # only one being clipped -- "invented it in pricing.py" printed as
    # "invented it in", which reads like a different verdict.
    col_w = {}
    for task_name in names:
        widest = max([len(cell_text(entry.get(task_name)))
                      for entry in by_backend.values()] + [len(task_name)])
        col_w[task_name] = min(_CELL_CAP, widest)

    width = max([len(b) for b in by_backend] + [7])
    head = (f"{'backend':<{width}}  {'wire':<9}  "
            + "  ".join(f"{n:<{col_w[n]}}" for n in names) + "  pass")
    lines = [head, "-" * len(head)]
    clipped: list[tuple[str, str, str]] = []
    for name, entry in sorted(by_backend.items(), key=lambda kv: score(kv[1])):
        cells = []
        for task_name in names:
            text = cell_text(entry.get(task_name))
            w = col_w[task_name]
            if len(text) > w:
                clipped.append((name, task_name, text))
                text = text[:w - 1] + "~"
            cells.append(f"{text:<{w}}")
        passed = sum(1 for n in names if entry.get(n, {}).get("ok"))
        ran = contested(entry)
        tally = f"{passed}/{len(ran)}" if ran else "no contest"
        if ran and len(ran) < len(names):
            tally += f" ({len(names) - len(ran)} nc)"
        lines.append(f"{name:<{width}}  {entry['wire']:<9}  " + "  ".join(cells)
                     + f"  {tally}")
    if clipped:
        # A `~` in the grid means the reason did not fit, never that the reason
        # was short. Print every clipped cell in full so the table is still the
        # whole result and not a lossy summary of it.
        lines.append("")
        lines.append("~ truncated above, in full:")
        seen = set()
        for lane, task_name, text in clipped:
            if (lane, task_name) in seen:
                continue
            seen.add((lane, task_name))
            lines.append(f"  {lane}  {task_name}  {text}")
    return "\n".join(lines)


def lane_board(backends, harnesses_available: list[str] | None = None) -> str:
    """What is reachable from this box and what running it costs, before it runs.

    The question a person asks first is not "which harness won" but "what can I
    run without being billed for it". That is answerable without spending
    anything - `config.lane_kind` already knows - so it should never require a
    trial run to discover. Free lanes come first because the free tier is the
    one to develop against.
    """
    rows = []
    for b in backends:
        kind = config.lane_kind(b.model, b.name, b.base_url)
        price = config.price_for(b.model, b.name)
        if kind == config.FREE:
            cost = "free"
        elif kind == config.METERED and price:
            cost = f"${price[0]:g}/${price[1]:g} per Mtok"
        elif kind == config.PLAN:
            cost = "plan quota - no dollar price exists"
        else:
            cost = "per token, price unknown - set ONE_CLI_PRICING"
        rows.append((kind, b.name, b.wire, cost, b.base_url))
    order = {config.FREE: 0, config.METERED: 1, config.PLAN: 2, config.UNKNOWN: 3}
    rows.sort(key=lambda r: (order.get(r[0], 3), r[1]))
    width = max([len(r[1]) for r in rows], default=4)
    lines = [f"{'lane':<{width}}  {'kind':<7}  {'wire':<9}  cost"]
    for kind, name, wire, cost, _url in rows:
        lines.append(f"{name:<{width}}  {kind:<7}  {wire:<9}  {cost}")
    tally = {k: sum(1 for r in rows if r[0] == k)
             for k in (config.FREE, config.METERED, config.PLAN, config.UNKNOWN)}
    lines.append("")
    lines.append(f"{tally[config.FREE]} free, {tally[config.METERED]} metered, "
                 f"{tally[config.PLAN]} plan, {tally[config.UNKNOWN]} unpriced. "
                 f"--free runs the first group and costs nothing; --budget can "
                 f"only bound the second.")
    if harnesses_available is not None:
        lines.append(f"harnesses installed here: "
                     f"{', '.join(harnesses_available) or 'none'}")
    return "\n".join(lines)


def select_backends(patterns: list[str], include_local: bool) -> list[config.Backend]:
    found = config.discover(include_local=include_local)
    if not patterns:
        return found
    by_name = {b.name: b for b in found}
    out, seen = [], set()
    for pattern in patterns:
        # `endpoint=model` benches one endpoint's other models: an aggregator
        # like Ollama Cloud serves 19 of them behind a single drop-in, and
        # standing up 19 near-identical .env files to reach them is silly.
        if "=" in pattern:
            host, _, model = pattern.partition("=")
            base = by_name.get(host)
            if not base:
                continue
            clone = replace(base, name=f"{host}={model}", model=model,
                            label=f"{model} via {host}")
            if clone.name not in seen:
                seen.add(clone.name)
                out.append(clone)
            continue
        # An exact name is a request for that one backend, not for everything
        # whose name contains it - "qwen" must not drag in every local tag.
        exact = [b for b in found if b.name == pattern]
        if exact:
            if exact[0].name not in seen:
                seen.add(exact[0].name)
                out.append(exact[0])
            continue
        for backend in found:
            hay = f"{backend.name} {backend.model}".lower()
            if pattern.lower() in hay and backend.name not in seen:
                seen.add(backend.name)
                out.append(backend)
    return out


def corpus_tasks() -> tuple[Sequence[Task], list[str]]:
    """On-disk tasks, and the reason any directory did not become one.

    Imported late for the same reason `scenarios` is: corpus imports us.
    """
    from onecli.evals import corpus
    return corpus.validate()


def suite(name: str) -> list[Task]:
    """Task list by suite name. `scenarios` imports us, so it is imported late."""
    if name == "corpus":
        return list(corpus_tasks()[0])
    if name == "basic":
        return list(TASKS)
    from onecli.evals import scenarios
    if name == "real":
        return list(scenarios.SCENARIOS)
    if name == "hard":
        return list(scenarios.HARD_SCENARIOS)
    return (list(TASKS) + list(scenarios.SCENARIOS)
            + list(scenarios.HARD_SCENARIOS) + list(corpus_tasks()[0]))


def main(argv: list[str] | None = None) -> int:
    # Stamp who spawned the cells. A child `one` writes this into its run
    # record, so the origin of a cell's process is recorded rather than inferred.
    os.environ["ONE_CLI_CALLER"] = "one-bench"
    ap = argparse.ArgumentParser(prog="one-bench", description=__doc__.splitlines()[0])
    ap.add_argument("patterns", nargs="*",
                    help="Substrings of backend name/model. Empty = every backend.")
    ap.add_argument("-t", "--task", action="append", default=[],
                    help="Restrict to a task by name. Repeatable.")
    ap.add_argument("--suite", default="basic",
                    choices=["basic", "real", "hard", "corpus", "all"],
                    help="basic = the original three, real = the repo "
                         "scenarios, hard = the discriminating tier.")
    ap.add_argument("--timeout", type=float, default=420.0, help="Per cell, seconds.")
    ap.add_argument("-j", "--jobs", type=int, default=4,
                    help="Parallel cloud backends. A local llama-server is always serial.")
    ap.add_argument("--no-local", action="store_true",
                    help="Skip llama.cpp endpoints; drop-ins only.")
    ap.add_argument("--out", default="", help="Run dir. Default .scratch/bench/<stamp>.")
    ap.add_argument("--list", action="store_true", help="Show the selection and exit.")
    args = ap.parse_args(argv)

    backends = select_backends(args.patterns, not args.no_local)
    if not backends:
        print("no backend matched", file=sys.stderr)
        return 2
    tasks = suite(args.suite)

    # A corpus directory that failed to load is a task you think you are
    # running and are not, so it is named on stderr every run, never swallowed.
    if args.suite in ("corpus", "all"):
        for problem in corpus_tasks()[1]:
            print(f"corpus: {problem}", file=sys.stderr)

    if args.task:
        by_name = {t.name: t for t in tasks}
        unknown = [n for n in args.task if n not in by_name]
        if unknown:
            print(f"unknown task: {', '.join(unknown)}", file=sys.stderr)
            return 2
        tasks = [by_name[n] for n in args.task]

    if args.list:
        for b in backends:
            print(f"{b.name:52} {b.wire:10} {b.base_url}")
        for t in tasks:
            dims = []
            mask = getattr(t, "mask", "")
            if mask:
                dims.append(f"mask={mask}")
            gates = getattr(t, "gates", None)
            if gates:
                dims.append(" ".join(f"{k}={v}" for k, v in gates.items()))
            print(f"  task {t.name:32} {' '.join(dims)}")
        print(f"\n{len(backends)} backends x {len(tasks)} tasks", file=sys.stderr)
        return 0

    root = Path(args.out) if args.out else (
        Path.cwd() / ".scratch" / "bench" / time.strftime("%Y%m%d-%H%M%S"))
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "rows.jsonl"
    print(f"one-bench: {len(backends)} backends x {len(tasks)} tasks -> {root}",
          file=sys.stderr)

    def progress(row: dict) -> None:
        mark = "ok" if row["ok"] else (row.get("why") or row["stopped"])
        print(f"  {row.get('harness', 'one'):<4} {row['backend']:<40} "
              f"{row['task']:<8} {mark:<12} "
              f"{row['seconds']:>6.1f}s", file=sys.stderr, flush=True)

    harnesses = ("one",)
    rows = run_matrix(backends, tasks, root, args.timeout, args.jobs, out_path,
                      on_row=progress, harnesses=harnesses)
    report = table(rows, tasks) + "\n\n" + cost_table(rows)
    (root / "table.txt").write_text(report + "\n")
    print("\n" + report)
    passed = sum(1 for r in rows if r["ok"])
    print(f"\n{passed}/{len(rows)} cells passed - rows: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

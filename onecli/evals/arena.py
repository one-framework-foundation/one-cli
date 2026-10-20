"""The arena: the same task, the same endpoint, several agent harnesses.

`bench.py` answers "which model can hold a tool loop". This answers a different
question - "does the harness matter, and how much" - by holding the model and
the endpoint fixed and swapping the agent around it: our loop, pi, opencode,
qwen-code, codex.

Two things beyond pass/fail are recorded, because pass/fail stops being enough
the moment the task is realistic:

* **What it touched.** Files created, modified and deleted, and lines changed,
  measured against the pristine fixture. A pass that rewrote four modules is not
  the same result as a pass that changed one operator, and the fixture carries a
  deprecated decoy specifically so collateral edits are visible.
* **What a judge thinks of the change.** An independent model scores the diff
  blind - it is never told which harness or which model produced it - on
  correctness, minimality and whether the change fits the codebase. The judge is
  advisory: the checker decides pass/fail, the judge only explains quality
  differences between two passes.

Judge lane: `ONE_JUDGE=<backend>`. There is no default judge model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config, context, harnesses, providers
from . import arena_parity, bench, scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
# No judge lane ships as a default: the operator names one with ONE_JUDGE. A
# lane never judges its own diff; see `judge()`.
JUDGE_DEFAULT = os.environ.get("ONE_JUDGE", "")

JUDGE_SYSTEM = (
    "You review a single code change made by an automated agent. You are given "
    "the task it was asked to do and the complete diff of what it did. You do "
    "not know who made it. Score strictly and briefly."
)

JUDGE_RUBRIC = """\
Task the agent was given:
---
{prompt}
---

Diff of everything that changed in the working tree:
---
{diff}
---

Score three things, each 0-5:

- correctness: does the change actually accomplish the task, with no bug
  introduced? 5 = right and complete, 0 = does not do the task.
- minimality: is this the smallest change that does the job? Penalise
  refactors, reformatting, renames, new abstractions, new files, and edits to
  files the task did not require. 5 = surgical, 0 = sprawling.
- fit: does it match the surrounding code - its units, its style, its
  conventions - and leave tests and deprecated code alone? 5 = indistinguishable
  from the existing code, 0 = alien.

Reply with ONE line of JSON and nothing else:
{{"correctness": <0-5>, "minimality": <0-5>, "fit": <0-5>, "note": "<12 words max>"}}
"""


# --------------------------------------------------------------------------
# what changed on disk
# --------------------------------------------------------------------------

def snapshot(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        try:
            out[str(path.relative_to(root))] = path.read_text()
        except (UnicodeDecodeError, OSError):
            out[str(path.relative_to(root))] = "<binary>"
    return out


def diff_report(before: dict[str, str], after: dict[str, str]) -> tuple[dict, str]:
    created = sorted(set(after) - set(before))
    deleted = sorted(set(before) - set(after))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    lines = 0
    chunks: list[str] = []
    import difflib

    for path in changed + created:
        old = before.get(path, "").splitlines(keepends=True)
        new = after.get(path, "").splitlines(keepends=True)
        delta = list(difflib.unified_diff(old, new, f"a/{path}", f"b/{path}", n=2))
        lines += sum(1 for row in delta
                     if row[:1] in "+-" and not row.startswith(("+++", "---")))
        chunks.append("".join(delta))
    for path in deleted:
        chunks.append(f"--- a/{path}\n+++ /dev/null\n")
        lines += len(before[path].splitlines())
    stats = {"files_created": created, "files_changed": changed,
             "files_deleted": deleted, "lines_touched": lines}
    return stats, "".join(chunks)[:12000]


# --------------------------------------------------------------------------
# the judge
# --------------------------------------------------------------------------

_JUDGE_CACHE: dict[str, config.Backend | None] = {}


def judge_backend() -> config.Backend | None:
    if "backend" not in _JUDGE_CACHE:
        try:
            _JUDGE_CACHE["backend"] = _resolve_one(JUDGE_DEFAULT)
        except Exception:
            _JUDGE_CACHE["backend"] = None
    return _JUDGE_CACHE["backend"]


def _resolve_one(pattern: str) -> config.Backend:
    picked = bench.select_backends([pattern], include_local=False)
    if not picked:
        raise RuntimeError(f"no backend matches {pattern!r}")
    return picked[0]


def judge(prompt: str, diff: str, contestant: config.Backend | None = None,
          budget: "bench.Budget | None" = None) -> dict:
    backend = judge_backend()
    if backend is None or not diff.strip():
        return {}
    # A lane must not score its own diff. The judge is advisory - the checker
    # decides pass/fail - but an advisory number a model gave itself is worse
    # than no number, because it reads like the others.
    if contestant is not None and contestant.name.partition("=")[0] == \
            backend.name.partition("=")[0]:
        return {"judge_error": "skipped: judge is this lane"}
    # The judge is a request on a lane like any cell, so the ceiling is checked
    # before it and the reply's usage is charged after it.
    if budget is not None and budget.exhausted():
        return {"judge_error": "skipped: --budget reached"}
    try:
        wire = providers.build(backend, timeout=180.0)
        turn = wire.chat([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_RUBRIC.format(prompt=prompt, diff=diff)},
        ], temperature=0.0, max_tokens=400)
        if budget is not None:
            used_in, used_out = context.usage_tokens(turn.usage)
            priced = bench.cost_fields(backend.model, backend.name, backend.base_url)
            budget.charge(bench.spend(priced, {"input_tokens": used_in,
                                               "output_tokens": used_out}))
        text = (turn.text or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return {"judge_error": text[:120]}
        payload = json.loads(text[start:end + 1])
        scores = {k: payload.get(k) for k in ("correctness", "minimality", "fit")}
        if any(not isinstance(v, (int, float)) for v in scores.values()):
            return {"judge_error": "non-numeric score"}
        scores["judge_note"] = str(payload.get("note", ""))[:80]
        scores["judge_total"] = round(sum(float(scores[k]) for k in
                                          ("correctness", "minimality", "fit")), 1)
        return scores
    except Exception as exc:  # a judge failure must never fail a cell
        return {"judge_error": f"{type(exc).__name__}: {exc}"[:120]}


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def _stream(launch: harnesses.Launch) -> dict:
    """How the child's stdin is wired.

    A harness that wants a body on stdin says so in `launch.stdin`. Every other
    harness gets DEVNULL, never this process's own fd: `input=None` means
    INHERIT, and whatever this process's stdin holds (a hook's pipe, say)
    would become a prompt the child pays a backend to answer.
    """
    return {"input": launch.stdin} if launch.stdin else {"stdin": subprocess.DEVNULL}


def run_cell(harness: harnesses.Harness, backend: config.Backend, task: bench.Task,
             root: Path, timeout: float, system: str, do_judge: bool,
             budget: "bench.Budget | None" = None) -> dict:
    slug = f"{harness.name}-{backend.name}".replace("/", "_").replace(":", "_")
    # Absolute, always: every adapter hands these paths to a child process that
    # runs with its cwd inside the workspace, so a relative path here silently
    # points the child at a config that does not exist.
    cell_root = (root / slug / task.name).resolve()
    if cell_root.exists():
        shutil.rmtree(cell_root)
    workspace = cell_root / "ws"
    cfg_dir = cell_root / "cfg"
    workspace.mkdir(parents=True)
    cfg_dir.mkdir(parents=True, mode=0o700)
    try:
        task.setup(workspace)
        before = snapshot(workspace)

        row = {
            "harness": harness.name, "backend": backend.name,
            "model": backend.model, "wire": backend.wire, "task": task.name,
            "system_prompt_chars": len(system), "ok": False, "steps": 0,
            "tool_calls": 0, "seconds": 0.0, "stopped": "timeout", "error": "",
            "why": "", "lines_touched": 0, "files_changed": [], "files_created": [],
            "files_deleted": [],
        }
        # Third-party harnesses report no usage at all, so their token fields stay
        # None: unknown, not zero. Ours fills them in from the meter below.
        row.update(bench.cost_fields(backend.model, backend.name, backend.base_url))
        if backend.wire not in harness.wires:
            row.update(stopped="skipped", why="n/a wire",
                       error=f"{harness.name} cannot drive the {backend.wire} wire")
            return row

        _, _, override = backend.name.partition("=")
        cell = harnesses.Cell(backend=backend, model_id=override, workspace=workspace,
                              prompt=task.prompt, system=system,
                              read_only=task.read_only, timeout=timeout,
                              cfg_dir=cfg_dir, max_steps=task.max_steps)
        launch = harness.build(cell)
        started = time.monotonic()
        try:
            proc = subprocess.run(launch.cmd, cwd=launch.cwd, env=launch.env,
                                  capture_output=True, text=True, timeout=timeout,
                                  **_stream(launch))
        except subprocess.TimeoutExpired as expired:
            row["seconds"] = round(time.monotonic() - started, 1)
            row["why"] = "timeout"
            row["error"] = f"cell exceeded {timeout:.0f}s"
            # A timeout is the cell most in need of its transcript and was the one
            # cell that kept none: the early return skipped the writes below, so
            # "timeout, 0 steps, 0 tools" was the whole record of what happened.
            _dump(cell_root, expired.stdout, expired.stderr)
            return row
        except OSError as exc:
            row.update(seconds=0.0, stopped="crash", why="no binary", error=str(exc)[:200])
            return row
        row["seconds"] = round(time.monotonic() - started, 1)

        result = harness.parse(proc.stdout, proc.stderr)
        row.update(steps=result.steps, tool_calls=result.tool_calls,
                   stopped=result.stopped, error=result.error[:200])
        row.update(bench.spend(row, result.cost))

        after = snapshot(workspace)
        stats, diff = diff_report(before, after)
        row.update(stats)

        ok, why = task.check(workspace, result.text)
        row["ok"] = ok
        if not ok:
            row["why"] = (bench._short_error(str(row["error"] or ""))
                          if row["stopped"] in ("error", "crash") else (why or row["stopped"]))
        if do_judge and diff.strip():
            row.update(judge(task.prompt, diff, contestant=backend, budget=budget))
        (cell_root / "answer.txt").write_text(result.text or "")
        (cell_root / "diff.patch").write_text(diff)
        # What the harness actually said on the way. A cell that fails is a claim
        # about a harness, and without the child's own output there is nothing to
        # check that claim against - the run before this one had a cell whose final
        # text blamed "garbled tool output" and no way to see whether it was right.
        _dump(cell_root, proc.stdout, proc.stderr)
        return row
    finally:
        shutil.rmtree(cfg_dir)


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------

def _dump(cell_root: Path, stdout, stderr) -> None:
    """What the harness actually said, whether it finished or ran out of time.

    A cell that fails is a claim about a harness, and without the child's own
    output there is nothing to check that claim against - a run before this one
    had a cell whose final text blamed "garbled tool output" and no way to see
    whether it was right.
    """
    for name, blob in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        if isinstance(blob, bytes):
            blob = blob.decode(errors="replace")
        (cell_root / name).write_text(blob or "")


def run(harness_names: list[str], backends: list[config.Backend],
        tasks: list[bench.Task], root: Path, timeout: float, jobs: int,
        out_path: Path, system: str, do_judge: bool,
        budget: "bench.Budget | None" = None,
        skip: set[tuple[str, str]] | None = None) -> list[dict]:
    rows: list[dict] = []
    sink = out_path.open("a")

    def emit(row: dict) -> None:
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        mark = "ok " if row["ok"] else (row.get("why") or "fail")[:14]
        print(f"  {row['harness']:9} {row['backend']:26} {row['task']:12} "
              f"{mark:15} {row['seconds']:6.1f}s", flush=True)

    def unit(harness_name: str, backend: config.Backend) -> None:
        harness = harnesses.HARNESSES[harness_name]
        for task in tasks:
            # The ceiling is checked BEFORE the request, not after: a cell that
            # has already run has already been billed, so a budget enforced on
            # the way out is a report rather than a limit.
            if budget is not None and budget.exhausted():
                budget.skip(harness_name, backend.name, task.name)
                continue
            row = run_cell(harness, backend, task, root, timeout, system, do_judge,
                           budget=budget)
            if budget is not None:
                budget.charge(row)
            emit(row)

    def backend_unit(backend: config.Backend) -> None:
        for harness_name in harness_names:
            # A probe verdict is per (harness, lane): a harness excluded on one
            # lane still runs on every other one.
            if skip and (harness_name, backend.name) in skip:
                continue
            unit(harness_name, backend)

    # A local llama-server holds its one model for the life of the process (see
    # bench.run_matrix), so there is nothing to unload between cells. Local
    # lanes run serially - one GPU, one queue - and remote ones fan out.
    local = [b for b in backends if bench.is_local(b)]
    remote = [b for b in backends if not bench.is_local(b)]
    if remote:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            list(pool.map(backend_unit, remote))
    for backend in local:
        backend_unit(backend)
    sink.close()
    return rows


def table(rows: list[dict], tasks: list[bench.Task]) -> str:
    names = [t.name for t in tasks]
    by: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        by.setdefault((row["backend"], row["harness"]), {})[row["task"]] = row
    width = max([len(f"{b} / {h}") for b, h in by] + [20])
    head = f"{'backend / harness':{width}} " + " ".join(f"{n:16}" for n in names)
    lines = [head + " pass  touched  judge", "-" * (len(head) + 22)]
    for (backend, harness), cells in sorted(
            by.items(), key=lambda kv: (-sum(1 for c in kv[1].values() if c["ok"]),
                                        sum(c["seconds"] for c in kv[1].values()))):
        parts = []
        for name in names:
            cell = cells.get(name)
            if not cell:
                parts.append(f"{'-':16}")
                continue
            mark = "ok" if cell["ok"] else (cell.get("why") or "fail")[:11]
            parts.append(f"{mark + ' ' + str(int(cell['seconds'])) + 's':16}")
        passed = sum(1 for c in cells.values() if c["ok"])
        touched = sum(c.get("lines_touched", 0) for c in cells.values())
        judged = [c["judge_total"] for c in cells.values() if c.get("judge_total")]
        judge_str = f"{sum(judged) / len(judged):.1f}/15" if judged else "-"
        lines.append(f"{backend + ' / ' + harness:{width}} " + " ".join(parts) +
                     f" {passed}/{len(names)}  {touched:5}    {judge_str}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Stamp who spawned the cells. A child `one` writes this into its run
    # record, so the origin of a cell's process is recorded rather than inferred.
    os.environ["ONE_CLI_CALLER"] = "one-arena"
    parser = argparse.ArgumentParser(
        prog="one-arena",
        description="Run the same task through several agent harnesses on one endpoint.")
    parser.add_argument("patterns", nargs="*", help="Backend selection, as one-bench.")
    parser.add_argument("--harness", action="append", default=[],
                        help="Repeatable. Default: every harness present on this box.")
    parser.add_argument("--suite", default="real",
                        choices=["real", "basic", "hard", "both", "all"],
                        help="real = the repo scenarios, basic = one-bench's "
                             "three, hard = the discriminating tier.")
    parser.add_argument("-t", "--task", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    parser.add_argument("--no-system", action="store_true",
                        help="Drop the long system prompt (the control run).")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--probe-attempts", type=int, default=arena_parity.PROBE_ATTEMPTS,
                        help="Probe retries per (harness, lane) before excluding it. "
                             "One sample of a stochastic agent is not evidence.")
    parser.add_argument("--probe-first-lane-only", action="store_true",
                        help="Cheap mode: probe only the first lane and apply its "
                             "verdict to that lane alone. Every other lane runs "
                             "unprobed. Default is to probe every lane.")
    parser.add_argument("--no-parity-probe", action="store_true",
                        help="Skip the pre-flight observed probe (dims 9-11). The "
                             "probe is ON by default; this opt-out is for someone "
                             "who knows they are skipping the fairness gate.")
    parser.add_argument("--free", action="store_true",
                        help="Only lanes declared free (explicit zero price). "
                             "Costs nothing, so this is the tier to develop against.")
    parser.add_argument("--budget", type=float, default=None,
                        help="Stop once the run has spent this many dollars. "
                             "Only bounds lanes with a known per-token price - "
                             "a flat-rate plan is refused rather than pretended "
                             "to be bounded.")
    parser.add_argument("--out", default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--lanes", action="store_true",
                        help="Print every reachable lane and what it costs, "
                             "then exit. Spends nothing.")
    args = parser.parse_args(argv)

    # A local llama-server is only reachable when explicitly named: a bare
    # arena run must not drag in every model the box is serving, but naming
    # one is a request for it.
    picked = bench.select_backends(
        args.patterns,
        include_local=bool(args.patterns) or args.free or args.lanes)
    if args.lanes:
        # Answered before any selection error can matter: a person running
        # --lanes is asking what exists, so "no backend matched" is the wrong
        # reply to give them.
        installed = [n for n, h in harnesses.HARNESSES.items() if h.available()]
        print(bench.lane_board(picked, installed))
        return 0
    if args.free:
        picked = bench.free_only(picked)
        if not picked:
            print("no free lane reachable - no lane with an explicit zero price is serving")
            return 1
    if not picked:
        print("no backend matched")
        return 1

    budget = bench.Budget(args.budget)
    # The judge spends on its own lane, so it is held to the same rule as the
    # lanes under test: a ceiling that cannot price the judge bounds nothing.
    judged = [] if args.no_judge or not budget.active else [
        b for b in [judge_backend()] if b is not None]
    unbounded = budget.unbounded(picked + judged) if budget.active else []
    if unbounded:
        plans = [n for n, kind in unbounded if kind == config.PLAN]
        unpriced = [n for n, kind in unbounded if kind == config.UNKNOWN]
        if plans:
            print(f"--budget cannot bound a flat-rate plan: {', '.join(plans)}. "
                  f"Those lanes bill quota, not dollars.")
        if unpriced:
            print(f"--budget cannot bound a lane whose price we do not know: "
                  f"{', '.join(unpriced)}. Name the price in ONE_CLI_PRICING, "
                  f"or drop the lane.")
        print("Drop those lanes, or drop --budget.")
        return 2

    names = args.harness or [n for n, h in harnesses.HARNESSES.items() if h.available()]
    missing = [n for n in names if n not in harnesses.HARNESSES]
    if missing:
        print(f"unknown harness: {', '.join(missing)}")
        return 2
    unavailable = [n for n in names if not harnesses.HARNESSES[n].available()]
    if unavailable:
        print(f"not installed, skipping: {', '.join(unavailable)}")
        names = [n for n in names if n not in unavailable]

    tasks = list(scenarios.SCENARIOS) if args.suite in ("real", "both", "all") else []
    if args.suite in ("basic", "both", "all"):
        tasks += list(bench.TASKS)
    if args.suite in ("hard", "all"):
        tasks += list(scenarios.HARD_SCENARIOS)
    if args.task:
        tasks = [t for t in tasks if t.name in args.task]
    system = "" if args.no_system else scenarios.SYSTEM_PROMPT

    if args.list:
        for backend in picked:
            for name in names:
                print(f"{name:10} {backend.name:28} {backend.wire:10} {backend.base_url}")
        print(f"\n{len(picked)} backends x {len(names)} harnesses x {len(tasks)} tasks "
              f"= {len(picked) * len(names) * len(tasks)} cells; "
              f"system prompt {len(system)} chars")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = (Path(args.out).resolve() if args.out
            else Path.cwd() / ".scratch/arena" / stamp)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "rows.jsonl"
    print(f"{len(picked)} backends x {len(names)} harnesses x {len(tasks)} tasks "
          f"-> {out_path}")

    # The observed pre-flight probe (dims 9-11). ON by default: a harness that
    # cannot read the fixture is EXCLUDED here rather than scored 0 in the run.
    # `parity.md` is always written, whether the probe ran or not - a run with
    # no parity report is unpublishable.
    declared_names = list(names)
    probes: dict[str, dict[str, arena_parity.ProbeResult]] = {}
    skip: set[tuple[str, str]] = set()
    probe_lanes: list[str] = []
    if not args.no_parity_probe:
        probe_backends = [picked[0]] if args.probe_first_lane_only else picked
        probe_lanes = [b.name for b in probe_backends]
        print(f"parity probe on {len(probe_backends)} lane(s): {len(names)} harnesses, "
              f"{args.probe_attempts} attempts each")
        probes = arena_parity.probe_matrix(names, probe_backends, root, args.timeout,
                                           args.probe_attempts, args.jobs, budget)
        skip = arena_parity.excluded_pairs(probes)
        for harness_name, backend_name in sorted(skip):
            print(f"excluded by probe: {harness_name} on {backend_name}")
    (root / "parity.md").write_text(
        arena_parity.parity_markdown(declared_names, probes, probe_lanes))
    runnable = [n for n in names
                if any((n, b.name) not in skip for b in picked)]
    if not runnable:
        print("every harness failed the parity probe on every lane - nothing to run")
        return 1
    if not args.no_judge and judge_backend() is None:
        print(f"judge lane {JUDGE_DEFAULT} unreachable - scoring without it")

    rows = run(names, picked, tasks, root, args.timeout, args.jobs, out_path,
               system, not args.no_judge, budget, skip)
    print()
    print(table(rows, tasks))
    print()
    print(bench.cost_table(rows))
    print(budget.report())
    passed = sum(1 for r in rows if r["ok"])
    print(f"\n{passed}/{len(rows)} cells passed - rows: {out_path}")

    # An empty column is not a score. A harness that was asked for and ran zero
    # cells means the run is broken, not that the harness scored 0/0 - printing
    # a TOTAL line beside an empty column is how a benchmark lies.
    ran = {r["harness"] for r in rows}
    empty = [n for n in declared_names if n not in ran]
    if empty:
        print(f"\nBROKEN RUN: requested harness(es) ran zero cells: {', '.join(empty)}")
        for name in empty:
            for backend in picked:
                res = (probes.get(backend.name) or {}).get(name)
                if res is not None and not res.ok:
                    print(f"  {name} on {backend.name}: {res.reason} ({res.score})")
        print("These numbers are not publishable. See parity.md.")
        (root / "parity.md").write_text(
            arena_parity.parity_markdown(declared_names, probes, probe_lanes, empty))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

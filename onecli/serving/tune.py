"""Measure the serving recipes and remember which one actually won.

The rule this module exists to enforce: a lane is launched with the fastest
config MEASURED on that host, never the fastest config someone read about. A
recipe with no measurement behind it is served anyway, but it is reported as
unverified rather than quietly presented as a choice.

The ledger is per host and keyed by (model, runner). Change the llama.cpp
build and the old number stops applying, which the key makes visible instead
of letting a stale speedup ride.
"""

from __future__ import annotations

import http.client
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .. import secureio
from . import hostfacts, recipes
from .catalog import Model

LEDGER = Path.home() / ".one-cli" / "tuning.json"
SCHEMA = 1

# A scratch port for the sweep, chosen high so it cannot collide with a live
# lane. The sweep never touches a port something is already serving on.
TUNE_PORT = 18990

# The measurement workload. Three shapes because they stress a drafter
# differently: writing new code from nothing, restating code it was just
# given, and a small edit inside a large body. Speculative decoding is worth
# most on the last two, and a bench that only writes fresh code understates it.
TASKS = [
    {"name": "fresh",
     "prompt": "Write a Python function `merge_intervals(intervals)` that "
               "merges overlapping intervals and returns them sorted. Include "
               "a short docstring. Code only."},
    {"name": "rewrite",
     "prompt": "Rewrite this function to use a single pass and no sorting, "
               "keeping the same signature and docstring:\n\n"
               "def total(rows):\n"
               "    \"\"\"Sum the amount field of every paid row.\"\"\"\n"
               "    paid = [r for r in rows if r['status'] == 'paid']\n"
               "    paid = sorted(paid, key=lambda r: r['id'])\n"
               "    return sum(r['amount'] for r in paid)\n\nCode only."},
    {"name": "edit",
     "prompt": "In the function below, change only the timeout from 30 to 120 "
               "and return the whole function unchanged otherwise:\n\n"
               "def fetch(url, retries=3):\n"
               "    for attempt in range(retries):\n"
               "        try:\n"
               "            return requests.get(url, timeout=30).json()\n"
               "        except requests.Timeout:\n"
               "            continue\n"
               "    raise RuntimeError('gave up on ' + url)\n\nCode only."},
]

MAX_TOKENS = 220


@dataclass
class Sample:
    task: str
    tok_s: float = 0.0
    tokens: int = 0
    draft_n: int = 0
    draft_accepted: int = 0
    error: str = ""

    @property
    def acceptance(self) -> float:
        return self.draft_accepted / self.draft_n if self.draft_n else 0.0


@dataclass
class Result:
    recipe: str
    flags: list[str] = field(default_factory=list)
    tok_s: float = 0.0
    acceptance: float = 0.0
    samples: list[Sample] = field(default_factory=list)
    error: str = ""

    def ok(self) -> bool:
        return not self.error and self.tok_s > 0


def _post(base_url: str, model: str, prompt: str,
          timeout: float = 180.0) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "one-cli/tune"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def measure(base_url: str, model: str = "", tasks: list[dict] | None = None,
            warmup: bool = True) -> Result:
    """Measure an endpoint that is ALREADY running.

    Reads the server's own `timings` rather than timing the socket: wall-clock
    at the client folds in prompt processing and the network, and a recipe
    that only changes generation speed would be judged on the wrong number.
    """
    tasks = tasks or TASKS
    out = Result(recipe="measured")
    if warmup:
        try:
            _post(base_url, model, "Say ok.", timeout=120)
        except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError):
            pass
    tot_tokens = tot_ms = 0.0
    draft_n = draft_ok = 0
    for t in tasks:
        s = Sample(task=t["name"])
        try:
            data = _post(base_url, model, t["prompt"])
        except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError) as exc:
            s.error = str(exc)
            out.samples.append(s)
            continue
        tm = data.get("timings") or {}
        s.tokens = int(tm.get("predicted_n", 0) or 0)
        s.tok_s = float(tm.get("predicted_per_second", 0) or 0)
        s.draft_n = int(tm.get("draft_n", 0) or 0)
        s.draft_accepted = int(tm.get("draft_n_accepted", 0) or 0)
        if not s.tok_s and s.tokens:
            ms = float(tm.get("predicted_ms", 0) or 0)
            s.tok_s = s.tokens / (ms / 1000.0) if ms else 0.0
        tot_tokens += s.tokens
        tot_ms += float(tm.get("predicted_ms", 0) or 0)
        draft_n += s.draft_n
        draft_ok += s.draft_accepted
        out.samples.append(s)
    if tot_ms:
        # Token-weighted, not a mean of means: a task that generated twice as
        # much should count twice, or a short refusal skews the recipe.
        out.tok_s = round(tot_tokens / (tot_ms / 1000.0), 2)
    out.acceptance = round(draft_ok / draft_n, 4) if draft_n else 0.0
    errs = [s.error for s in out.samples if s.error]
    if errs and not out.tok_s:
        out.error = errs[0]
    return out


def _wait_health(port: int, timeout: float = 600.0, proc=None) -> bool:
    """Wait for /health; a `proc` that exits first ends the wait early."""
    url = f"http://127.0.0.1:{port}/health"
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                if json.loads(r.read().decode("utf-8")).get("status") == "ok":
                    return True
        except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    return False


def _stop(facts: hostfacts.HostFacts, name: str, proc) -> None:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    if facts.docker:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       check=False, timeout=120)


def run_candidate(model: Model, recipe: recipes.Recipe,
                  facts: hostfacts.HostFacts, port: int = TUNE_PORT,
                  context: int = 0, parallel: int = 1,
                  root: Path | None = None,
                  log=None) -> Result:
    """Serve one recipe on a scratch port, measure it, and take it down.

    parallel defaults to 1 here even though production lanes run -np 2: a
    sweep wants the recipe isolated, and the production number is re-measured
    against the live lane afterwards.
    """
    name = f"one-tune-{port}"
    cmd = recipes.command(model, recipe, port, facts, context=context,
                          parallel=parallel, root=root, container=name)
    # A dangling container from an interrupted sweep would make the next run
    # measure the PREVIOUS recipe and report it under this one's name.
    if facts.docker and not facts.llama_server:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       check=False, timeout=120)
    proc = None
    res = Result(recipe=recipe.name, flags=list(recipe.flags))
    try:
        if facts.llama_server:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        else:
            started = subprocess.run(cmd, capture_output=True, text=True,
                                     check=False, timeout=300)
            if started.returncode != 0:
                res.error = (started.stderr or started.stdout).strip()[:400]
                return res
        if log:
            log(f"  {recipe.name}: waiting for the model to load")
        if not _wait_health(port):
            res.error = "server never reported healthy"
            return res
        measured = measure(f"http://127.0.0.1:{port}",
                           model.alias or model.name)
        res.tok_s = measured.tok_s
        res.acceptance = measured.acceptance
        res.samples = measured.samples
        res.error = measured.error
        return res
    finally:
        _stop(facts, name, proc)


def load_ledger() -> dict:
    try:
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "entries": {}}


def _key(model: Model, facts: hostfacts.HostFacts) -> str:
    return f"{model.name}@{facts.hostname}"


def other_lanes(facts: hostfacts.HostFacts) -> int:
    """How many OTHER model lanes were up while the sweep ran.

    A box already serving three models measures slower than an idle one, and
    the distill's baseline came out at 5.7 tok/s under load against 11.3 idle.
    The comparison BETWEEN candidates stays valid - they all ran under the
    same contention - but the absolute number is about a busy machine, so it
    is recorded rather than presented as the model's speed.
    """
    from . import serve  # late: serve reads the ledger this module writes
    try:
        return len([lane for lane in serve.running(facts) if lane.port != TUNE_PORT])
    except (OSError, ValueError):
        return 0


def record(model: Model, facts: hostfacts.HostFacts, results: list[Result],
           skipped: list[tuple[str, str]] | None = None,
           concurrent_lanes: int = 0) -> dict:
    """Write the sweep into the ledger and return the entry."""
    ok = [r for r in results if r.ok()]
    if not ok:
        raise RuntimeError("no candidate produced a measurement")
    base = next((r for r in ok if r.recipe == "baseline"), None)
    winner = max(ok, key=lambda r: r.tok_s)
    entry = {
        "model": model.name,
        "host": facts.hostname,
        "runner": facts.runner(),
        "gpu": facts.gpu.name,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "winner": winner.recipe,
        "flags": winner.flags,
        "tok_s": winner.tok_s,
        "acceptance": winner.acceptance,
        "baseline_tok_s": base.tok_s if base else 0.0,
        "speedup": round(winner.tok_s / base.tok_s, 2) if base and base.tok_s else 0.0,
        "candidates": [{"recipe": r.recipe, "tok_s": r.tok_s,
                        "acceptance": r.acceptance, "flags": r.flags,
                        "error": r.error} for r in results],
        "skipped": [{"recipe": n, "why": w} for n, w in (skipped or [])],
        "concurrent_lanes": concurrent_lanes,
    }
    if concurrent_lanes:
        entry["contended"] = (
            f"{concurrent_lanes} other lane(s) were serving on this host "
            f"during the sweep; the winner is valid, the absolute tok/s is "
            f"about a busy machine")
    data = load_ledger()
    data.setdefault("entries", {})[_key(model, facts)] = entry
    data["schema"] = SCHEMA
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    secureio.atomic_write_text(LEDGER, json.dumps(data, indent=2) + "\n")
    return entry


def best(model: Model, facts: hostfacts.HostFacts | None = None) -> dict | None:
    """The verified-fastest entry for this model on this host, or None.

    Returns None rather than a guess when the runner has changed since the
    measurement: the number was true about a build that is no longer here.
    """
    facts = facts or hostfacts.collect()
    entry = load_ledger().get("entries", {}).get(_key(model, facts))
    if not entry:
        return None
    if entry.get("runner") and entry["runner"] != facts.runner():
        entry = dict(entry)
        entry["stale"] = (f"measured on runner {entry['runner']}, this host "
                          f"now runs {facts.runner()}")
    return entry


def recipe_for(model: Model, facts: hostfacts.HostFacts | None = None
               ) -> tuple[recipes.Recipe, dict | None]:
    """The recipe a lane should launch with, and the evidence behind it.

    With a verified entry: the winner, replayed from the ledger. Without one:
    baseline, and the caller is expected to say so out loud.
    """
    facts = facts or hostfacts.collect()
    entry = best(model, facts)
    if entry and not entry.get("stale"):
        return (recipes.Recipe(name=entry["winner"], flags=list(entry["flags"]),
                               why=f"measured {entry['tok_s']} tok/s on "
                                   f"{entry['host']} at {entry['measured_at']}"),
                entry)
    return (recipes.Recipe(name="baseline", flags=[],
                           why="no verified measurement on this host yet"),
            entry)


def sweep(model: Model, facts: hostfacts.HostFacts | None = None,
          all_ngram: bool = False, port: int = TUNE_PORT,
          context: int = 0, root: Path | None = None, log=None) -> dict:
    """Measure every viable recipe for a model on this host, then record it."""
    facts = facts or hostfacts.collect()
    if not facts.can_serve():
        raise RuntimeError("no llama-server binary and no usable docker on "
                           "this host - nothing to measure with")
    if not model.present(root):
        raise RuntimeError(f"{model.file} is not on disk "
                           f"(`one models pull {model.name}`)")
    spec = recipes.supported_spec_types(facts)
    cands = recipes.candidates(model, facts, spec_types=spec,
                               all_ngram=all_ngram, root=root)
    skips = recipes.skipped(model, facts, spec_types=spec, root=root)
    busy = other_lanes(facts)
    if busy and log:
        log(f"note: {busy} other lane(s) are serving here - the winner will "
            f"be right, the absolute tok/s will be low")
    results = []
    for r in cands:
        if log:
            log(f"measuring {r.name}")
        res = run_candidate(model, r, facts, port=port, context=context,
                            root=root, log=log)
        if log:
            if res.ok():
                extra = (f", {res.acceptance:.0%} draft accepted"
                         if res.acceptance else "")
                log(f"  {r.name}: {res.tok_s} tok/s{extra}")
            else:
                log(f"  {r.name}: FAILED - {res.error}")
        results.append(res)
    return record(model, facts, results, skips, concurrent_lanes=busy)

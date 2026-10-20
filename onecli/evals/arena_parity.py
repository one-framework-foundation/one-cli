"""The arena parity protocol: declared checks, the observed probe, the report.

Three things live here, one per half of the parity protocol:

* `check_declaration` proves a harness's DECLARED parity (dims 4-8, 12 on the
  `Harness`) against what `build(cell)` actually produces. A harness that says
  MATCHED on a flag its command does not carry fails.
* `run_probe` is the OBSERVED half (dims 9, 10, 11): a throwaway read-only cell
  against a one-file fixture that forces each harness to read a known value and
  to be refused a write. A harness that cannot read the fixture is EXCLUDED with
  a reason - it is never scored 0, because a zero must mean the agent failed the
  task, never that the adapter failed the agent.
* `parity_markdown` writes the per-run `parity.md`. A run without one is
  unpublishable.

This is the *arena* parity protocol (harness fairness). It is unrelated to
`onecli/evals/parity.py`, which is the one-cli-vs-Claude-Code feature table.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .. import config, harnesses
from . import bench
from ..harnesses import ABSENT, MATCHED, PARITY_DIMENSIONS


# --------------------------------------------------------------------------
# declared parity: does the built command back up what the adapter claims
# --------------------------------------------------------------------------

# The dimensions whose value is legible in the built command or the config the
# adapter writes. 12 (ambient) is declared but not asserted here: it does not
# show up as a token to grep for. 8 (thinking) IS checkable - a harness claiming
# MATCHED must carry a `--thinking` flag on the built command.


def _written(cell: harnesses.Cell) -> dict[str, str]:
    out: dict[str, str] = {}
    for base in (cell.cfg_dir, cell.workspace):
        for path in sorted(base.rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                try:
                    out[str(path)] = path.read_text()
                except (UnicodeDecodeError, OSError):
                    out[str(path)] = "<binary>"
    return out


def _haystack(harness: harnesses.Harness, cell: harnesses.Cell):
    """Build the harness and return (launch, everything it emitted as one blob).

    The blob is the command plus every config file the adapter dropped, so a
    value that reaches the harness through a JSON config (crush's token cap,
    droid's) counts exactly as one on argv does.
    """
    launch = harness.build(cell)
    parts = [str(tok) for tok in launch.cmd]
    if launch.stdin:
        parts.append(launch.stdin)
    parts.extend(_written(cell).values())
    return launch, "\n".join(parts)


def _has_system_role(launch, cell: harnesses.Cell) -> bool:
    """A system-role channel: an explicit flag, or a dedicated identity file."""
    if any(tok in ("--system", "--system-prompt") for tok in launch.cmd):
        return True
    for path in cell.cfg_dir.rglob("SOUL.md"):
        try:
            if cell.system and cell.system in path.read_text():
                return True
        except OSError:
            pass
    return False


def check_declaration(harness: harnesses.Harness,
                      cell: harnesses.Cell) -> list[str]:
    """Every way this harness's parity table disagrees with its built command.

    Empty means the declaration is true. The cell must carry distinctive values
    (a system marker in `system`, a marker in `prompt`) so the concatenation
    check for a prepended system prompt cannot match by accident.
    """
    parity = harness.parity
    if not parity:
        return [f"{harness.name}: no parity declared"]
    launch, hay = _haystack(harness, cell)
    bad: list[str] = []

    def matched(dim: int) -> bool:
        return parity[dim].status == MATCHED

    if matched(5) and str(cell.max_steps) not in hay:
        bad.append(f"dim5 step budget: MATCHED but cell.max_steps={cell.max_steps} "
                   f"is not in the built command")
    if matched(6) and str(int(cell.timeout)) not in hay:
        bad.append(f"dim6 wall timeout: MATCHED but cell.timeout={int(cell.timeout)} "
                   f"is not in the built command")
    if matched(7) and "8192" not in hay:
        bad.append("dim7 output token budget: MATCHED but 8192 is not in the built command")
    if matched(8) and "--thinking" not in launch.cmd:
        bad.append("dim8 thinking level: MATCHED but no --thinking flag on the built command")

    role = _has_system_role(launch, cell)
    concat = f"{cell.system}\n\n{cell.prompt}"
    if matched(4) and not role:
        bad.append("dim4 system prompt role: MATCHED but no --system/--system-prompt "
                   "flag and no SOUL.md carrying the system text")
    if parity[4].status == ABSENT and concat not in hay:
        bad.append("dim4 system prompt role: ABSENT (declared as prepended) but the "
                   "system+prompt concatenation is not present in the built command")
    return bad


# --------------------------------------------------------------------------
# the observed probe (dimensions 9, 10, 11)
# --------------------------------------------------------------------------

PROBE_SECRET = "PARITY-PROBE-VALUE-4713"
PROBE_FILE = "known.txt"
PROBE_PROMPT = (
    "Do these three things, in order.\n"
    f"1. Read the file {PROBE_FILE} in the current working directory and quote "
    "its exact contents on a line that begins with 'FILE: '.\n"
    "2. On a line that begins with 'TOOLS: ', list the exact names of every tool "
    "you have available to you, separated by commas.\n"
    "3. Attempt to create a new file called probe_write.txt containing the word "
    "wrote. If you are refused or have no write tool, say so plainly.\n"
)

# Known stripped-surface fingerprints (see the openclaw adapter), kept as data
# so a probe that trips one can name it in its exclusion reason.
_FINGERPRINTS: list[tuple[frozenset[str], str]] = [
    (frozenset({"session_status"}), "minimal-profile fingerprint: session_status only"),
    (frozenset({"web_fetch", "web_search", "memory_search"}),
     "sandbox web+memory allowlist fingerprint"),
]


def tool_fingerprint(tools: list[str]) -> str:
    have = frozenset(t.strip().lower() for t in tools if t.strip())
    for names, label in _FINGERPRINTS:
        if have == names:
            return label
    return ""


def extract_tools(answer: str) -> list[str]:
    """Best-effort tool list from the agent's own 'TOOLS:' line.

    Tolerant of a leading enumerator: a model asked for a line "beginning with
    TOOLS:" routinely writes "2. TOOLS: ..." because the probe numbers its three
    steps. Without this the tool list comes back empty even on a clean read, so
    the parity report cannot record the observed surface (dim 9).
    """
    for line in (answer or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:\d+[.)]\s*|[-*]\s*)?TOOLS:\s*(.*)$",
                         stripped, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(";", ",").replace("|", ",")
            names = [tok.strip(" `*-\t") for tok in raw.split(",")]
            return [n for n in names if n]
    return []


# Substrings that make a tool name read-shaped. A harness whose own TOOLS: line
# carries one of these HAS a read tool, whatever else went wrong on the turn -
# so the exclusion reason may not claim it has none.
READ_TOOL_HINTS = ("read", "view", "cat", "open", "grep", "search_file",
                   "str_replace", "fs_", "file")

# Names that are not file reads even if a hint substring ever matches them;
# web and memory surfaces are excluded explicitly.
_NOT_READ = ("web_search", "web_fetch", "memory_search", "session_status")


def has_read_tool(tools: list[str]) -> bool:
    """Does this self-reported tool list contain anything file-read shaped."""
    for raw in tools:
        name = raw.strip().lower()
        if not name or name in _NOT_READ:
            continue
        if any(hint in name for hint in READ_TOOL_HINTS):
            return True
    return False


def evaluate_probe(answer: str, wrote: bool, tools: list[str]) -> tuple[bool, str]:
    """The pure decision: did this harness read, and was the write refused.

    Kept side-effect-free so the test can drive it directly with a synthetic
    answer - which is how the probe is proven to bite without a live lane.

    The verdict is split, because the old single reason ("no read tool /
    workspace rooting failed") asserted a capability the very same sample
    disproved whenever the harness had just listed `read` among its tools. The
    exclusion still stands; only the claim is fixed.
    """
    read_ok = PROBE_SECRET in (answer or "")
    if not read_ok:
        fp = tool_fingerprint(tools)
        if has_read_tool(tools):
            reason = ("excluded: did not quote the known value (has a read tool; "
                      "may be a flaky turn)")
        else:
            reason = ("excluded: no read tool / workspace rooting failed - did not "
                      "quote the known value")
        if fp:
            reason += f" [{fp}]"
        return False, reason
    if wrote:
        return False, ("excluded: read-only not enforced - a write took effect on a "
                       "read-only cell")
    return True, ""


@dataclass
class ProbeResult:
    harness: str
    ok: bool
    reason: str = ""
    read_ok: bool = False
    write_refused: bool = False
    tools: list[str] = field(default_factory=list)
    error: str = ""
    timed_out: bool = False
    # How many attempts were spent, and which one passed (0 = none did). A
    # single sample of a stochastic agent is not evidence, so the report shows
    # the score: "passed 1/3".
    attempts: int = 1
    passed_on: int = 0
    # What the attempt cost, when the lane is priced and the harness metered it.
    usd: float | None = None

    @property
    def score(self) -> str:
        return f"passed {1 if self.ok else 0}/{self.attempts}"


def _list_ws(ws: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(ws.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            try:
                out[str(path.relative_to(ws))] = path.read_text()
            except (UnicodeDecodeError, OSError):
                out[str(path.relative_to(ws))] = "<binary>"
    return out


PROBE_ATTEMPTS = 3


def run_probe(harness: harnesses.Harness, backend: config.Backend,
              root: Path, timeout: float = 120.0,
              attempts: int = PROBE_ATTEMPTS,
              budget: "bench.Budget | None" = None) -> ProbeResult:
    """Probe one harness on one backend, up to `attempts` times.

    An agent is stochastic: one blank turn is a coin flip, not a capability
    verdict, and excluding on one failing sample would let the gate blind the
    very harness it protects. So pass on ANY attempt, and exclude only when every
    attempt fails. A timeout is a retryable attempt, never a verdict on its own.

    A probe attempt is a real request on the lane, so `--budget` bounds it the
    way it bounds a cell: checked before the request, charged after it.
    """
    last = ProbeResult(harness.name, False, reason="excluded: probe never ran")
    for attempt in range(1, max(1, attempts) + 1):
        if budget is not None and budget.exhausted():
            return ProbeResult(harness.name, False, attempts=attempt - 1,
                               reason="excluded: --budget reached before the probe")
        last = _probe_once(harness, backend, root, timeout, attempt)
        if budget is not None:
            budget.charge({"usd": last.usd})
        last.attempts = attempt
        if last.ok:
            last.passed_on = attempt
            return last
    last.attempts = max(1, attempts)
    return last


def _probe_once(harness: harnesses.Harness, backend: config.Backend,
                root: Path, timeout: float, attempt: int = 1) -> ProbeResult:
    """One throwaway read-only cell for one harness. Never scores; only gates."""
    slug = f"{harness.name}-{backend.name}-{attempt}".replace("/", "_").replace(":", "_")
    cell_root = (root / "__probe__" / slug).resolve()
    if cell_root.exists():
        shutil.rmtree(cell_root)
    ws = cell_root / "ws"
    cfg = cell_root / "cfg"
    ws.mkdir(parents=True)
    cfg.mkdir(parents=True, mode=0o700)
    try:
        (ws / PROBE_FILE).write_text(PROBE_SECRET + "\n")
        before = _list_ws(ws)

        if backend.wire not in harness.wires:
            return ProbeResult(harness.name, False,
                               reason=f"excluded: cannot drive the {backend.wire} wire")

        _, _, override = backend.name.partition("=")
        cell = harnesses.Cell(backend=backend, model_id=override, workspace=ws,
                              prompt=PROBE_PROMPT, system="", read_only=True,
                              timeout=timeout, cfg_dir=cfg)
        launch = harness.build(cell)
        stream = {"input": launch.stdin} if launch.stdin else {"stdin": subprocess.DEVNULL}
        try:
            proc = subprocess.run(launch.cmd, cwd=launch.cwd, env=launch.env,
                                  capture_output=True, text=True, timeout=timeout,
                                  **stream)
        except subprocess.TimeoutExpired:
            # Its own verdict, never folded into the read-tool claim: a probe that
            # ran out of wall clock said nothing at all about the tool surface.
            return ProbeResult(harness.name, False, timed_out=True,
                               reason=f"excluded: probe timed out after {timeout:.0f}s "
                                      f"(no verdict on tools)")
        except OSError as exc:
            return ProbeResult(harness.name, False,
                               reason="excluded: probe could not launch", error=str(exc)[:200])

        parsed = harness.parse(proc.stdout, proc.stderr)
        text = parsed.text
        priced = bench.cost_fields(backend.model, backend.name, backend.base_url)
        usd = bench.spend(priced, parsed.cost).get("usd")
        after = _list_ws(ws)
        wrote = any(name != PROBE_FILE and (name not in before or before[name] != body)
                    for name, body in after.items())
        tools = extract_tools(text)
        ok, reason = evaluate_probe(text, wrote, tools)
        return ProbeResult(harness.name, ok, reason=reason,
                           read_ok=PROBE_SECRET in (text or ""),
                           write_refused=not wrote, tools=tools, usd=usd)
    finally:
        shutil.rmtree(cfg)


def probe_harnesses(names: list[str], backend: config.Backend, root: Path,
                    timeout: float = 120.0, attempts: int = PROBE_ATTEMPTS,
                    budget: "bench.Budget | None" = None) -> dict[str, ProbeResult]:
    """Probe every named harness against ONE backend. Serial: it is pre-flight,
    one cheap cell each, and running them all at once buys nothing."""
    results: dict[str, ProbeResult] = {}
    for name in names:
        harness = harnesses.HARNESSES[name]
        if not harness.available():
            results[name] = ProbeResult(name, False, reason="excluded: binary not installed")
            continue
        started = time.monotonic()
        res = run_probe(harness, backend, root, timeout, attempts, budget)
        results[name] = res
        verdict = f"ok ({res.score})" if res.ok else f"{res.reason[:36]} ({res.score})"
        print(f"  probe {name:9} on {backend.name:22} {verdict:56} "
              f"{time.monotonic() - started:5.1f}s", flush=True)
    return results


def probe_matrix(names: list[str], backends: list[config.Backend], root: Path,
                 timeout: float = 120.0, attempts: int = PROBE_ATTEMPTS,
                 jobs: int = 4,
                 budget: "bench.Budget | None" = None) -> dict[str, dict[str, ProbeResult]]:
    """Probe every harness on EVERY backend, keyed backend -> harness -> result.

    A probe verdict is a fact about a (harness, lane) pair, never about the
    harness: probing one lane and dropping the loser everywhere would let a
    single unlucky sample remove a harness from the whole benchmark. Lanes are
    probed in parallel because they are independent remote endpoints.
    """
    out: dict[str, dict[str, ProbeResult]] = {}
    if not backends:
        return out
    if len(backends) == 1:
        return {backends[0].name: probe_harnesses(names, backends[0], root,
                                                  timeout, attempts, budget)}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(probe_harnesses, names, b, root, timeout, attempts, budget): b
                   for b in backends}
        for fut, backend in futures.items():
            out[backend.name] = fut.result()
    return out


def excluded_pairs(matrix: dict[str, dict[str, ProbeResult]]) -> set[tuple[str, str]]:
    """The (harness, backend) pairs the probe refused. Nothing wider."""
    return {(name, backend)
            for backend, results in matrix.items()
            for name, res in results.items() if not res.ok}


# --------------------------------------------------------------------------
# the per-run report
# --------------------------------------------------------------------------

def parity_markdown(names: list[str], probes: dict[str, dict[str, ProbeResult]],
                    backend_labels: list[str] | None = None,
                    broken: list[str] | None = None) -> str:
    """The declared table, the observed probe per lane, and the exclusions.

    `probes` is keyed backend -> harness -> result. `broken` names harnesses
    that were requested and ran zero cells; when it is non-empty the report
    opens with the failure, because a comparison table with an empty column is
    not a result.
    """
    lines: list[str] = ["# Arena parity report", ""]
    if broken:
        lines += ["## BROKEN RUN - DO NOT PUBLISH THESE NUMBERS", "",
                  "These harnesses were requested and ran **zero cells**: "
                  + ", ".join(f"`{n}`" for n in broken) + ".", "",
                  "An empty column is not a score of zero. Any total printed "
                  "beside it is meaningless. Fix the exclusions below and re-run.",
                  ""]
    if backend_labels:
        lines += ["Probe lanes: " + ", ".join(f"`{b}`" for b in backend_labels), ""]
    lines += ["A run whose parity report is missing is unpublishable. See the "
              "arena parity protocol notes for the required fields.", ""]

    lines += ["## Declared parity (dimensions 4-8, 12)", ""]
    header = "| harness | " + " | ".join(
        f"{n} {PARITY_DIMENSIONS[n]}" for n in (4, 5, 6, 7, 8, 12)) + " |"
    lines.append(header)
    lines.append("| --- " * (len(PARITY_DIMENSIONS) + 1) + "|")
    for name in names:
        harness = harnesses.HARNESSES.get(name)
        if not harness or not harness.parity:
            continue
        cells = " | ".join(harness.parity[n].status for n in (4, 5, 6, 7, 8, 12))
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    absents = []
    for name in names:
        harness = harnesses.HARNESSES.get(name)
        if not harness:
            continue
        for dim in (4, 5, 6, 7, 8, 12):
            entry = harness.parity.get(dim)
            if entry and entry.status == ABSENT:
                absents.append(f"- **{name}** dim{dim} ({PARITY_DIMENSIONS[dim]}): {entry.note}")
    if absents:
        lines += ["### ABSENT dimensions (declared defects)", ""] + absents + [""]

    lines += ["## Observed probe (dimensions 9-11)", "",
              "One row per (harness, lane) pair: a probe verdict is a fact about "
              "a pair, never about a harness.", ""]
    lines += ["| lane | harness | probe | attempts | read | write refused | tools |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for backend in sorted(probes):
        for name in names:
            res = probes[backend].get(name)
            if res is None:
                lines.append(f"| {backend} | {name} | not probed | - | - | - | - |")
                continue
            tools = ", ".join(res.tools) if res.tools else "-"
            lines.append(f"| {backend} | {name} | {'PASS' if res.ok else 'EXCLUDED'} | "
                         f"{res.score} | {'yes' if res.read_ok else 'no'} | "
                         f"{'yes' if res.write_refused else 'no'} | {tools} |")
    lines.append("")

    excluded = [(b, n, r) for b in sorted(probes)
                for n, r in probes[b].items() if not r.ok]
    if excluded:
        lines += ["## Exclusions", "",
                  "These (harness, lane) pairs failed the pre-flight probe and are "
                  "removed from the run - that lane only. None is scored 0.", ""]
        for backend, name, res in excluded:
            lines.append(f"- **{name}** on `{backend}` ({res.score}): {res.reason}")
        lines.append("")
    return "\n".join(lines)

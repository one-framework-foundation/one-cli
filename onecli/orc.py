"""Orchestration config - who drives, who works, and what that run cost.

`--orchestrator NAME` is one sentence: drive on NAME, send the workers to
`--model`. That sentence is the whole vocabulary, and it runs out the moment a
tree has more than one kind of worker in it. A sweep that wants its explorer on
a 32k local box and its reviewer on something that can actually review has no
way to say so on a command line without writing a preset file per backend - and
a preset file is the wrong place, because the pin belongs to THIS run's economy
rather than to what the agent is for.

An `.orc.json` is that missing noun. It names an orchestrator, a map of preset
name to backend, and the two budgets a tree spends (`max_depth`, `max_steps`).
It is read exactly the way `subagent.load_presets` reads an agents file, for
the same reason: a broken config file must cost a warning, never the run.

**Precedence, highest first, and every resolution carries which one it was:**

    preset-pin   `model` on the preset itself (the agent knows what it needs)
    orc-file     the `workers` entry for that preset name
    flag         --orchestrator + --model, the default for every unpinned child
    inherit      nothing named a backend; the child runs the parent's chain

This is `Spawner.pin_for` verbatim - `preset.model or workers[name] or worker`
- and the order is by SPECIFICITY, not by how late it binds. A pin on a preset
is a statement about the AGENT (this reviewer is useless below a certain tier)
and survives being dropped into someone else's orc file. A `workers` entry is a
statement about one preset in THIS run. A flag is a statement about every
unpinned child at once, so it is the broadest thing said and it loses to both.
Inverting this would make the per-preset pin unreachable from a command line
that also passes --orchestrator, which is the common case.

The launch echo exists because none of that is visible otherwise. A four-line
tree printed before the first request says which backend each preset actually
landed on AND why, which is the difference between "the cheap lane is broken"
and "the cheap lane was never used".

The run record is the same question asked afterwards. One JSON per run beside
the sessions and the jobs, under ~/.one-cli/runs, holding the resolved config
with its sources, per-node wall time and step counts, whatever the driver told
us about tokens, and how it ended.

**How it ended is not what it cost to get there.** `exit_status` answers
"did this run finish", and a recommended config cannot be built on that alone:
two runs that both answered are not the same run if one of them fell down its
fallback chain twice, resampled eleven malformed tool calls and compacted a
32k child three times. So the record carries the FAILURE MODES beside the
outcome, at run level and per node:

    failover        which rungs the chain dropped, and why (`Chain.history`)
    retries         requests the wire retried, split by the class it gave
    compactions     how many times the context was summarized away
    loop_stops      how many loop-guard advisories the model was shown
    stopped         the per-node reason, which `Spawn.stopped` already holds

Every one of these is read off a producer that was already counting it - the
chain's own history, the wire's retry counters, `Agent.compactions`, the loop
guard's advisories. Nothing here is a second tally kept beside the first, and
nothing here is a field waiting for a producer: `None` means "no producer could
answer this for this node" and is deliberately not `0`, because a permanently
zero counter reads as evidence that nothing happened.

The sharpest of those `None`s is `failover` on a pinned child. `chain_for`
gives a pinned child a chain of exactly ONE rung on purpose, so it has no rung
to drop - the field is absent rather than empty, because an empty list would
say "the chain held". That asymmetry between a pinned and an unpinned node is
the thing worth seeing, not the thing to smooth over.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from onecli import render, progress_policy as progress, secureio

# Beside ~/.one-cli/sessions, ~/.one-cli/jobs and ~/.one-cli/spills. One state
# root for the CLI; a second one is how a `--list` grows two answers.
RUN_DIR = Path.home() / ".one-cli" / "runs"

# Where a bare `--orc NAME` is looked for, in order. The project file first: an
# orc config describes a repo's economy far more often than a person's.
SEARCH = ("./.orc.json", "~/.one-cli/orc.json", "~/.one-cli/orc/{name}.json")

# The four rungs of the precedence chain, as they are printed and recorded.
FLAG = "flag"
PIN = "preset-pin"
FILE = "orc-file"
DEFAULT = "default"
# Not a rung: nothing named a backend, so the child runs on the parent's chain.
INHERIT = "inherit"

# The keys that make a JSON object an orc config rather than a map of them.
FIELDS = ("orchestrator", "workers", "max_depth", "max_steps",
          "task_class", "response_gate", "progress_policy")


@dataclass(frozen=True)
class Orc:
    """One named orchestration config, as read off disk.

    Every field is optional. A config that only names `workers` is a useful
    config - it says where the fan-out goes and leaves the driving alone.
    """

    name: str = ""
    orchestrator: str = ""
    workers: dict[str, str] = field(default_factory=dict)
    max_depth: int = 0
    max_steps: int = 0
    # What KIND of work this run is - "research", "build", "review", anything.
    # Free-form on purpose: see `clean_task_class`.
    task_class: str = ""
    # The file it came from, for the expanded echo. "" for one built in a test.
    path: str = ""
    response_gate: str = ""
    progress_policy: dict | None = None

    def worker_for(self, preset: str) -> str:
        return self.workers.get(preset, "")


def clean_task_class(value) -> str:
    """A task class, normalised. Lowercased and stripped, and nothing else.

    No vocabulary is enforced. `research`/`build`/`review`/`refactor`/`debug`
    are suggested in --help and in the docs because a suggestion that is wrong
    for your run costs nothing, whereas an enum that is wrong for your run
    makes you pick the nearest lie - and a record full of nearest lies groups
    into nothing. Case and whitespace ARE normalised, because "Build " and
    "build" being two classes is not expressiveness, it is a typo surviving.
    """
    return str(value or "").strip().lower()


def _response_gate(value):
    from onecli.response_gate import mode
    return mode(value) if value != "" else ""


def _orc_from(name: str, body: dict, path: str) -> Orc:
    workers = body.get("workers")
    return Orc(
        name=name or str(body.get("name") or ""),
        orchestrator=str(body.get("orchestrator") or ""),
        workers=({str(k): str(v) for k, v in workers.items()}
                 if isinstance(workers, dict) else {}),
        max_depth=int(body.get("max_depth") or 0),
        max_steps=int(body.get("max_steps") or 0),
        task_class=clean_task_class(body.get("task_class")),
        path=path,
        response_gate=_response_gate(body.get("response_gate", "")),
        progress_policy=(progress.resolve(body["progress_policy"]) if "progress_policy" in body else None),
    )


def load_orcs(paths: list[str] | None) -> tuple[dict[str, Orc], list[str]]:
    """Read orchestration configs from JSON files. Returns (orcs, errors).

    Mirrors `subagent.load_presets` deliberately, down to the accepted shapes:
    a bare map of name to body, the same map under an "orcs" key, or a list of
    objects each carrying its own name. A broken file is named and skipped, and
    an operator who cannot read one config still gets the run.

    One shape load_presets does not have: a file holding a SINGLE config, with
    `orchestrator` and `workers` at the top level. That is what people write
    for a project's own `.orc.json`, and it is told apart from a map of configs
    by whether any orc field is present at the top level - a map of configs has
    only names up there.
    """
    orcs: dict[str, Orc] = {}
    errors: list[str] = []
    for path in paths or []:
        full = os.path.expanduser(path)
        try:
            raw = json.loads(open(full, encoding="utf-8").read())
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if isinstance(raw, dict) and isinstance(raw.get("orcs"), (dict, list)):
            raw = raw["orcs"]
        if isinstance(raw, dict) and any(k in raw for k in FIELDS):
            stem = Path(full).stem.lstrip(".") or "orc"
            name = str(raw.get("name") or stem)
            try:
                orcs[name] = _orc_from(name, raw, full)
            except (ValueError, TypeError) as exc:
                errors.append(f"{path}: {name}: {exc}")
            continue
        if isinstance(raw, dict):
            items = [(str(k), v) for k, v in raw.items()]
        elif isinstance(raw, list):
            items = [(str(b.get("name") or ""), b) for b in raw if isinstance(b, dict)]
        else:
            errors.append(f"{path}: expected an object or a list of orc configs")
            continue
        for name, body in items:
            if not name or not isinstance(body, dict):
                errors.append(f"{path}: an orc config with no name was skipped")
                continue
            try:
                orcs[name] = _orc_from(name, body, full)
            except (ValueError, TypeError) as exc:
                errors.append(f"{path}: {name}: {exc}")
    return orcs, errors


def find(ref: str) -> tuple[Orc | None, list[str]]:
    """Resolve `--orc NAME|PATH` to one config. Returns (orc, errors).

    A ref that names a readable file is that file. Anything else is a name,
    looked for in the project's `.orc.json`, the user's `~/.one-cli/orc.json`,
    and `~/.one-cli/orc/<name>.json`. Not finding it is an error string, not an
    exception: an orc config is an optimisation, and a run that cannot find one
    should say so and run on the flags it was given.
    """
    ref = (ref or "").strip()
    if not ref:
        return None, []
    expanded = os.path.expanduser(ref)
    if os.path.isfile(expanded):
        orcs, errors = load_orcs([expanded])
        if not orcs:
            return None, errors or [f"{ref}: no orc config in this file"]
        # A file named directly is answered by its only config, or by the one
        # whose name matches the filename. Anything else is ambiguous and says so.
        if len(orcs) == 1:
            return next(iter(orcs.values())), errors
        stem = Path(expanded).stem.lstrip(".")
        if stem in orcs:
            return orcs[stem], errors
        return None, errors + [
            f"{ref}: holds {len(orcs)} configs ({', '.join(sorted(orcs))}); "
            f"name one with --orc"]
    errors = []
    for template in SEARCH:
        path = os.path.expanduser(template.format(name=ref))
        if not os.path.isfile(path):
            continue
        orcs, errs = load_orcs([path])
        errors += errs
        if ref in orcs:
            return orcs[ref], errors
    return None, errors + [f"no orc config named {ref!r} (looked in "
                           f"{', '.join(SEARCH)})"]


def catalogue(root: str | Path | None = None) -> tuple[list[Orc], list[str]]:
    """Every layout this run could resolve, in SEARCH order. (orcs, errors).

    `find` answers one name; this answers "what is there at all", which is the
    question an operator has before they know a name to type. It walks the same
    SEARCH paths so the two can never disagree about where a config lives - the
    `{name}` rung is a directory, so it is globbed rather than formatted.

    `root` replaces `~/.one-cli` so a test can point the whole search at a
    tmp_path instead of the operator's real state directory.
    """
    base = str(Path(root).expanduser()) if root else str(
        Path.home() / ".one-cli")
    seen: dict[str, Orc] = {}
    errors: list[str] = []
    for template in SEARCH:
        path = template.replace("~/.one-cli", base)
        path = os.path.expanduser(path)
        if "{name}" in path:
            folder = Path(path).parent
            files = sorted(str(f) for f in folder.glob("*.json")) \
                if folder.is_dir() else []
        else:
            files = [path] if os.path.isfile(path) else []
        for f in files:
            orcs, errs = load_orcs([f])
            errors += errs
            for name, o in orcs.items():
                # SEARCH order is precedence order: the project file wins, so
                # a name already seen is not overwritten by a later rung.
                seen.setdefault(name, o)
    return list(seen.values()), errors


def summarise(orcs: list[Orc]) -> str:
    """`--orc-list`: one line per layout, or a plain sentence when there are none.

    An empty table is the one output that reads as a bug. Say it in words.
    """
    if not orcs:
        return render.yellow(
            "no orc configs found (looked in " + ", ".join(SEARCH) + ")")
    lines = [render.bold("orchestration configs")]
    for o in orcs:
        pins = len(o.workers)
        lines.append(render.node(
            o.name, f"{o.orchestrator or '(no orchestrator)'} · "
                    f"{pins} worker pin{'' if pins == 1 else 's'}"))
        lines.append(render.leaf(o.path or "(built in)"))
    return "\n".join(lines)


def describe(orc: Orc, presets: dict) -> str:
    """`--orc-show`: one layout as it resolves against the presets that exist.

    `resolve` walks the PRESETS, so a `workers` entry naming a preset that is
    not loaded is skipped without a word - correct at runtime (a stale pin must
    not cost the run) and invisible, which is how a config keeps a typo for
    months. Here the same skip is drawn as a bad leaf: same behaviour, said out
    loud.
    """
    lines = [render.node(f"orc · {orc.name}",
                         orc.orchestrator or "(no orchestrator)", mutating=True)]
    for name in sorted(orc.workers):
        if name in presets:
            lines.append(render.leaf(f"{name} → {orc.workers[name]} ({FILE})"))
        else:
            lines.append(render.leaf(
                f"{name} → {orc.workers[name]} (no such preset - ignored)",
                bad=True))
    for name in sorted(presets):
        if name in orc.workers:
            continue
        pin = getattr(presets[name], "model", "")
        lines.append(render.leaf(
            f"{name} → {pin} ({PIN})" if pin
            else f"{name} → (--model, or the parent's chain)"))
    lines.append(render.leaf(f"max_depth {orc.max_depth} · "
                             f"max_steps {orc.max_steps} · "
                             f"task_class {orc.task_class or '(none)'}"))
    lines.append(render.leaf(f"config: {orc.path or '(built in)'}"))
    return "\n".join(lines)


# -- what the flags, the presets and the file add up to ---------------------


@dataclass(frozen=True)
class Slot:
    """One node's backend and the rung of the chain that decided it."""

    name: str
    backend: str
    source: str
    response_gate: str = "off"
    progress_policy: dict | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "backend": self.backend, "source": self.source,
                "response_gate": self.response_gate, "progress_policy": self.progress_policy}


@dataclass(frozen=True)
class Resolved:
    """The whole orchestration config after every rung has had its say."""

    orchestrator: Slot
    workers: tuple[Slot, ...] = ()
    max_depth: int = 0
    max_steps: int = 0
    orc: str = ""
    path: str = ""
    task_class: str = ""

    def as_dict(self) -> dict:
        return {
            "orc": self.orc,
            "path": self.path,
            "task_class": self.task_class,
            "orchestrator": self.orchestrator.as_dict(),
            "workers": [w.as_dict() for w in self.workers],
            "max_depth": self.max_depth,
            "max_steps": self.max_steps,
        }


def resolve_worker(preset, orc: Orc | None, default: str) -> tuple[str, str]:
    """The backend one preset runs on, and which rung said so.

    The precedence chain of the module docstring, in the order it is written
    there. There is no per-preset command-line flag, so `flag` reaches a worker
    only through `default` - which is `--model` under `--orchestrator`, and is
    labelled `default` because that is what it is from the preset's side: the
    thing every child gets when nothing spoke about it in particular.
    """
    if getattr(preset, "model", ""):
        return preset.model, PIN
    if orc is not None:
        pinned = orc.worker_for(preset.name)
        if pinned:
            return pinned, FILE
    if default:
        return default, DEFAULT
    return "", INHERIT


# What the ECHO prints where nothing named a backend. They are sentences for a
# person, not endpoint names, and `record_for` must never let one reach a
# record's `backend` field - `--orc-report` groups on that field, so a display
# string there reads as an endpoint that does not exist.
NO_WORKER_NAMED = "(parent's chain)"
NO_ORCHESTRATOR_NAMED = "(--model)"


def resolve(presets: dict, orc: Orc | None, orchestrator: str,
            orchestrator_source: str, worker_default: str,
            max_depth: int, max_steps: int,
            task_class: str = "", response_gate: str = "", progress_policy=None) -> Resolved:
    """Every node of the tree, with the rung that decided each one."""
    policy = response_gate or (orc.response_gate if orc else "") or "enforce"
    cadence = progress.resolve(progress_policy, base=progress.resolve(orc.progress_policy if orc else None))
    workers = []
    for name in sorted(presets):
        backend, source = resolve_worker(presets[name], orc, worker_default)
        workers.append(Slot(name=name, backend=backend or NO_WORKER_NAMED,
                            source=source,
                            response_gate=presets[name].response_gate or policy,
                            progress_policy=progress.resolve(presets[name].progress_policy, base=cadence)))
    return Resolved(
        orchestrator=Slot(name="orchestrator",
                          backend=orchestrator or NO_ORCHESTRATOR_NAMED,
                          source=orchestrator_source, response_gate=policy, progress_policy=cadence),
        workers=tuple(workers),
        max_depth=max_depth,
        max_steps=max_steps,
        orc=orc.name if orc else "",
        path=orc.path if orc else "",
        # The flag wins over the file; either may be absent.
        task_class=clean_task_class(task_class) or (
            orc.task_class if orc else ""),
    )


def echo(resolved: Resolved, explain: bool = False) -> str:
    """The launch echo: what each node resolved to, and why.

    Drawn with `render.node` and `render.leaf` so it is the same spine the run
    itself scrolls down, and so `--plain` and `NO_COLOR` are already handled -
    render decides that once, for everything.
    """
    title = "orchestrator"
    if resolved.orc:
        title += f" · {resolved.orc}"
    lines = [render.node(title,
                         f"{resolved.orchestrator.backend} "
                         f"({resolved.orchestrator.source})", mutating=True)]
    for slot in resolved.workers:
        lines.append(render.leaf(f"{slot.name} → {slot.backend} ({slot.source})"))
    if explain:
        if resolved.path:
            lines.append(render.leaf(f"config: {resolved.path}"))
        lines.append(render.leaf(f"max_depth {resolved.max_depth} · "
                                 f"max_steps {resolved.max_steps} · "
                                 f"task_class "
                                 f"{resolved.task_class or '(none)'}"))
    return "\n".join(lines)


# -- the run record ---------------------------------------------------------


@dataclass
class Node:
    """One agent in the tree and what it spent.

    `input_tokens`/`output_tokens`/`usd` are None where nothing reported them.
    A subagent has no counters of its own by design - the tree shares ONE meter
    so a ceiling is a ceiling for the whole tree - so a child records its wall
    time and its steps, and the money is the root's line for all of them.
    """

    name: str
    # Which level of the tree this node ran at: the root is 0, its children 1,
    # their children 2. A flat node list cannot otherwise say whether a run
    # was two levels or three, and "a different backend at every level" is a
    # claim about levels. Absent on every record written before this, where it
    # reads 0 - see `records()` on why the raw dict, not this default, is what
    # the report counts.
    depth: int = 0
    # The configured backend NAME this node ran on, and the model id that name
    # resolved to. Two fields because two backends can serve the same model id
    # - `lane-a` and `lane-b` both serve `shared-model` - so grouping by a
    # model id answers a different question from grouping by an endpoint, and
    # only the endpoint is what a cost comparison is about. Root and child use
    # the same vocabulary in each: `backend` is always a name here, never an id.
    backend: str = ""
    model: str = ""
    source: str = ""
    seconds: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    driver: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None
    metered: bool = True
    # -- what it cost to get there, not just whether it ended ---------------
    # Why the answer arrived when it did. `status` says a run ended; these say
    # which configuration was rough getting there.
    #
    # None everywhere means "no producer could answer this for this node", and
    # is deliberately not 0: an always-zero counter reads as evidence that
    # nothing happened, which is a stronger claim than we can make. A driver
    # child keeps no counters we can read, so it carries None.
    #
    # `failover` is None for a PINNED node too, and that asymmetry is the
    # point: `Spawner.chain_for` gives a pinned child exactly one rung, so it
    # has no rung to drop. An empty list would say "the chain held".
    stopped: str = ""
    failover: list[dict] | None = None
    retries: int | None = None
    retry_reasons: dict | None = None
    compactions: int | None = None
    loop_stops: int | None = None


@dataclass
class Phase:
    """One unit of work off a fixed checklist, and what it cost.

    The ledger a phase list needs. A phase is not a node: a node is an
    agent that ran, a phase is a piece of the JOB, and in the manager loop one
    long-lived implementer serves several phases in a row. Two phases with the
    same `backend` and the same `agent` are the same node measured twice, which
    is exactly the reading a node list cannot give.

    `verdict` is what the phase came to, and it is deliberately not the same
    word as `stopped`: `stopped` is how the agent loop ended (`answered`,
    `budget`, `error`), and `verdict` is whether the checklist item was carried
    - `done` when the loop answered, otherwise the word the loop stopped on. A
    phase can end `answered` and still be the phase a stall detector wants, and
    telling those apart needs both fields to exist.

    `mode` and `wording` carry the two things a paired comparison tests: whether ONE
    re-steered implementer beats a fresh child per phase (`loop` vs `fanout`),
    and whether asking for a phase "extremely well" beats asking for it
    "perfectly". Both are on the record because a paired run whose two halves
    cannot be told apart afterwards is not a pair, it is two runs.

    Tokens are None where nothing reported them - the same rule as `Node`.
    """

    index: int = 0
    name: str = ""
    agent: str = ""
    mode: str = ""
    wording: str = ""
    started: str = ""
    finished: str = ""
    seconds: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    stopped: str = ""
    verdict: str = ""
    backend: str = ""
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class Record:
    """One run, written once, when it is over."""

    session: str = ""
    started: str = ""
    finished: str = ""
    seconds: float = 0.0
    exit_status: str = ""
    exit_code: int = 0
    # Denormalised out of `config` on purpose: grouping every run of a kind is
    # the whole point, and a top-level key is what a `jq` or a `glob` can reach
    # without knowing the config's shape.
    task_class: str = ""
    # The run-level roll-up of the per-node failure modes. Denormalised for the
    # same reason `task_class` is: "which configurations were rough" is a
    # question asked across every record at once, and a top-level key is what a
    # `jq` can reach without walking the node list. Counters sum over the
    # nodes that HAVE them; `failover` is every rung any node dropped.
    failover: list[dict] = field(default_factory=list)
    retries: int = 0
    retry_reasons: dict = field(default_factory=dict)
    compactions: int = 0
    loop_stops: int = 0
    config: dict = field(default_factory=dict)
    nodes: list[Node] = field(default_factory=list)
    # Empty for every run that was not driven off a checklist, which is most of
    # them. An empty list and a missing key are the same reading here and are
    # allowed to be: a run with no phases has none, whenever it was written.
    phases: list[Phase] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def _sum_reasons(nodes: list[Node]) -> dict:
    """The run's retry reasons: every node's map, added up. Happened-only."""
    out: dict[str, int] = {}
    for n in nodes:
        for reason, count in (n.retry_reasons or {}).items():
            out[reason] = out.get(reason, 0) + int(count)
    return out


def path_for(session_id: str) -> Path:
    return RUN_DIR / f"{session_id}.json"


def write(record: Record) -> Path:
    """Write the record and return where it went. Never raises.

    A run that has already produced its answer must not fail at the last step
    over a full disk or a read-only home. The record is accounting; losing it
    costs a question later, and raising here costs the answer now.
    """
    p = path_for(record.session)
    try:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        secureio.atomic_write_text(p, json.dumps(record.as_dict(), indent=2, default=str))
    except OSError:
        pass
    return p


def read(session_id: str) -> Record:
    """Read one back. Raises FileNotFoundError when there is none."""
    p = path_for(session_id)
    if not p.exists():
        raise FileNotFoundError(f"no run record {session_id} at {p}")
    raw = json.loads(p.read_text())
    nodes = [Node(**n) for n in raw.get("nodes") or []]
    phases = [Phase(**p) for p in raw.get("phases") or []]
    return Record(session=raw.get("session") or "",
                  started=raw.get("started") or "",
                  finished=raw.get("finished") or "",
                  seconds=float(raw.get("seconds") or 0.0),
                  exit_status=raw.get("exit_status") or "",
                  exit_code=int(raw.get("exit_code") or 0),
                  task_class=clean_task_class(raw.get("task_class")),
                  failover=raw.get("failover") or [],
                  retries=int(raw.get("retries") or 0),
                  retry_reasons=raw.get("retry_reasons") or {},
                  compactions=int(raw.get("compactions") or 0),
                  loop_stops=int(raw.get("loop_stops") or 0),
                  config=raw.get("config") or {},
                  nodes=nodes,
                  phases=phases)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_for(session_id: str, resolved: Resolved, result, spawns,
               seconds: float, exit_code: int, started_at: float = 0.0,
               chain_name: str = "", exit_status: str = "",
               chain_model: str = "",
               phases: list["Phase"] | None = None) -> Record:
    """Build the record from what the run finished holding.

    `result` is an `agent.Result`, `spawns` the parent Spawner's list. The root
    node carries the cost because the meter is the tree's, not the root's - see
    `Node`.

    `result` may be None: a run killed by Ctrl-C or by an exception still has a
    record to write, and the caller passes what it holds - which is the spawns
    and the clock, and no Result. `exit_status` is how that run says what
    happened to it, since there is no `result.stopped` to read. Left empty it
    falls back to the Result's own word.
    """
    cost = dict(getattr(result, "cost", None) or {})

    def endpoint(name: str) -> str:
        """A backend name for the record, or `(unnamed)`.

        The last seam between the echo's vocabulary and the census's. Both
        fallbacks below reach for a `Slot.backend`, which holds a sentence
        when nothing named a backend; grouping on that sentence invents an
        endpoint. Saying we do not know which one it was is the honest answer
        and the one the reader already knows how to read.
        """
        if not name or name in (NO_WORKER_NAMED, NO_ORCHESTRATOR_NAMED):
            return UNKNOWN_BACKEND
        return name

    root = Node(
        name="orchestrator",
        depth=0,
        backend=endpoint(chain_name or resolved.orchestrator.backend),
        model=chain_model,
        source=resolved.orchestrator.source,
        seconds=round(seconds, 3),
        steps=len(getattr(result, "steps", None) or []),
        tool_calls=int(getattr(result, "tool_calls", 0) or 0),
        input_tokens=cost.get("input_tokens"),
        output_tokens=cost.get("output_tokens"),
        usd=cost.get("usd"),
        metered=not cost.get("estimated", False),
        stopped=str(getattr(result, "stopped", "") or ""),
        # The root's chain is the run's chain, so its history is every rung the
        # RUN dropped. Unlike a child, the root is never pinned to one rung.
        failover=list(getattr(result, "fallbacks", None) or []),
        retries=int(getattr(result, "retries", 0) or 0),
        retry_reasons=dict(getattr(result, "retry_reasons", None) or {}),
        compactions=len(getattr(result, "compactions", None) or []),
        loop_stops=int(getattr(result, "loop_stops", 0) or 0),
    )
    by_name = {s.name: s for s in resolved.workers}
    nodes = [root]
    for sp in spawns or []:
        slot = by_name.get(sp.agent)
        nodes.append(Node(
            name=sp.agent,
            depth=sp.depth,
            backend=endpoint(sp.backend or (slot.backend if slot else "")),
            model=sp.model,
            source=slot.source if slot else INHERIT,
            seconds=round(sp.seconds, 3),
            steps=sp.steps,
            tool_calls=sp.tool_calls,
            driver=sp.driver,
            metered=sp.metered,
            stopped=sp.stopped,
            failover=sp.failover,
            retries=sp.retries,
            retry_reasons=sp.retry_reasons,
            compactions=sp.compactions,
            loop_stops=sp.loop_stops,
        ))
    finished = time.time()
    return Record(
        session=session_id,
        started=datetime.fromtimestamp(
            started_at or (finished - seconds), timezone.utc).isoformat(
                timespec="seconds"),
        finished=datetime.fromtimestamp(finished, timezone.utc).isoformat(
            timespec="seconds"),
        seconds=round(seconds, 3),
        exit_status=exit_status or str(getattr(result, "stopped", "") or ""),
        exit_code=exit_code,
        task_class=resolved.task_class,
        failover=[f for n in nodes for f in (n.failover or [])],
        retries=sum(n.retries or 0 for n in nodes),
        retry_reasons=_sum_reasons(nodes),
        compactions=sum(n.compactions or 0 for n in nodes),
        loop_stops=sum(n.loop_stops or 0 for n in nodes),
        config=resolved.as_dict(),
        nodes=nodes,
        phases=list(phases or []),
    )



# -- reading the records back ----------------------------------------------
#
# A record that can only be written answers nothing, so the reader below
# exists before anything that would recommend a configuration.

# What a task class reads as when the record does not carry one. Two different
# facts, kept apart on purpose: `""` is a run that declared no class, a MISSING
# key is a record from a format with no class field. Merging them would let
# such records vote in a grouping they cannot belong to.
NO_CLASS = "(none)"
UNRECORDED = "(unrecorded)"
# A node whose `backend` is empty; current records always name one.
UNKNOWN_BACKEND = "(unnamed)"


def records() -> tuple[list[dict], list[str]]:
    """Every run record on disk, RAW. Returns (records, errors).

    Raw dicts rather than `Record`s, deliberately. `read()` is a typed round
    trip that fills a missing key with its default - an absent `task_class`
    becomes `""`, an absent `retries` becomes `0` - which is right for one
    record and wrong for a census over ninety of them: the default cannot be
    told apart from a value, so every record written before a field existed
    would vote as if it had carried it. Here the ABSENCE is the reading, so
    nothing is coerced on the way in.

    A file that will not parse is named and skipped, the same contract
    `load_orcs` has: one unreadable record must not cost the other ninety-five.
    """
    out: list[dict] = []
    errors: list[str] = []
    try:
        files = sorted(RUN_DIR.glob("*.json"))
    except OSError as exc:
        return [], [f"{RUN_DIR}: {exc}"]
    for f in files:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{f.name}: {exc}")
            continue
        if not isinstance(raw, dict):
            errors.append(f"{f.name}: not a run record")
            continue
        out.append(raw)
    return out, errors


def _reading(value) -> float | None:
    """A numeric reading, or None for "nobody could answer this".

    `None` in, `None` out - the docstring's rule, enforced at the one place
    every column enters the summary. A non-number is also None rather than a
    guess: a record is a file on disk and may have been edited by hand.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _task_label(raw: dict) -> str:
    if "task_class" not in raw:
        return UNRECORDED
    return clean_task_class(raw.get("task_class")) or NO_CLASS


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


@dataclass
class Group:
    """One backend under one task class, and every node that ran there.

    The group key is the BACKEND NAME, never the model id, because two endpoints can serve one id - `lane-a` and `lane-b` both
    serve `shared-model` - so grouping on the id merges two lanes into one
    row on exactly the field a cost comparison is built on. The ids are shown
    beside the name because the pairing is worth seeing; they do not group.
    """

    backend: str
    task_class: str
    nodes: int = 0
    runs: set = field(default_factory=set)
    models: set = field(default_factory=set)
    # Which levels of the tree this group's nodes ran at. A SET, not a key:
    # the grouping stays (backend, task class) so the rows every earlier
    # reading was taken over do not fracture, but a row that is a grandchild
    # and a row that is the orchestrator are no longer indistinguishable.
    # A record written before the field carries no depth and adds nothing
    # here, so the cell reads `-` rather than claiming level 0.
    depths: set = field(default_factory=set)
    readings: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    stopped: dict = field(default_factory=dict)

    def add(self, key: str, value) -> None:
        """Record one node's reading of one column, if it had one."""
        got = _reading(value)
        if got is not None:
            self.readings.setdefault(key, []).append(got)

    def model_cell(self) -> str:
        if not self.models:
            return "-"
        if len(self.models) == 1:
            return next(iter(self.models))
        return f"{len(self.models)} ids"


def _depth_cell(g: "Group") -> str:
    """Which levels this row's nodes ran at, or `-` for a record that predates
    the field. Never `0` by default: level 0 is the orchestrator, and saying it
    of a node nobody recorded a level for would be a claim, not a reading."""
    if not g.depths:
        return "-"
    return ",".join(str(d) for d in sorted(g.depths))


def _cell(values: list[float], nodes: int, fmt) -> str:
    """One summary cell. Three states, and none of them is `0`.

        `-`           nothing in this group carried the column
        `2.4s 7/19`   seven nodes carried it, twelve could not answer
        `2.4s!`       one reading, so it is a number and not a median

    The `!` is the whole point of the exercise: a median over three of
    ninety-six runs that prints like a median over ninety-six is the way a
    report lies without a single wrong number in it.
    """
    got = len(values)
    if not got:
        return "-"
    text = fmt(median(values))
    if got < nodes:
        text += f" {got}/{nodes}"
    if got == 1:
        text += "!"
    return text


def _sum_cell(values: list[float], nodes: int) -> str:
    """A counter column: the total, and what it is a total OVER.

    A sum, not a median - the question a failure-mode column answers is "how
    often did this configuration drop a rung", and the median of a column that
    is mostly zero is zero however rough the tail was. No `!`: a sum of one
    reading is still a sum, so the only thing worth marking is coverage.
    """
    if not values:
        return "-"
    text = f"{sum(values):g}"
    if len(values) < nodes:
        text += f" {len(values)}/{nodes}"
    return text


def _secs(v: float) -> str:
    return f"{v:.1f}s"


def _plain(v: float) -> str:
    return f"{v:g}"


def _tokens(v: float) -> str:
    return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"


def _usd(v: float) -> str:
    return f"${v:.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Columns sized from their content, the way `bench.table` sizes its own.

    Plain text on purpose: a report is pasted into a message and grepped out of
    scrollback, and neither survives being drawn in anything cleverer.
    """
    widths = [max([len(h)] + [len(r[i]) for r in rows])
              for i, h in enumerate(headers)]
    head = "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()
    out = ["  " + head, "  " + "-" * len(head)]
    for row in rows:
        out.append("  " + "  ".join(c.ljust(w) for c, w in
                                    zip(row, widths)).rstrip())
    return out


def group(runs: list[dict]) -> tuple[list["Group"], int]:
    """Every node in every record, grouped on (backend, task class).

    Returns the groups, biggest first, and how many records carried no node at
    all - a record with nothing under it is not evidence about any backend, so
    it is counted and named rather than dropped or attributed.
    """
    groups: dict[tuple[str, str], Group] = {}
    nodeless = 0
    for raw in runs:
        label = _task_label(raw)
        nodes = [n for n in (raw.get("nodes") or []) if isinstance(n, dict)]
        if not nodes:
            nodeless += 1
            continue
        for n in nodes:
            backend = str(n.get("backend") or "") or UNKNOWN_BACKEND
            key = (backend, label)
            g = groups.get(key)
            if g is None:
                g = groups[key] = Group(backend=backend, task_class=label)
            g.nodes += 1
            g.runs.add(str(raw.get("session") or ""))
            if n.get("model"):
                g.models.add(str(n["model"]))
            if "depth" in n and _reading(n.get("depth")) is not None:
                g.depths.add(int(n["depth"]))
            for col in ("seconds", "steps", "input_tokens", "output_tokens",
                        "usd", "retries", "compactions", "loop_stops"):
                g.add(col, n.get(col))
            # `failover` is a LIST of the rungs dropped, and `None` on a pinned
            # node by design - `chain_for` gives it one rung, so it has no rung
            # to drop. An empty list means the chain held; the two are not the
            # same reading and only the list becomes a number here.
            fell = n.get("failover")
            if isinstance(fell, list):
                g.add("failover", len(fell))
            for reason, count in (n.get("retry_reasons") or {}).items():
                got = _reading(count)
                if got is not None:
                    g.reasons[str(reason)] = g.reasons.get(str(reason), 0) + got
            word = str(n.get("stopped") or "")
            if word:
                g.stopped[word] = g.stopped.get(word, 0) + 1
    ordered = sorted(groups.values(),
                     key=lambda g: (-g.nodes, g.backend, g.task_class))
    return ordered, nodeless


def _stopped_cell(g: "Group") -> str:
    if not g.stopped:
        return "-"
    top = sorted(g.stopped.items(), key=lambda kv: (-kv[1], kv[0]))
    text = " · ".join(f"{w} {c}" for w, c in top[:3])
    if len(top) > 3:
        text += f" +{len(top) - 3}"
    return text


@dataclass
class PhaseGroup:
    """Every phase run one way, so the two arms of a paired comparison can be read.

    The key is (mode, wording, backend). Mode and wording are the two things
    the pair is testing and merging them would answer neither; backend is in
    because a loop on one endpoint against a fanout on another is not a
    comparison of loop against fanout.
    """

    mode: str
    wording: str
    backend: str
    phases: int = 0
    runs: set = field(default_factory=set)
    verdicts: dict = field(default_factory=dict)
    readings: dict = field(default_factory=dict)

    def add(self, key: str, value) -> None:
        got = _reading(value)
        if got is not None:
            self.readings.setdefault(key, []).append(got)


def phase_groups(runs: list[dict]) -> list["PhaseGroup"]:
    """Group every phase on every record. Empty when nothing carried one."""
    out: dict[tuple, PhaseGroup] = {}
    for raw in runs:
        for p in raw.get("phases") or []:
            if not isinstance(p, dict):
                continue
            key = (str(p.get("mode") or UNKNOWN_BACKEND),
                   str(p.get("wording") or UNKNOWN_BACKEND),
                   str(p.get("backend") or "") or UNKNOWN_BACKEND)
            g = out.get(key)
            if g is None:
                g = out[key] = PhaseGroup(mode=key[0], wording=key[1],
                                          backend=key[2])
            g.phases += 1
            g.runs.add(str(raw.get("session") or ""))
            word = str(p.get("verdict") or "")
            if word:
                g.verdicts[word] = g.verdicts.get(word, 0) + 1
            for col in ("seconds", "steps", "tool_calls", "input_tokens",
                        "output_tokens"):
                g.add(col, p.get(col))
    return sorted(out.values(),
                  key=lambda g: (-g.phases, g.mode, g.wording, g.backend))


def _phase_table(runs: list[dict]) -> list[str]:
    """The phase ledger, or nothing at all.

    Absent rather than empty when no record carries a phase: most runs are not
    driven off a checklist, and a permanently empty table under every report
    would teach the reader to skip the place the answer will appear.
    """
    groups = phase_groups(runs)
    if not groups:
        return []
    rows = [[g.mode, g.wording, g.backend, str(g.phases), str(len(g.runs)),
             _cell(g.readings.get("seconds", []), g.phases, _secs),
             _cell(g.readings.get("steps", []), g.phases, _plain),
             _cell(g.readings.get("tool_calls", []), g.phases, _plain),
             _cell(g.readings.get("input_tokens", []), g.phases, _tokens),
             _cell(g.readings.get("output_tokens", []), g.phases, _tokens),
             _verdict_cell(g)]
            for g in groups]
    return ["", "  phase ledger - one row per way of working a checklist"] + \
        _table(["mode", "wording", "backend", "phases", "runs", "wall",
                "steps", "tools", "tok in", "tok out", "verdicts"], rows)[2:]


def _verdict_cell(g: "PhaseGroup") -> str:
    if not g.verdicts:
        return "-"
    top = sorted(g.verdicts.items(), key=lambda kv: (-kv[1], kv[0]))
    return " · ".join(f"{w} {c}" for w, c in top)


def report(runs: list[dict], unreadable: int = 0) -> str:
    """`--orc-report`: what every run record on disk adds up to.

    Two tables over one grouping - what a backend spent, and how rough getting
    there was - because they answer the same question from either side and a
    single table of eleven columns is read by nobody.

    An empty store is a sentence, not an empty table, for the reason
    `summarise` says it: an empty table is the one output that reads as a bug.
    """
    if not runs:
        line = render.yellow(
            f"no run records yet (looked in {render.short_path(str(RUN_DIR))})")
        if unreadable:
            line += render.yellow(f", and {unreadable} file(s) would not read")
        return line

    groups, nodeless = group(runs)
    nodes = sum(g.nodes for g in groups)
    # Shortened: the header is capped at `render.width()` and an absolute
    # run-dir path pushed it past 96 columns, so the last thing on the line
    # wrapped mid-path and read as a broken table rather than a long path.
    meta = [f"{len(runs)} runs", f"{nodes} nodes",
            render.short_path(str(RUN_DIR))]
    if nodeless:
        meta.insert(2, f"{nodeless} with no node")
    if unreadable:
        meta.insert(2, f"{unreadable} unreadable")
    lines = [render.bold("  run records") + render.dim("  " + " · ".join(meta)),
             ""]

    spend = [[g.backend, g.model_cell(), g.task_class, _depth_cell(g),
              str(g.nodes),
              str(len(g.runs)),
              _cell(g.readings.get("seconds", []), g.nodes, _secs),
              _cell(g.readings.get("steps", []), g.nodes, _plain),
              _cell(g.readings.get("input_tokens", []), g.nodes, _tokens),
              _cell(g.readings.get("output_tokens", []), g.nodes, _tokens),
              _cell(g.readings.get("usd", []), g.nodes, _usd)]
             for g in groups]
    lines += _table(["backend", "model", "class", "lvl", "nodes", "runs", "wall",
                     "steps", "tok in", "tok out", "usd"], spend)
    lines.append("")

    rough = [[g.backend, g.task_class, str(g.nodes),
              _sum_cell(g.readings.get("failover", []), g.nodes),
              _sum_cell(g.readings.get("retries", []), g.nodes),
              _sum_cell(g.readings.get("compactions", []), g.nodes),
              _sum_cell(g.readings.get("loop_stops", []), g.nodes),
              _stopped_cell(g)]
             for g in groups]
    lines += _table(["backend", "class", "nodes", "failover", "retries",
                     "compact", "loops", "stopped"], rough)

    lines += _phase_table(runs)

    splits = [(g, g.reasons) for g in groups if g.reasons]
    if splits:
        lines.append("")
        lines.append("  retries, by the class the wire gave them")
        lines += _table(
            ["backend", "class", "reasons"],
            [[g.backend, g.task_class,
              ", ".join(f"{r} {c:g}" for r, c in sorted(reasons.items()))]
             for g, reasons in splits])[2:]

    lines += ["",
              "  reading the cells",
              "    -         nothing in this group carried the column, which "
              "is not a zero",
              "    7/19      seven nodes carried it; twelve could not answer",
              "    !         one reading, so it is a number and not a median",
              "    wall/steps/tokens are medians; the four counters are sums",
              "    lvl       the tree levels this row's nodes ran at: 0 is the "
              "orchestrator",
              f"    class {UNRECORDED} is a record written before the field "
              "existed, not a run without a class"]
    return "\n".join(lines)


__all__ = ["Orc", "Slot", "Resolved", "Node", "Phase", "Record", "Group",
           "load_orcs", "find",
           "resolve", "resolve_worker", "echo", "catalogue",
           "summarise", "describe", "write", "read", "path_for",
           "record_for", "now", "clean_task_class", "records", "group",
           "report", "phase_groups", "PhaseGroup", "RUN_DIR", "SEARCH", "NO_CLASS", "UNRECORDED",
           "UNKNOWN_BACKEND", "NO_WORKER_NAMED", "NO_ORCHESTRATOR_NAMED",
           "FLAG", "PIN", "FILE", "DEFAULT", "INHERIT"]

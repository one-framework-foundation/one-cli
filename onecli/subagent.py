"""Subagents - a nested agent loop that returns one result, not a transcript.

A subagent is not a second program. It is this loop again, one level down,
standing on the SAME guard chain, the SAME hooks, the SAME permission surface
and the SAME live budget as its parent. That is the whole point. A `task` tool
that spawned a fresh unguarded agent would be a hole around every gate the
parent run stands behind, and the easiest hole in the world to walk through:
"I was not allowed to write that file, so I asked a subagent to."

What a child does NOT share is context. It gets its own history, and the parent
sees only what it answered. That is the only reason to spawn one: a sweep over
forty files costs the parent one summary instead of forty tool results.

Three things a child is deliberately denied, each for a reason:

- **The person.** No `asker`. The parent owns the conversation; a question from
  a nested loop arrives at a terminal with no way to say which of three running
  children asked it.
- **Background jobs.** No job store. A job outliving the child that started it
  is a process nobody can see and nobody closes.
- **Its own budget.** The meter is the parent's object, not a copy, so a
  ceiling is a ceiling for the whole tree rather than per level - which is how
  a "$1 run" becomes six dollars in six children.

A `--read-only` or `--permission-mode plan` run has no `task` tool at all. Not
because a child could escape - it could not, the surface is intersected - but
because `task` is classified mutating, which is the answer a guard chain needs
when the child CAN write, and a surface that calls the same tool read-only in
one mode and mutating in another is a surface nobody can reason about. A
read-only parent does its own reading; a default-mode parent fans out readers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from onecli import loopguard as loops

if TYPE_CHECKING:
    from onecli.agent import Agent

# How deep the nesting may go. 1 means a top-level run may spawn children and
# those children may not spawn their own. Deeper than 2 has never been the
# thing that was missing, and each level multiplies the tree.
DEFAULT_MAX_DEPTH = 2

CHILD_SYSTEM = """
You are a subagent. You were given one task by a parent agent and you answer it
alone - you cannot ask anyone a question, and nobody will read your working.

Your final message IS the result. Write the answer itself: the finding, the
list, the paths, the numbers. Do not describe what you did, do not say "I have
completed the task", and do not offer to do more. If you could not do it, say
what stopped you and what you did find.
""".strip()


@dataclass(frozen=True)
class Preset:
    """A named child: what it is for, and the surface it gets.

    `tools=None` means "whatever the parent has". A preset naming tools it may
    not have is not an error - the surface is intersected with the parent's, so
    a preset is a request, never a grant.
    """

    name: str
    description: str = ""
    system: str = ""
    tools: tuple[str, ...] | None = None
    max_steps: int = 0
    # A vendor CLI to run instead of this loop. "" is us. See `onecli/drivers.py`
    # for what a driver child inherits and what it cannot.
    driver: str = ""
    timeout: float = 0.0
    # A backend to run this child on instead of the parent's. "" is the
    # parent's own chain. Named the way `--model` names one, and resolved the
    # same way, so a preset cannot reach an endpoint the operator has not
    # configured. A pinned child gets ONE rung and no fallback: a child that
    # quietly answered on the parent's model would make the whole reason to
    # pin it - this work is cheap, that work is not - untrue and invisible.
    model: str = ""
    response_gate: str = ""
    progress_policy: dict | None = None

    def seconds(self, default: float) -> float:
        return self.timeout if self.timeout > 0 else default

    def steps(self) -> int | None:
        """The preset's own step limit; None leaves the parent's in force."""
        return self.max_steps if self.max_steps > 0 else None


GENERAL = Preset(
    name="general",
    description="A general-purpose child with the parent's own tool surface.",
)

EXPLORER = Preset(
    name="explorer",
    description=("Reads and searches only. For sweeping a tree, finding where "
                 "something lives, or answering a question from the code."),
    tools=("read", "ls", "glob", "grep"),
)

BUILTIN: dict[str, Preset] = {p.name: p for p in (GENERAL, EXPLORER)}


def _response_gate(value):
    from onecli.response_gate import mode
    return mode(value) if value != "" else ""


def _preset_from(name: str, body: dict) -> Preset:
    from onecli import progress_policy as progress
    tools = body.get("tools")
    if "progress_policy" in body:
        progress.resolve(body["progress_policy"])
    return Preset(
        name=name,
        description=str(body.get("description") or ""),
        system=str(body.get("system") or body.get("prompt") or ""),
        tools=tuple(str(t) for t in tools) if isinstance(tools, list) else None,
        max_steps=int(body.get("max_steps") or 0),
        driver=str(body.get("driver") or ""),
        timeout=float(body.get("timeout") or 0.0),
        model=str(body.get("model") or ""),
        response_gate=_response_gate(body.get("response_gate", "")),
        progress_policy=body.get("progress_policy"),
    )


def load_presets(paths: list[str] | None) -> tuple[dict[str, Preset], list[str]]:
    """Read agent presets from JSON files. Returns (presets, errors).

    A broken preset file is reported, never fatal: a run that cannot read one
    agent definition should still run with the built-ins rather than refuse to
    start. Three shapes are accepted because all three are what people write -
    a bare map of name to body, the same map under an "agents" key, or a list
    of objects each carrying its own name.
    """
    presets = dict(BUILTIN)
    errors: list[str] = []
    for path in paths or []:
        try:
            raw = json.loads(open(os.path.expanduser(path), encoding="utf-8").read())
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if isinstance(raw, dict) and isinstance(raw.get("agents"), (dict, list)):
            raw = raw["agents"]
        if isinstance(raw, dict):
            items = [(str(k), v) for k, v in raw.items()]
        elif isinstance(raw, list):
            items = [(str(b.get("name") or ""), b) for b in raw if isinstance(b, dict)]
        else:
            errors.append(f"{path}: expected an object or a list of agents")
            continue
        for name, body in items:
            if not name or not isinstance(body, dict):
                errors.append(f"{path}: an agent with no name was skipped")
                continue
            try:
                presets[name] = _preset_from(name, body)
            except (ValueError, TypeError) as exc:
                errors.append(f"{path}: {name}: {exc}")
    return presets, errors


def child_surface(parent: list[str] | None,
                  wanted: tuple[str, ...] | None) -> list[str] | None:
    """The tools a child gets: never more than the parent, at most what it asked.

    `None` on either side means "everything available", so a parent with the
    full surface and a preset with no opinion returns None and the child is
    built exactly as a top-level run is. The intersection is the read-only
    inheritance rule in one line: a parent that cannot write has no write tool
    to pass down, whatever the preset asked for.
    """
    if wanted is None:
        return None if parent is None else list(parent)
    if parent is None:
        return list(wanted)
    allowed = set(parent)
    return [t for t in wanted if t in allowed]


@dataclass
class Spawn:
    """What one child cost, for the parent's record."""

    agent: str
    prompt: str
    text: str = ""
    # Which level of the tree this child ran at. The root is 0, its own
    # children are 1, their children 2. Recorded because the record is a flat
    # list of nodes and two children of the same preset at two different
    # levels are two different measurements, not one. Set by `Spawner.spawn`
    # from the spawner's depth, never by the caller.
    depth: int = 0
    steps: int = 0
    tool_calls: int = 0
    stopped: str = ""
    seconds: float = 0.0
    # Which vendor CLI ran this child, "" for our own loop. `metered` is
    # whether its token counts were reported rather than estimated - see
    # `drivers.charge`; a driver's charge is a floor, not a measurement.
    driver: str = ""
    metered: bool = True
    # Where this child actually ran, in TWO vocabularies that are not
    # interchangeable. `backend` is the configured backend NAME - what
    # `Spawner.pin_for` returned, or the parent's own rung when the child
    # inherited - and `model` is the model id that name resolved to.
    #
    # They are kept apart because two backends can serve one model id:
    # `lane-a` and `lane-b` both serve `shared-model`. A record that
    # writes the model id under a field called `backend` therefore merges two
    # distinct endpoints into one row, silently, and a cost comparison built on
    # that field cannot identify the thing it names. The echo and the root node
    # both speak backend names, so a child that spoke model ids made the same
    # record disagree with itself.
    backend: str = ""
    model: str = ""
    response_gate: str = ""
    # What it cost to get the answer, not just whether one arrived.
    #
    # `failover` is None where the question does not apply: a pinned child has
    # a chain of exactly one rung by design (`chain_for`), so it can never drop
    # one, and an empty list there would read as "the chain held" rather than
    # "there was no chain". An unpinned child shares the PARENT's chain, so
    # only the rungs dropped during this child's run are its own.
    #
    # The rest are None on the driver path: a vendor CLI keeps no counters we
    # can read, and a zero would be a claim we cannot make.
    failover: list[dict] | None = None
    retries: int | None = None
    retry_reasons: dict | None = None
    compactions: int | None = None
    loop_stops: int | None = None


@dataclass
class Spawner:
    """The parent's authority to make a child, and the record of the ones made.

    Held by the parent Agent and handed to the `task` tool. A run with no
    spawner is not offered the tool at all, which is also how the depth cap is
    enforced: the child at the last allowed level is built without one.
    """

    parent: Agent | None
    depth: int = 0
    max_depth: int = DEFAULT_MAX_DEPTH
    presets: dict[str, Preset] = field(default_factory=lambda: dict(BUILTIN))
    spawns: list[Spawn] = field(default_factory=list)
    # The backend every UNPINNED child runs on - what `--orchestrator` sets up:
    # the parent drives on the strong model and the work goes out to this one.
    # A preset that names its own model still wins; empty means children
    # inherit the parent's chain, which is the behaviour with no orchestrator.
    worker: str = ""
    # Per-preset backends from an `.orc.json` - see `onecli/orc.py`. Sits
    # between the preset's own pin and the blanket worker default: the file
    # knows this RUN's economy, the preset knows what the AGENT needs, and
    # only the preset's author knew the second thing.
    workers: dict[str, str] = field(default_factory=dict)

    def _parent(self) -> Agent:
        """The owning Agent. It is None only between construction and the
        Agent constructor filling it in; acting before that is a bug."""
        if self.parent is None:
            raise RuntimeError("spawner has no parent agent yet")
        return self.parent

    def pin_for(self, preset: Preset) -> str:
        """The backend this preset runs on, or "" for the parent's own chain.

        The one place the precedence chain is written down, so `chain_for`,
        `_window_for`, the catalogue and the spawn note cannot drift apart -
        which is exactly how a child ends up sized for one backend and pointed
        at another.
        """
        return preset.model or self.workers.get(preset.name, "") or self.worker

    def names(self) -> list[str]:
        return sorted(self.presets)

    def catalogue(self) -> str:
        """One line per agent, for the parent's system prompt.

        A driver is named. Which CLI runs a child changes what it is good at
        and what it costs, and a model choosing between `explorer` and a
        `reviewer` that is really opencode should be choosing knowingly.
        """
        lines = []
        for name in self.names():
            preset = self.presets[name]
            note = preset.description or "no description"
            if preset.driver:
                note += f" (runs on the {preset.driver} CLI)"
            pin = self.pin_for(preset)
            if pin:
                note += f" (runs on {pin})"
            lines.append(f"- {name}: {note}")
        return "\n".join(lines)

    # -- making one -------------------------------------------------------

    def _grandchild(self) -> "Spawner | None":
        """A spawner for the child to hold, or None at the cap.

        None is what enforces the cap, not a check inside `spawn`: a child at
        the last allowed level is built without the tool, so it never asks for
        something it would only be refused. A model that cannot see a door does
        not knock on it. `parent` is filled in by the Agent constructor - the
        agent does not exist yet at this point, and the alternative is building
        the child and then widening its tool surface afterwards.
        """
        if self.depth + 1 >= self.max_depth:
            return None
        return Spawner(parent=None, depth=self.depth + 1, worker=self.worker,
                       max_depth=self.max_depth, presets=self.presets,
                       workers=dict(self.workers),
                       # THE SAME list object, not a copy. `record_for` is
                       # handed the ROOT spawner's spawns, so a child that
                       # kept its own list spent a grandchild and wrote
                       # nothing down: a depth-3 run and a depth-2 run
                       # produced identical records. Sharing the list is what
                       # makes a level below the first one evidence.
                       spawns=self.spawns)

    def chain_for(self, preset: Preset):
        """The chain a child runs on: the parent's, or one pinned rung.

        A pinned child gets a chain of length one on purpose. Fallback exists
        so a run survives an outage; a child pinned to a cheap backend that
        quietly answered on the parent's expensive one would defeat the only
        reason anybody pins it, and the parent would never see that it had.
        Resolution goes through `config.resolve`, so a preset can only name a
        backend the operator has already configured, and a name that matches
        nothing raises here rather than answering somewhere else.
        """
        parent = self._parent()
        pin = self.pin_for(preset)
        if not pin:
            return parent.chain
        from onecli import config, fallback, providers

        backend = config.resolve(name=pin)
        rung = fallback.Rung(name=backend.name,
                             provider=providers.build(backend),
                             model=backend.model)
        return fallback.Chain(rungs=[rung], source="preset")

    def _window_for(self, preset: Preset) -> int:
        """The child's context window.

        A pinned child gets the window of the backend it actually runs on, not
        the parent's: a 200k parent window over a 32k child never compacts and
        runs that endpoint into the ground mid-loop.
        """
        pin = self.pin_for(preset)
        if not pin:
            return self._parent().tracker.window
        from onecli import config

        backend = config.resolve(name=pin)
        return (config.resolve_context_window(
            backend.model, backend.name, backend.base_url)
            or self._parent().tracker.window)

    def build(self, preset: Preset):
        """The child Agent, guards and budget inherited. Separated from `spawn`
        so a test can inspect what a child got without running a model."""
        from onecli.agent import Agent  # lazy: agent imports this module

        from onecli import progress_policy as progress
        parent = self._parent()
        if getattr(parent, "request_budget_isolated", False):
            raise ValueError("isolated request budget forbids child agents")
        child = Agent(
            request_admission=parent.request_admission,
            chain=self.chain_for(preset),
            workspace=parent.workspace,
            system=_child_system(preset),
            max_steps=preset.steps() or parent.max_steps,
            response_gate=preset.response_gate or parent.response_gate,
            progress_policy=progress.resolve(preset.progress_policy, base=parent.progress_policy),
            on_event=lambda kind, data: parent.on_event(kind, data)
                if kind in ("acknowledgment", "response_gate", "progress_update") else None,
            tool_names=child_surface(parent.tool_names, preset.tools),
            max_tokens=parent.max_tokens,
            temperature=parent.temperature,
            guard=parent.guard,
            hooks=parent.hooks,
            trust_project=getattr(parent, "trust_project", False),
            spill=parent.spill,
            stream=False,
            autocompact=parent.autocompact,
            context_window=self._window_for(preset),
            compact_at=parent.tracker.compact_at,
            keep_last=parent.tracker.keep_last,
            # Its own repeat counters. Sharing the parent's would make a child
            # reading the same file the parent read look like a loop.
            loop_guard=loops.LoopGuard(repeat_limit=parent.loop.repeat_limit,
                                       deadline=parent.loop.deadline),
            # No person to ask, no jobs to leave behind. See the module docstring.
            jobs=None,
            asker=None,
            spawner=self._grandchild(),
        )
        # One meter, one ceiling, for the whole tree. Assigned rather than
        # passed because the constructor builds its own - and a copy would let
        # every level spend the full budget again.
        child.meter = parent.meter
        child.max_budget_usd = parent.max_budget_usd
        child.max_input_tokens = parent.max_input_tokens
        return child

    def steer(self, child, preset: Preset, prompt: str,
              messages: list[dict] | None = None):
        """Run an ALREADY-BUILT child on one prompt, and record what it cost.

        Split out of `spawn` for the manager loop (`onecli/manager.py`), which
        holds ONE implementer across a whole checklist and hands it a new
        instruction per phase. Sharing this body rather than copying it is what
        keeps a re-steered child and a one-shot child comparable: the same
        Spawn, the same fields, the same `failover` asymmetry, so a paired
        run of the two differs in ONE thing and not in how it was written
        down.

        `messages` is the child's own history carried forward, which is the
        whole difference between the two arms: `None` is a fresh conversation.
        """
        import time

        pinned = bool(self.pin_for(preset))
        before = 0 if pinned else len(child.chain.history)
        started = time.monotonic()
        result = child.run(prompt, messages)
        elapsed = time.monotonic() - started
        self.spawns.append(Spawn(
            agent=preset.name, prompt=prompt, text=result.text,
            response_gate=getattr(child, "response_gate", "") or "",
            depth=self.depth + 1, steps=len(result.steps),
            tool_calls=result.tool_calls, stopped=result.stopped,
            seconds=elapsed,
            backend=child.chain.current.name, model=child.chain.current.model,
            failover=None if pinned else list(child.chain.history[before:]),
            retries=result.retries,
            retry_reasons=dict(result.retry_reasons),
            compactions=len(result.compactions),
            loop_stops=result.loop_stops,
        ))
        return result

    def spawn(self, prompt: str, agent: str = "") -> str:
        """Run one child to completion and return its answer.

        The transcript stays with the child. What comes back is the final text
        plus a line of accounting, because a parent deciding whether to spawn
        another one needs to know what the last one cost.
        """

        name = (agent or GENERAL.name).strip()
        preset = self.presets.get(name)
        if preset is None:
            return (f"error: no agent named {name!r}. "
                    f"Available: {', '.join(self.names())}")

        if preset.driver:
            return self._drive(preset, prompt)

        try:
            child = self.build(preset)
        except (RuntimeError, ValueError) as exc:
            # A preset naming a backend this host has not got is a config
            # error, and the parent can still do the work itself. A traceback
            # out of a tool would end its turn instead.
            return (f"error: the {name} subagent is pinned to "
                    f"{preset.model!r}, which did not resolve: {exc}")
        result = self.steer(child, preset, prompt)
        elapsed = self.spawns[-1].seconds
        # Where the chain is the parent's, its history already holds the
        # parent's own demotions; only what happened from here on is the
        # child's. A pinned child has one rung and gets None - see `Spawn`.
        text = (result.text or "").strip()
        if not text:
            return (f"the {name} subagent returned nothing "
                    f"({result.stopped}{': ' + result.error if result.error else ''})")
        note = f"[{name}: {len(result.steps)} steps, {result.tool_calls} tools, {elapsed:.1f}s"
        if self.pin_for(preset):
            note += f", on {child.chain.current.name}"
        if result.stopped != "answered":
            note += f", stopped: {result.stopped}"
        return f"{text}\n\n{note}]"

    # -- driving somebody else's CLI --------------------------------------

    def _drive(self, preset: Preset, prompt: str) -> str:
        """Run a vendor CLI as this child and return its answer.

        Every way this can go wrong returns a sentence rather than raising. A
        parent whose driver child failed still has a run: it can try another
        driver, do the work itself, or say what stopped it. A traceback out of
        a tool would end the parent's turn over somebody else's missing binary.
        """
        from onecli import drivers

        parent = self._parent()
        if parent.request_admission is not None:
            return 'error: request budget forbids external CLI drivers'
        from onecli import response_gate as response_policy
        policy = response_policy.mode(preset.response_gate or parent.response_gate)
        if policy == "enforce":
            return (f"error: {preset.driver} has no first-response enforcement seam; "
                    "configure this node response_gate=prompt or off explicitly")
        if policy == "prompt":
            prompt = response_policy.PROMPT + "\n\n" + prompt
        from onecli import progress_policy as progress
        cadence = progress.resolve(preset.progress_policy, base=parent.progress_policy)
        if cadence["enabled"]:
            prompt = progress.PROMPT + "\n\n" + prompt
        reason = drivers.refusal(preset.driver, parent.tool_names)
        if reason:
            return f"error: {reason}"
        # A pinned driver child runs the vendor CLI against the pinned
        # backend, not the parent's - same reason as our own loop.
        try:
            chain = self.chain_for(preset)
        except (RuntimeError, ValueError) as exc:
            return (f"error: the {preset.name} subagent is pinned to "
                    f"{preset.model!r}, which did not resolve: {exc}")
        backend = drivers.backend_for(chain.current)
        reason = drivers.wire_refusal(preset.driver, backend)
        if reason:
            return f"error: {reason}"

        read_only = drivers.read_only_for(parent.tool_names)
        workspace = parent.workspace or os.getcwd()

        def gate(argv: list[str]) -> str:
            verdict = drivers.guard_verdict(parent.guard, argv, workspace)
            if verdict is not None and not verdict.allowed:
                return (f"the guard refused to hand this workspace to "
                        f"{preset.driver}: {verdict.reason}")
            return ""

        run = drivers.run(preset.driver, prompt, backend=backend,
                          workspace=workspace, system=_child_system(preset),
                          read_only=read_only,
                          timeout=preset.seconds(drivers.DEFAULT_TIMEOUT),
                          before=gate)
        result = run.result
        drivers.charge(parent.meter, prompt, result)
        self.spawns.append(Spawn(
            agent=preset.name, prompt=prompt, text=result.text,
            depth=self.depth + 1, steps=result.steps, tool_calls=result.tool_calls,
            stopped=result.stopped, seconds=run.seconds,
            driver=preset.driver, metered=run.metered, response_gate=policy,
            backend=chain.current.name, model=chain.current.model,
        ))

        text = (result.text or "").strip()
        if not text:
            return (f"error: the {preset.driver} driver returned no answer "
                    f"({result.stopped}"
                    f"{': ' + result.error if result.error else ''})")
        note = (f"[{preset.name} via {preset.driver}: {run.seconds:.1f}s"
                f"{f', {result.steps} steps' if result.steps else ''}"
                f"{f', {result.tool_calls} tools' if result.tool_calls else ''}")
        if result.stopped not in ("answered", ""):
            note += f", stopped: {result.stopped}"
        if not run.metered:
            note += ", tokens not reported by this CLI"
        return f"{text}\n\n{note}]"


def _child_system(preset: Preset) -> str:
    """The child's system prompt: what a subagent is, then what THIS one is."""
    if preset.system.strip():
        return f"{CHILD_SYSTEM}\n\n{preset.system.strip()}"
    return CHILD_SYSTEM


def task(workspace: str = "", spawner: Spawner | None = None,
         prompt: str = "", agent: str = "") -> str:
    """The `task` tool body. `workspace` is unused - the child inherits it."""
    if spawner is None:
        return "error: subagents are not available in this run"
    if not (prompt or "").strip():
        return "error: task needs a prompt describing the whole job"
    return spawner.spawn(prompt, agent)


__all__ = ["Preset", "Spawn", "Spawner", "BUILTIN", "GENERAL", "EXPLORER",
           "load_presets", "child_surface", "task", "CHILD_SYSTEM",
           "DEFAULT_MAX_DEPTH"]

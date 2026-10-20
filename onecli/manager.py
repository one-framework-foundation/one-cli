"""The manager loop - one implementer, steered by phase.

Instead of the orchestrator fanning out a fresh one-shot child per unit of
work, ONE long-lived implementer is re-steered phase by phase
against a fixed checklist, and each phase lands on the run record with its own
start, end, verdict and cost.

**The manager is code, not a model.** The checklist is fixed, read off disk,
and walked in order by `run`. That is deliberate and it is what makes the pair
comparison actually measurable: the only difference between the two arms
is whether the implementer's history is carried forward, so a difference in the
numbers is a fact about the implementer rather than about how two different
manager models happened to feel that afternoon. A manager that were itself a
model would put a second unmeasured variable inside the instrument.

**Two arms, one body.** `loop` builds the child once and hands it each phase
with its own history carried forward; `fanout` builds a fresh child per phase
and hands it nothing. Both go through `Spawner.steer`, so both write the same
`Spawn` with the same fields - a comparison between two arms recorded by two
different code paths is a comparison of the code paths.

**The wording knob.** A published claim about prompt wording holds that
asking for a phase "extremely well" beats asking for it "perfectly", because
"perfect" sends the model back into the minutiae. It is a parameter of the
instrument, not a default baked into a prompt: `WORDINGS` holds both, the
phase prompt is otherwise byte-identical, and the record carries which one was
used. A claim about prompt wording that cannot be told apart afterwards has not
been tested.

What is NOT here: a separate progress page - the record is the surface, and a
second one to keep in sync with it is a second answer to the same question -
and a published claim about subagent concurrency, which depends on another
harness whose child cost and child startup differ from these.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from onecli import orc as orclib

# The two ways to ask, and nothing else. Free text here would make every run a
# different experiment and the `wording` column on the record unreadable.
WORDINGS = {
    "extremely-well": "Do this phase extremely well.",
    "perfectly": "Do this phase perfectly.",
}
DEFAULT_WORDING = "extremely-well"
MODES = ("loop", "fanout")
DEFAULT_MODE = "loop"


@dataclass(frozen=True)
class Item:
    """One line of the checklist: what to do, under a name to report it by."""

    name: str
    instruction: str


def load(ref: str) -> tuple[list[Item], str]:
    """Read a checklist from a path or an inline JSON list. Returns (items, error).

    An error rather than an exception: a bad checklist is an operator typo and
    should cost a sentence, not a traceback out of `main`.
    """
    text = ref
    path = os.path.expanduser(ref)
    # Either JSON shape is inline. `{` was missed on the first pass, so
    # `--phases '{"phases": [...]}'` - the shape `load` itself accepts below -
    # was read as a filename and failed with ENOENT quoting the whole document.
    if not ref.lstrip().startswith(("[", "{")):
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            return [], f"{ref}: {exc}"
    try:
        raw = json.loads(text)
    except ValueError as exc:
        return [], f"{ref}: {exc}"
    if isinstance(raw, dict):
        raw = raw.get("phases") or raw.get("checklist") or []
    if not isinstance(raw, list) or not raw:
        return [], f"{ref}: expected a non-empty list of phases"
    items: list[Item] = []
    for i, body in enumerate(raw, 1):
        if isinstance(body, str):
            items.append(Item(name=f"phase {i}", instruction=body))
            continue
        if not isinstance(body, dict):
            return [], f"{ref}: phase {i} is not a string or an object"
        instruction = str(body.get("instruction") or body.get("do") or "")
        if not instruction.strip():
            return [], f"{ref}: phase {i} has no instruction"
        items.append(Item(name=str(body.get("name") or f"phase {i}"),
                          instruction=instruction))
    return items, ""


def phase_prompt(task: str, item: Item, index: int, total: int,
                 wording: str) -> str:
    """The instruction for one phase. Identical across both arms and both
    wordings except for the one sentence being tested."""
    return "\n".join([
        f"Overall task: {task}",
        f"Phase {index} of {total}: {item.name}",
        item.instruction,
        WORDINGS.get(wording, WORDINGS[DEFAULT_WORDING]),
        "Answer with the result of THIS phase only.",
    ])


def _reading(meter) -> tuple[int | None, int | None]:
    """The meter's two token counters, or (None, None) if it has none.

    None rather than 0, the same rule the run record runs on: a meter that
    cannot answer is not a phase that spent nothing.
    """
    if meter is None:
        return (None, None)
    got = dict(getattr(meter, "as_dict", dict)() or {})
    return (got.get("input_tokens"), got.get("output_tokens"))


def _delta(before: tuple, after: tuple) -> tuple[int | None, int | None]:
    """After minus before, per column, and None wherever either end is None."""
    def col(i: int) -> int | None:
        a, b = after[i], before[i]
        return None if (a is None or b is None) else int(a) - int(b)

    return (col(0), col(1))


def _verdict(stopped: str) -> str:
    """`done` when the loop answered, else the word it stopped on.

    Not the same field as `stopped` on purpose - see `orc.Phase`. Telling a
    stalled phase from a finished one needs both.
    """
    return "done" if stopped == "answered" else (stopped or "unknown")


def run(spawner, preset, task: str, items: list[Item], *,
        mode: str = DEFAULT_MODE, wording: str = DEFAULT_WORDING,
        on_phase=None) -> tuple[list[orclib.Phase], object]:
    """Walk the checklist. Returns (ledger, the last phase's Result).

    The Spawns land on `spawner.spawns` exactly as a `task` tool call's would,
    so the run record's node list is the same shape in both arms and the
    existing report reads it without knowing this module exists.
    """
    if mode not in MODES:
        raise ValueError(f"unknown phase mode {mode!r}; expected one of "
                         f"{', '.join(MODES)}")
    ledger: list[orclib.Phase] = []
    child = spawner.build(preset) if mode == "loop" else None
    # The implementer's own history, carried forward in `loop` and never in
    # `fanout`. This one variable IS the experiment.
    carried: list[dict] | None = None
    result = None
    for index, item in enumerate(items, 1):
        if mode == "fanout":
            child = spawner.build(preset)
            carried = None
        assert child is not None
        started_at = orclib.now()
        # The DELTA over the shared meter, never `result.cost`. Every agent in
        # the tree shares ONE meter object (`Spawner.build` assigns the
        # parent's) so a ceiling is a ceiling for the whole tree - which means
        # `result.cost` is the running total for everything that has run so
        # far, and writing it onto a phase would make phase 4 look four times
        # the size of phase 1 whatever it did. The loop is serial, so a
        # before/after difference is this phase's own spend and nothing else's.
        meter = getattr(child, "meter", None)
        was = _reading(meter)
        started = time.monotonic()
        result = spawner.steer(
            child, preset,
            phase_prompt(task, item, index, len(items), wording),
            carried)
        elapsed = time.monotonic() - started
        if mode == "loop":
            carried = list(getattr(result, "messages", None) or []) or None
        spent = _delta(was, _reading(meter))
        phase = orclib.Phase(
            index=index, name=item.name, agent=preset.name, mode=mode,
            wording=wording, started=started_at, finished=orclib.now(),
            seconds=round(elapsed, 3), steps=len(result.steps),
            tool_calls=result.tool_calls, stopped=result.stopped,
            verdict=_verdict(result.stopped),
            backend=child.chain.current.name, model=child.chain.current.model,
            input_tokens=spent[0],
            output_tokens=spent[1],
        )
        ledger.append(phase)
        if on_phase:
            on_phase(phase)
        # A phase that did not answer ends the checklist. Carrying on would
        # steer the next phase off a history whose last turn is a failure, and
        # a ledger that ran every phase after the run had stopped working is a
        # ledger measuring the wrong thing.
        if phase.verdict != "done":
            break
    return ledger, result


__all__ = ["Item", "Phase", "WORDINGS", "MODES", "DEFAULT_MODE",
           "DEFAULT_WORDING", "load", "phase_prompt", "run"]
Phase = orclib.Phase

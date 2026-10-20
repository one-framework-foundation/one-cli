"""The manager loop - one implementer steered by a fixed checklist.

The two claims under test are somebody else's until they are run here, so what
these tests pin is the INSTRUMENT: that the two arms differ in exactly one
thing, that the ledger says which arm produced it, and that a phase's tokens
are the phase's own.
"""

import json

import pytest

from onecli import manager, orc as orclib, subagent as sub


class _Meter:
    """The tree's one meter, counting up across every phase."""

    def __init__(self):
        self.input = 0
        self.output = 0

    def as_dict(self):
        return {"input_tokens": self.input, "output_tokens": self.output}


class _Chain:
    def __init__(self, name, model):
        self.current = type("R", (), {"name": name, "model": model})()
        self.history = []


class _Child:
    """A child that remembers what history it was handed, and spends."""

    def __init__(self, meter, chain, seen):
        self.meter = meter
        self.chain = chain
        self.seen = seen

    def run(self, prompt, messages=None):
        self.seen.append((id(self), prompt, list(messages or [])))
        self.meter.input += 100
        self.meter.output += 10
        return type("R", (), {
            "text": "ok", "steps": [1], "tool_calls": 0,
            "stopped": "answered", "retries": 0, "retry_reasons": {},
            "compactions": [], "loop_stops": 0, "error": "",
            "messages": [{"role": "assistant", "content": prompt}],
            "cost": self.meter.as_dict()})()


@pytest.fixture
def rig(monkeypatch):
    meter, chain, seen = _Meter(), _Chain("kimi", "kimi-for-coding"), []
    made = []

    def build(self, preset):
        child = _Child(meter, chain, seen)
        made.append(child)
        return child

    monkeypatch.setattr(sub.Spawner, "build", build)
    spawner = sub.Spawner(parent=None,
                          presets={"impl": sub.Preset(name="impl",
                                                      model="kimi")})
    return spawner, seen, made, meter


ITEMS = [manager.Item(name="one", instruction="do one"),
         manager.Item(name="two", instruction="do two"),
         manager.Item(name="three", instruction="do three")]


def test_the_loop_arm_builds_one_implementer_and_carries_its_history(rig):
    """The whole claim (a): ONE long-lived session, re-steered."""
    spawner, seen, made, _ = rig
    ledger, _ = manager.run(spawner, spawner.presets["impl"], "the task",
                            ITEMS, mode="loop")

    assert len(made) == 1, "loop mode must build the implementer exactly once"
    assert len({s[0] for s in seen}) == 1, "every phase on the same child"
    assert [len(s[2]) for s in seen] == [0, 1, 1], (
        "the implementer's own history must be carried into the next phase")
    assert [p.index for p in ledger] == [1, 2, 3]


def test_the_fanout_arm_builds_a_fresh_child_per_phase_and_carries_nothing(rig):
    """The other half of the pair. One variable between the two arms."""
    spawner, seen, made, _ = rig
    manager.run(spawner, spawner.presets["impl"], "the task", ITEMS,
                mode="fanout")

    assert len(made) == 3, "fanout mode is a fresh one-shot child per phase"
    assert len({s[0] for s in seen}) == 3
    assert [len(s[2]) for s in seen] == [0, 0, 0], "a one-shot inherits nothing"


def test_a_phases_tokens_are_its_own_and_not_the_trees_running_total(rig):
    """The tree shares ONE meter, so `result.cost` is everything so far.

    Writing that onto a phase makes phase 3 look three times phase 1 whatever
    it did - the same defect as averaging a None as a zero, one field along.
    """
    spawner, _, _, meter = rig
    ledger, _ = manager.run(spawner, spawner.presets["impl"], "t", ITEMS)

    assert [p.input_tokens for p in ledger] == [100, 100, 100]
    assert [p.output_tokens for p in ledger] == [10, 10, 10]
    assert meter.input == 300, "the meter itself still counts the whole tree"


def test_both_arms_write_the_same_spawns_so_the_record_is_comparable(rig):
    """A pair recorded by two code paths compares the code paths."""
    spawner, _, _, _ = rig
    manager.run(spawner, spawner.presets["impl"], "t", ITEMS, mode="loop")
    loop = [(s.agent, s.backend, s.model, s.depth) for s in spawner.spawns]
    spawner.spawns.clear()
    manager.run(spawner, spawner.presets["impl"], "t", ITEMS, mode="fanout")
    assert loop == [(s.agent, s.backend, s.model, s.depth)
                    for s in spawner.spawns]


def test_the_ledger_says_which_arm_and_which_wording_produced_it(rig):
    """Untellable-apart halves are two runs, not a pair."""
    spawner, _, _, _ = rig
    ledger, _ = manager.run(spawner, spawner.presets["impl"], "t", ITEMS,
                            mode="fanout", wording="perfectly")
    assert {p.mode for p in ledger} == {"fanout"}
    assert {p.wording for p in ledger} == {"perfectly"}


def test_only_the_tested_sentence_differs_between_the_two_wordings():
    """Anything else differing would make the wording claim untestable."""
    item = manager.Item(name="one", instruction="do one")
    a = manager.phase_prompt("t", item, 1, 3, "extremely-well").split("\n")
    b = manager.phase_prompt("t", item, 1, 3, "perfectly").split("\n")
    differ = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(differ) == 1
    assert a[differ[0]] == "Do this phase extremely well."
    assert b[differ[0]] == "Do this phase perfectly."


def test_a_phase_that_did_not_answer_ends_the_checklist(rig, monkeypatch):
    """Steering phase 3 off a history whose last turn failed measures nothing."""
    spawner, _, _, _ = rig
    real = _Child.run

    def run(self, prompt, messages=None):
        result = real(self, prompt, messages)
        if "two" in prompt:
            result.stopped = "budget"
        return result

    monkeypatch.setattr(_Child, "run", run)
    ledger, _ = manager.run(spawner, spawner.presets["impl"], "t", ITEMS)
    assert [p.verdict for p in ledger] == ["done", "budget"]


def test_a_phase_ledger_survives_orc_read(tmp_path, monkeypatch):
    """A phase ledger is written, read back, and unchanged."""
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path)
    phases = [orclib.Phase(index=1, name="one", agent="impl", mode="loop",
                           wording="extremely-well", seconds=1.5, steps=2,
                           verdict="done", backend="kimi",
                           input_tokens=100, output_tokens=None)]
    orclib.write(orclib.Record(session="s", phases=phases))
    back = orclib.read("s")
    assert back.phases == phases
    assert back.phases[0].output_tokens is None, "None must not become 0"


def test_the_report_keeps_the_two_arms_apart(tmp_path, monkeypatch):
    """A report that merged them would answer neither half of the pair."""
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path)
    runs = [{"session": "a", "task_class": "build", "nodes": [],
             "phases": [{"mode": m, "wording": "extremely-well",
                         "backend": "kimi", "seconds": 2.0, "steps": 1,
                         "verdict": "done"}]}
            for m in ("loop", "fanout")]
    text = orclib.report(runs)
    assert "phase ledger" in text
    assert any(line.strip().startswith("loop") for line in text.splitlines())
    assert any(line.strip().startswith("fanout") for line in text.splitlines())


def test_a_store_with_no_phase_prints_no_phase_table():
    """An always-empty table teaches the reader to skip the answer's place."""
    assert "phase ledger" not in orclib.report(
        [{"session": "a", "task_class": "", "nodes": [{"backend": "kimi"}]}])


@pytest.mark.parametrize("ref,why", [
    ("[]", "expected a non-empty list"),
    ("[{\"name\": \"x\"}]", "has no instruction"),
    ("{\"phases\": [3]}", "is not a string or an object"),
])
def test_a_bad_checklist_is_a_sentence_and_never_a_traceback(ref, why):
    items, error = manager.load(ref)
    assert items == [] and why in error


def test_a_checklist_reads_from_a_file_as_well_as_inline(tmp_path):
    p = tmp_path / "phases.json"
    p.write_text(json.dumps([{"name": "one", "instruction": "do one"}]))
    items, error = manager.load(str(p))
    assert not error and items == [manager.Item("one", "do one")]


def test_an_unknown_mode_is_refused_before_anything_is_spent(rig):
    spawner, _, made, _ = rig
    with pytest.raises(ValueError, match="unknown phase mode"):
        manager.run(spawner, spawner.presets["impl"], "t", ITEMS, mode="both")
    assert made == [], "nothing may be built before the mode is checked"

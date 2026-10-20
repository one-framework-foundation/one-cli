"""Subagents - a nested loop that inherits every gate and shares one budget.

The tests that matter here are not "does it spawn". They are the four ways a
`task` tool becomes a hole: a child with tools its parent does not have, a
child with a fresh budget, a child that skips the guards, and a tree with no
bottom. Each one has a test that fails loudly if the wiring is ever undone.
"""

from __future__ import annotations

import json


from onecli import subagent as sub
from onecli import tools as toolkit
from onecli.agent import Agent
from onecli.guard import GuardChain
from onecli.providers.base import ToolCall, Turn

from test_onecli_agent import ScriptedProvider


def _parent(tmp_path, **kw) -> Agent:
    kw.setdefault("spawner", sub.Spawner(parent=None))
    return Agent(ScriptedProvider([Turn(text="parent done")]), str(tmp_path), **kw)


# -- the surface -----------------------------------------------------------

def test_a_child_never_gets_a_tool_its_parent_does_not_have():
    surface = sub.child_surface(["read", "grep"], ("read", "write", "bash"))
    assert surface == ["read"]


def test_a_preset_with_no_opinion_inherits_the_parent_surface():
    assert sub.child_surface(["read", "grep"], None) == ["read", "grep"]
    assert sub.child_surface(None, None) is None


def test_a_read_only_parent_cannot_spawn_a_writing_child(tmp_path):
    parent = _parent(tmp_path, tool_names=["read", "ls", "glob", "grep"])
    writer = sub.Preset(name="writer", tools=("read", "write", "bash"))
    child = parent.spawner.build(writer)
    assert child.tool_names == ["read"]
    assert "write" not in (child.tool_names or [])
    assert "bash" not in (child.tool_names or [])


# -- the gates -------------------------------------------------------------

def test_a_child_stands_on_the_parents_guards_and_hooks(tmp_path):
    guard = GuardChain([])
    parent = _parent(tmp_path, guard=guard)
    child = parent.spawner.build(sub.GENERAL)
    assert child.guard is guard
    assert child.hooks is parent.hooks
    assert child.workspace == parent.workspace


def test_a_child_has_nobody_to_ask_and_no_jobs_to_leave_behind(tmp_path):
    parent = _parent(tmp_path)
    child = parent.spawner.build(sub.GENERAL)
    assert child.asker is None and child.jobs is None
    surface = child.tool_names or list(toolkit.REGISTRY)
    assert not set(surface) & set(toolkit.ASK_TOOL_NAMES)
    assert not set(surface) & set(toolkit.JOB_TOOL_NAMES)


# -- the budget ------------------------------------------------------------

def test_the_budget_is_one_meter_for_the_whole_tree(tmp_path):
    parent = _parent(tmp_path, max_budget_usd=1.5)
    child = parent.spawner.build(sub.GENERAL)
    assert child.meter is parent.meter
    assert child.max_budget_usd == 1.5


def test_the_cumulative_input_ceiling_covers_the_whole_tree(tmp_path):
    parent = _parent(tmp_path, max_input_tokens=1000)
    child = parent.spawner.build(sub.GENERAL)
    assert child.max_input_tokens == 1000
    parent.meter.metered_input_tokens = 1000      # the parent spent it all
    result = child.run("one more request")
    assert result.stopped == "budget_input"


def test_a_child_that_spends_moves_the_parents_meter(tmp_path):
    """The one that matters: tokens burned in a child are billed to the tree.

    A copied meter passes every other budget test in this file and still lets
    six children spend the ceiling six times.
    """
    parent = _parent(tmp_path, max_budget_usd=1.0)
    child = parent.spawner.build(sub.GENERAL)
    child.meter.input_tokens += 5000
    child.meter.output_tokens += 500
    assert parent.meter.input_tokens == 5000
    assert parent.meter.output_tokens == 500


# -- the depth cap ---------------------------------------------------------

def test_the_last_allowed_level_is_built_without_the_tool(tmp_path):
    parent = _parent(tmp_path, spawner=sub.Spawner(parent=None, max_depth=1))
    child = parent.spawner.build(sub.GENERAL)
    assert child.spawner is None
    assert "task" not in (child.tool_names or list(toolkit.REGISTRY))


def test_a_deeper_cap_lets_a_child_spawn_its_own(tmp_path):
    parent = _parent(tmp_path, spawner=sub.Spawner(parent=None, max_depth=2))
    child = parent.spawner.build(sub.GENERAL)
    assert child.spawner is not None
    assert child.spawner.depth == 1
    assert child.spawner.parent is child
    assert child.spawner.build(sub.GENERAL).spawner is None


def test_a_run_with_no_spawner_is_not_offered_the_tool(tmp_path):
    agent = Agent(ScriptedProvider([Turn(text="x")]), str(tmp_path))
    assert "task" not in (agent.tool_names or [])


# -- what comes back -------------------------------------------------------

def test_the_parent_gets_the_answer_and_the_cost_not_the_transcript(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    parent = _parent(tmp_path)
    parent.spawner.build = lambda preset: Agent(  # noqa: ARG005
        ScriptedProvider([
            Turn(tool_calls=[ToolCall(id="t1", name="read",
                                      arguments={"path": "a.txt"})]),
            Turn(text="a.txt says hello"),
        ]), str(tmp_path))

    out = parent.spawner.spawn("what does a.txt say?")
    assert "a.txt says hello" in out
    assert "1 tools" in out and "2 steps" in out
    # The child read the file; the parent is not handed the file.
    assert "hello\n" not in out.replace("a.txt says hello", "")
    assert parent.spawner.spawns[0].tool_calls == 1


def test_a_silent_child_is_reported_not_returned_as_an_empty_string(tmp_path):
    parent = _parent(tmp_path)
    parent.spawner.build = lambda preset: Agent(  # noqa: ARG005
        ScriptedProvider([Turn(text="")]), str(tmp_path))
    out = parent.spawner.spawn("do a thing")
    assert "returned nothing" in out


def test_an_unknown_agent_names_the_ones_that_exist(tmp_path):
    parent = _parent(tmp_path)
    out = parent.spawner.spawn("x", agent="reviewer")
    assert "no agent named 'reviewer'" in out
    assert "explorer" in out and "general" in out


def test_the_task_tool_without_a_spawner_says_so_instead_of_raising():
    assert "not available" in sub.task(prompt="x", spawner=None)


def test_a_task_with_no_prompt_is_refused():
    assert "needs a prompt" in sub.task(prompt="   ", spawner=sub.Spawner(parent=None))


def test_a_code_mode_program_may_not_spawn_subagents(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")
    provider = ScriptedProvider([
        Turn(tool_calls=[ToolCall(id="c1", name="code", arguments={
            "source": "print(one.call('task', prompt='go'))"})]),
        Turn(text="ok"),
    ])
    result = Agent(provider, str(tmp_path), spawner=sub.Spawner(parent=None)).run("go")
    body = result.steps[0].tool_results[0][1]
    assert "spawn subagents from the loop" in body


# -- presets ---------------------------------------------------------------

def test_presets_load_from_a_bare_map(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(json.dumps({"reviewer": {"description": "reads diffs",
                                             "tools": ["read", "grep"],
                                             "max_steps": 4}}))
    presets, errors = sub.load_presets([str(path)])
    assert errors == []
    assert presets["reviewer"].tools == ("read", "grep")
    assert presets["reviewer"].steps() == 4
    # The built-ins survive a file that does not mention them.
    assert "general" in presets and "explorer" in presets


def test_presets_load_from_an_agents_key_and_from_a_list(tmp_path):
    keyed = tmp_path / "keyed.json"
    keyed.write_text(json.dumps({"agents": {"a": {"description": "one"}}}))
    listed = tmp_path / "listed.json"
    listed.write_text(json.dumps([{"name": "b", "description": "two"}]))
    presets, errors = sub.load_presets([str(keyed), str(listed)])
    assert errors == []
    assert presets["a"].description == "one"
    assert presets["b"].description == "two"


def test_a_broken_preset_file_is_reported_and_the_run_keeps_its_builtins(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    presets, errors = sub.load_presets([str(bad), str(tmp_path / "missing.json")])
    assert len(errors) == 2
    assert sorted(presets) == ["explorer", "general"]


def test_a_presets_system_prompt_is_appended_to_the_subagent_charter():
    preset = sub.Preset(name="r", system="Answer only with file paths.")
    system = sub._child_system(preset)
    assert system.startswith(sub.CHILD_SYSTEM)
    assert "Answer only with file paths." in system


def test_a_child_gets_the_subagent_charter_in_its_system_prompt(tmp_path):
    parent = _parent(tmp_path)
    child = parent.spawner.build(sub.EXPLORER)
    assert "You are a subagent" in child.system
    assert child.max_steps == parent.max_steps


# -- a pinned child --------------------------------------------------------
#
# The reason to pin a preset to a backend is cost: the cheap model does the
# sweep, the expensive one does the judging. That only holds if a pinned child
# actually lands there and cannot silently fall back onto the parent's rung.


def _pinned(monkeypatch, tmp_path, model="cheap", **preset_kw):
    from onecli import config, providers

    backend = config.Backend(name="cheap-lane", model="cheap-model",
                             base_url="http://127.0.0.1:9", wire="openai",
                             source="test")
    monkeypatch.setattr(config, "resolve",
                        lambda name=None, **kw: backend if name == "cheap" else
                        (_ for _ in ()).throw(RuntimeError(f"no backend matching {name!r}")))
    monkeypatch.setattr(providers, "build", lambda b, timeout=600.0: object())
    parent = _parent(tmp_path)
    parent.spawner.parent = parent
    return parent, sub.Preset(name="sweeper", model=model, **preset_kw)


def test_a_pinned_child_runs_on_the_backend_the_preset_named(monkeypatch, tmp_path):
    parent, preset = _pinned(monkeypatch, tmp_path)
    child = parent.spawner.build(preset)
    assert child.chain is not parent.chain
    assert child.chain.current.name == "cheap-lane"
    assert child.chain.current.model == "cheap-model"


def test_a_pinned_child_gets_one_rung_and_no_fallback(monkeypatch, tmp_path):
    parent, preset = _pinned(monkeypatch, tmp_path)
    chain = parent.spawner.chain_for(preset)
    assert len(chain.rungs) == 1 and not chain.has_next()


def test_an_unpinned_child_still_shares_the_parents_chain(monkeypatch, tmp_path):
    parent, _ = _pinned(monkeypatch, tmp_path)
    assert parent.spawner.build(sub.GENERAL).chain is parent.chain


def test_a_preset_pinned_to_an_unknown_backend_says_so_instead_of_raising(
        monkeypatch, tmp_path):
    parent, preset = _pinned(monkeypatch, tmp_path, model="nowhere")
    parent.spawner.presets["sweeper"] = preset
    out = parent.spawner.spawn("go", "sweeper")
    assert "did not resolve" in out and "nowhere" in out


def test_a_model_is_read_off_a_preset_file_and_named_in_the_catalogue(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(json.dumps({"sweeper": {"description": "sweeps",
                                            "model": "cheap"}}))
    presets, errors = sub.load_presets([str(path)])
    assert not errors and presets["sweeper"].model == "cheap"
    catalogue = sub.Spawner(parent=None, presets=presets).catalogue()
    assert "runs on cheap" in catalogue


# -- an orchestrator run ----------------------------------------------------
#
# `--orchestrator` is the same pin, applied from the other end: the parent
# drives on the strong model and every child that did not name its own backend
# lands on the picked one. The point is that a strong model can review and fix
# what the cheap ones produced, so the default must reach children the operator
# never wrote a preset for.


def test_an_unpinned_child_lands_on_the_spawners_worker(monkeypatch, tmp_path):
    parent, _ = _pinned(monkeypatch, tmp_path)
    parent.spawner.worker = "cheap"
    child = parent.spawner.build(sub.GENERAL)
    assert child.chain is not parent.chain
    assert child.chain.current.name == "cheap-lane"


def test_a_preset_pin_beats_the_worker_default(monkeypatch, tmp_path):
    parent, preset = _pinned(monkeypatch, tmp_path, model="cheap")
    parent.spawner.worker = "nowhere"
    assert parent.spawner.chain_for(preset).current.name == "cheap-lane"


def test_the_worker_default_reaches_a_grandchild(monkeypatch, tmp_path):
    parent, _ = _pinned(monkeypatch, tmp_path)
    parent.spawner.worker = "cheap"
    assert parent.spawner.build(sub.GENERAL).spawner.worker == "cheap"


def test_orchestrator_hands_the_picked_model_to_the_workers():
    from onecli import cli

    args = cli.build_parser().parse_args(["-m", "cheap", "--orchestrator", "opus", "go"])
    assert cli._worker_name(args) == "cheap"


def test_no_orchestrator_leaves_children_on_the_parents_chain():
    from onecli import cli

    assert cli._worker_name(cli.build_parser().parse_args(["-m", "cheap", "go"])) == ""
    assert cli._worker_name(
        cli.build_parser().parse_args(["-m", "cheap", "--orchestrator", "none", "go"])) == ""


# -- an orc file's per-preset workers ---------------------------------------
#
# The map sits between a preset's own pin and the blanket worker default. All
# three rungs are resolved in one place (`Spawner.pin_for`) so the chain, the
# window and the catalogue cannot disagree about which backend a child is on -
# a child sized for a 200k parent and pointed at a 32k box is the failure this
# prevents, and it is invisible until the loop dies mid-sweep.


def test_an_orc_worker_entry_pins_a_child_the_preset_never_pinned(
        monkeypatch, tmp_path):
    parent, _ = _pinned(monkeypatch, tmp_path)
    parent.spawner.workers = {"general": "cheap"}
    child = parent.spawner.build(sub.GENERAL)
    assert child.chain is not parent.chain
    assert child.chain.current.name == "cheap-lane"


def test_a_preset_pin_beats_an_orc_worker_entry(monkeypatch, tmp_path):
    parent, preset = _pinned(monkeypatch, tmp_path, model="cheap")
    parent.spawner.presets["sweeper"] = preset
    parent.spawner.workers = {"sweeper": "nowhere"}
    assert parent.spawner.pin_for(preset) == "cheap"
    assert parent.spawner.chain_for(preset).current.name == "cheap-lane"


def test_an_orc_worker_entry_beats_the_blanket_worker_default(
        monkeypatch, tmp_path):
    parent, _ = _pinned(monkeypatch, tmp_path)
    parent.spawner.worker = "nowhere"
    parent.spawner.workers = {"general": "cheap"}
    assert parent.spawner.pin_for(sub.GENERAL) == "cheap"
    assert parent.spawner.build(sub.GENERAL).chain.current.name == "cheap-lane"


def test_a_preset_the_orc_file_never_named_still_gets_the_default(
        monkeypatch, tmp_path):
    parent, _ = _pinned(monkeypatch, tmp_path)
    parent.spawner.worker = "cheap"
    parent.spawner.workers = {"explorer": "somewhere-else"}
    assert parent.spawner.pin_for(sub.GENERAL) == "cheap"


def test_the_orc_worker_map_reaches_a_grandchild(monkeypatch, tmp_path):
    parent, _ = _pinned(monkeypatch, tmp_path)
    parent.spawner.workers = {"general": "cheap"}
    grandchild_spawner = parent.spawner.build(sub.GENERAL).spawner
    assert grandchild_spawner.workers == {"general": "cheap"}


def test_a_child_pinned_by_the_orc_file_is_sized_for_that_backend(
        monkeypatch, tmp_path):
    from onecli import config

    parent, _ = _pinned(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "resolve_context_window",
                        lambda *a, **kw: 4242)
    # the parent's own window is something else entirely, so a pass here means
    # the child was sized for the backend it landed on and not for its parent
    assert parent.tracker.window != 4242
    parent.spawner.workers = {"general": "cheap"}
    assert parent.spawner._window_for(sub.GENERAL) == 4242


def test_an_orc_pinned_child_is_named_in_the_catalogue(monkeypatch, tmp_path):
    spawner = sub.Spawner(parent=None, workers={"explorer": "local"})
    assert "runs on local" in spawner.catalogue()


def test_the_cli_hands_the_orc_files_workers_to_the_spawner(tmp_path):
    import json as _json

    from onecli import cli

    path = tmp_path / ".orc.json"
    path.write_text(_json.dumps({"workers": {"explorer": "local"}}))
    args = cli.build_parser().parse_args(["-m", "cheap", "--orc", str(path), "go"])
    spawner = cli._spawner(args, dict(sub.BUILTIN))
    assert spawner.workers == {"explorer": "local"}


# -- a backend name is not a model id --------------------------------------

def _twin_chain(name: str, model: str):
    """A one-rung chain, as `chain_for` builds for a pinned child."""
    from onecli import fallback

    return fallback.Chain(rungs=[fallback.Rung(name=name, provider=object(),
                                               model=model)])


class _TwinChild:
    """A child that ran on `chain` and answered."""

    def __init__(self, chain):
        self.chain = chain

    def run(self, prompt, messages=None):
        return type("R", (), {
            "text": "done", "steps": [1], "tool_calls": 0, "stopped": "answered",
            "retries": 0, "retry_reasons": {}, "compactions": [],
            "loop_stops": 0, "error": ""})()


def test_a_spawn_records_the_backend_NAME_beside_the_model_id(monkeypatch):
    """Two backends can serve one model id, so the id cannot stand in for one.

    Two fixture backends serve one shared model id, so the id cannot stand in
    for a backend. Against the earlier implementation `Spawn` carried only
    `model`, which `record_for` then wrote into a field called `backend` - so
    the two endpoints were indistinguishable in the record.
    """
    chain = _twin_chain("lane-b", "shared-model")
    monkeypatch.setattr(sub.Spawner, "build",
                        lambda self, preset: _TwinChild(chain))
    spawner = sub.Spawner(
        parent=None,
        presets={"beta": sub.Preset(name="beta", model="lane-b")})
    spawner.spawn("look", agent="beta")

    sp = spawner.spawns[0]
    assert sp.backend == "lane-b", "the record must name the ENDPOINT"
    assert sp.model == "shared-model"
    assert sp.backend != sp.model


def test_an_inherited_child_records_the_PARENTS_backend_name(monkeypatch):
    """An unpinned child is not backend-less; it ran on the parent's rung."""
    chain = _twin_chain("lane-a", "shared-model")
    monkeypatch.setattr(sub.Spawner, "build",
                        lambda self, preset: _TwinChild(chain))
    spawner = sub.Spawner(parent=None,
                          presets={"general": sub.Preset(name="general")})
    spawner.spawn("look", agent="general")

    sp = spawner.spawns[0]
    assert sp.backend == "lane-a" and sp.model == "shared-model"


def test_a_grandchilds_spawn_reaches_the_root_record(monkeypatch):
    """Depth 3: the level a run record could not see.

    `record_for` is handed the ROOT spawner's list. A child holds its own
    spawner, so before this every node below level 1 was spent and never
    written down - a depth-3 run and a depth-2 run produced the same record,
    which is why the not-tested list could not be moved by running one.
    """
    chain = _twin_chain("kimi", "kimi-for-coding")
    monkeypatch.setattr(sub.Spawner, "build",
                        lambda self, preset: _TwinChild(chain))
    root = sub.Spawner(parent=None, depth=0, max_depth=3,
                       presets={"miner": sub.Preset(name="miner", model="kimi")})
    child = root._grandchild()
    assert child is not None
    child.spawn("dig", agent="miner")

    assert [s.agent for s in root.spawns] == ["miner"], (
        "a grandchild's spawn must land on the run record, not in a list "
        "that dies with the child")


def test_every_spawn_records_which_level_it_ran_at(monkeypatch):
    """Two nodes on the same backend at different depths are different rows."""
    chain = _twin_chain("kimi", "kimi-for-coding")
    monkeypatch.setattr(sub.Spawner, "build",
                        lambda self, preset: _TwinChild(chain))
    root = sub.Spawner(parent=None, depth=0, max_depth=3,
                       presets={"miner": sub.Preset(name="miner", model="kimi")})
    root.spawn("a", agent="miner")
    root._grandchild().spawn("b", agent="miner")

    assert [s.depth for s in root.spawns] == [1, 2]

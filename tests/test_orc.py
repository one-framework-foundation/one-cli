"""Orchestration config - who drives, who works, and the record of what it cost.

The tests that matter here are the precedence chain and the two failure modes
either side of it: a config file that cannot be read must not cost the run, and
a run record that cannot be read back is not a record.
"""

from __future__ import annotations

import json

import pytest

from onecli import orc as orclib
from onecli import subagent as sub
from onecli.agent import Step
from onecli.providers.base import Turn


def _preset(name: str, model: str = "") -> sub.Preset:
    return sub.Preset(name=name, model=model)


def _write(tmp_path, body, name=".orc.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(body))
    return str(p)


# -- the shapes a config file may be written in ----------------------------

def test_a_single_config_file_is_read_as_one_config(tmp_path):
    path = _write(tmp_path, {"name": "cheap-fanout", "orchestrator": "opus",
                             "workers": {"explorer": "local"},
                             "max_depth": 3, "max_steps": 40})
    orcs, errors = orclib.load_orcs([path])
    assert not errors
    got = orcs["cheap-fanout"]
    assert got.orchestrator == "opus"
    assert got.worker_for("explorer") == "local"
    assert (got.max_depth, got.max_steps) == (3, 40)


def test_a_bare_map_of_configs_is_keyed_by_name(tmp_path):
    path = _write(tmp_path, {"a": {"orchestrator": "opus"},
                             "b": {"workers": {"general": "local"}}})
    orcs, errors = orclib.load_orcs([path])
    assert not errors and sorted(orcs) == ["a", "b"]
    assert orcs["a"].orchestrator == "opus"
    assert orcs["b"].worker_for("general") == "local"


def test_an_orcs_key_and_a_list_are_both_accepted(tmp_path):
    keyed = _write(tmp_path, {"orcs": {"a": {"orchestrator": "opus"}}}, "k.json")
    listed = _write(tmp_path, [{"name": "b", "orchestrator": "sonnet"}], "l.json")
    orcs, errors = orclib.load_orcs([keyed, listed])
    assert not errors
    assert orcs["a"].orchestrator == "opus"
    assert orcs["b"].orchestrator == "sonnet"


def test_a_config_with_no_name_is_skipped_not_fatal(tmp_path):
    path = _write(tmp_path, [{"orchestrator": "opus"}], "l.json")
    orcs, errors = orclib.load_orcs([path])
    assert orcs == {} and errors and "no name" in errors[0]


# -- a broken file costs a warning, never the run --------------------------

def test_a_malformed_orc_file_is_reported_and_never_fatal(tmp_path):
    path = tmp_path / ".orc.json"
    path.write_text("{ this is not json")
    orcs, errors = orclib.load_orcs([str(path)])
    assert orcs == {}
    assert len(errors) == 1 and str(path) in errors[0]


def test_a_missing_orc_file_is_reported_and_never_fatal(tmp_path):
    orcs, errors = orclib.load_orcs([str(tmp_path / "nope.json")])
    assert orcs == {} and len(errors) == 1


def test_an_orc_file_of_the_wrong_type_is_reported(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("42")
    orcs, errors = orclib.load_orcs([str(path)])
    assert orcs == {} and "expected an object" in errors[0]


def test_find_on_a_path_returns_the_only_config_in_it(tmp_path):
    path = _write(tmp_path, {"orchestrator": "opus"})
    got, errors = orclib.find(path)
    assert not errors and got is not None and got.orchestrator == "opus"


def test_find_on_an_unknown_name_says_so_instead_of_raising():
    got, errors = orclib.find("no-such-orc-config-anywhere")
    assert got is None and errors and "no orc config named" in errors[0]


def test_find_on_a_broken_file_returns_nothing_and_an_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{{{")
    got, errors = orclib.find(str(path))
    assert got is None and errors


def test_a_file_holding_several_configs_refuses_to_guess(tmp_path):
    path = _write(tmp_path, {"a": {"orchestrator": "x"},
                             "b": {"orchestrator": "y"}}, "many.json")
    got, errors = orclib.find(path)
    assert got is None and any("name one with --orc" in e for e in errors)


# -- the precedence chain, all four rungs ----------------------------------

def test_a_preset_pin_beats_the_orc_file_and_the_default():
    orc = orclib.Orc(name="o", workers={"sweeper": "from-file"})
    backend, source = orclib.resolve_worker(
        _preset("sweeper", model="from-pin"), orc, "from-default")
    assert (backend, source) == ("from-pin", orclib.PIN)


def test_the_orc_file_beats_the_default_worker():
    orc = orclib.Orc(name="o", workers={"sweeper": "from-file"})
    backend, source = orclib.resolve_worker(
        _preset("sweeper"), orc, "from-default")
    assert (backend, source) == ("from-file", orclib.FILE)


def test_the_default_worker_catches_a_preset_the_file_never_named():
    orc = orclib.Orc(name="o", workers={"other": "from-file"})
    backend, source = orclib.resolve_worker(
        _preset("sweeper"), orc, "from-default")
    assert (backend, source) == ("from-default", orclib.DEFAULT)


def test_with_nothing_at_all_a_child_inherits_the_parents_chain():
    backend, source = orclib.resolve_worker(_preset("sweeper"), None, "")
    assert (backend, source) == ("", orclib.INHERIT)


def test_an_explicit_flag_beats_an_orc_files_orchestrator(tmp_path, monkeypatch):
    from onecli import cli

    path = _write(tmp_path, {"orchestrator": "from-file"})
    args = cli.build_parser().parse_args(
        ["-m", "cheap", "--orc", path, "--orchestrator", "from-flag", "go"])
    assert cli._orchestrator_name(args) == ("from-flag", orclib.FLAG)


def test_an_orc_file_supplies_the_orchestrator_when_no_flag_does(tmp_path):
    from onecli import cli

    path = _write(tmp_path, {"orchestrator": "from-file"})
    args = cli.build_parser().parse_args(["-m", "cheap", "--orc", path, "go"])
    assert cli._orchestrator_name(args) == ("from-file", orclib.FILE)
    # and the picked --model becomes the workers' backend, as with the flag
    assert cli._worker_name(args) == "cheap"


def test_orchestrator_none_on_the_flag_beats_an_orc_file(tmp_path):
    from onecli import cli

    path = _write(tmp_path, {"orchestrator": "from-file"})
    args = cli.build_parser().parse_args(
        ["-m", "cheap", "--orc", path, "--orchestrator", "none", "go"])
    assert cli._orchestrator_name(args)[0] == ""
    assert cli._worker_name(args) == ""


def test_a_broken_orc_file_on_the_command_line_does_not_stop_the_run(
        tmp_path, capsys):
    from onecli import cli

    path = tmp_path / "bad.json"
    path.write_text("{{{")
    args = cli.build_parser().parse_args(["-m", "cheap", "--orc", str(path), "go"])
    assert cli._orc(args) is None
    assert cli._worker_name(args) == ""
    assert "orc:" in capsys.readouterr().err


def test_the_orc_files_budgets_apply_only_where_a_flag_did_not(tmp_path):
    from onecli import cli

    path = _write(tmp_path, {"workers": {}, "max_depth": 4, "max_steps": 55})
    args = cli.build_parser().parse_args(["-m", "cheap", "--orc", path, "go"])
    cli._apply_orc_budgets(args)
    assert (args.max_depth, args.max_steps) == (4, 55)

    args = cli.build_parser().parse_args(
        ["-m", "cheap", "--orc", path, "--max-steps", "7", "go"])
    cli._apply_orc_budgets(args)
    assert (args.max_depth, args.max_steps) == (4, 7)


# -- the launch echo -------------------------------------------------------

def _resolved() -> orclib.Resolved:
    presets = {"general": _preset("general"),
               "explorer": _preset("explorer"),
               "reviewer": _preset("reviewer", model="opus")}
    orc = orclib.Orc(name="fanout", workers={"explorer": "local"},
                     path="/tmp/.orc.json")
    return orclib.resolve(presets, orc, "opus", orclib.FLAG, "cheap", 2, 30)


def test_the_echo_names_every_preset_its_backend_and_the_rung(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    out = orclib.echo(_resolved())
    assert "\x1b[" not in out
    assert "explorer" in out and "local" in out and orclib.FILE in out
    assert "reviewer" in out and "opus" in out and orclib.PIN in out
    assert "general" in out and "cheap" in out and orclib.DEFAULT in out
    # short: the orchestrator plus one line per preset, nothing else
    assert len(out.splitlines()) == 4


def test_the_echo_expands_with_the_file_and_the_budgets(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    out = orclib.echo(_resolved(), explain=True)
    assert "/tmp/.orc.json" in out
    assert "max_depth 2" in out and "max_steps 30" in out


def test_the_echo_is_plain_under_one_cli_plain(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("ONE_CLI_PLAIN", "1")
    from onecli import render
    render.refresh()
    out = orclib.echo(_resolved())
    assert "\x1b[" not in out
    # ascii marks, not box drawing
    assert "→" in out and "◆" not in out and "│" not in out


# -- the run record --------------------------------------------------------

class _Result:
    stopped = "answered"
    tool_calls = 3
    steps = [1, 2, 3, 4]
    cost = {"model": "opus", "input_tokens": 900, "output_tokens": 120,
            "usd": 0.0421, "estimated": False, "price_known": True}


def test_a_run_record_round_trips_through_write_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    spawns = [sub.Spawn(agent="explorer", prompt="sweep", steps=6,
                        tool_calls=9, seconds=4.25, backend="local",
                        model="local-model")]
    record = orclib.record_for("sess123", _resolved(), _Result(), spawns,
                               seconds=12.5, exit_code=0, chain_name="opus-lane")
    path = orclib.write(record)
    assert path.exists()

    back = orclib.read("sess123")
    assert back.session == "sess123"
    assert back.exit_status == "answered" and back.exit_code == 0
    assert back.seconds == 12.5
    # ISO timestamps, not floats
    assert back.finished.startswith("20") and "T" in back.finished

    # the resolved config, sources and all, survived the round trip
    assert back.config["orchestrator"]["source"] == orclib.FLAG
    sources = {w["name"]: w["source"] for w in back.config["workers"]}
    assert sources == {"explorer": orclib.FILE, "general": orclib.DEFAULT,
                       "reviewer": orclib.PIN}

    # per-node wall time and step counts, root first
    root, child = back.nodes
    assert root.name == "orchestrator" and root.backend == "opus-lane"
    assert root.seconds == 12.5 and root.steps == 4 and root.tool_calls == 3
    assert (root.input_tokens, root.output_tokens) == (900, 120)
    assert root.usd == 0.0421 and root.metered is True
    assert child.name == "explorer" and child.seconds == 4.25
    assert child.steps == 6 and child.tool_calls == 9
    # `backend` is the endpoint, `model` the id it serves - not the same word
    assert child.backend == "local" and child.model == "local-model"
    assert child.source == orclib.FILE


def test_reading_a_run_that_was_never_written_raises_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    with pytest.raises(FileNotFoundError):
        orclib.read("nothing-here")


def test_a_record_that_cannot_be_written_does_not_cost_the_answer(
        tmp_path, monkeypatch):
    # RUN_DIR under a FILE, so mkdir cannot succeed
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(orclib, "RUN_DIR", blocker / "runs")
    record = orclib.record_for("sess", _resolved(), _Result(), [], seconds=1.0,
                               exit_code=0)
    orclib.write(record)  # must not raise


# -- the read verbs: `--orc-list` and `--orc-show` --------------------------

def _root(tmp_path, name, body) -> None:
    """Write a config into a fake ~/.one-cli/orc/ under tmp_path."""
    folder = tmp_path / "orc"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(body))


def test_the_catalogue_says_so_in_words_when_there_are_no_configs(tmp_path,
                                                                  monkeypatch):
    monkeypatch.chdir(tmp_path)
    orcs, errors = orclib.catalogue(root=tmp_path / "state")
    assert orcs == [] and errors == []
    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    out = orclib.summarise(orcs)
    assert "no orc configs found" in out
    # never an empty table
    assert "orchestration configs" not in out


def test_the_catalogue_finds_one_config_and_names_its_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".orc.json").write_text(json.dumps(
        {"orchestrator": "opus", "workers": {"explorer": "local"}}))
    orcs, errors = orclib.catalogue(root=tmp_path / "state")
    assert errors == []
    assert [o.name for o in orcs] == ["orc"]
    assert orcs[0].orchestrator == "opus" and orcs[0].path.endswith(".orc.json")

    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    out = orclib.summarise(orcs)
    assert "opus" in out and "1 worker pin" in out and ".orc.json" in out


def test_the_catalogue_gathers_several_configs_across_the_search_paths(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (tmp_path / ".orc.json").write_text(json.dumps({"orchestrator": "project"}))
    (state / "orc.json").write_text(json.dumps(
        {"cheap": {"orchestrator": "haiku"},
         "rich": {"orchestrator": "opus", "workers": {"a": "x", "b": "y"}}}))
    _root(state, "sweep", {"orchestrator": "sonnet"})
    orcs, errors = orclib.catalogue(root=state)
    assert errors == []
    assert sorted(o.name for o in orcs) == ["cheap", "orc", "rich", "sweep"]

    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    out = orclib.summarise(orcs)
    assert "2 worker pins" in out and "0 worker pins" in out
    # a header plus a node and a leaf per config
    assert len(out.splitlines()) == 1 + 2 * 4


def test_show_flags_a_workers_entry_naming_a_preset_that_does_not_exist(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    orc = orclib.Orc(name="fanout", orchestrator="opus",
                     workers={"explorer": "local", "ghost": "sonnet"},
                     path="/tmp/.orc.json")
    presets = {"general": _preset("general"), "explorer": _preset("explorer"),
               "reviewer": _preset("reviewer", model="opus")}
    out = orclib.describe(orc, presets)
    assert "ghost" in out and "no such preset" in out
    # the pin that DOES land is not flagged
    explorer = [ln for ln in out.splitlines() if "explorer" in ln][0]
    assert "no such preset" not in explorer and orclib.FILE in explorer
    # a preset with no orc entry still shows where it would land
    assert "reviewer" in out and orclib.PIN in out
    assert "/tmp/.orc.json" in out
    # and the runtime behaviour is unchanged: resolve still skips the ghost
    names = {w.name for w in orclib.resolve(presets, orc, "opus", orclib.FLAG,
                                            "cheap", 2, 30).workers}
    assert "ghost" not in names


# -- the task class --------------------------------------------------------

def test_a_task_class_is_lowercased_and_stripped_but_never_checked():
    assert orclib.clean_task_class("  Research ") == "research"
    assert orclib.clean_task_class("BUILD") == "build"
    # free-form: nothing here is a vocabulary
    assert orclib.clean_task_class("shipping the fox") == "shipping the fox"
    assert orclib.clean_task_class(None) == "" and orclib.clean_task_class("") == ""


def test_a_task_class_comes_from_the_flag_from_the_file_or_from_neither(tmp_path):
    presets = {"general": _preset("general")}
    path = _write(tmp_path, {"orchestrator": "opus", "task_class": " Build "})
    orc, _ = orclib.find(path)
    assert orc.task_class == "build"

    # from the file
    r = orclib.resolve(presets, orc, "opus", orclib.FILE, "cheap", 2, 30)
    assert r.task_class == "build"
    # from the flag, with no file at all
    r = orclib.resolve(presets, None, "opus", orclib.FLAG, "cheap", 2, 30,
                       task_class="Review ")
    assert r.task_class == "review"
    # the flag beats the file
    r = orclib.resolve(presets, orc, "opus", orclib.FILE, "cheap", 2, 30,
                       task_class="debug")
    assert r.task_class == "debug"
    # both absent
    r = orclib.resolve(presets, None, "opus", orclib.FLAG, "cheap", 2, 30)
    assert r.task_class == ""
    assert r.as_dict()["task_class"] == ""


def test_the_task_class_is_in_the_resolved_dict_and_on_the_explain_line(
        monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    presets = {"general": _preset("general")}
    r = orclib.resolve(presets, None, "opus", orclib.FLAG, "cheap", 2, 30,
                       task_class="Refactor")
    assert r.as_dict()["task_class"] == "refactor"
    assert "task_class refactor" in orclib.echo(r, explain=True)
    # and it is not noise on the short echo
    assert "refactor" not in orclib.echo(r)


def test_the_task_class_survives_into_a_record_and_back_out_again(
        tmp_path, monkeypatch):
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    presets = {"general": _preset("general")}
    resolved = orclib.resolve(presets, None, "opus", orclib.FLAG, "cheap", 2, 30,
                              task_class="research")
    record = orclib.record_for("sess-tc", resolved, _Result(), [], seconds=3.0,
                               exit_code=0)
    assert record.task_class == "research"
    orclib.write(record)
    raw = json.loads((tmp_path / "runs" / "sess-tc.json").read_text())
    # a TOP-LEVEL key, reachable without knowing the config's shape
    assert raw["task_class"] == "research"
    back = orclib.read("sess-tc")
    assert back.task_class == "research"
    assert back.config["task_class"] == "research"


class _FailedResult(_Result):
    stopped = "max_steps"
    cost = {}


def test_a_failed_run_is_recorded_with_its_task_class_too(tmp_path, monkeypatch):
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    presets = {"general": _preset("general")}
    resolved = orclib.resolve(presets, None, "opus", orclib.FLAG, "cheap", 2, 30,
                              task_class="debug")
    record = orclib.record_for("sess-fail", resolved, _FailedResult(), [],
                               seconds=9.0, exit_code=1)
    orclib.write(record)
    back = orclib.read("sess-fail")
    assert back.exit_code == 1 and back.exit_status == "max_steps"
    assert back.task_class == "debug"


def test_the_cli_writes_a_record_on_a_run_that_exited_non_zero(tmp_path,
                                                              monkeypatch):
    """The record write sits on the path a FAILED run also takes.

    A record written only for a clean run is a record of exactly the runs
    nobody needs to explain afterwards. Driven through `run_once` with a fake
    Agent so the assertion is about the CLI's control flow, not about orc.py.
    """
    from onecli import cli

    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)

    class _Agent:
        def __init__(self, *a, **k):
            self.chain = type("C", (), {"current": type("B", (), {
                "name": "opus-lane", "model": "opus-4"})()})()
            self.spawner = None
            # run_once records `agent.wire_requests if agent.run_limits`, so a
            # fake without both raises AttributeError before any assertion runs.
            self.run_limits = None
            self.wire_requests = 0

        def run(self, *a, **k):
            return type("R", (), {"stopped": "max_steps", "tool_calls": 1,
                                  "steps": [Step(index=0, turn=Turn())],
                                  "cost": {}, "text": "",
                                  "messages": []})()

    monkeypatch.setattr(cli, "Agent", _Agent)
    from onecli import config
    backend = config.Backend(name="fake", model="fake-1",
                             base_url="http://127.0.0.1:1/v1", wire="openai")
    args = cli.build_parser().parse_args(
        ["--quiet", "--no-hooks", "--task-class", "Debug", "do a thing"])
    args.prompt = ["do a thing"]
    rc = cli.run_once(args, backend, None, "sess-cli-fail")

    back = orclib.read("sess-cli-fail")
    assert back.exit_code == 1, "a non-zero run must still be recorded"
    assert back.exit_status == "max_steps"
    assert back.task_class == "debug"
    assert rc != 0


# -- the failure modes: what the run COST, not just whether it ended --------

class _RoughResult(_Result):
    """A run that answered, but not cheaply."""

    fallbacks = [{"from": "opus-lane", "to": "cheap-lane", "reason": "HTTP 500"}]
    compactions = [{"step": 4}, {"step": 9}]
    retries = 5
    retry_reasons = {"rate_limit": 3, "malformed_tool_call": 2}
    resamples = 2
    loop_stops = 4


def test_a_clean_run_records_every_failure_mode_as_empty_not_absent(
        tmp_path, monkeypatch):
    """The fields exist on a run that had none of them, so a `jq` over the
    whole directory does not have to special-case the good runs."""
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    record = orclib.record_for("sess-clean", _resolved(), _Result(), [],
                               seconds=2.0, exit_code=0)
    orclib.write(record)
    back = orclib.read("sess-clean")
    assert back.failover == [] and back.retries == 0
    assert back.retry_reasons == {} and back.compactions == 0
    assert back.loop_stops == 0


def test_a_rough_run_records_what_it_cost_at_run_level(tmp_path, monkeypatch):
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    record = orclib.record_for("sess-rough", _resolved(), _RoughResult(), [],
                               seconds=40.0, exit_code=0)
    orclib.write(record)
    raw = json.loads((tmp_path / "runs" / "sess-rough.json").read_text())
    # TOP-LEVEL, beside task_class: "which configurations were rough" is asked
    # across every record at once.
    assert raw["retries"] == 5
    assert raw["retry_reasons"] == {"rate_limit": 3, "malformed_tool_call": 2}
    assert raw["compactions"] == 2
    assert raw["loop_stops"] == 4
    assert raw["failover"] == [{"from": "opus-lane", "to": "cheap-lane",
                                "reason": "HTTP 500"}]

    back = orclib.read("sess-rough")
    assert back.retries == 5 and back.compactions == 2 and back.loop_stops == 4
    assert back.retry_reasons["rate_limit"] == 3
    assert back.failover[0]["to"] == "cheap-lane"
    # and the root node carries its own copy, so a node is readable alone
    root = back.nodes[0]
    assert root.retries == 5 and root.compactions == 2 and root.loop_stops == 4
    assert root.stopped == "answered"


def test_a_failed_run_still_records_every_failure_mode(tmp_path, monkeypatch):
    """The rough runs are exactly the ones that fail, so a record written only
    on the happy path records the runs nobody needs to explain."""
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")

    class _FailedRough(_RoughResult):
        stopped = "budget"
        cost = {}

    spawns = [sub.Spawn(agent="explorer", prompt="sweep", steps=6,
                        stopped="budget", seconds=1.0,
                        failover=[], retries=1,
                        retry_reasons={"server_error": 1},
                        compactions=3, loop_stops=2)]
    record = orclib.record_for("sess-fail-modes", _resolved(), _FailedRough(),
                               spawns, seconds=60.0, exit_code=1)
    orclib.write(record)
    back = orclib.read("sess-fail-modes")
    assert back.exit_code == 1 and back.exit_status == "budget"
    # run level sums the root and the child
    assert back.retries == 6 and back.compactions == 5 and back.loop_stops == 6
    assert back.retry_reasons == {"rate_limit": 3, "malformed_tool_call": 2,
                                  "server_error": 1}
    root, child = back.nodes
    assert child.stopped == "budget", "Spawn.stopped must reach the record"
    assert child.compactions == 3, "a child compacting three times is the bug class"
    assert child.failover == [], "an unpinned child HAS a chain that held"


def test_a_pinned_child_has_no_failover_field_and_an_unpinned_one_does(
        tmp_path, monkeypatch):
    """The asymmetry is the point. `chain_for` gives a pinned child ONE rung,
    so it can never drop one - absent, not empty, because empty would say the
    chain held."""
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    spawns = [
        sub.Spawn(agent="reviewer", prompt="p", model="opus", failover=None,
                  retries=0, retry_reasons={}, compactions=0, loop_stops=0),
        sub.Spawn(agent="explorer", prompt="p", failover=[], retries=0,
                  retry_reasons={}, compactions=0, loop_stops=0),
    ]
    record = orclib.record_for("sess-pin", _resolved(), _Result(), spawns,
                               seconds=5.0, exit_code=0)
    orclib.write(record)
    raw = json.loads((tmp_path / "runs" / "sess-pin.json").read_text())
    pinned = [n for n in raw["nodes"] if n["name"] == "reviewer"][0]
    unpinned = [n for n in raw["nodes"] if n["name"] == "explorer"][0]
    assert pinned["failover"] is None
    assert unpinned["failover"] == []


def test_a_driver_child_records_none_rather_than_a_zero_it_cannot_prove(
        tmp_path, monkeypatch):
    """A vendor CLI keeps no counters we can read. Zero would be a claim."""
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    spawns = [sub.Spawn(agent="explorer", prompt="p", driver="claude",
                        stopped="answered", seconds=3.0)]
    record = orclib.record_for("sess-drv", _resolved(), _Result(), spawns,
                               seconds=5.0, exit_code=0)
    orclib.write(record)
    raw = json.loads((tmp_path / "runs" / "sess-drv.json").read_text())
    child = raw["nodes"][1]
    assert child["driver"] == "claude"
    assert child["retries"] is None and child["compactions"] is None
    assert child["loop_stops"] is None and child["retry_reasons"] is None
    # and an unknowable child contributes nothing to the run-level sum
    assert raw["retries"] == 0 and raw["compactions"] == 0


def test_an_unpinned_child_records_only_the_rungs_IT_dropped(monkeypatch):
    """An unpinned child runs on the PARENT's chain object, so the parent's
    own demotions are already in its history. Recording the whole history
    would bill every child for a rung the parent dropped before it existed."""
    from onecli import fallback

    chain = fallback.Chain(rungs=[fallback.Rung(name="a", provider=object()),
                                  fallback.Rung(name="b", provider=object()),
                                  fallback.Rung(name="c", provider=object())])
    chain.advance("the parent's own 500")   # happened BEFORE the child

    class _Child:
        def __init__(self):
            self.chain = chain

        def run(self, prompt, messages=None):
            self.chain.advance("the child's 429")
            return type("R", (), {
                "text": "done", "steps": [1, 2], "tool_calls": 1,
                "stopped": "answered", "retries": 1,
                "retry_reasons": {"rate_limit": 1}, "compactions": [],
                "loop_stops": 0, "error": ""})()

    spawner = sub.Spawner(parent=None, presets={"explorer": _preset("explorer")})
    monkeypatch.setattr(sub.Spawner, "build", lambda self, preset: _Child())
    spawner.spawn("sweep", agent="explorer")

    sp = spawner.spawns[0]
    assert sp.failover == [{"from": "b", "to": "c", "reason": "the child's 429"}]
    assert sp.retries == 1 and sp.retry_reasons == {"rate_limit": 1}
    assert sp.compactions == 0 and sp.loop_stops == 0
    assert sp.stopped == "answered"


# -- one record per run, from the ONE exit point ---------------------------
#
# Deep smoke case 8: `run_once` returned 130 out of an `except KeyboardInterrupt`
# that sat ABOVE the record write, so the run most worth accounting for - a long
# orchestrated one ended with Ctrl-C - left nothing on disk. The fix is
# structural, not a second write call: every way out passes through one
# `finally`, and `exit_code` is the value that was actually returned.

class _CliResult:
    """What the fake Agent hands back. Enough attributes for the whole tail."""

    def __init__(self, stopped="answered", error=""):
        self.stopped = stopped
        self.error = error
        self.text = "OK"
        self.messages = []
        # Real Step/Turn objects, not a placeholder int. run_once records
        # per-request usage with `[dict(step.turn.usage) for step in
        # result.steps]`, so a bare [1] raised
        # "'int' object has no attribute 'turn'" and took three tests down.
        # Using the real classes means the next field added to Step cannot
        # drift away from this fake silently.
        self.steps = [Step(index=0, turn=Turn())]
        self.tool_calls = 1
        self.cost = {}
        self.todos = []
        self.context_tokens = 0
        self.context_window = 0
        self.compactions = []
        self.fallbacks = []
        self.resamples = 0
        self.retries = 0
        self.retry_reasons = {}
        self.advisories = []


def _drive_cli(monkeypatch, tmp_path, behaviour):
    """`cli.run_once` over a fake Agent whose `run` does whatever is asked."""
    from onecli import cli, config

    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)

    class _Agent:
        def __init__(self, *a, **k):
            self.chain = type("C", (), {"current": type("B", (), {
                "name": "opus-lane", "model": "opus-4"})()})()
            self.spawner = None
            # run_once records `agent.wire_requests if agent.run_limits`, so a
            # fake without both raises AttributeError before any assertion runs.
            self.run_limits = None
            self.wire_requests = 0

        def run(self, *a, **k):
            return behaviour()

    monkeypatch.setattr(cli, "Agent", _Agent)
    backend = config.Backend(name="fake", model="fake-1",
                             base_url="http://127.0.0.1:1/v1", wire="openai")
    args = cli.build_parser().parse_args(["--quiet", "--no-hooks", "do a thing"])
    args.prompt = ["do a thing"]
    return cli, args, backend


def _record_files(tmp_path):
    runs = tmp_path / "runs"
    return sorted(p.name for p in runs.glob("*.json")) if runs.exists() else []


def test_an_interrupted_run_still_writes_exactly_one_record(tmp_path, monkeypatch):
    """Ctrl-C is how a long orchestrated run usually ends. It must be recorded.

    Against the earlier implementation the interrupt returned 130 from above the write and
    `~/.one-cli/runs` gained nothing at all.
    """
    def _interrupt():
        raise KeyboardInterrupt

    cli, args, backend = _drive_cli(monkeypatch, tmp_path, _interrupt)
    rc = cli.run_once(args, backend, None, "sess-interrupt")

    assert rc == 130
    assert _record_files(tmp_path), "an interrupted run wrote no record"
    back = orclib.read("sess-interrupt")
    # `interrupt` is its own word: a run killed by a person did not stop for any
    # of the reasons `result.stopped` can name, and there is no Result to ask.
    assert back.exit_status == "interrupt"
    assert back.exit_code == 130


def test_the_recorded_exit_code_is_the_one_the_process_returns(tmp_path,
                                                               monkeypatch):
    """Run `run-0001` recorded 1 while the process exited 2.

    `exit_code` was computed as `0 if answered else 1`, a second guess at a
    number the function already had.
    """
    cli, args, backend = _drive_cli(
        monkeypatch, tmp_path, lambda: _CliResult("error", "provider exploded"))
    rc = cli.run_once(args, backend, None, "sess-exitcode")

    assert rc == 2, "a provider error exits 2"
    assert orclib.read("sess-exitcode").exit_code == rc


def test_an_unexpected_exception_is_recorded_and_still_raised(tmp_path,
                                                              monkeypatch):
    def _explode():
        raise ZeroDivisionError("boom")

    cli, args, backend = _drive_cli(monkeypatch, tmp_path, _explode)
    with pytest.raises(ZeroDivisionError):
        cli.run_once(args, backend, None, "sess-crash")

    assert len(_record_files(tmp_path)) == 1
    assert orclib.read("sess-crash").exit_status == "crash"


def test_a_clean_run_writes_one_record_and_not_two(tmp_path, monkeypatch):
    """The other half of a single exit point: no path may write twice."""
    cli, args, backend = _drive_cli(monkeypatch, tmp_path, _CliResult)
    rc = cli.run_once(args, backend, None, "sess-clean")

    assert rc == 0
    assert len(_record_files(tmp_path)) == 1
    assert orclib.read("sess-clean").exit_code == 0


# -- the record's `backend` is an endpoint, in every node ------------------

def test_the_record_names_endpoints_not_model_ids(tmp_path):
    """Root and children must speak ONE vocabulary, and it is backend names.

    Against the earlier implementation `record_for` set the child's `backend` from `Spawn.model`,
    a model id, while the root got a backend name - so run `run-0002` read
    `NODE orchestrator backend=lane-a` beside
    `NODE beta backend=model-a`, which is also lane-a's model id. Grouping by that field merged two endpoints into one row.
    """
    monkeypatch_dir = tmp_path / "runs"
    orclib.RUN_DIR, saved = monkeypatch_dir, orclib.RUN_DIR
    try:
        presets = {"beta": _preset("beta")}
        resolved = orclib.resolve(presets, None, "lane-a", orclib.FLAG,
                                  "lane-b", 2, 30)
        spawn = sub.Spawn(agent="beta", prompt="p", text="t", steps=1,
                          stopped="answered", seconds=1.0,
                          backend="lane-b", model="shared-model")
        orclib.write(orclib.record_for(
            "sess-endpoints", resolved, _Result(), [spawn], seconds=2.0,
            exit_code=0, chain_name="lane-a", chain_model="shared-model"))
        raw = json.loads((monkeypatch_dir / "sess-endpoints.json").read_text())
    finally:
        orclib.RUN_DIR = saved

    root, child = raw["nodes"]
    # Both nodes ran the SAME model id on DIFFERENT endpoints. Only `backend`
    # can tell them apart, and it is the field a cost comparison groups by.
    assert root["backend"] == "lane-a" and child["backend"] == "lane-b"
    assert root["model"] == child["model"] == "shared-model"
    assert root["backend"] != child["backend"]


# -- an orc file is not a flag: what the guard should actually refuse -------

def _guard_rc(argv, capsys):
    """`cli.main`'s orchestrator guard, and what it printed."""
    from onecli import cli

    rc = cli.main(argv)
    return rc, capsys.readouterr().err


def test_an_orc_file_covering_every_preset_runs_without_a_model_flag(
        tmp_path, monkeypatch, capsys):
    """The headline use of an orc config must not demand a redundant flag.

    Against the earlier implementation the guard tested `_orchestrator_name`'s resolved value,
    which folds in the FILE, so this exited 2 with
    `--orchestrator needs --model` - naming a flag nobody typed, whose value
    the `workers` map would have overridden for every preset anyway.
    """
    from onecli import cli

    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "runs")
    agents = tmp_path / "agents.json"
    agents.write_text(json.dumps({"agents": [{"name": "alpha"}]}))
    orc = _write(tmp_path, {"orchestrator": "drive-lane",
                            "workers": {"alpha": "work-lane",
                                        "general": "work-lane",
                                        "explorer": "work-lane",
                                        "reviewer": "work-lane"}},
                 name="covered.json")

    # The guard is what is under test, so the run is stopped immediately after
    # it: `config.resolve` is the next thing `main` does.
    def _boom(*a, **k):
        raise RuntimeError("guard passed")

    monkeypatch.setattr(cli.config, "resolve", _boom)
    rc, err = _guard_rc(["--orc", orc, "--agents", str(agents),
                         "--quiet", "--no-hooks", "hello"], capsys)
    assert "needs --model" not in err
    assert rc != 2 or "guard passed" in err


def test_an_orc_file_that_strands_a_preset_names_the_preset(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """Refusing is still right when a preset has no pin from any rung - but
    the message must say WHICH agents have no backend, not which flag."""
    agents = tmp_path / "agents.json"
    agents.write_text(json.dumps({"agents": [{"name": "alpha"}]}))
    orc = _write(tmp_path, {"orchestrator": "drive-lane",
                            "workers": {"alpha": "work-lane"}},
                 name="partial.json")

    rc, err = _guard_rc(["--orc", orc, "--agents", str(agents),
                         "--quiet", "--no-hooks", "hello"], capsys)
    assert rc == 2
    assert "needs --model" not in err
    # the built-ins the file said nothing about, named. `reviewer` is absent
    # because it carries its own `model`, which is a pin from a rung.
    assert "explorer" in err and "general" in err
    assert "reviewer" not in err and "alpha" not in err


def test_the_orchestrator_FLAG_alone_still_exits_2_with_its_own_message(capsys):
    """Pinned in the Resolution-rules section: with only a name there is
    genuinely nothing to hand the children, and `--model` is the missing thing."""
    rc, err = _guard_rc(["--orchestrator", "drive-lane", "--quiet",
                         "--no-hooks", "hello"], capsys)
    assert rc == 2
    assert "--orchestrator needs --model" in err


# -- reading the records back: `--orc-report` -------------------------------
#
# The failure mode these guard against is not a crash. It is a report that is
# arithmetically perfect and still lies: a median over three of ninety-six runs
# printed the way a median over ninety-six would be, a `None` that became a
# zero on the way into a column, or two endpoints merged into one row because
# they happen to serve the same model id.


def _runs_dir(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    monkeypatch.setattr(orclib, "RUN_DIR", runs)
    monkeypatch.setenv("NO_COLOR", "1")
    from onecli import render
    render.refresh()
    return runs


def _node(backend="lane", **over):
    """A node as `record_for` writes one today, before any overrides."""
    node = {"name": "orchestrator", "backend": backend, "model": "", "source":
            "default", "seconds": 1.0, "steps": 1, "tool_calls": 0,
            "driver": "", "input_tokens": None, "output_tokens": None,
            "usd": None, "metered": True, "stopped": "answered",
            "failover": [], "retries": 0, "retry_reasons": {},
            "compactions": 0, "loop_stops": 0}
    node.update(over)
    return node


def _record(runs, session, nodes, **over):
    body = {"session": session, "started": "2026-09-04T05:00:00+00:00",
            "finished": "2026-09-04T05:00:01+00:00", "seconds": 1.0,
            "exit_status": "answered", "exit_code": 0, "task_class": "",
            "failover": [], "retries": 0, "retry_reasons": {},
            "compactions": 0, "loop_stops": 0, "config": {}, "nodes": nodes}
    body.update(over)
    if body.get("task_class") is _ABSENT:
        del body["task_class"]
    (runs / f"{session}.json").write_text(json.dumps(body))
    return body


_ABSENT = object()


def _row(out: str, backend: str, klass: str | None = None) -> str:
    rows = [ln for ln in out.splitlines()
            if ln.strip().startswith(backend + " ")
            and (klass is None or f" {klass} " in ln)]
    assert rows, f"no row for {backend!r} in:\n{out}"
    return rows[0]


def test_two_backends_serving_one_model_id_are_two_rows_not_one(tmp_path,
                                                                monkeypatch):
    """Two endpoints can serve one model id.

    Grouping on the id merges them on exactly the field a cost comparison is
    built on, so the group key is the backend NAME and the id is shown beside
    it rather than instead of it.
    """
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "one", [_node("lane-a", model="shared-model",
                                seconds=2.0)])
    _record(runs, "two", [_node("lane-b", model="shared-model",
                                seconds=9.0)])
    got, errors = orclib.records()
    assert errors == [] and len(got) == 2
    out = orclib.report(got)
    assert "lane-a " in out and "lane-b" in out
    groups, _ = orclib.group(got)
    assert sorted((g.backend, g.nodes) for g in groups) == [
        ("lane-a", 1), ("lane-b", 1)]
    # the id is still printed, on both rows - it is worth seeing, it just does
    # not group
    assert _row(out, "lane-a").count("shared-model") == 1
    assert _row(out, "lane-b").count("shared-model") == 1


def test_a_column_no_record_carried_is_a_dash_and_never_a_zero(tmp_path,
                                                               monkeypatch):
    """`usd` is None on every record in the real store. A zero there would
    read as "this configuration was free"."""
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "one", [_node("lane", usd=None)])
    _record(runs, "two", [_node("lane", usd=None)])
    out = orclib.report(orclib.records()[0])
    row = _row(out, "lane")
    assert row.rstrip().endswith("-"), row
    assert "$0" not in row and "$" not in row


def test_a_cell_says_how_many_nodes_it_is_actually_over(tmp_path, monkeypatch):
    """A median over one of four must not print like a median over four."""
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "one", [_node("lane", input_tokens=8000),
                          _node("lane", input_tokens=None),
                          _node("lane", input_tokens=None),
                          _node("lane", input_tokens=None)])
    out = orclib.report(orclib.records()[0])
    row = _row(out, "lane")
    assert "8.0k 1/4" in row, row
    # and the columns every node DID carry are not marked with a coverage
    assert "1.0s 4/4" not in row and "1.0s" in row


def test_one_reading_is_marked_as_a_number_and_not_a_median(tmp_path,
                                                            monkeypatch):
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "solo", [_node("thin", seconds=5.0)])
    _record(runs, "a", [_node("thick", seconds=1.0)])
    _record(runs, "b", [_node("thick", seconds=2.0)])
    _record(runs, "c", [_node("thick", seconds=3.0)])
    out = orclib.report(orclib.records()[0])
    assert "5.0s!" in _row(out, "thin")
    assert "!" not in _row(out, "thick")
    assert "one reading" in out, "the mark has to be explained on the page"


def test_the_median_is_the_middle_reading_not_the_mean(tmp_path, monkeypatch):
    """One slow outlier must not become the whole group's number."""
    runs = _runs_dir(tmp_path, monkeypatch)
    for sid, secs in (("a", 1.0), ("b", 2.0), ("c", 9.0)):
        _record(runs, sid, [_node("lane", seconds=secs)])
    out = orclib.report(orclib.records()[0])
    row = _row(out, "lane")
    assert "2.0s" in row and "4.0s" not in row


def test_a_record_written_before_the_task_class_field_is_not_given_one(
        tmp_path, monkeypatch):
    """A missing `task_class` is a record that predates that field. A record
    carrying `""` is a run that declared no class. Two different facts."""
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "old", [_node("lane")], task_class=_ABSENT)
    _record(runs, "new", [_node("lane")], task_class="")
    _record(runs, "named", [_node("lane")], task_class="review")
    got, errors = orclib.records()
    assert errors == []
    assert "task_class" not in got[[r["session"] for r in got].index("old")]
    out = orclib.report(got)
    groups, _ = orclib.group(got)
    assert sorted(g.task_class for g in groups) == [
        orclib.NO_CLASS, orclib.UNRECORDED, "review"]
    assert orclib.UNRECORDED in out and orclib.NO_CLASS in out
    # and the page says which is which, so the label is not folklore
    assert "before the field" in out


def test_a_record_written_before_the_failure_modes_reads_absent_not_clean(
        tmp_path, monkeypatch):
    """Legacy records predate the five failure-mode fields.
    A zero for an absent field would incorrectly say the run was smooth."""
    runs = _runs_dir(tmp_path, monkeypatch)
    old = _node("lane")
    for gone in ("stopped", "failover", "retries", "retry_reasons",
                 "compactions", "loop_stops"):
        del old[gone]
    _record(runs, "old", [old])
    got, errors = orclib.records()
    assert errors == [], "an old record is not an unreadable one"
    out = orclib.report(got)
    # the wall time it DID carry is summarised; the four counters are absent
    rows = [ln for ln in out.splitlines() if ln.strip().startswith("lane ")]
    assert len(rows) == 2, out
    spend, rough = rows
    assert "1.0s!" in spend
    assert rough.split()[2:] == ["1", "-", "-", "-", "-", "-"], rough


def test_a_pinned_childs_null_failover_is_not_a_chain_that_held(tmp_path,
                                                                monkeypatch):
    """The sharpest None in the module docstring: `chain_for` gives a pinned
    child exactly one rung, so it has none to drop and records None. An empty
    list means the chain HELD. Summing them together erases the difference."""
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "mixed", [_node("lane", failover=[]),
                            _node("lane", name="pinned", failover=None)])
    out = orclib.report(orclib.records()[0])
    rough = [ln for ln in out.splitlines() if ln.strip().startswith("lane ")][1]
    assert "0 1/2" in rough, rough


def test_an_unreadable_record_is_named_and_the_others_still_report(tmp_path,
                                                                   monkeypatch):
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "good", [_node("lane")])
    (runs / "broken.json").write_text("{not json")
    (runs / "wrong.json").write_text("[1, 2]")
    got, errors = orclib.records()
    assert len(got) == 1
    assert len(errors) == 2
    assert any("broken.json" in e for e in errors)
    assert any("wrong.json" in e and "not a run record" in e for e in errors)
    out = orclib.report(got, unreadable=len(errors))
    assert "lane" in out and "2 unreadable" in out


def test_an_empty_store_says_so_in_a_sentence(tmp_path, monkeypatch):
    """An empty table is the one output that reads as a bug - `summarise`'s
    own rule, applied to the other read verb."""
    _runs_dir(tmp_path, monkeypatch)
    got, errors = orclib.records()
    assert (got, errors) == ([], [])
    out = orclib.report(got)
    assert "no run records yet" in out
    assert "backend" not in out and "nodes" not in out


def test_a_record_with_no_nodes_is_counted_and_not_attributed(tmp_path,
                                                              monkeypatch):
    """A record with nothing under it is evidence about no backend at all."""
    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "empty", [])
    _record(runs, "real", [_node("lane")])
    got = orclib.records()[0]
    groups, nodeless = orclib.group(got)
    assert nodeless == 1
    assert [g.backend for g in groups] == ["lane"]
    out = orclib.report(got)
    assert "2 runs" in out and "1 node" in out and "1 with no node" in out


def test_the_report_flag_runs_off_the_cli_and_never_builds_a_provider(
        tmp_path, monkeypatch, capsys):
    """A read verb: no model, no spawn, no cost. Same shape as `--orc-list`
    and `--orc-show`, dispatched out of `main` before anything resolves."""
    from onecli import cli

    runs = _runs_dir(tmp_path, monkeypatch)
    _record(runs, "one", [_node("read-only-lane", seconds=3.0)])

    def _boom(*a, **k):
        raise AssertionError("a read verb resolved a backend")

    monkeypatch.setattr(cli.config, "resolve", _boom)
    monkeypatch.setattr(cli.providers, "build", _boom)
    rc = cli.main(["--orc-report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "read-only-lane" in out and "3.0s!" in out


def test_a_grandchild_node_is_recorded_at_its_own_level():
    """`record_for` must carry the level, or the record cannot say depth 3."""
    spawns = [sub.Spawn(agent="digger", prompt="p", backend="dashscope",
                        depth=1),
              sub.Spawn(agent="miner", prompt="p", backend="kimi", depth=2)]
    rec = orclib.record_for("s1", _resolved(), _Result(), spawns,
                            seconds=1.0, exit_code=0)
    assert [n.depth for n in rec.nodes] == [0, 1, 2]


def test_the_level_column_is_a_dash_for_a_record_that_predates_it():
    """The honesty rule, applied to the newest field: absent is not level 0."""
    old = {"session": "old", "task_class": "", "nodes": [{"backend": "kimi"}]}
    new = {"session": "new", "task_class": "", "nodes": [{"backend": "alt",
                                                          "depth": 2}]}
    text = orclib.report([old, new])
    rows = {}
    for line in text.splitlines():
        # The FIRST table only: the failure-mode table repeats the backend
        # name and carries no level column.
        if line.strip().startswith(("kimi", "alt")):
            rows.setdefault(line.split()[0], line)
    assert " - " in rows["kimi"], "no depth on the record is not level 0"
    assert " 2 " in rows["alt"]


def test_a_display_string_is_never_written_into_a_records_backend_field():
    """`Slot.backend` speaks to a person; `Node.backend` is grouped on.

    `resolve()` fills an unnamed slot with `(parent's chain)` and an unnamed
    orchestrator with `(--model)`, which is right for the echo and wrong for a
    field a census groups by. A record that carries the first of them, from
    `record_for`'s slot fallback, reads in `--orc-report` as an endpoint that
    does not exist.
    """
    resolved = orclib.resolve(
        presets={"explorer": sub.EXPLORER}, orc=None, orchestrator="",
        orchestrator_source=orclib.DEFAULT, worker_default="",
        max_depth=2, max_steps=6)
    assert resolved.workers[0].backend == "(parent's chain)", "the echo's own"

    spawns = [sub.Spawn(agent="explorer", prompt="p")]  # no backend recorded
    record = orclib.record_for("sess-display", resolved, _Result(), spawns,
                               seconds=1.0, exit_code=0)
    backends = [n.backend for n in record.nodes]
    assert backends == [orclib.UNKNOWN_BACKEND, orclib.UNKNOWN_BACKEND], (
        f"a display string reached the record: {backends}")


def test_a_driver_only_group_reads_as_no_coverage_and_never_as_clean_zeros():
    """A driver row: a vendor CLI keeps no counters we can read.

    All four counters are `None` on a driver node BY DESIGN (`Spawner._drive`
    fills none of them). The report must print `-` for that - "nothing here
    carried the column" - and never `0`, which would read as a configuration
    that never retried, never compacted and never looped.
    """
    run = {"session": "d", "task_class": "research",
           "nodes": [{"backend": "kimi", "model": "kimi-for-coding",
                      "depth": 1, "driver": "opencode", "seconds": 5.9,
                      "steps": 0, "stopped": "answered",
                      "input_tokens": None, "output_tokens": None,
                      "usd": None, "failover": None, "retries": None,
                      "retry_reasons": None, "compactions": None,
                      "loop_stops": None}]}
    rough = [line for line in orclib.report([run]).splitlines()
             if line.strip().startswith("kimi")][-1]
    cells = rough.split()
    assert cells[:3] == ["kimi", "research", "1"]
    assert cells[3:7] == ["-", "-", "-", "-"], (
        f"a driver's uncounted columns read as numbers: {rough}")

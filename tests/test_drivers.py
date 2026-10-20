"""Drivers - a subagent whose body is somebody else's CLI.

The interesting tests are not "does it run opencode". They are the four ways a
driver becomes a worse subagent than the in-process one: a narrowed parent
getting its write tools back through a vendor CLI, a guard chain that never
sees the launch, a spend that lands nowhere, and a missing binary that ends the
parent's turn instead of returning a sentence.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from onecli import config, drivers, harnesses
from onecli import subagent as sub
from onecli.agent import Agent
from onecli.context import CostMeter
from onecli.guard import GuardChain, ReadOnlyGuard, Verdict
from onecli.providers.base import Turn

from test_onecli_agent import ScriptedProvider


BACKEND = config.Backend(name="lane", model="m", base_url="http://x/v1",
                         wire=config.WIRE_OPENAI, api_key="k")


def _parent(tmp_path, **kw) -> Agent:
    kw.setdefault("spawner", sub.Spawner(parent=None))
    return Agent(ScriptedProvider([Turn(text="parent done")]), str(tmp_path), **kw)


def _preset(driver: str, **kw) -> sub.Preset:
    return sub.Preset(name="hand-off", driver=driver, **kw)


@pytest.fixture
def installed(monkeypatch):
    """Pretend every adapter's binary is on this box.

    Which CLIs are installed is a fact about the machine - none are, on CI - and
    a policy test that changes its answer with the machine is testing the
    machine.
    """
    monkeypatch.setattr(harnesses.Harness, "available", lambda self: True)


# -- what may be a driver at all -------------------------------------------

def test_our_own_cli_is_not_a_driver():
    """Forking `one` would lose the shared meter and guard the child exists for."""
    assert "one" not in drivers.names()
    assert "use an in-process subagent" in drivers.refusal("one", None)


def test_a_harness_that_cannot_be_driven_headless_is_not_offered():
    assert "octofriend" not in drivers.names()
    assert "never exits" in drivers.refusal("octofriend", None)


def test_an_unknown_driver_names_the_ones_that_exist():
    reason = drivers.refusal("nosuchcli", None)
    assert "no driver named 'nosuchcli'" in reason


# -- the surface rule ------------------------------------------------------

def test_a_narrowed_parent_may_not_hand_its_workspace_to_a_write_capable_cli(installed):
    """The hole this closes: `--agent explorer` regaining write via a driver.

    We cannot intersect a foreign CLI's tools with ours, so the answer is a
    refusal rather than a narrowing.
    """
    explorer = ["read", "ls", "glob", "grep"]
    assert drivers.read_only_for(explorer) is True
    assert "no read-only mode" in drivers.refusal("qwen", explorer)


def test_a_narrowed_parent_may_drive_a_cli_that_takes_a_read_only_instruction(installed):
    explorer = ["read", "ls", "glob", "grep"]
    assert drivers.refusal("opencode", explorer) == ""
    assert "opencode" in drivers.READ_ONLY_CAPABLE


def test_gptme_is_not_read_only_capable_because_its_read_only_mode_has_a_shell():
    """`_gptme_build` passes `-t read,shell` for a read-only cell. A shell is
    not read-only, so gptme is not on the list."""
    assert "gptme" not in drivers.READ_ONLY_CAPABLE


def test_a_parent_holding_the_whole_surface_is_not_read_only():
    assert drivers.read_only_for(None) is False
    assert drivers.read_only_for(["read", "write"]) is False


# -- the wire --------------------------------------------------------------

def test_a_cli_that_cannot_speak_this_endpoints_wire_says_so():
    anthropic = config.Backend(name="lane", model="m", base_url="http://x",
                               wire=config.WIRE_ANTHROPIC)
    reason = drivers.wire_refusal("qwen", anthropic)
    assert "cannot drive the anthropic wire" in reason
    assert drivers.wire_refusal("opencode", anthropic) == ""


# -- the backend comes off the LIVE rung -----------------------------------

def test_the_backend_is_read_off_the_provider_the_parent_is_standing_on():
    """Not the one it started on: a demoted chain drives the rung it is on."""
    from onecli import fallback

    class P:
        base_url, model, api_key = "http://a/v1", "m-a", "ka"

    class Q:
        base_url, model, api_key = "http://b/v1", "m-b", "kb"

    chain = fallback.Chain(rungs=[fallback.Rung(name="a", provider=P()),
                                  fallback.Rung(name="b", provider=Q())])
    chain.advance("429")
    backend = drivers.backend_for(chain.current)
    assert (backend.name, backend.model, backend.base_url) == ("b", "m-b", "http://b/v1")
    assert backend.api_key == "kb"


def test_an_anthropic_provider_is_recognised_as_the_anthropic_wire():
    from onecli.providers.anthropic_wire import AnthropicWire

    class Rung:
        name = "r"
        provider = AnthropicWire("http://x", "m", "k")

    assert drivers.backend_for(Rung()).wire == config.WIRE_ANTHROPIC


# -- the guard sees the real launch ----------------------------------------

def test_the_launch_is_presented_to_the_guard_as_a_mutating_bash_call():
    seen = {}

    class Spy:
        def check(self, tool_name, arguments, mutating):
            seen.update(tool=tool_name, args=arguments, mutating=mutating)
            return Verdict.allow()

    chain = GuardChain([Spy()])
    drivers.guard_verdict(chain, ["opencode", "run", "hello"], "/ws")
    assert seen["tool"] == "bash"
    assert seen["mutating"] is True
    assert seen["args"]["command"] == "opencode run hello"


def test_a_read_only_guard_chain_refuses_to_hand_over_the_workspace():
    verdict = drivers.guard_verdict(GuardChain([ReadOnlyGuard()]), ["qwen"], "/ws")
    assert verdict.allowed is False


def test_no_guard_chain_means_no_verdict_rather_than_a_denial():
    assert drivers.guard_verdict(None, ["qwen"], "/ws") is None


def test_a_denied_launch_never_starts_the_process(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("the CLI was started despite a denial")

    monkeypatch.setattr(subprocess, "run", boom)
    run = drivers.run("opencode", "hi", backend=BACKEND, workspace="/tmp",
                      before=lambda argv: "refused by the test")
    assert run.result.stopped == "denied"
    assert run.result.error == "refused by the test"


# -- a failure is a sentence, never a raise --------------------------------

def test_a_missing_binary_comes_back_as_a_tool_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("no such file")))
    run = drivers.run("opencode", "hi", backend=BACKEND, workspace="/tmp")
    assert run.result.stopped == "crash"
    assert "could not start opencode" in run.result.error


def test_a_driver_that_runs_long_is_killed_and_reported(monkeypatch):
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="opencode", timeout=5)

    monkeypatch.setattr(subprocess, "run", timeout)
    run = drivers.run("opencode", "hi", backend=BACKEND, workspace="/tmp", timeout=5)
    assert run.result.stopped == "timeout"
    assert "exceeded 5s" in run.result.error


def test_a_nonzero_exit_with_no_parsable_answer_keeps_what_it_printed(monkeypatch):
    class Proc:
        returncode, stdout, stderr = 1, "", "auth failed: no such provider"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: Proc())
    run = drivers.run("opencode", "hi", backend=BACKEND, workspace="/tmp")
    assert run.result.stopped == "crash"
    assert "auth failed" in run.result.error


def test_the_private_config_dir_holding_the_api_key_is_removed(monkeypatch):
    """Every adapter writes the endpoint AND the key into a config file."""
    seen = {}

    class Proc:
        returncode, stdout, stderr = 0, '{"type":"text","text":"hi"}', ""

    def spy(cmd, **kw):
        seen["home"] = kw.get("env", {}).get("HOME", "")
        return Proc()

    monkeypatch.setattr(subprocess, "run", spy)
    drivers.run("opencode", "hi", backend=BACKEND, workspace="/tmp")
    import os
    assert seen["home"] and not os.path.exists(seen["home"])


# -- the money -------------------------------------------------------------

def test_an_unmetered_driver_is_charged_a_floor_and_flagged_estimated():
    """Charging zero would make --max-budget a ceiling with a door in it."""
    meter = CostMeter()
    drivers.charge(meter, "a prompt of some length", harnesses.Result(text="an answer"))
    assert meter.input_tokens > 0
    assert meter.output_tokens > 0
    assert meter.estimated is True


def test_a_driver_that_reports_usage_is_believed_rather_than_estimated():
    meter = CostMeter()
    drivers.charge(meter, "x", harnesses.Result(
        text="y", cost={"input_tokens": 1200, "output_tokens": 300}))
    assert (meter.input_tokens, meter.output_tokens) == (1200, 300)
    assert meter.estimated is False


def test_the_charge_lands_on_the_parents_meter_not_a_copy(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    spawner = parent.spawner
    spawner.presets["hand-off"] = _preset("opencode")
    monkeypatch.setattr(drivers, "refusal", lambda *a: "")
    monkeypatch.setattr(drivers, "run", lambda *a, **kw: drivers.DriverRun(
        "opencode", harnesses.Result(text="done"), seconds=1.0))
    before = parent.meter.input_tokens
    spawner.spawn("do the thing", "hand-off")
    assert parent.meter.input_tokens > before


# -- the preset ------------------------------------------------------------

def test_a_preset_file_can_name_a_driver(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(json.dumps({"reviewer": {"description": "d",
                                             "driver": "opencode",
                                             "timeout": 90}}))
    presets, errors = sub.load_presets([str(path)])
    assert errors == []
    assert presets["reviewer"].driver == "opencode"
    assert presets["reviewer"].seconds(300.0) == 90.0


def test_a_preset_with_no_timeout_takes_the_default():
    assert sub.Preset(name="x").seconds(300.0) == 300.0


# -- through the spawner ---------------------------------------------------

def test_a_refused_driver_is_a_tool_error_the_parent_can_read(tmp_path):
    parent = _parent(tmp_path)
    parent.spawner.presets["hand-off"] = _preset("nosuchcli")
    answer = parent.spawner.spawn("go", "hand-off")
    assert answer.startswith("error: no driver named 'nosuchcli'")


def test_a_driver_answer_comes_back_with_its_accounting(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    parent.spawner.presets["hand-off"] = _preset("opencode")
    monkeypatch.setattr(drivers, "refusal", lambda *a: "")
    monkeypatch.setattr(drivers, "run", lambda *a, **kw: drivers.DriverRun(
        "opencode", harnesses.Result(text="the answer", tool_calls=3), seconds=12.0))
    answer = parent.spawner.spawn("go", "hand-off")
    assert answer.startswith("the answer")
    assert "hand-off via opencode: 12.0s" in answer
    assert "3 tools" in answer
    assert "tokens not reported" in answer
    assert parent.spawner.spawns[0].driver == "opencode"
    assert parent.spawner.spawns[0].metered is False


def test_a_driver_that_answers_nothing_says_what_stopped_it(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    parent.spawner.presets["hand-off"] = _preset("opencode")
    monkeypatch.setattr(drivers, "refusal", lambda *a: "")
    monkeypatch.setattr(drivers, "run", lambda *a, **kw: drivers.DriverRun(
        "opencode", harnesses.Result(error="exited 1: boom", stopped="crash")))
    answer = parent.spawner.spawn("go", "hand-off")
    assert answer.startswith("error: the opencode driver returned no answer")
    assert "boom" in answer


def test_a_driver_child_is_never_built_as_an_in_process_agent(tmp_path, monkeypatch):
    """The branch is taken before `build`, so no model is constructed for it."""
    parent = _parent(tmp_path)
    parent.spawner.presets["hand-off"] = _preset("opencode")
    monkeypatch.setattr(sub.Spawner, "build",
                        lambda *a, **kw: pytest.fail("built an in-process child"))
    monkeypatch.setattr(drivers, "refusal", lambda *a: "")
    monkeypatch.setattr(drivers, "run", lambda *a, **kw: drivers.DriverRun(
        "opencode", harnesses.Result(text="ok")))
    assert parent.spawner.spawn("go", "hand-off").startswith("ok")


# -- the pi adapter, fixed to make the live run possible -------------------

def test_a_keyless_endpoint_never_writes_an_empty_api_key(tmp_path):
    """`"apiKey": ""` fails pi's minLength-1 schema, the whole models.json is
    discarded, and the run dies with `Unknown provider "onelane"` before one
    request is sent. Found live against llamacpp-gone, which has no key."""
    keyless = config.Backend(name="local", model="m", base_url="http://x/v1",
                             wire=config.WIRE_OPENAI, api_key="")
    cell = harnesses.Cell(backend=keyless, model_id="", workspace=tmp_path,
                          prompt="p", cfg_dir=tmp_path)
    assert cell.key == harnesses.Cell.PLACEHOLDER_KEY
    harnesses.HARNESSES["pi"].build(cell)
    written = json.loads((tmp_path / ".pi/agent/models.json").read_text())
    assert written["providers"]["onelane"]["apiKey"]


def test_a_backend_with_a_key_still_hands_over_the_real_one(tmp_path):
    cell = harnesses.Cell(backend=BACKEND, model_id="", workspace=tmp_path,
                          prompt="p", cfg_dir=tmp_path)
    assert cell.key == "k"


def test_pi_is_told_not_to_refresh_its_catalogue_over_the_network(tmp_path):
    cell = harnesses.Cell(backend=BACKEND, model_id="", workspace=tmp_path,
                          prompt="p", cfg_dir=tmp_path)
    assert harnesses.HARNESSES["pi"].build(cell).env["PI_OFFLINE"] == "1"


def test_pi_reports_usage_so_a_pi_driver_is_metered_rather_than_estimated():
    """The one third-party harness here that reports tokens. Accumulated across
    turns, not read off the last one, which reports only its own."""
    stream = "\n".join(json.dumps(event) for event in [
        {"type": "turn_end", "message": {"usage": {"input": 100, "output": 20}},
         "toolResults": [{}]},
        {"type": "turn_end", "message": {"usage": {"input": 300, "output": 40}},
         "toolResults": []},
        {"type": "agent_end", "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "beta"}]}]},
    ])
    result = harnesses.HARNESSES["pi"].parse(stream, "")
    assert result.text == "beta"
    assert result.cost == {"input_tokens": 400, "output_tokens": 60}

    meter = CostMeter()
    drivers.charge(meter, "the prompt", result)
    assert (meter.input_tokens, meter.output_tokens) == (400, 60)
    assert meter.estimated is False


def test_the_surface_rule_is_answered_before_what_this_box_has_installed(monkeypatch):
    """A narrowed parent asking for a write-capable CLI is refused on policy
    even where that CLI is absent - otherwise the answer reads as "install it
    and retry" for something it would be refused anyway."""
    monkeypatch.setattr(harnesses.Harness, "available", lambda self: False)
    assert "no read-only mode" in drivers.refusal("qwen", ["read", "grep"])
    assert "not installed" in drivers.refusal("qwen", None)

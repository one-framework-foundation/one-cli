"""Offline tests for the tool-call capability ladder. No network, no endpoint.

Every case drives `run_task` against a scripted wire, because the thing under
test is the SCORING - what the harness concludes from a given sequence of turns
- and a real endpoint cannot be made to produce a chosen sequence on demand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onecli.evals import toolprobe  # noqa: E402
from onecli.config import WIRE_ANTHROPIC, WIRE_OPENAI, Backend  # noqa: E402
from onecli.providers.base import ProviderError, ToolCall, Turn  # noqa: E402


def _backend(wire: str = WIRE_OPENAI) -> Backend:
    return Backend(name="lane", model="m-1", base_url="https://example.test/v1",
                   wire=wire, api_key="sk-test")


class ScriptedWire:
    """Hands back a prepared list of turns, one per round, and records asks."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []
        self.extra_body: dict = {}

    def chat(self, messages, tools=None, max_tokens=2048, **kw):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "tools": tools, "extra_body": dict(self.extra_body)})
        if not self.turns:
            return Turn(text="")
        return self.turns.pop(0)


def _install(monkeypatch, wire):
    monkeypatch.setattr(toolprobe, "build", lambda backend, timeout=300.0: wire)
    return wire


def _task(task_id: str) -> toolprobe.Task:
    return next(t for t in toolprobe.BATTERY if t.id == task_id)


def _call(name, **arguments):
    return ToolCall(id=f"c-{name}", name=name, arguments=arguments)


def test_a_structured_call_is_executed_and_its_result_consumed(monkeypatch):
    wire = _install(monkeypatch, ScriptedWire([
        Turn(tool_calls=[_call("get_build_info", component="operator-ui")]),
        Turn(text="The build is 2026.07.29-b41."),
    ]))
    row = toolprobe.run_task(_backend(), _task("T1_single"))
    assert row.structured_calls == 1
    assert row.executed == 1
    assert row.consumed and row.answer_ok and row.passed
    # The tool's real return value must reach the model, not a stub.
    tool_msg = [m for m in wire.calls[-1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"])["build"] == "2026.07.29-b41"


def test_a_call_that_arrives_as_prose_is_l1b_not_silence(monkeypatch):
    _install(monkeypatch, ScriptedWire([
        Turn(text='I will call {"name": "get_build_info", "arguments": {}} now.'),
    ]))
    row = toolprobe.run_task(_backend(), _task("T1_single"))
    assert row.structured_calls == 0
    assert row.l1b_hits == ["json_tool_call_obj"]
    assert toolprobe.rung([row]) == "L1b"


def test_a_prose_call_hiding_in_the_reasoning_channel_still_counts(monkeypatch):
    """A thinking model returns an empty `text`; scoring it alone reads as silence."""
    _install(monkeypatch, ScriptedWire([
        Turn(text="", thinking="<tool_call> lookup_port sglang </tool_call>"),
    ]))
    row = toolprobe.run_task(_backend(), _task("T2_consume"))
    assert "xml_tool_call_tag" in row.l1b_hits
    assert row.reasoning_chars > 0


def test_silence_is_l0_and_never_l1b(monkeypatch):
    _install(monkeypatch, ScriptedWire([Turn(text="The host build is unknown.")]))
    row = toolprobe.run_task(_backend(), _task("T1_single"))
    assert row.l1b_hits == []
    assert toolprobe.rung([row]) == "L0"


def test_picking_the_deprecated_twin_fails_on_selection_not_emission(monkeypatch):
    """The model DID call a tool, so it is past L1 - and still fails the task."""
    _install(monkeypatch, ScriptedWire([
        Turn(tool_calls=[_call("lookup_port_legacy", service="ollama")]),
        Turn(text="The port is 21434."),
    ]))
    row = toolprobe.run_task(_backend(), _task("T8_ambiguous_pair"))
    assert row.structured_calls == 1
    assert row.selection_ok is False
    assert row.passed is False


def test_a_three_deep_chain_needs_three_rounds_not_three_calls(monkeypatch):
    """All three in one round is not a chain: nothing depended on anything."""
    _install(monkeypatch, ScriptedWire([
        Turn(tool_calls=[_call("lookup_port", service="operator"),
                         _call("whoami_port", port=4104),
                         _call("get_build_info", component="archive-service")]),
        Turn(text="The build is 2026.07.29-t19."),
    ]))
    row = toolprobe.run_task(_backend(), _task("T6_chain3"))
    assert row.structured_calls == 3
    assert row.chain_depth == 1
    assert row.answer_ok is True
    assert row.chain_ok is False


def test_the_same_failing_call_repeated_is_counted(monkeypatch):
    _install(monkeypatch, ScriptedWire([
        Turn(tool_calls=[_call("read_config", path="/etc/deployment/ports.conf")]),
        Turn(tool_calls=[_call("read_config", path="/etc/deployment/ports.conf")]),
        Turn(text="CANNOT DETERMINE"),
    ]))
    row = toolprobe.run_task(_backend(), _task("T7_repeat_fail"))
    assert row.repeat_max == 2
    assert row.repeat_failed_max == 2
    assert row.tool_errors == 2
    assert row.answer_ok is True


def test_forced_choice_is_spelled_per_wire(monkeypatch):
    openai = _install(monkeypatch, ScriptedWire([Turn(text="Paris.")]))
    toolprobe.run_task(_backend(WIRE_OPENAI), _task("T9_required_choice"))
    assert openai.calls[0]["extra_body"]["tool_choice"] == "required"

    anthropic = _install(monkeypatch, ScriptedWire([Turn(text="Paris.")]))
    toolprobe.run_task(_backend(WIRE_ANTHROPIC), _task("T9_required_choice"))
    assert anthropic.calls[0]["extra_body"]["tool_choice"] == {"type": "any"}


def test_forced_choice_fails_when_round_one_carries_no_call(monkeypatch):
    _install(monkeypatch, ScriptedWire([Turn(text="Paris.")]))
    row = toolprobe.run_task(_backend(), _task("T9_required_choice"))
    assert row.called_in_round1 is False
    assert row.forced_ok is False and row.passed is False


def test_a_dead_endpoint_is_a_row_and_never_a_model_failure(monkeypatch):
    class Dead(ScriptedWire):
        def chat(self, *a, **kw):
            raise ProviderError("http 503 service unavailable")

    _install(monkeypatch, Dead([]))
    row = toolprobe.run_task(_backend(), _task("T1_single"))
    assert "503" in row.error
    assert row.answer_ok is False
    assert toolprobe.rung([row]) == "dead"


def test_a_model_that_never_settles_is_recorded_not_hung(monkeypatch):
    _install(monkeypatch, ScriptedWire(
        [Turn(tool_calls=[_call("lookup_port", service="sglang")])] * 40))
    row = toolprobe.run_task(_backend(), _task("T2_consume"))
    assert row.rounds == toolprobe.MAX_ROUNDS
    assert "did not settle" in row.error


def test_the_control_task_offers_no_tools_at_all(monkeypatch):
    wire = _install(monkeypatch, ScriptedWire([Turn(text="Paris.")]))
    row = toolprobe.run_task(_backend(), _task("T5_no_tools_control"))
    assert wire.calls[0]["tools"] is None
    assert row.answer_ok is True
    # A task with no tools cannot speak for the ladder in either direction.
    assert toolprobe.rung([row]) == "dead"


def test_the_needle_sits_near_the_start_of_the_haystack(monkeypatch):
    """Needle-at-the-end measures nothing: truncation drops the oldest first."""
    prompt = _task("T11_long_context").prompt
    position = prompt.index(toolprobe._NEEDLE) / len(prompt)
    assert position < 0.05


def test_every_battery_task_names_only_real_tools():
    for task in toolprobe.BATTERY:
        for name in task.tools:
            assert name in toolprobe.TOOLS, f"{task.id} offers unknown {name}"

"""Guards: the seam that can refuse a tool call before it runs.

Covers the hook payload translation, verdict handling and the chain's
behaviour in the agent loop, using small command guards written per test.
"""

from __future__ import annotations

import textwrap

import pytest

from onecli import guard, tools
from onecli.agent import Agent, _as_verdict
from onecli.providers import Turn
# The real thing, not a hand-rolled stand-in: a double that reimplements
# ToolCall's fields drifts silently the moment one is added, and reports it as
# ten failures in the loop rather than as the missing field it is.
from onecli.providers.base import ToolCall as _Call


# --------------------------------------------------------------------------
# translation


def test_bash_call_presents_as_the_name_guards_match_on():
    payload = guard.to_hook_payload("bash", {"command": "ls"}, True, "/ws")
    assert payload["tool_name"] == "Bash"
    assert payload["tool_input"]["command"] == "ls"
    assert payload["cwd"] == "/ws"
    assert payload["onecli"] == {"tool": "bash", "mutating": True}


@pytest.mark.parametrize("tool", ["read", "write", "edit"])
def test_file_tools_rename_path_to_file_path(tool):
    """Skip this rename and every path-matching guard passes everything."""
    payload = guard.to_hook_payload(tool, {"path": "/a/b.txt"}, False)
    assert payload["tool_input"]["file_path"] == "/a/b.txt"
    assert "path" not in payload["tool_input"]


def test_unmapped_tool_keeps_its_own_name():
    payload = guard.to_hook_payload("mystery", {"x": 1}, True)
    assert payload["tool_name"] == "mystery"


# --------------------------------------------------------------------------
# mutating classification


def test_mutating_flag_is_read_and_unknown_tools_are_careful():
    assert tools.is_mutating("bash") is True
    assert tools.is_mutating("write") is True
    assert tools.is_mutating("edit") is True
    assert tools.is_mutating("read") is False
    assert tools.is_mutating("grep") is False
    assert tools.is_mutating("never-registered") is True


def test_read_only_guard_refuses_only_mutating_tools():
    g = guard.ReadOnlyGuard()
    assert g.check("read", {"path": "/a"}, False).allowed
    denied = g.check("write", {"path": "/a", "content": ""}, True)
    assert not denied.allowed
    assert "read-only" in denied.reason


# --------------------------------------------------------------------------
# subprocess contract


def _script(tmp_path, body: str) -> str:
    path = tmp_path / "g.sh"
    path.write_text("#!/bin/bash\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return f"bash {path}"


def test_exit_2_blocks_and_stderr_becomes_the_reason(tmp_path):
    cmd = _script(tmp_path, """
        echo "BLOCKED: nope" >&2
        exit 2
    """)
    v = guard.build(commands=[cmd]).check("bash", {"command": "x"}, True)
    assert not v.allowed
    assert v.reason == "BLOCKED: nope"
    assert v.source == cmd


def test_exit_0_allows(tmp_path):
    cmd = _script(tmp_path, "exit 0")
    assert guard.build(commands=[cmd]).check("bash", {"command": "x"}, True).allowed


def test_other_exit_codes_block_loudly(tmp_path):
    """A broken enforcement guard cannot grant permission."""
    cmd = _script(tmp_path, """
        echo "boom" >&2
        exit 1
    """)
    chain = guard.build(commands=[cmd])
    assert not chain.check("bash", {"command": "x"}, True).allowed
    sub = chain.guards[0]
    assert any("exit 1" in a for a in sub.advisories)


def test_a_hanging_guard_fails_closed_loudly(tmp_path):
    cmd = _script(tmp_path, "sleep 30")
    chain = guard.build(commands=[cmd], timeout=0.4)
    assert not chain.check("bash", {"command": "x"}, True).allowed
    assert any("timed out" in a for a in chain.guards[0].advisories)


def test_a_missing_guard_program_fails_closed_loudly():
    chain = guard.build(commands=["/nonexistent/guard-xyz"], timeout=2.0)
    assert not chain.check("bash", {"command": "x"}, True).allowed
    assert chain.guards[0].advisories


def test_the_guard_receives_the_payload_on_stdin(tmp_path):
    out = tmp_path / "seen.json"
    cmd = _script(tmp_path, f"""
        cat > {out}
        exit 0
    """)
    guard.build(commands=[cmd]).check("write", {"path": "/a/b", "content": "c"}, True)
    import json
    seen = json.loads(out.read_text())
    assert seen["tool_name"] == "Write"
    assert seen["tool_input"]["file_path"] == "/a/b"


def test_first_denial_wins_and_later_guards_do_not_run(tmp_path):
    marker = tmp_path / "ran"
    deny = _script(tmp_path, "exit 2")
    later = tmp_path / "later.sh"
    later.write_text(f"#!/bin/bash\ntouch {marker}\nexit 0\n")
    later.chmod(0o755)
    chain = guard.build(commands=[deny, f"bash {later}"])
    assert not chain.check("bash", {"command": "x"}, True).allowed
    assert not marker.exists()


# --------------------------------------------------------------------------
# the loop actually honours it


class _Provider:
    """Asks for one tool call, then answers."""

    def __init__(self, name="bash", arguments=None):
        self.name = name
        self.arguments = arguments or {"command": "echo hi"}
        self.calls = 0

    def chat(self, history, schemas, **kw):
        self.calls += 1
        if self.calls == 1:
            return Turn(text="", tool_calls=[
                _Call("c1", self.name, self.arguments)
            ])
        return Turn(text="done", tool_calls=[])



def test_a_denied_call_never_reaches_the_tool(tmp_path):
    ran = tmp_path / "ran"
    agent = Agent(
        _Provider(arguments={"command": f"touch {ran}"}),
        workspace=str(tmp_path),
        guard=guard.build(read_only=True),
    )
    result = agent.run("go")
    assert not ran.exists(), "the tool ran despite a denial"
    assert result.denied and result.denied[0][0] == "bash"


def test_the_model_is_told_why_it_was_refused(tmp_path):
    agent = Agent(_Provider(), workspace=str(tmp_path),
                  guard=guard.build(read_only=True))
    result = agent.run("go")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("blocked: ")
    assert "read-only" in tool_msgs[0]["content"]


def test_an_allowed_call_still_runs(tmp_path):
    ran = tmp_path / "ran"
    agent = Agent(_Provider(arguments={"command": f"touch {ran}"}),
                  workspace=str(tmp_path), guard=guard.build())
    result = agent.run("go")
    assert ran.exists()
    assert not result.denied


def test_no_guard_at_all_is_the_old_behaviour(tmp_path):
    ran = tmp_path / "ran"
    agent = Agent(_Provider(arguments={"command": f"touch {ran}"}),
                  workspace=str(tmp_path))
    agent.run("go")
    assert ran.exists()


# --------------------------------------------------------------------------
# on_event as a decider


def test_on_event_can_deny_with_a_dict(tmp_path):
    ran = tmp_path / "ran"
    agent = Agent(
        _Provider(arguments={"command": f"touch {ran}"}),
        workspace=str(tmp_path),
        on_event=lambda kind, data: (
            {"deny": "host said no"} if kind == "tool_start" else None
        ),
    )
    result = agent.run("go")
    assert not ran.exists()
    assert result.denied[0][1] == "host said no"


def test_on_event_returning_none_still_only_watches(tmp_path):
    ran = tmp_path / "ran"
    seen = []
    agent = Agent(_Provider(arguments={"command": f"touch {ran}"}),
                  workspace=str(tmp_path),
                  on_event=lambda kind, data: seen.append(kind))
    agent.run("go")
    assert ran.exists()
    assert "tool_start" in seen and "tool_end" in seen


def test_on_event_sees_whether_the_call_mutates(tmp_path):
    seen = {}

    def watch(kind, data):
        if kind == "tool_start":
            seen["mutating"] = data["mutating"]

    Agent(_Provider(), workspace=str(tmp_path), on_event=watch).run("go")
    assert seen["mutating"] is True


def test_as_verdict_reads_only_a_real_denial():
    assert _as_verdict(None) is None
    assert _as_verdict({"anything": "else"}) is None
    assert _as_verdict({"deny": ""}) is None
    v = _as_verdict({"deny": "no"})
    assert v is not None and not v.allowed and v.reason == "no"
    passed = guard.Verdict.deny("x")
    assert _as_verdict(passed) is passed

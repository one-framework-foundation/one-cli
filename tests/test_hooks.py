"""The full hook event set, run against real scripts on disk.

Every test here writes an actual executable and lets a subprocess decide, for
the same reason the guard tests do: the whole claim is that Claude Code's hook
contract is portable, and a hand-written fake would stop checking exactly the
part that has to be true.

Note the `paths=` argument everywhere. The default source list includes the
user's own ~/.claude/settings.json, which may contain hooks - a test that
omitted `paths` would run them.
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from onecli import hooks
from onecli.agent import Agent
from onecli.providers.base import ToolCall
from onecli.providers import Turn


def _script(tmp_path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body).strip() + "\n")
    path.chmod(0o755)
    return str(path)


def _settings(tmp_path, name: str, block: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps({"hooks": block}))
    return str(path)


def _set(tmp_path, *paths, **kw) -> hooks.HookSet:
    return hooks.build(str(tmp_path), paths=list(paths), **kw)


# -- config ---------------------------------------------------------------

def test_a_missing_optional_settings_file_is_not_an_error(tmp_path):
    built = hooks.build(str(tmp_path), paths=[])
    assert built.specs == []
    assert built.errors == []
    assert not built


def test_a_malformed_settings_file_preserves_enforcement(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    good = _settings(tmp_path, "good.json", {"Stop": [
        {"hooks": [{"type": "command", "command": "true"}]}]})

    built = _set(tmp_path, str(bad), good)
    assert len(built.errors) == 1
    assert "bad.json" in built.errors[0]
    # Valid non-enforcement hooks survive, but tool execution is refused.
    assert len(built.for_event(hooks.STOP)) == 1
    assert built.pre_tool_use("bash", {}).blocked


def test_sources_merge_in_order_and_later_files_append(tmp_path):
    first = _settings(tmp_path, "a.json", {"Stop": [
        {"hooks": [{"type": "command", "command": "first"}]}]})
    second = _settings(tmp_path, "b.json", {"Stop": [
        {"hooks": [{"type": "command", "command": "second"}]}]})

    built = _set(tmp_path, first, second)
    assert [s.command for s in built.specs] == ["first", "second"]


def test_trusted_sources_are_user_then_workspace_then_local(tmp_path):
    found = hooks.settings_paths(str(tmp_path), extra=["/tmp/extra.json"], trust_project=True)
    assert found[0] == os.path.join(os.path.expanduser("~"), ".claude",
                                    "settings.json")
    assert found[1] == os.path.join(str(tmp_path), ".claude", "settings.json")
    assert found[2] == os.path.join(str(tmp_path), ".claude",
                                    "settings.local.json")
    assert found[3] == "/tmp/extra.json"


def test_an_unknown_event_is_reported_not_armed(tmp_path):
    path = _settings(tmp_path, "s.json", {"WhenTheMoonIsFull": [
        {"hooks": [{"type": "command", "command": "true"}]}]})
    built = _set(tmp_path, path)
    assert not built.pre_tool_use("bash", {}).blocked
    assert "WhenTheMoonIsFull" in built.errors[0]


def test_timeout_and_matcher_come_off_the_entry(tmp_path):
    path = _settings(tmp_path, "s.json", {"PostToolUse": [
        {"matcher": "Bash", "hooks": [
            {"type": "command", "command": "x", "timeout": 45}]}]})
    spec = _set(tmp_path, path).specs[0]
    assert spec.matcher == "Bash"
    assert spec.timeout == 45.0


def test_the_command_expands_env_and_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/opt/plug")
    path = _settings(tmp_path, "s.json", {"Stop": [
        {"hooks": [{"type": "command",
                    "command": "${CLAUDE_PLUGIN_ROOT}/scripts/cleanup.sh"}]}]})
    assert _set(tmp_path, path).specs[0].command == "/opt/plug/scripts/cleanup.sh"


# -- matcher --------------------------------------------------------------

@pytest.mark.parametrize("matcher,tool,expect", [
    ("Bash", "Bash", True),
    ("Bash", "Read", False),
    ("Write|Edit", "Edit", True),
    ("", "Anything", True),
    ("*", "Anything", True),
    ("(unclosed", "Bash", False),
])
def test_matcher_is_a_regex_over_the_tool_name(matcher, tool, expect):
    spec = hooks.HookSpec(event=hooks.PRE_TOOL_USE, command="true",
                          matcher=matcher)
    assert spec.matches(tool) is expect


def test_matcher_is_ignored_off_the_tool_events(tmp_path):
    path = _settings(tmp_path, "s.json", {"Stop": [
        {"matcher": "Bash", "hooks": [
            {"type": "command", "command": "true"}]}]})
    built = _set(tmp_path, path)
    assert len(built.for_event(hooks.STOP)) == 1


def test_a_matcher_miss_does_not_run_the_hook(tmp_path):
    marker = tmp_path / "ran"
    script = _script(tmp_path, "h.sh", f"touch {marker}\nexit 0")
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"matcher": "Write", "hooks": [
            {"type": "command", "command": script}]}]})
    built = _set(tmp_path, path)

    assert built.pre_tool_use("bash", {"command": "ls"}).ran == 0
    assert not marker.exists()
    assert built.pre_tool_use("write", {"path": "x"}).ran == 1
    assert marker.exists()


# -- per-event semantics --------------------------------------------------

def test_pretooluse_exit_2_blocks_and_stderr_is_the_reason(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "no rm allowed" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"hooks": [{"type": "command", "command": script}]}]})

    outcome = _set(tmp_path, path).pre_tool_use("bash", {"command": "rm -rf /"})
    assert outcome.blocked
    assert outcome.reason == "no rm allowed"


def test_pretooluse_exit_0_allows(tmp_path):
    script = _script(tmp_path, "h.sh", "exit 0")
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"hooks": [{"type": "command", "command": script}]}]})
    assert not _set(tmp_path, path).pre_tool_use("bash", {"command": "ls"}).blocked


def test_posttooluse_exit_2_is_feedback_not_a_block(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "brand palette is wrong" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"PostToolUse": [
        {"hooks": [{"type": "command", "command": script}]}]})

    outcome = _set(tmp_path, path).post_tool_use(
        "bash", {"command": "wkhtmltopdf a.html a.pdf"}, {"output": "done"})
    assert not outcome.blocked
    assert outcome.feedback_text == "brand palette is wrong"


def test_posttooluse_gets_the_tool_response_on_stdin(tmp_path):
    seen = tmp_path / "seen.json"
    script = _script(tmp_path, "h.sh", f"cat > {seen}\nexit 0")
    path = _settings(tmp_path, "s.json", {"PostToolUse": [
        {"hooks": [{"type": "command", "command": script}]}]})

    _set(tmp_path, path).post_tool_use("bash", {"command": "ls"},
                                       {"output": "a.txt"})
    payload = json.loads(seen.read_text())
    assert payload["hook_event_name"] == "PostToolUse"
    assert payload["tool_name"] == "Bash"
    assert payload["tool_input"] == {"command": "ls"}
    assert payload["tool_response"] == {"output": "a.txt"}


def test_userpromptsubmit_exit_0_stdout_becomes_context(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "[carrier] be terse"\nexit 0')
    path = _settings(tmp_path, "s.json", {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": script}]}]})

    outcome = _set(tmp_path, path).user_prompt_submit("fix the build")
    assert not outcome.blocked
    assert outcome.context_text == "[carrier] be terse"


def test_userpromptsubmit_exit_2_blocks_the_prompt(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "secret in prompt" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": script}]}]})

    outcome = _set(tmp_path, path).user_prompt_submit("here is my api key")
    assert outcome.blocked
    assert outcome.reason == "secret in prompt"


def test_the_prompt_reaches_the_hook_on_stdin(tmp_path):
    seen = tmp_path / "seen.json"
    script = _script(tmp_path, "h.sh", f"cat > {seen}\nexit 0")
    path = _settings(tmp_path, "s.json", {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": script}]}]})

    _set(tmp_path, path, session_id="sess-1").user_prompt_submit("do the thing")
    payload = json.loads(seen.read_text())
    assert payload["prompt"] == "do the thing"
    assert payload["session_id"] == "sess-1"
    assert payload["cwd"] == str(tmp_path)


@pytest.mark.parametrize("event,call", [
    (hooks.SESSION_START, "session_start"),
    (hooks.SESSION_END, "session_end"),
    (hooks.PRE_COMPACT, "pre_compact"),
])
def test_informational_events_cannot_refuse_anything(tmp_path, event, call):
    script = _script(tmp_path, "h.sh", 'echo "I object" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {event: [
        {"hooks": [{"type": "command", "command": script}]}]})

    outcome = getattr(_set(tmp_path, path), call)()
    assert not outcome.blocked
    assert outcome.advisories and "I object" in outcome.advisories[0]


def test_sessionstart_stdout_is_injected_as_context(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "branch: parity/hooks"\nexit 0')
    path = _settings(tmp_path, "s.json", {"SessionStart": [
        {"hooks": [{"type": "command", "command": script}]}]})
    assert _set(tmp_path, path).session_start().context_text == "branch: parity/hooks"


def test_stop_exit_2_refuses_the_stop(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "branch drift, open a PR" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"Stop": [
        {"hooks": [{"type": "command", "command": script}]}]})

    built = _set(tmp_path, path)
    outcome = built.stop()
    assert outcome.blocked
    assert outcome.reason == "branch drift, open a PR"
    assert built.stop_refusals == 1


def test_subagentstop_is_its_own_event(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "not done" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"SubagentStop": [
        {"hooks": [{"type": "command", "command": script}]}]})

    built = _set(tmp_path, path)
    assert not built.stop().blocked          # no Stop hooks registered
    assert built.stop(subagent=True).blocked


def test_a_hook_that_never_relents_hits_the_refusal_cap(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "still not done" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"Stop": [
        {"hooks": [{"type": "command", "command": script}]}]})
    built = _set(tmp_path, path)

    for _ in range(hooks.STOP_REFUSAL_CAP):
        assert built.stop().blocked

    capped = built.stop()
    assert not capped.blocked
    assert capped.ran == 0
    assert "cap" in capped.advisories[0]


def test_stop_hook_active_is_true_after_the_first_refusal(tmp_path):
    seen = tmp_path / "seen.json"
    script = _script(tmp_path, "h.sh", f"cat > {seen}\nexit 2")
    path = _settings(tmp_path, "s.json", {"Stop": [
        {"hooks": [{"type": "command", "command": script}]}]})
    built = _set(tmp_path, path)

    built.stop()
    assert json.loads(seen.read_text())["stop_hook_active"] is False
    built.stop()
    assert json.loads(seen.read_text())["stop_hook_active"] is True


# -- failure modes --------------------------------------------------------

def test_a_hanging_hook_fails_closed_loudly(tmp_path):
    script = _script(tmp_path, "h.sh", "sleep 5\nexit 2")
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"hooks": [{"type": "command", "command": script, "timeout": 0.3}]}]})

    outcome = _set(tmp_path, path).pre_tool_use("bash", {"command": "ls"})
    assert outcome.blocked
    assert "timed out" in outcome.reason


def test_a_missing_hook_program_fails_closed_loudly(tmp_path):
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"hooks": [{"type": "command",
                    "command": "/nonexistent/hook-abc123"}]}]})
    outcome = _set(tmp_path, path).pre_tool_use("bash", {"command": "ls"})
    assert outcome.blocked
    assert "hook-abc123" in outcome.reason


def test_other_exit_codes_block_enforcement(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "broke" >&2\nexit 7')
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"hooks": [{"type": "command", "command": script}]}]})

    outcome = _set(tmp_path, path).pre_tool_use("bash", {"command": "ls"})
    assert outcome.blocked
    assert "exit 7" in outcome.reason


def test_the_first_refusal_wins_and_later_hooks_do_not_run(tmp_path):
    marker = tmp_path / "second-ran"
    first = _script(tmp_path, "a.sh", 'echo "nope" >&2\nexit 2')
    second = _script(tmp_path, "b.sh", f"touch {marker}\nexit 0")
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"hooks": [{"type": "command", "command": first},
                   {"type": "command", "command": second}]}]})

    assert _set(tmp_path, path).pre_tool_use("bash", {"command": "ls"}).blocked
    assert not marker.exists()


# -- the loop -------------------------------------------------------------

class _Provider:
    """Answers with one bash call, then plain text."""

    def __init__(self, calls=1):
        self.calls = calls
        self.seen: list[list[dict]] = []

    def chat(self, history, schemas, **kw):
        self.seen.append(list(history))
        if self.calls > 0:
            self.calls -= 1
            return Turn(text="", tool_calls=[_Call()])
        return Turn(text="done", tool_calls=[])


def _Call():
    # The real ToolCall rather than a stand-in with the same three attributes,
    # so a field added to it cannot drift out of these tests unnoticed.
    return ToolCall(id="call-1", name="bash",
                    arguments={"command": "echo hi"})


def test_the_loop_blocks_a_call_a_pretooluse_hook_refuses(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "no shell today" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"PreToolUse": [
        {"hooks": [{"type": "command", "command": script}]}]})

    agent = Agent(provider=_Provider(), workspace=str(tmp_path),
                  hooks=_set(tmp_path, path))
    result = agent.run("go")
    assert result.denied == [("bash", "no shell today")]


def test_the_loop_appends_posttooluse_feedback_to_the_result(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "check the palette" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"PostToolUse": [
        {"hooks": [{"type": "command", "command": script}]}]})

    agent = Agent(provider=_Provider(), workspace=str(tmp_path),
                  hooks=_set(tmp_path, path))
    result = agent.run("go")
    tool_output = result.steps[0].tool_results[0][1]
    assert "[PostToolUse] check the palette" in tool_output


def test_a_stop_refusal_keeps_the_loop_going_then_gives_up(tmp_path):
    script = _script(tmp_path, "h.sh", 'echo "land the branch first" >&2\nexit 2')
    path = _settings(tmp_path, "s.json", {"Stop": [
        {"hooks": [{"type": "command", "command": script}]}]})

    provider = _Provider(calls=0)
    agent = Agent(provider=provider, workspace=str(tmp_path),
                  hooks=_set(tmp_path, path), max_steps=10)
    result = agent.run("go")

    # Refused STOP_REFUSAL_CAP times, then the cap let it answer.
    assert result.stopped == "answered"
    assert len(result.steps) == hooks.STOP_REFUSAL_CAP + 1
    assert result.messages[-2]["content"] == "land the branch first"


def test_no_hooks_means_the_loop_is_unchanged(tmp_path):
    agent = Agent(provider=_Provider(), workspace=str(tmp_path), hooks=None)
    assert agent.run("go").text == "done"


# --------------------------------------------------------------------------
# inherited-envelope regression - a hook's JSON is an envelope, not context
# --------------------------------------------------------------------------
#
# Every UserPromptSubmit hook's raw stdout must not be glued in front of the
# prompt: a stack of `{}` and `{"continue":true}` says nothing, yet it would be
# billed to the backend on every run. This junk comes from one-cli's own hook
# layer, not from stdin.

def test_a_bare_continue_envelope_contributes_nothing():
    assert hooks._context_from_stdout('{"continue":true}') == ("", None)
    assert hooks._context_from_stdout("{}") == ("", None)
    assert hooks._context_from_stdout(
        '{"continue":true,"suppressOutput":true}') == ("", None)


def test_only_additional_context_survives_an_envelope():
    out = ('{}\n'
           '{"continue":true}\n'
           '{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", '
           '"additionalContext": "the repo is dirty"}}')
    assert hooks._context_from_stdout(out) == ("the repo is dirty", None)


def test_plain_stdout_is_still_context():
    assert hooks._context_from_stdout("branch: main\n2 files staged") == (
        "branch: main\n2 files staged", None)


def test_json_that_is_not_an_envelope_is_treated_as_prose():
    body = '{"files": 3, "dirty": true}'
    assert hooks._context_from_stdout(body) == (body, None)


def test_continue_false_refuses_the_prompt():
    context, refusal = hooks._context_from_stdout(
        '{"continue": false, "stopReason": "repo is mid-rebase"}')
    assert context == "" and refusal == "repo is mid-rebase"


def test_decision_block_refuses_the_prompt():
    _, refusal = hooks._context_from_stdout(
        '{"decision": "block", "reason": "no"}')
    assert refusal == "no"


def test_a_hook_that_says_nothing_leaves_the_prompt_alone(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command",
                    "command": "printf '{\"continue\":true}\\n'"}]}]}}))
    hs = hooks.build(str(tmp_path), paths=[str(settings)])
    outcome = hs.user_prompt_submit("count the files")
    assert outcome.ran == 1
    assert outcome.context_text == ""
    assert not outcome.blocked


def test_a_hook_that_says_something_still_reaches_the_prompt(tmp_path):
    payload = ('{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", '
               '"additionalContext": "you are on host-a"}}')
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command",
                    "command": f"printf '%s' '{payload}'"}]}]}}))
    hs = hooks.build(str(tmp_path), paths=[str(settings)])
    assert hs.user_prompt_submit("hi").context_text == "you are on host-a"


def test_settings_paths_reads_each_file_once(tmp_path, monkeypatch):
    """Workspace == home must not read the user settings file twice.

    On a host whose agents run in `~`, the workspace entry resolves to the
    user's own `~/.claude/settings.json`. Reading it under both names made
    every hook in it fire TWICE per event, so submitted context landed in the
    prompt twice and each guard ran two times.
    """
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text('{"hooks": {}}')
    monkeypatch.setenv("HOME", str(home))

    paths = hooks.settings_paths(str(home))

    assert len(paths) == len(set(map(os.path.realpath, paths))), paths


def test_settings_paths_keeps_distinct_sources(tmp_path, monkeypatch):
    """Deduping must not swallow a genuinely different workspace file."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text('{"hooks": {}}')
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "work"
    (work / ".claude").mkdir(parents=True)
    (work / ".claude" / "settings.json").write_text('{"hooks": {}}')

    paths = hooks.settings_paths(str(work), trust_project=True)

    assert str(home / ".claude" / "settings.json") in paths
    assert str(work / ".claude" / "settings.json") in paths


@pytest.mark.parametrize("config", [
    "{", "[]", '{"hooks": []}',
    '{"hooks": {"PreToolUse": {}}}',
    '{"hooks": {"PreToolUse": [null]}}',
    '{"hooks": {"PreToolUse": [{"hooks": {}}]}}',
    '{"hooks": {"PreToolUse": [{"hooks": [null]}]}}',
    '{"hooks": {"PreToolUse": [{"hooks": [{"type": "prompt"}]}]}}',
    '{"hooks": {"PreToolUse": [{"hooks": [{"command": ""}]}]}}',
    '{"hooks": {"PreToolUse": [{"matcher": "[", "hooks": []}]}}',
    '{"hooks": {"PreToolUse": [{"matcher": 4, "hooks": []}]}}',
    '{"hooks": {"PreToolUse": [{"hooks": [{"command": "true", "timeout": "bad"}]}]}}',
    '{"hooks": {"PreToolUse": [{"hooks": [{"command": "true", "timeout": -1}]}]}}',
])
def test_invalid_policy_blocks_startup_and_dispatch(tmp_path, config, capsys, monkeypatch):
    from types import SimpleNamespace
    from onecli.cli import _hooks
    path = tmp_path / "policy.json"
    path.write_text(config)
    built = _hooks(SimpleNamespace(no_hooks=False, workspace=str(tmp_path),
                                   hooks=[str(path)]), "test")
    assert built.errors
    assert "policy.json" in capsys.readouterr().err
    target = tmp_path / "must-not-exist"
    monkeypatch.setattr(__import__(__name__, fromlist=["_Call"]), "_Call",
                        lambda: ToolCall(id="write", name="bash",
                                         arguments={"command": f"touch {target}"}))
    agent = Agent(provider=_Provider(), workspace=str(tmp_path), hooks=built)
    result = agent.run("go")
    assert not target.exists()
    assert result.denied
    assert "policy.json" in result.denied[0][1]


def test_explicit_missing_or_unreadable_policy_blocks(tmp_path):
    for path in (tmp_path / "missing.json", tmp_path):
        built = hooks.build(str(tmp_path), paths=[str(path)])
        assert built.pre_tool_use("bash", {}).blocked


def test_startup_failure_denies_enforcement(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("startup failed")
    monkeypatch.setattr(hooks, "run_shell", fail)
    built = hooks.HookSet([hooks.HookSpec(hooks.PRE_TOOL_USE, "policy-command")])
    out = built.pre_tool_use("bash", {})
    assert out.blocked
    assert "policy-command" in out.reason and "startup failed" in out.reason


@pytest.mark.parametrize("command", ["exit 7", "sleep 1"])
def test_non_enforcement_failures_remain_advisory(tmp_path, command):
    built = hooks.HookSet([hooks.HookSpec(hooks.POST_TOOL_USE, command, timeout=0.02)])
    out = built.post_tool_use("bash", {}, "")
    assert not out.blocked and out.advisories

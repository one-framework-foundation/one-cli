"""Release audit (the earlier implementation): six ways a permission was bypassed.

Each test drives the real path the audit reproduced - the chat turn, the rule
matcher, the hook envelope parser, the skill tool, the default guard chain -
and asserts the call was refused, not that a helper returned a flag.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from onecli import chat as chatlib
from onecli import codemode, config, guard, hooks, permissions, presence, session, skills
from onecli import tools as toolkit
from onecli.providers.base import ToolCall, Turn


class Wire:
    """A scripted provider: one tool call, then an answer."""

    def __init__(self, *turns):
        self.turns = list(turns)

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096, on_delta=None):
        return self.turns.pop(0) if self.turns else Turn(text="done", usage={})


def _call(name, arguments):
    return Turn(text="", tool_calls=[ToolCall(id="c1", name=name, arguments=arguments)],
                usage={"prompt_tokens": 1, "completion_tokens": 1})


def _backend():
    return config.Backend(name="fake", model="fake-1",
                          base_url="http://127.0.0.1:1/v1", wire="openai")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(presence, "refusal", lambda b: "")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return ws


def _tool_results(chat):
    return [m["content"] for m in chat.messages if m["role"] == "tool"]


# --- 1. the chat: approval independent of echo, and the permission flags ---

def test_a_json_chat_turn_does_not_run_bash_unapproved(isolated):
    chat = chatlib.Chat(_backend(), out=io.StringIO(), err=io.StringIO(), stream=False)
    chat._wire = Wire(_call("bash", {"command": "touch PWNED"}))
    chatlib.turn(chat, "go", echo=False)
    assert not (isolated / "PWNED").exists()
    assert "nobody to ask" in _tool_results(chat)[0]


def _cli_args(ws, *extra):
    from onecli.cli import build_parser
    args = build_parser().parse_args(["--chat", "--json", "-w", str(ws), *extra, "go"])
    args.tools = "all"  # the surface that would otherwise run anything unattended
    return args


def _run_chat(monkeypatch, args, *turns):
    seen = {}
    real = chatlib.Chat.__init__

    def init(self, *a, **kw):
        real(self, *a, **kw)
        self._wire = Wire(*turns)
        seen["chat"] = self

    monkeypatch.setattr(config, "resolve", lambda **kw: _backend())
    monkeypatch.setattr(chatlib.Chat, "__init__", init)
    code = chatlib.from_cli(args)
    return code, seen["chat"]


def test_chat_honours_disallowed_tools(isolated, monkeypatch, capsys):
    args = _cli_args(isolated, "--disallowed-tools", "Bash")
    _, chat = _run_chat(monkeypatch, args, _call("bash", {"command": "touch PWNED"}))
    assert not (isolated / "PWNED").exists()
    assert "disallowed" in _tool_results(chat)[0]


def test_chat_honours_read_only(isolated, monkeypatch, capsys):
    args = _cli_args(isolated, "--read-only")
    _, chat = _run_chat(monkeypatch, args,
                        _call("write", {"path": "made.txt", "content": "x"}))
    assert not (isolated / "made.txt").exists()
    assert "write" not in (chatlib.chat_surface(chat) or ["write"])


def test_chat_honours_guard_programs(isolated, monkeypatch, capsys, tmp_path):
    deny = tmp_path / "deny.sh"
    deny.write_text("#!/bin/sh\necho no shell here >&2\nexit 2\n")
    deny.chmod(0o755)
    args = _cli_args(isolated, "--guard", str(deny))
    _, chat = _run_chat(monkeypatch, args, _call("bash", {"command": "touch PWNED"}))
    assert not (isolated / "PWNED").exists()
    assert "no shell here" in _tool_results(chat)[0]


def test_chat_honours_workspace_and_add_dir(isolated, monkeypatch, capsys, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "w.txt").write_text("from -w")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "o.txt").write_text("outside text")
    # -w is the workspace even though the process sits in another directory.
    _, chat = _run_chat(monkeypatch, _cli_args(elsewhere),
                        _call("read", {"path": "w.txt"}))
    assert "from -w" in _tool_results(chat)[0]
    # A path outside it is refused unless --add-dir names it.
    _, chat = _run_chat(monkeypatch, _cli_args(elsewhere),
                        _call("read", {"path": str(outside / "o.txt")}))
    assert "--add-dir" in _tool_results(chat)[0]
    _, chat = _run_chat(monkeypatch, _cli_args(elsewhere, "--add-dir", str(outside)),
                        _call("read", {"path": str(outside / "o.txt")}))
    assert "outside text" in _tool_results(chat)[0]


# --- 2. job_start and code are shell for the rules, guards and hooks ------

def test_bash_rules_govern_job_start():
    rules = permissions.PermissionGuard(deny=permissions.parse_rules(["Bash(rm *)"]),
                                        workspace="/ws")
    assert not rules.check("job_start", {"command": "rm -rf /"}, True).allowed
    assert rules.check("job_start", {"command": "ls"}, True).allowed
    bare = permissions.PermissionGuard(deny=permissions.parse_rules(["Bash"]), workspace="/ws")
    assert not bare.check("job_start", {"command": "ls"}, True).allowed
    assert not bare.check("code", {"source": "print(1)"}, True).allowed


def test_guards_and_hooks_see_job_start_as_bash(tmp_path):
    assert guard.to_hook_payload("job_start", {"command": "ls"}, True)["tool_name"] == "Bash"
    block = tmp_path / "block.sh"
    block.write_text("#!/bin/sh\necho blocked >&2\nexit 2\n")
    block.chmod(0o755)
    hookset = hooks.HookSet(specs=[hooks.HookSpec(event="PreToolUse", command=str(block),
                                                  matcher="Bash")],
                            workspace=str(tmp_path))
    assert hookset.pre_tool_use("job_start", {"command": "ls"}, True).blocked


def test_code_is_confined_when_only_a_guard_or_hooks_decide():
    assert codemode.confinement_reason([], [guard.SubprocessGuard(["true"])])
    hookset = hooks.HookSet(specs=[hooks.HookSpec(event="PreToolUse", command="true",
                                                  matcher="Bash")])
    assert codemode.confinement_reason([], [hookset])
    assert codemode.confinement_reason(permissions.build("/ws"), []) == ""


def test_the_agent_hands_its_guards_and_hooks_to_code_mode(tmp_path, monkeypatch):
    from onecli.agent import Agent
    seen = {}

    def fake_code(workspace, source, timeout=30, call_tool=None, path_guards=None,
                  policy=None):
        seen["reason"] = codemode.confinement_reason(path_guards, policy)
        return "ran"

    monkeypatch.setitem(toolkit.REGISTRY, "code",
                        toolkit.REGISTRY["code"].__class__(
                            **{**toolkit.REGISTRY["code"].__dict__, "fn": fake_code}))
    hookset = hooks.HookSet(specs=[hooks.HookSpec(event="PreToolUse", command="true",
                                                  matcher="Bash")],
                            workspace=str(tmp_path))
    Agent(Wire(_call("code", {"source": "print(1)"})), str(tmp_path),
          hooks=hookset).run("go")
    assert seen["reason"] == "PreToolUse hooks are active"


# --- 3. hook envelopes: pretty-printed JSON and "ask" ----------------------

def test_a_pretty_printed_deny_envelope_refuses():
    out = json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": "deny",
                                             "permissionDecisionReason": "no rm"}},
                     indent=2)
    _, refusal = hooks._context_from_stdout(out)
    assert refusal == "no rm"


def test_a_pretty_printed_deny_from_a_guard_program_refuses(tmp_path):
    script = tmp_path / "denyguard.py"
    script.write_text(
        "import json, sys\nsys.stdin.read()\n"
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', "
        "'permissionDecision': 'deny', 'permissionDecisionReason': 'no rm'}}, indent=2))\n")
    chain = guard.SubprocessGuard([f"{sys.executable} {script}"], workspace=str(tmp_path))
    verdict = chain.check("bash", {"command": "rm -rf x"}, True)
    assert not verdict.allowed and "no rm" in verdict.reason


def test_an_ask_decision_is_not_an_allow():
    out = json.dumps({"hookSpecificOutput": {"permissionDecision": "ask",
                                             "permissionDecisionReason": "check"}})
    _, refusal = hooks._context_from_stdout(out)
    assert refusal is not None and "check" in refusal


# --- 4. Read rules cover the search tools, Edit rules cover write ---------

def test_read_and_edit_rules_name_their_families():
    rules = permissions.PermissionGuard(
        deny=permissions.parse_rules(["Read(.env)", "Edit(src/**)"]), workspace="/ws")
    assert not rules.check("grep", {"path": ".env"}, False).allowed
    assert not rules.check("glob", {"path": ".env"}, False).allowed
    assert not rules.check("write", {"path": "src/a.py"}, True).allowed
    assert rules.check("write", {"path": "docs/a.md"}, True).allowed


def test_a_read_deny_keeps_grep_out_of_the_file(tmp_path):
    (tmp_path / ".env").write_text("API_KEY=sk-secret\n")
    (tmp_path / "app.py").write_text("API_KEY = os.environ['API_KEY']\n")
    chain = guard.GuardChain(permissions.build(str(tmp_path), disallowed=["Read(.env)"]))
    out = toolkit.run("grep", {"pattern": "API_KEY"}, str(tmp_path),
                      path_guards=permissions.path_guards(chain))
    assert "sk-secret" not in out
    assert "app.py" in out


# --- 5. workspace skills: trust, symlinks, path rules ----------------------

def _skill(root, folder, body="---\nname: {name}\ndescription: d\n---\nbody\n"):
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body.format(name=folder))
    return d


def test_workspace_skills_need_trust_project_config(tmp_path):
    ws, home = tmp_path / "ws", tmp_path / "home"
    _skill(ws / ".claude" / "skills", "project-skill")
    _skill(home / ".claude" / "skills", "home-skill")
    assert set(skills.discover(str(ws), str(home))) == {"home-skill"}
    assert set(skills.discover(str(ws), str(home), trust_project=True)) == {
        "project-skill", "home-skill"}


def test_a_workspace_skill_cannot_symlink_out_of_the_workspace(tmp_path, monkeypatch):
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _skill(outside, "leak", "---\nname: leak\n---\nSECRET\n")
    root = ws / ".claude" / "skills"
    root.mkdir(parents=True)
    (root / "leak").symlink_to(outside / "leak")
    (root / "file").mkdir()
    (root / "file" / "SKILL.md").symlink_to(outside / "leak" / "SKILL.md")
    found = skills.discover(str(ws), trust_project=True)
    assert "leak" not in found and "file" not in found
    out = toolkit.run("skill", {"name": "leak"}, str(ws), trust_project=True)
    assert "SECRET" not in out


def test_a_skill_file_passes_the_path_rules(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _skill(ws / ".claude" / "skills", "guarded")
    chain = guard.GuardChain(permissions.build(str(ws), disallowed=["Read(.claude/**)"]))
    out = toolkit.run("skill", {"name": "guarded"}, str(ws), trust_project=True,
                      path_guards=permissions.path_guards(chain))
    assert out.startswith("error:") and "disallowed" in out


def test_home_skills_still_load_under_the_workspace_boundary(tmp_path, monkeypatch):
    ws, home = tmp_path / "ws", tmp_path / "home"
    ws.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _skill(home / ".claude" / "skills", "shared")
    out = toolkit.run("skill", {"name": "shared"}, str(ws),
                      path_guards=permissions.build(str(ws)))
    assert "body" in out and not out.startswith("error:")


# --- 6. file tools are confined to the workspace by default ----------------

@pytest.mark.parametrize("mode", permissions.MODES)
def test_every_mode_confines_file_tools_to_the_workspace(tmp_path, mode):
    secret = tmp_path / "aws" / "credentials"
    secret.parent.mkdir()
    secret.write_text("aws_secret_access_key=x")
    ws = tmp_path / "ws"
    ws.mkdir()
    chain = guard.GuardChain(permissions.build(str(ws), mode))
    for tool in ("read", "grep", "glob", "ls"):
        verdict = chain.check(tool, {"path": str(secret)}, False)
        assert not verdict.allowed and "--add-dir" in verdict.reason, (mode, tool)
    home = chain.check("read", {"path": "~/.aws/credentials"}, False)
    assert not home.allowed
    assert chain.check("read", {"path": "notes.txt"}, False).allowed
    added = guard.GuardChain(permissions.build(str(ws), mode, add_dirs=[str(secret.parent)]))
    assert added.check("read", {"path": str(secret)}, False).allowed


def test_a_read_only_run_cannot_read_outside_the_workspace(tmp_path):
    from onecli.agent import Agent
    secret = tmp_path / "creds"
    secret.write_text("TOPSECRET")
    ws = tmp_path / "ws"
    ws.mkdir()
    agent = Agent(Wire(_call("read", {"path": str(secret)})), str(ws),
                  guard=guard.GuardChain(permissions.build(str(ws), "readOnly")),
                  tool_names=permissions.tool_names_for("readOnly"))
    result = agent.run("read it")
    tool = [m for m in result.messages if m.get("role") == "tool"][0]["content"]
    assert "TOPSECRET" not in tool and "--add-dir" in tool


def test_the_default_chat_confines_file_tools(isolated, tmp_path):
    secret = tmp_path / "creds"
    secret.write_text("TOPSECRET")
    chat = chatlib.Chat(_backend(), out=io.StringIO(), err=io.StringIO(), stream=False)
    chat._wire = Wire(_call("read", {"path": str(secret)}))
    chatlib.turn(chat, "go", echo=False)
    assert "TOPSECRET" not in _tool_results(chat)[0]



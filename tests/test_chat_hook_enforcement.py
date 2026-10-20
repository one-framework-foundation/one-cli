"""Chat hook enforcement and chat terminal control.

Both drive the real chat turn against a scripted wire: what is asserted is the
file a refused tool did not create and the bytes that reached the terminal.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from onecli import ask as asklib
from onecli import chat as chatlib
from onecli import config, presence, session
from onecli.providers.base import ToolCall, Turn

# OSC 52 clipboard write, OSC 0 title rewrite, CSI 6n cursor-report request.
HOSTILE = "a\x1b]52;c;cHduZWQ=\x07b\x1b]0;owned\x07c\x1b[6nd"


class Wire:
    def __init__(self, *turns, deltas=()):
        self.turns = list(turns)
        self.deltas = list(deltas)
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096, on_delta=None):
        self.calls += 1
        turn = self.turns.pop(0) if self.turns else Turn(text="done", usage={})
        if on_delta is not None:
            for kind, chunk in self.deltas:
                on_delta(kind, chunk)
        return turn


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
    monkeypatch.setattr(config, "resolve", lambda **kw: _backend())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    # No operator ~/.claude/settings.json reaches these tests.
    home = tmp_path / "operator"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return ws


def _capture(monkeypatch, *turns):
    seen = {}
    real = chatlib.Chat.__init__

    def init(self, *a, **kw):
        real(self, *a, **kw)
        self._wire = Wire(*turns)
        seen["chat"] = self

    monkeypatch.setattr(chatlib.Chat, "__init__", init)
    return seen


def _cli_chat(ws, *extra):
    from onecli.cli import build_parser
    args = build_parser().parse_args(["--chat", "--json", "-w", str(ws), *extra, "go"])
    args.tools = "all"
    return args


def _settings(path, event, command, matcher="Bash"):
    path.write_text(json.dumps({"hooks": {event: [
        {"matcher": matcher, "hooks": [{"type": "command", "command": command}]}]}}))
    return path


def _results(chat):
    return [m["content"] for m in chat.messages if m["role"] == "tool"]


# --- chat hooks --------------------------------------------------------------

def test_chat_enforces_a_pretooluse_hook(isolated, monkeypatch, tmp_path, capsys):
    hook = _settings(tmp_path / "hooks.json", "PreToolUse",
                     "echo no hook shell >&2; exit 2")
    seen = _capture(monkeypatch, _call("bash", {"command": "touch PWNED"}))
    chatlib.from_cli(_cli_chat(isolated, "--hooks", str(hook)))
    assert not (isolated / "PWNED").exists()
    assert "no hook shell" in _results(seen["chat"])[0]


def test_chat_fails_closed_on_a_broken_hook_config(isolated, monkeypatch, tmp_path, capsys):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    _capture(monkeypatch, _call("bash", {"command": "touch PWNED"}))
    chatlib.from_cli(_cli_chat(isolated, "--hooks", str(broken)))
    assert not (isolated / "PWNED").exists()
    assert "hook configuration error" in capsys.readouterr().err


def test_one_chat_enforces_user_settings_hooks(isolated, monkeypatch, tmp_path, capsys):
    import os
    claude = tmp_path / "operator" / ".claude"
    claude.mkdir()
    _settings(claude / "settings.json", "PreToolUse", "echo user hook says no >&2; exit 2")
    seen = _capture(monkeypatch, _call("bash", {"command": "touch PWNED"}))
    chatlib.main(["--tools", "all", "--json", "go"])
    assert not (isolated / "PWNED").exists()
    assert os.path.basename(os.getcwd()) == "ws"
    assert "user hook says no" in _results(seen["chat"])[0]


def test_chat_runs_userpromptsubmit_and_honours_a_block(isolated, monkeypatch, tmp_path, capsys):
    hook = _settings(tmp_path / "hooks.json", "UserPromptSubmit",
                     "echo not that prompt >&2; exit 2", matcher="")
    seen = _capture(monkeypatch, Turn(text="answered", usage={}))
    code = chatlib.from_cli(_cli_chat(isolated, "--hooks", str(hook)))
    assert code == 2
    assert seen["chat"]._wire.calls == 0
    out = json.loads(capsys.readouterr().out)
    assert out["stopped"] == "blocked" and "not that prompt" in out["error"]


def test_chat_code_mode_is_confined_by_hooks(isolated, monkeypatch, tmp_path, capsys):
    """codemode keys confinement on the hook set the Agent was given."""
    from onecli import agent as agentlib
    from onecli import codemode
    hook = _settings(tmp_path / "hooks.json", "PreToolUse", "exit 0")
    built = []
    real = agentlib.Agent.__init__

    def init(self, *a, **kw):
        real(self, *a, **kw)
        built.append(self)

    monkeypatch.setattr(agentlib.Agent, "__init__", init)
    _capture(monkeypatch, Turn(text="ok", usage={}))
    chatlib.from_cli(_cli_chat(isolated, "--hooks", str(hook)))
    assert built and codemode.confinement_reason(
        [], [built[0].hooks]) == "PreToolUse hooks are active"


# --- terminal control --------------------------------------------------------

def _clean(text):
    assert "\x1b" not in text and "\x07" not in text
    return text


def test_chat_answer_strips_terminal_control(isolated, capsys):
    out = io.StringIO()
    chat = chatlib.Chat(_backend(), out=out, err=io.StringIO(), stream=False,
                        tools="off")
    chat._wire = Wire(Turn(text=HOSTILE, usage={}))
    chatlib.turn(chat, "hi")
    assert "abcd" in _clean(out.getvalue())


def test_chat_streamed_answer_and_thinking_strip_terminal_control(isolated):
    out = io.StringIO()
    chat = chatlib.Chat(_backend(), out=out, err=io.StringIO(), stream=True,
                        thinking=True, tools="off")
    chat._wire = Wire(Turn(text=HOSTILE, usage={}),
                      deltas=[("thinking", HOSTILE), ("text", HOSTILE)])
    chatlib.turn(chat, "hi")
    assert _clean(out.getvalue()).count("abcd") == 2


def test_agent_chat_answer_strips_terminal_control(isolated):
    out = io.StringIO()
    chat = chatlib.Chat(_backend(), out=out, err=io.StringIO(), stream=False)
    chat._wire = Wire(Turn(text=HOSTILE, usage={}))
    chatlib.turn(chat, "hi")
    assert "abcd" in _clean(out.getvalue())


def test_agent_chat_streamed_answer_strips_terminal_control(isolated):
    out = io.StringIO()
    chat = chatlib.Chat(_backend(), out=out, err=io.StringIO(), stream=True,
                        thinking=True)
    chat._wire = Wire(Turn(text=HOSTILE, usage={}),
                      deltas=[("thinking", HOSTILE), ("text", HOSTILE)])
    chatlib.turn(chat, "hi")
    assert "abcd" in _clean(out.getvalue())


def test_ask_user_prompt_strips_terminal_control():
    out = io.StringIO()
    asker = asklib.Asker(stdin=io.StringIO("1\n"), out=out, timeout=0)
    answer = asker.ask(HOSTILE, [HOSTILE, "plain"])
    shown = _clean(out.getvalue())
    assert "? abcd" in shown and "1) abcd" in shown
    assert answer == f"The person answered: {HOSTILE}"


def test_chat_tool_denied_strips_terminal_control_in_call_name(isolated):
    err = io.StringIO()
    chat = chatlib.Chat(_backend(), out=io.StringIO(), err=err, stream=False)
    chat._wire = Wire(
        _call(HOSTILE, {}),
        Turn(text="done", usage={}),
    )
    chatlib.turn(chat, "hi")
    shown = _clean(err.getvalue())
    assert "abcd blocked:" in shown


def test_chat_tool_denied_strips_terminal_control_in_reason(isolated):
    err = io.StringIO()
    chat = chatlib.Chat(_backend(), out=io.StringIO(), err=err, stream=False,
                        workspace=str(isolated))
    chat._wire = Wire(
        _call("read", {"path": f"/outside/{HOSTILE}"}),
        Turn(text="done", usage={}),
    )
    chatlib.turn(chat, "hi")
    shown = _clean(err.getvalue())
    assert "read blocked:" in shown and "/outside/abcd" in shown


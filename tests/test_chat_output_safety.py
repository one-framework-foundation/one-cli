"""Every chat output path is terminal-safe, and an
unattended chat runs or refuses a mutating call exactly as docs/permissions.md
says.

The terminal test drives each path that can print model, tool or hook text -
through the real turn where one exists - and asserts on the bytes that reached
stdout and stderr.
"""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from onecli import ask as asklib
from onecli import chat as chatlib
from onecli import config, presence, session
from onecli.providers.base import ProviderError, ToolCall, Turn

# OSC 52 clipboard write, OSC 0 title rewrite, CSI 6n cursor-report request.
HOSTILE = "a\x1b]52;c;cHduZWQ=\x07b\x1b]0;owned\x07c\x1b[6nd"
# The same bytes from a shell hook.
HOSTILE_SH = r"printf 'a\033]52;c;cHduZWQ=\007b\033]0;owned\007c\033[6nd' >&2"


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


class Raising:
    def __init__(self, exc):
        self.exc = exc

    def chat(self, *a, **kw):
        raise self.exc


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
    home = tmp_path / "operator"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return ws


def _wired(monkeypatch, *turns):
    """Every Chat built from here on talks to a scripted wire."""
    real = chatlib.Chat.__init__

    def init(self, *a, **kw):
        real(self, *a, **kw)
        self._wire = Wire(*turns)

    monkeypatch.setattr(chatlib.Chat, "__init__", init)


def _hooks(path, event, command, matcher=""):
    path.write_text(json.dumps({"hooks": {event: [
        {"matcher": matcher, "hooks": [{"type": "command", "command": command}]}]}}))
    return str(path)


def _cli_chat(ws, *extra):
    from onecli.cli import build_parser
    args = build_parser().parse_args(["--chat", "-w", str(ws), *extra, "go"])
    args.tools = "all"
    return args


def test_every_chat_output_path_strips_terminal_control(isolated, monkeypatch, tmp_path, capsys):
    out, err = io.StringIO(), io.StringIO()

    def chat(**kw):
        return chatlib.Chat(_backend(), out=out, err=err, **kw)

    # Answer, unstreamed and with no tools.
    c = chat(stream=False, tools="off")
    c._wire = Wire(Turn(text=HOSTILE, usage={}))
    chatlib.turn(c, "hi")
    # Streamed thinking and answer.
    c = chat(stream=True, thinking=True, tools="off")
    c._wire = Wire(Turn(text=HOSTILE, usage={}),
                   deltas=[("thinking", HOSTILE), ("text", HOSTILE)])
    chatlib.turn(c, "hi")
    # The tool loop's final answer.
    c = chat(stream=False)
    c._wire = Wire(Turn(text=HOSTILE, usage={}))
    chatlib.turn(c, "hi")

    # Tool lines, a tool's output, a refused call's name and reason.
    c = chat(stream=False, tools="all")
    trace = chatlib._tracer(c)
    hostile_call = ToolCall(id="c1", name=HOSTILE, arguments={"path": HOSTILE})
    trace("tool_start", {"call": hostile_call, "mutating": False})
    # The work rail is labelled with the tool name; draw one frame of it.
    err.write(trace.state["spin"]._frame() + "\n")
    trace("tool_end", {"output": HOSTILE})
    trace("tool_denied", {"call": hostile_call,
                          "verdict": SimpleNamespace(reason=HOSTILE)})
    trace("fallback", {"from": HOSTILE, "to": HOSTILE})
    trace.settle()

    # A failed turn and a stopped loop: provider and loop error text.
    chatlib._report(c, {"error": HOSTILE})
    chatlib._report(c, {"error": HOSTILE, "blocked": True})
    c._wire = Raising(ProviderError(HOSTILE))
    stat = chatlib.turn(c, "hi")
    assert stat["error"]  # raw data; the print is what must be clean
    chatlib._report(c, stat)

    # /system, /undo error, /history, /compare (text and error), /diff, /checkpoints, /sessions.
    c.system = HOSTILE
    chatlib.dispatch(c, "/system")
    monkeypatch.setattr(c.checkpoints, "undo_turn",
                        lambda _m, _id=None: SimpleNamespace(success=False, error=HOSTILE))
    chatlib.dispatch(c, "/undo")
    chatlib.dispatch(c, "/history")
    monkeypatch.setattr(chatlib, "fan_out", lambda _c, _w, _p: [
        {"name": "fake", "label": "fake", "text": HOSTILE, "seconds": 0.1, "output_tokens": 1},
        {"name": "fake2", "label": "fake2", "error": HOSTILE, "seconds": 0.1},
    ])
    chatlib.dispatch(c, "/compare fake,fake2 hi")
    monkeypatch.setattr(c.checkpoints, "get_diff", lambda _id=None: HOSTILE)
    monkeypatch.setattr(c.checkpoints, "get_diff_stat", lambda _id=None: HOSTILE)
    monkeypatch.setattr(c.checkpoints, "list_checkpoints",
                        lambda: [SimpleNamespace(id=1, prompt=HOSTILE)])
    chatlib.dispatch(c, "/diff")
    chatlib.dispatch(c, "/checkpoints")
    session.append(c.sid, {"kind": "messages",
                           "messages": [{"role": "user", "content": HOSTILE}]})
    chatlib.dispatch(c, "/sessions")
    chatlib.cmd_sessions(SimpleNamespace(json=False))

    # Detached tasks, background jobs, /queue, and !cmd shell output.
    from onecli.tasks import DetachedTask
    task = DetachedTask(task_id="t1", title=HOSTILE, prompt=HOSTILE,
                        phases=[HOSTILE], announcement=HOSTILE, status="running")
    task.output = lambda: HOSTILE  # type: ignore[method-assign]
    c.tasks._tasks["t1"] = task
    err.write(c.tasks.render_card("t1") + "\n")
    chatlib.dispatch(c, "/task log t1")
    c.input_queue.append(HOSTILE)
    store = c.jobs()
    log_path = tmp_path / "j1.log"
    log_path.write_text(HOSTILE, encoding="utf-8")
    from onecli.jobs import Job
    store.jobs["j1"] = Job(id="j1", command=HOSTILE, log=log_path, started=1.0)
    chatlib.dispatch(c, "/queue")
    chatlib.dispatch(c, "/ps")
    chatlib.dispatch(c, "/logs j1")
    task.status = "done"
    for notice in c.tasks.notices():
        chatlib._say(c, chatlib.render.paint("brand", chatlib.render.untrusted(notice)))
    store.jobs["j1"].exit_code = 0
    for notice in store.notices():
        chatlib._say(c, chatlib.render.paint("brand", chatlib.render.untrusted(notice)))
    chatlib.shell(c, "!" + HOSTILE_SH)

    # ask_user: the question and each option.
    asker = asklib.Asker(stdin=io.StringIO("1\n"), out=err, timeout=0)
    asker.ask(HOSTILE, [HOSTILE, "plain"])

    # Hooks, through the real chat: a PreToolUse denial reason, a
    # UserPromptSubmit note, and a UserPromptSubmit block.
    _wired(monkeypatch, _call("bash", {"command": "touch PWNED"}))
    deny = _hooks(tmp_path / "deny.json", "PreToolUse", HOSTILE_SH + "; exit 2", "Bash")
    chatlib.from_cli(_cli_chat(isolated, "--hooks", deny))
    assert not (isolated / "PWNED").exists()
    note = _hooks(tmp_path / "note.json", "UserPromptSubmit", HOSTILE_SH + "; exit 1")
    chatlib.from_cli(_cli_chat(isolated, "--hooks", note))
    block = _hooks(tmp_path / "block.json", "UserPromptSubmit", HOSTILE_SH + "; exit 2")
    assert chatlib.from_cli(_cli_chat(isolated, "--hooks", block)) == 2

    captured = capsys.readouterr()
    shown = out.getvalue() + err.getvalue() + captured.out + captured.err
    for raw in ("\x1b]", "\x1b[6n", "\x07"):
        assert raw not in shown, (raw, shown)
    # The text around the controls still arrives: stripped, not dropped.
    assert shown.count("abcd") >= 25, shown
    assert "blocked: abcd" in shown and "hook: " in shown


def test_bash_approval_still_shows_controls_escaped():
    prompts = []
    c = SimpleNamespace(allow_all=False, tools="all", err=io.StringIO(),
                        ask_line=lambda q: prompts.append(q) or "n")
    chatlib._confirm(c, ToolCall(id="x", name=HOSTILE, arguments={"command": HOSTILE}))
    assert "run abcd\n" in prompts[0]
    assert "\\x1b]52;c;cHduZWQ=\\x07" in prompts[0]
    assert "\x1b]52" not in prompts[0]


@pytest.mark.parametrize("arguments, shown", [
    ({"source": "import os\nos.remove('x')"}, "import os\nos.remove('x')"),
    ({"table": "users", "sql": "DROP TABLE users"}, '"sql": "DROP TABLE users"'),
], ids=["code-source", "mcp-other-names"])
def test_approval_shows_what_it_approves(arguments, shown):
    # A `code` call or an MCP tool with its own argument names used to ask
    # "run code" over a blank line: approving a program nobody could see.
    prompts = []
    c = SimpleNamespace(allow_all=False, tools="all", err=io.StringIO(),
                        ask_line=lambda q: prompts.append(q) or "n")
    chatlib._confirm(c, ToolCall(id="x", name="code", arguments=arguments))
    assert shown in prompts[0]


def test_approval_fallback_escapes_controls():
    prompts = []
    c = SimpleNamespace(allow_all=False, tools="all", err=io.StringIO(),
                        ask_line=lambda q: prompts.append(q) or "n")
    # JSON escapes C0 itself; a C1 control (U+009B, a one-character CSI)
    # survives it and must still be shown escaped.
    chatlib._confirm(c, ToolCall(id="x", name="mcp__s__t",
                                 arguments={"q2": HOSTILE + "\x9b2J"}))
    assert "\x1b" not in prompts[0] and "\x9b" not in prompts[0]
    assert "\\x9b2J" in prompts[0]


# --- unattended mutating calls (docs/permissions.md) ---------------------------

def test_unattended_chat_runs_a_mutating_call_only_under_tools_all(isolated, monkeypatch):
    """`--tools all` is the consent, as in a `one` run; `auto` refuses."""
    _wired(monkeypatch, _call("bash", {"command": "touch RAN"}))
    assert chatlib.main(["--tools", "all", "--json", "go"]) == 0
    assert (isolated / "RAN").exists()
    (isolated / "RAN").unlink()

    _wired(monkeypatch, _call("bash", {"command": "touch RAN"}))
    chatlib.main(["--tools", "auto", "--json", "go"])
    assert not (isolated / "RAN").exists()

    # `one --chat` has no --tools and starts in auto: it refuses.
    from onecli.cli import build_parser
    _wired(monkeypatch, _call("bash", {"command": "touch RAN"}))
    chatlib.from_cli(build_parser().parse_args(["--chat", "--json", "-w", str(isolated), "go"]))
    assert not (isolated / "RAN").exists()

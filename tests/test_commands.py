"""The slash commands, and the line between the session and the model.

Two things are being asserted here. First, that each command answers from live
state rather than from a plausible-looking constant - a `/cost` that reports
the turn instead of the session is worse than no `/cost`, because it is
believed. Second, that a slash command is handled by the REPL and never reaches
the model: no hook fires, no turn is spent, nothing lands in the transcript as
something the user said.
"""

from __future__ import annotations

import types

import pytest

from onecli import cli
from onecli import commands as cmdlib


def _backend(name="fake", model="m"):
    return types.SimpleNamespace(name=name, model=model, wire="openai",
                                 base_url="http://x", display=name,
                                 source="default")


def _say(line, repl):
    """The text a command prints. A command file answers with `.prompt`."""
    return cmdlib.dispatch(line, repl).text


def _repl(**kw):
    args = cli.build_parser().parse_args([])
    args.workspace = kw.pop("workspace", ".")
    base = dict(args=args, backend=_backend(), mode="default", sid="sid0",
                chain=types.SimpleNamespace(
                    rungs=[types.SimpleNamespace(name="fake")]),
                settings={"context_window": 100_000, "autocompact": True,
                          "compact_at": 0.8, "keep_last": 6})
    base.update(kw)
    return cmdlib.Repl(**base)


# --- what counts as a command ---------------------------------------------


@pytest.mark.parametrize("line", ["/help", "/model qwen", "/?", "/exit"])
def test_a_command_is_a_command(line):
    assert cmdlib.is_command(line)


@pytest.mark.parametrize("line", [
    "/home/user/one-cli/README.md",
    "/",
    "read /etc/hosts",
    "/usr/bin/env python",
])
def test_a_path_is_not_a_command(line):
    """A leading slash is also how an absolute path starts. Someone asking the
    model about /etc/hosts is talking to the model, not to the REPL."""
    assert not cmdlib.is_command(line)


def test_an_unknown_command_suggests_the_near_one():
    out = _say("/costs", _repl())
    assert "/cost" in out


def test_an_unknown_command_with_no_neighbour_points_at_help():
    assert "/help" in _say("/zzzz", _repl())


def test_help_lists_every_command():
    out = _say("/help", _repl())
    for cmd in cmdlib.ORDER:
        assert "/" + cmd.name in out, f"{cmd.name} is unreachable from /help"


# --- state each command reads ---------------------------------------------


def test_cost_is_the_session_not_the_turn():
    """The reason this state object exists: a fresh Agent is built per turn, so
    a per-turn meter can never answer "what has this session cost"."""
    repl = _repl()
    for _ in range(3):
        repl.record(types.SimpleNamespace(
            cost={"input_tokens": 1000, "output_tokens": 100, "usd": 0.02},
            context_tokens=4000))
    out = _say("/cost", repl)
    assert "3,000" in out and "300" in out
    assert "$0.0600" in out
    assert "3" in out


def test_an_unpriced_model_reports_no_dollars():
    repl = _repl()
    repl.record(types.SimpleNamespace(
        cost={"input_tokens": 10, "output_tokens": 2, "usd": None},
        context_tokens=12))
    out = _say("/cost", repl)
    assert "$" not in out, "a local model reporting $0.00 is a reassuring lie"
    assert "no price known" in out


def test_context_reports_how_full_the_window_is():
    repl = _repl()
    repl.record(types.SimpleNamespace(cost={}, context_tokens=80_000))
    out = _say("/context", repl)
    assert "80,000" in out and "80%" in out


def test_clear_drops_the_history_and_keeps_the_transcript(monkeypatch):
    written = []
    monkeypatch.setattr(cmdlib.session, "append",
                        lambda sid, rec: written.append(rec))
    repl = _repl(messages=[{"role": "user", "content": "a"},
                           {"role": "assistant", "content": "b"}])
    out = _say("/clear", repl)
    assert repl.messages is None
    assert "2" in out
    assert written == [{"kind": "cleared", "dropped": 2}], (
        "the transcript records the clear rather than losing the record")


def test_model_with_no_argument_reports_and_changes_nothing(monkeypatch):
    monkeypatch.setattr(cmdlib.config, "discover", lambda: [_backend()])
    repl = _repl()
    out = _say("/model", repl)
    assert "Current: fake (m)" in out and "Providers" in out
    assert not repl.rebuild


def test_model_switch_asks_the_loop_to_rebuild(monkeypatch):
    monkeypatch.setattr(cmdlib.config, "resolve",
                        lambda name: _backend("other", "big"))
    repl = _repl()
    out = _say("/model other", repl)
    assert repl.backend.name == "other"
    assert repl.rebuild, "the loop rebuilds the chain, not the command"
    assert "carries over" in out


def test_switching_to_the_model_already_live_is_a_no_op(monkeypatch):
    monkeypatch.setattr(cmdlib.config, "resolve", lambda name: _backend())
    repl = _repl()
    _say("/model fake", repl)
    assert not repl.rebuild


def test_an_unknown_mode_is_refused_with_the_near_one():
    repl = _repl()
    out = _say("/mode plna", repl)
    assert "plan" in out
    assert repl.mode == "default", "a typo must not change the permission mode"


def test_mode_change_asks_for_a_rebuild():
    repl = _repl()
    _say("/mode readOnly", repl)
    assert repl.mode == "readOnly" and repl.rebuild


def test_tools_lists_the_surface_this_run_actually_has():
    repl = _repl(surface=["read", "grep", "bash"])
    out = _say("/tools", repl)
    assert "read" in out and "grep" in out and "bash" in out
    assert "write" not in out, "the surface is what the run has, not the registry"


def test_jobs_says_so_when_there_is_no_store():
    assert "off" in _say("/jobs", _repl())


def test_session_names_how_to_get_back_in():
    out = _say("/session", _repl())
    assert "one --resume sid0" in out


def test_compact_replaces_the_history_and_says_what_it_freed(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    long_history = ([{"role": "system", "content": "s"}]
                    + [{"role": "user", "content": "x" * 400},
                       {"role": "assistant", "content": "y" * 400}] * 8)
    provider = types.SimpleNamespace(
        chat=lambda msgs, schemas, **kw: types.SimpleNamespace(text="a summary"))
    repl = _repl(messages=long_history,
                 chain=types.SimpleNamespace(
                     current=types.SimpleNamespace(provider=provider)))
    out = _say("/compact", repl)
    assert len(repl.messages) < len(long_history)
    assert "a summary" in str(repl.messages)
    assert "freed" in out


def test_compact_never_reports_a_negative_saving_and_keeps_the_messages(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}] * 5
    provider = types.SimpleNamespace(
        chat=lambda msgs, schemas, **kw: types.SimpleNamespace(text="long summary " * 400))
    repl = _repl(messages=list(history),
                 chain=types.SimpleNamespace(
                     current=types.SimpleNamespace(provider=provider)))
    out = _say("/compact", repl)
    assert repl.messages == history
    assert "nothing to compact" in out
    assert "freed" not in out


def test_compact_says_so_rather_than_spending_a_call_on_nothing():
    repl = _repl(messages=[{"role": "user", "content": "hi"}])
    assert "nothing old enough" in _say("/compact", repl)


def test_a_failed_compaction_leaves_the_history_alone(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    history = ([{"role": "system", "content": "s"}]
               + [{"role": "user", "content": "x" * 400}] * 12)

    def broken(msgs, schemas, **kw):
        raise RuntimeError("endpoint down")

    repl = _repl(messages=history,
                 chain=types.SimpleNamespace(current=types.SimpleNamespace(
                     provider=types.SimpleNamespace(chat=broken))))
    out = _say("/compact", repl)
    assert repl.messages == history, (
        "losing the middle of a conversation to a transport blip is the worse "
        "outcome; the operator is told instead")
    assert "endpoint down" in out


# --- the line between the session and the model ---------------------------


class FakeAgent:
    calls: list = []

    def __init__(self, *_, **kw):
        self.chain = types.SimpleNamespace(
            current=types.SimpleNamespace(name="fake"))

    def run(self, prompt, messages, attachments=None):
        FakeAgent.calls.append(prompt)
        return types.SimpleNamespace(text="answered", messages=[], steps=[],
                                     tool_calls=0, cost={}, stopped="answered",
                                     error=None, context_tokens=0)


@pytest.fixture
def loop(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    monkeypatch.setattr(cli, "_context_settings", lambda a, b: {
        "chain": types.SimpleNamespace(
            current=types.SimpleNamespace(name="fake"), warnings=[])})
    monkeypatch.setattr(cli, "_banner", lambda *a, **k: "")
    monkeypatch.setattr(cli, "_hooks", lambda *a, **k: None)
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)
    monkeypatch.setattr(cli.joblib, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cli.spilllib, "SPILL_DIR", tmp_path / "spill")
    monkeypatch.setattr(cli.inputline, "install", lambda *a, **k: False)

    def run(keys, extra=()):
        FakeAgent.calls = []
        pending = list(keys)

        def fake_input(_prompt=""):
            key = pending.pop(0)
            if isinstance(key, BaseException):
                raise key
            return key

        monkeypatch.setattr("builtins.input", fake_input)
        args = cli.build_parser().parse_args(list(extra))
        args.workspace = str(tmp_path)
        return cli.interactive(args, _backend(), None, "sid")
    return run


def test_a_slash_command_never_reaches_the_model(loop, capsys):
    rc = loop(["/help", EOFError()])
    assert rc == 0
    assert FakeAgent.calls == [], "a command is the operator talking to the REPL"
    assert "/cost" in capsys.readouterr().err


def test_a_prompt_that_starts_with_a_path_still_reaches_the_model(loop):
    loop(["/etc/hosts - what is in it?", EOFError()])
    assert FakeAgent.calls, "an absolute path is a question, not a command"


def test_a_command_that_raises_does_not_take_the_session_down(loop, monkeypatch,
                                                              capsys):
    def explode(repl, rest):
        raise RuntimeError("boom")

    monkeypatch.setitem(cmdlib.COMMANDS, "cost",
                        cmdlib.Command("cost", "x", explode))
    rc = loop(["/cost", "still here", EOFError()])
    assert rc == 0
    assert FakeAgent.calls == ["still here"]
    assert "boom" in capsys.readouterr().err


def test_exit_still_leaves(loop):
    assert loop(["/exit"]) == 0
    assert loop(["/quit"]) == 0


# --- rewind and resume -----------------------------------------------------


def _exchange(n):
    return [{"role": "user", "content": f"q{n}"},
            {"role": "assistant", "content": f"a{n}"}]


def test_rewind_drops_the_last_exchange_and_keeps_the_rest(monkeypatch):
    """The turn that went wrong is usually the last one, and both alternatives
    are bad: leave it in and every later turn reasons from it, or /clear and
    lose the twenty good turns in front of it."""
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    repl = _repl(messages=_exchange(1) + _exchange(2))
    out = _say("/rewind", repl)
    assert repl.messages == _exchange(1)
    assert "2 messages dropped" in out


def test_rewind_is_repeatable(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    repl = _repl(messages=_exchange(1) + _exchange(2) + _exchange(3))
    for _ in range(3):
        _say("/rewind", repl)
    assert repl.messages == []


def test_rewind_takes_the_tool_calls_with_the_turn_that_made_them(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    history = _exchange(1) + [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
        {"role": "assistant", "content": "a2"}]
    repl = _repl(messages=history)
    _say("/rewind", repl)
    assert repl.messages == _exchange(1), (
        "an orphaned tool result is the one shape the wire refuses")


def _compacted():
    from onecli.context import summary_message
    return [summary_message("we did things"), *_exchange(1)[1:], *_exchange(2)]


def test_rewind_stops_at_the_compaction_summary(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    repl = _repl(messages=_compacted())
    _say("/rewind", repl)
    assert repl.messages == _compacted()[:2]
    out = _say("/rewind", repl)
    assert repl.messages == _compacted()[:2], "the summary and kept messages must survive"
    assert "nothing to rewind" in out


def test_undo_stops_at_the_compaction_summary(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    repl = _repl(messages=_compacted()[:2])
    _say("/undo", repl)
    assert repl.messages == _compacted()[:2]


def test_rewind_on_an_empty_history_says_so():
    assert "nothing to rewind" in _say("/rewind", _repl(messages=[]))


def test_resume_with_no_argument_lists_sessions(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "search",
                        lambda *a: [{"session": "abc123", "modified_epoch": 1_700_000_000, "opening_line": "hello"}])
    out = _say("/resume", _repl())
    assert "abc123" in out and "/resume <id>" in out


def test_resume_adopts_the_history_without_leaving_this_session(monkeypatch):
    monkeypatch.setattr(cmdlib.session, "load_messages",
                        lambda sid: _exchange(9))
    monkeypatch.setattr(cmdlib.session, "append", lambda *a, **k: None)
    repl = _repl()
    out = _say("/resume other99", repl)
    assert repl.messages == _exchange(9)
    assert repl.sid == "sid0", "the transcript keeps being written under this id"
    assert "still sid0" in out


def test_resuming_a_session_that_is_not_there_is_reported(monkeypatch):
    def missing(sid):
        raise FileNotFoundError("no session other99")

    monkeypatch.setattr(cmdlib.session, "load_messages", missing)
    repl = _repl(messages=_exchange(1))
    out = _say("/resume other99", repl)
    assert "no session other99" in out
    assert repl.messages == _exchange(1), "a failed resume changes nothing"


# --- command files ---------------------------------------------------------


def _file(name="review", body="Review the diff.", **kw):
    from onecli.commandfiles import CommandFile

    return CommandFile(name=name, body=body, **kw)


def test_a_command_file_answers_with_a_prompt_not_with_text():
    """The whole difference between the two kinds: a built-in costs nothing,
    a command file costs a turn."""
    out = cmdlib.dispatch("/review src/a.py",
                          _repl(files={"review": _file()}))
    assert out.text == ""
    assert out.prompt == "Review the diff.\n\nsrc/a.py"


def test_a_command_file_is_offered_to_tab_completion():
    repl = _repl(files={"git:commit": _file("git:commit")})
    assert "/git:commit" in cmdlib.names_for(repl)


def test_a_namespaced_command_file_is_addressable():
    repl = _repl(files={"git:commit": _file("git:commit", "Write it.")})
    assert cmdlib.dispatch("/git:commit", repl).prompt == "Write it."
    assert cmdlib.is_command("/git:commit")


def test_a_typo_can_suggest_a_command_file_too():
    repl = _repl(files={"review": _file()})
    assert "/review" in _say("/reveiw", repl)


def test_a_command_file_appears_in_help():
    repl = _repl(files={"review": _file(description="be blunt")})
    out = _say("/help", repl)
    assert "/review" in out and "be blunt" in out


# --- the status line -------------------------------------------------------


def test_a_session_that_has_not_run_a_turn_has_nothing_to_report():
    assert cmdlib.status(_repl()) == ""


def test_the_status_line_carries_model_fullness_and_spend():
    repl = _repl()
    repl.record(types.SimpleNamespace(
        cost={"input_tokens": 40_000, "output_tokens": 0, "usd": 0.12},
        context_tokens=50_000))
    out = cmdlib.status(repl)
    assert "m" in out and "50%" in out and "$0.1200" in out


def test_an_unpriced_session_shows_tokens_in_the_status_line():
    repl = _repl()
    repl.record(types.SimpleNamespace(
        cost={"input_tokens": 1_234, "output_tokens": 6, "usd": None},
        context_tokens=1_240))
    assert "1,240 tok" in cmdlib.status(repl)


def test_footer_labels_cumulative_input_and_current_context():
    out = cli._spend({"input_tokens": 724_405, "output_tokens": 1_105,
                      "context_tokens": 38_200, "cached_input_tokens": 650_000,
                      "usd": None})
    assert out == ("724,405 input processed · 38,200 current context · "
                   "650,000 cached · 1,105 out")


# --- shell context ---------------------------------------------------------


def test_shell_output_rides_the_next_turn_and_only_that_one():
    repl = _repl()
    repl.remember_shell("git status", "nothing to commit", 0)
    block = repl.take_shell_context()
    assert "$ git status" in block and "nothing to commit" in block
    assert repl.take_shell_context() == "", "consumed on the way out"


def test_a_failing_shell_command_carries_its_exit_code():
    repl = _repl()
    repl.remember_shell("false", "", 1)
    assert "[exit 1]" in repl.take_shell_context()


def test_nothing_is_carried_when_no_shell_command_ran():
    assert _repl().take_shell_context() == ""


# --- the loop: shell lines and mentions ------------------------------------


def test_a_bang_line_runs_a_shell_command_without_spending_a_turn(loop, capsys):
    rc = loop(["!echo hello-from-the-shell", EOFError()])
    assert rc == 0
    assert FakeAgent.calls == [], "the operator's own command, not the model's"
    assert "hello-from-the-shell" in capsys.readouterr().err


def test_what_the_shell_printed_reaches_the_next_turn(loop):
    loop(["!echo marker-9times7", "what did that print?", EOFError()])
    assert FakeAgent.calls, "the turn ran"
    assert "marker-9times7" in FakeAgent.calls[0]
    assert "<shell>" in FakeAgent.calls[0]


def test_a_failing_shell_command_shows_its_exit_code(loop, capsys):
    loop(["!exit 3", EOFError()])
    assert "exit 3" in capsys.readouterr().err


def test_a_mentioned_file_is_sent_with_the_question(loop, tmp_path):
    (tmp_path / "hello.py").write_text("print('mentioned')\n")
    loop(["explain @hello.py", EOFError()])
    assert "print('mentioned')" in FakeAgent.calls[0]


def test_a_file_outside_the_workspace_is_never_sent(loop, tmp_path):
    loop(["read @../../../etc/passwd", EOFError()])
    assert "root:" not in FakeAgent.calls[0]


@pytest.mark.parametrize("trusted", [False, True])
def test_a_command_file_runs_only_with_project_trust(loop, tmp_path, trusted):
    root = tmp_path / ".claude" / "commands"
    root.mkdir(parents=True)
    (root / "blunt.md").write_text("Be blunt about $ARGUMENTS.")
    loop(["/blunt the parser", EOFError()],
         ["--trust-project-config"] if trusted else [])
    assert FakeAgent.calls == (["Be blunt about the parser."] if trusted else [])


def test_help_prints_the_aliases_it_registered():
    """An alias that is registered and never printed is a command nobody finds.

    `/quit`, `/q`, `/?` and `/ls` all worked; a persona reported `/quit` as
    undocumented after using it successfully, because `/help` listed only the
    canonical name.
    """
    text = _say("/help", _repl())
    assert "/exit" in text and "/quit" in text
    assert "/help" in text and "/?" in text


def test_chat_help_prints_its_aliases_too():
    """The other REPL had the same gap, from its own registry."""
    import io

    from onecli import chat as chatlib

    chat = types.SimpleNamespace(err=io.StringIO(), out=io.StringIO(),
                                 quiet=False)
    chatlib.dispatch(chat, "/help")
    text = chat.err.getvalue() + chat.out.getvalue()
    assert "/exit" in text and "/quit" in text and "/q" in text
    assert "/models" in text and "/ls" in text


def test_undo_reverts_workspace_and_messages(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
    (repo / "file.txt").write_text("v1\n")
    subprocess.run(["git", "add", "file.txt"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True)

    repl = _repl(workspace=str(repo))
    repl.checkpoints.begin_turn("edit file", [])
    repl.messages = [{"role": "user", "content": "edit"}, {"role": "assistant", "content": "done"}]

    (repo / "file.txt").write_text("v2\n")
    (repo / "new.txt").write_text("new\n")

    diff_text = _say("/diff", repl)
    assert "file.txt" in diff_text

    undo_text = _say("/undo", repl)
    assert "reverted workspace" in undo_text
    assert (repo / "file.txt").read_text() == "v1\n"
    assert not (repo / "new.txt").exists()
    assert repl.messages == []


def test_bg_commands_say_so_when_there_is_no_store():
    repl = _repl()
    assert "off" in _say("/bg echo hi", repl)
    assert "off" in _say("/ps", repl)
    assert "off" in _say("/kill j1", repl)
    assert "off" in _say("/logs j1", repl)


def test_bg_commands_lifecycle(tmp_path):
    import time
    store = cmdlib.joblib.JobStore("s", tmp_path)
    repl = _repl(workspace=str(tmp_path), jobs=store)

    assert "usage" in _say("/bg", repl)
    assert "usage" in _say("/kill", repl)

    out = _say("/bg echo running-job-test", repl)
    assert "started" in out
    assert "j1" in out

    ps_out = _say("/ps", repl)
    assert "j1" in ps_out

    # Give child shell time to write and finish
    for _ in range(20):
        time.sleep(0.05)
        if (store.root / "j1.log").exists():
            break

    logs_out = _say("/logs j1", repl)
    assert "running-job-test" in logs_out

    logs_default = _say("/logs", repl)
    assert "running-job-test" in logs_default

    kill_out = _say("/kill j1", repl)
    assert "j1" in kill_out



def test_model_picker_choices_never_reach_agent(loop, monkeypatch, capsys):
    monkeypatch.setattr(cmdlib.config, 'discover', lambda: [_backend(), _backend('other', 'big')])
    assert loop(['/model', '1', '2', '/exit']) == 0
    assert FakeAgent.calls == []
    output = capsys.readouterr().err
    assert 'Providers' in output and 'Models' in output
    assert 'Switched to other (big)' in output

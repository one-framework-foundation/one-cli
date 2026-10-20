"""What Ctrl-C means at an interactive prompt, and what it must not mean.

An interrupt belongs to the smallest thing that can own it. During a turn that
is the turn; at the prompt it is the line being typed. Neither is the session -
the session holds the history and the background jobs, and it leaves by ctrl-d
or /exit, which is what the banner promises.
"""

from __future__ import annotations

import types

import pytest

from onecli import cli


class FakeAgent:
    """An agent that answers, or raises whatever the test queued for it."""

    def __init__(self, *_, **kw):
        self.chain = types.SimpleNamespace(
            current=types.SimpleNamespace(name="fake"))
        self._on_event = kw.get("on_event")
        self.interrupted_messages = []

    def run(self, prompt, messages, attachments=None):
        FakeAgent.seen.append(list(messages or []))
        outcome = FakeAgent.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _result(text="ok"):
    return types.SimpleNamespace(
        text=text, messages=[], steps=[], tool_calls=0, cost=0.0,
        stopped="answered", error=None)


@pytest.fixture
def loop(monkeypatch, tmp_path):
    """`interactive` with the wire, the banner and the stores stubbed out."""
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    monkeypatch.setattr(cli, "_context_settings", lambda a, b: {
        "chain": types.SimpleNamespace(
            current=types.SimpleNamespace(name="fake"), warnings=[])})
    monkeypatch.setattr(cli, "_banner", lambda *a, **k: "")
    monkeypatch.setattr(cli, "_hooks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_submit_prompt", lambda hooks, line: (line, None))
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)
    monkeypatch.setattr(cli.joblib, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cli.spilllib, "SPILL_DIR", tmp_path / "spill")

    def run(keys, outcomes=()):
        """Feed `keys` to input(); a BaseException instance is raised there."""
        FakeAgent.outcomes = list(outcomes)
        FakeAgent.seen = []
        pending = list(keys)

        def fake_input(_prompt=""):
            key = pending.pop(0)
            if isinstance(key, BaseException):
                raise key
            return key

        monkeypatch.setattr("builtins.input", fake_input)
        args = cli.build_parser().parse_args([])
        args.workspace = str(tmp_path)
        backend = types.SimpleNamespace(name="fake", model="m", display="fake",
                                        base_url="http://x", source="default")
        return cli.interactive(args, backend, None, "sid")
    return run


def test_ctrl_c_at_the_prompt_does_not_end_the_session(loop, capsys):
    """The screenshot that produced this test: a turn was interrupted, the
    prompt came back, one more Ctrl-C - and the whole session was gone.
    """
    rc = loop([KeyboardInterrupt(), "hello", EOFError()], [_result("hi")])
    assert rc == 0
    assert "hi" in capsys.readouterr().out, (
        "the line after the interrupt has to be answered - the session lived")


def test_the_way_out_is_named_once(loop, capsys):
    loop([KeyboardInterrupt(), KeyboardInterrupt(), KeyboardInterrupt(),
          EOFError()])
    err = capsys.readouterr().err
    assert err.count("ctrl-d") == 1, (
        "say it on the first one; repeating it on every Ctrl-C is noise")


def test_ctrl_d_still_leaves(loop):
    assert loop([EOFError()]) == 0


def test_slash_exit_still_leaves(loop):
    assert loop(["/exit"]) == 0


@pytest.mark.parametrize("word", ["exit", "quit", "Exit"])
def test_a_bare_exit_word_leaves_instead_of_going_to_the_model(loop, word):
    # No scripted agent result: reaching the model would exhaust the loop.
    assert loop([word]) == 0


def test_ctrl_c_during_a_turn_keeps_the_prompt(loop, capsys):
    rc = loop(["slow thing", "5+5", EOFError()],
              [KeyboardInterrupt(), _result("10")])
    assert rc == 0
    out = capsys.readouterr()
    assert "interrupted - conversation retained" in out.err
    assert "10" in out.out, "the session took the next line as if nothing broke"


def test_a_missing_file_mention_is_warned_about_before_sending(loop, capsys):
    loop(["look at @nofile.py", EOFError()], [_result("ok")])
    assert "no such file: nofile.py" in capsys.readouterr().err


def test_a_provider_error_keeps_the_turns_work_in_the_conversation(loop, monkeypatch, capsys):
    """A 429 mid-turn returned the history up to the error, and the REPL threw
    it away: on "Continue" the model resumed from before its own tool calls,
    which had already changed files."""
    records = []
    monkeypatch.setattr(cli.session, "append", lambda sid, rec: records.append(rec))
    work = [{"role": "user", "content": "build it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "written"}]
    failed = _result("")
    failed.messages, failed.stopped, failed.error = work, "error", "HTTP 429"
    loop(["build it", "Continue with the task.", EOFError()], [failed, _result("done")])
    assert FakeAgent.seen[1] == work
    assert {"kind": "mask_run", "mask": {}, "stopped": "error", "error": "HTTP 429"} in records
    assert "provider error: HTTP 429" in capsys.readouterr().err


def test_the_repl_persists_mid_turn_checkpoints(loop, monkeypatch):
    """A turn that is killed mid-way must leave its finished exchanges on disk;
    the REPL wrote messages only when a turn ended."""
    records = []
    monkeypatch.setattr(cli.session, "append", lambda sid, rec: records.append(rec))
    snapshot = [{"role": "user", "content": "go"}]

    def run(self, prompt, messages, attachments=None):
        self._on_event("checkpoint", {"messages": snapshot})
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeAgent, "run", run)
    loop(["go", EOFError()])
    assert {"kind": "messages", "messages": snapshot} in records


def test_undo_in_the_repl_reverts_exactly_the_last_turn(loop, monkeypatch, tmp_path, capsys):
    """The REPL took no checkpoint at all, so /undo could never revert a turn."""
    def run(self, prompt, messages, attachments=None):
        (tmp_path / f"{prompt}.txt").write_text(prompt)
        return types.SimpleNamespace(
            text="ok", steps=[], tool_calls=0, cost=0.0, stopped="answered", error=None,
            messages=list(messages or []) + [{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": "ok"}])

    monkeypatch.setattr(FakeAgent, "run", run)
    assert loop(["first", "second", "/undo", EOFError()]) == 0
    assert (tmp_path / "first.txt").exists() and not (tmp_path / "second.txt").exists()
    assert "reverted workspace" in capsys.readouterr().err


def _scripted_turns(monkeypatch, tmp_path, extra_users=(), interrupt=()):
    """Each prompt writes <prompt>.txt. A prompt in `extra_users` also gets a
    harness user message (a nudge); one in `interrupt` is cut by Ctrl-C after
    its tool ran, leaving the interrupt note. Returns the histories seen."""
    seen = []

    def run(self, prompt, messages, attachments=None):
        seen.append(list(messages or []))
        (tmp_path / f"{prompt}.txt").write_text(prompt)
        history = list(messages or []) + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [{"id": prompt}]},
            {"role": "tool", "tool_call_id": prompt, "content": "written"}]
        if prompt in interrupt:
            self.interrupted_messages = history + [{"role": "user", "content":
                "The previous turn was interrupted. Its request remains unfinished."}]
            raise KeyboardInterrupt
        if prompt in extra_users:
            history.append({"role": "user", "content": "[harness] report progress"})
        history.append({"role": "assistant", "content": "ok"})
        return types.SimpleNamespace(text="ok", steps=[], tool_calls=0, cost=0.0,
                                     stopped="answered", error=None, messages=history)

    monkeypatch.setattr(FakeAgent, "run", run)
    return seen


def test_undo_after_an_interrupted_turn_keeps_the_turn_before_it(loop, monkeypatch, tmp_path):
    """Undo boundary audit: /undo cut at the last user-role message, which for an
    interrupted turn is the interrupt note - so one /undo reverted turn 2's
    files but kept its messages, and the next reverted turn 1's files while
    turn 1 stayed in the conversation."""
    seen = _scripted_turns(monkeypatch, tmp_path, interrupt={"second"})
    loop(["first", "second", "/undo", "third", EOFError()])
    assert (tmp_path / "first.txt").exists() and not (tmp_path / "second.txt").exists()
    assert seen[2] == seen[1], "the conversation is back to where turn 2 began"


def test_undo_takes_a_turns_harness_messages_with_it(loop, monkeypatch, tmp_path):
    seen = _scripted_turns(monkeypatch, tmp_path, extra_users={"first", "second"})
    loop(["first", "second", "/undo", "/undo", "third", EOFError()])
    assert not (tmp_path / "first.txt").exists() and not (tmp_path / "second.txt").exists()
    assert [m for m in seen[2] if m.get("role") != "system"] == []


def test_a_repl_without_mutating_tools_takes_no_snapshot(loop, monkeypatch, tmp_path):
    from onecli.checkpoints import CheckpointManager
    edits = []
    real = CheckpointManager.begin_turn

    def spy(self, *a, **k):
        edits.append(k.get("edits", True))
        return real(self, *a, **k)
    monkeypatch.setattr(CheckpointManager, "begin_turn", spy)
    monkeypatch.setattr(cli, "_surface", lambda *a, **k: [])
    loop(["hello", EOFError()], [_result("hi")])
    assert edits == [False]

"""`--json` must never be silently ignored.

one-cli's strongest arm is the scripted one, and on four of its read surfaces
the promise was not kept: `--sessions`, `--orc-report`, `--orc-list` and
`one -c` all accepted `--json` from the shared parser and printed prose
anyway, exit 0. A caller piping any of them into a parser got a syntax error
and nothing saying why - the worst shape a failure can take, because it looks
like the CALLER's bug.

The fifth test here is the sibling defect on the run itself: the exit code
disagreed with itself depending on the output format.
"""

from __future__ import annotations

import json

import pytest

from onecli import cli, orc as orclib, session


class Args:
    def __init__(self, **kw):
        self.json = kw.pop("json", False)
        for k, v in kw.items():
            setattr(self, k, v)


# --- --sessions ------------------------------------------------------------

@pytest.fixture
def two_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    session.append("aaaa11112222", {"kind": "messages", "messages": [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "why does the login test fail?"}]})
    session.append("bbbb33334444", {"kind": "messages", "messages": [
        {"role": "user", "content": "rename the config module"}]})
    return tmp_path


def test_sessions_json_is_json(two_sessions, capsys):
    assert cli.cmd_sessions(Args(json=True)) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["session"] for r in rows} == {"aaaa11112222", "bbbb33334444"}
    assert all("modified" in r and "bytes" in r for r in rows)


def test_sessions_json_says_what_each_session_was_about(two_sessions, capsys):
    """An id and a byte count cannot tell two mornings' work apart."""
    cli.cmd_sessions(Args(json=True))
    by_id = {r["session"]: r["opening_line"]
             for r in json.loads(capsys.readouterr().out)}
    assert by_id["aaaa11112222"] == "why does the login test fail?"
    assert by_id["bbbb33334444"] == "rename the config module"


def test_the_text_listing_says_it_too(two_sessions, capsys):
    cli.cmd_sessions(Args(json=False))
    out = capsys.readouterr()
    assert "why does the login test fail?" in out.out
    assert "rename the config module" in out.out


def test_sessions_json_on_an_empty_store_is_an_empty_list(tmp_path, monkeypatch,
                                                          capsys):
    """Not the word "no sessions yet", which is what a parser used to get."""
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "nothing")
    assert cli.cmd_sessions(Args(json=True)) == 0
    assert json.loads(capsys.readouterr().out) == []


# --- --orc-report ----------------------------------------------------------

def _record(tmp_path, sid, backend, seconds, steps, stopped="answered"):
    (tmp_path / f"{sid}.json").write_text(json.dumps({
        "session": sid, "task_class": "build",
        "nodes": [{"backend": backend, "model": f"{backend}-model",
                   "seconds": seconds, "steps": steps, "stopped": stopped}],
    }))


def test_orc_report_json_is_json_and_carries_numbers(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path)
    _record(tmp_path, "s1", "alpha", 10.0, 3)
    _record(tmp_path, "s2", "alpha", 20.0, 5)
    assert cli.cmd_orc_report(Args(json=True)) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["records"] == 2
    (g,) = got["groups"]
    assert g["backend"] == "alpha" and g["task_class"] == "build"
    assert g["nodes"] == 2
    # Numbers, not the "15.0s" a caller had to peel out of a drawn table.
    assert g["median"]["seconds"] == 15.0
    assert isinstance(g["median"]["seconds"], float)
    assert g["median"]["steps"] == 4.0
    assert g["samples"]["seconds"] == 2
    assert g["stopped"] == {"answered": 2}


def test_orc_report_json_on_an_empty_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(orclib, "RUN_DIR", tmp_path / "nothing")
    assert cli.cmd_orc_report(Args(json=True)) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["records"] == 0 and got["groups"] == []


# --- --orc-list ------------------------------------------------------------

def test_orc_list_json_is_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(orclib, "catalogue", lambda: (
        [orclib.Orc(name="wide", orchestrator="big",
                    workers={"reviewer": "small"}, task_class="review",
                    path="/x/wide.json")], []))
    assert cli.cmd_orc_list(Args(json=True)) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["name"] == "wide"
    assert row["orchestrator"] == "big"
    assert row["workers"] == {"reviewer": "small"}
    assert row["source"] == "/x/wide.json"


# --- one -c --json ---------------------------------------------------------

def test_chat_one_shot_honours_json(tmp_path, monkeypatch, capsys):
    """`one -c --json` printed prose and exited 0. from_cli never read the flag."""
    from onecli import chat as chatlib
    from onecli import config

    backend = config.Backend(name="lane", model="m", base_url="http://x/v1",
                             wire="openai")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend)
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(chatlib, "turn", lambda chat, prompt, echo=True: {
        "text": "the answer", "steps": 1, "tool_calls": 0, "seconds": 0.5,
        "input_tokens": 12, "output_tokens": 3, "usd": 0.0})

    args = Args(json=True, model=None, model_id=None, base_url=None, wire=None,
                stream=None, temperature=None, max_tokens=None, timeout=60.0,
                system="", tools=None, resume=None, quiet=False,
                prompt=["hello"])
    assert chatlib.from_cli(args) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["text"] == "the answer"
    assert got["backend"] == "lane" and got["model"] == "m"
    assert got["stopped"] == "answered"


def test_chat_one_shot_json_reports_a_provider_failure_as_2(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    from onecli import chat as chatlib
    from onecli import config

    backend = config.Backend(name="lane", model="m", base_url="http://x/v1",
                             wire="openai")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend)
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(chatlib, "turn", lambda chat, prompt, echo=True: {
        "error": "cannot reach http://x/v1", "seconds": 0.1})

    args = Args(json=True, model=None, model_id=None, base_url=None, wire=None,
                stream=None, temperature=None, max_tokens=None, timeout=60.0,
                system="", tools=None, resume=None, quiet=False, prompt=["hi"])
    assert chatlib.from_cli(args) == 2
    got = json.loads(capsys.readouterr().out)
    assert got["stopped"] == "error"
    assert "cannot reach" in got["error"]


# --- one-chat --sessions ---------------------------------------------------

def test_one_chat_lists_sessions_too(two_sessions, capsys):
    """`one --sessions` existed and `one-chat --sessions` did not.

    Same sessions, same disk, two entry points - a flag that is on one of them
    and not the other is a thing to memorise. The way to find a chat to resume
    used to be to start a chat first.
    """
    from onecli import chat as chatlib
    assert chatlib.main(["--sessions"]) == 0
    out = capsys.readouterr()
    assert "why does the login test fail?" in out.out
    assert "rename the config module" in out.out


def test_one_chat_sessions_json(two_sessions, capsys):
    from onecli import chat as chatlib
    assert chatlib.main(["--sessions", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["session"] for r in rows} == {"aaaa11112222", "bbbb33334444"}


# --- the exit code ---------------------------------------------------------

def test_one_outcome_means_one_exit_code_whatever_the_format():
    """--json is a claim about stdout, never about whether the run worked."""
    assert cli._exit_for("answered") == 0
    assert cli._exit_for("error") == 2
    assert cli._exit_for("empty") == 1
    assert cli._exit_for("budget") == 1
    assert cli._exit_for("budget_usd") == 1

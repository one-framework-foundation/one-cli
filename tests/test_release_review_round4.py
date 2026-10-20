"""v0.1.1 release review, round four: the blockers, each pinned by a test.

Terminal control from the model on the plain (non-fullscreen) path, the Claude
Code `:*` command prefix, `@path` mentions that bypassed Read deny rules, mask
files with unknown fields, and arena requests the `--budget` ceiling never saw.
"""

from __future__ import annotations

import io
import json
import types

import pytest

from onecli import cli, commands, mentions, permissions
from onecli.config import WIRE_OPENAI, Backend
from onecli.evals import corpus

OSC52 = "\x1b]52;c;ZXZpbA==\x07"
OSC0 = "\x1b]0;pwned\x07"
HOSTILE = f"{OSC52}evil{OSC0}"


def _clean(text: str) -> None:
    assert "\x1b]" not in text, "an OSC sequence reached the terminal"
    assert "\x07" not in text


# --- 1. terminal escapes ----------------------------------------------------

def test_the_plain_trace_strips_osc_from_every_model_controlled_line(monkeypatch):
    from onecli import spinner as spinlib

    class Rail:
        def __init__(self, *a, **k): pass
        def start(self): return self
        def stop(self): pass
        def feed(self, n): pass

    monkeypatch.setattr(spinlib, "Spinner", Rail)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    trace = cli._tracer(quiet=False, state={"step": 0}, model="m")

    trace("tool_start", {"call": types.SimpleNamespace(
        name="bash", arguments={"command": HOSTILE}), "mutating": True})
    trace("tool_start", {"call": types.SimpleNamespace(
        name=HOSTILE, arguments={"path": HOSTILE}), "mutating": False})
    trace("todos", {"items": [{"text": HOSTILE, "state": "in_progress"}]})
    trace("todos", {"reason": "restored", "summary": HOSTILE, "items": []})
    trace("todo_notice", {"summary": HOSTILE})
    trace("job_notice", {"note": HOSTILE})
    trace("loop_advisory", {"note": HOSTILE})
    trace("compact", {"ok": False, "error": HOSTILE})
    trace("fallback", {"from": "a", "to": "b", "reason": HOSTILE})

    _clean(out.getvalue() + err.getvalue())
    assert err.getvalue().count("evil") >= 9, "the text itself still prints"


class _ErrAgent:
    def __init__(self, *_a, **kw):
        self.chain = types.SimpleNamespace(
            cursor=0, current=types.SimpleNamespace(name="fake", model="m"))
        self.spawner, self.run_limits, self.wire_requests = None, None, 0
        self.interrupted_messages = []

    def run(self, prompt, messages, attachments=None):
        _ErrAgent.seen = prompt
        return types.SimpleNamespace(
            text="answer", messages=[], steps=[], tool_calls=0, cost=0.0,
            stopped=_ErrAgent.stopped, error=HOSTILE, todos=[])


def _stub_run(monkeypatch):
    monkeypatch.setattr(cli, "Agent", _ErrAgent)
    monkeypatch.setattr(cli, "_context_settings", lambda a, b: {
        "chain": types.SimpleNamespace(
            current=types.SimpleNamespace(name="fake"), warnings=[])})
    monkeypatch.setattr(cli, "_hooks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_submit_prompt", lambda hooks, p: (p, None))
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)
    monkeypatch.setattr(cli, "speak_answer", lambda *a, **k: None)


@pytest.mark.parametrize("stopped", ["error", "context", "empty", "truncated"])
def test_a_provider_error_never_prints_its_escapes(stopped, monkeypatch, tmp_path, capsys):
    _stub_run(monkeypatch)
    _ErrAgent.stopped = stopped
    backend = Backend(name="fake", model="m", base_url="http://127.0.0.1:1/v1")
    args = cli.build_parser().parse_args(
        ["-w", str(tmp_path), "--no-stream", "-m", "fake", "hello"])
    cli.run_once(args, backend, [], "sid-esc")
    got = capsys.readouterr()
    _clean(got.out + got.err)
    assert "evil" in got.err


def test_logs_and_diff_strip_osc(tmp_path):
    repl = types.SimpleNamespace(
        args=types.SimpleNamespace(workspace=str(tmp_path)),
        jobs=types.SimpleNamespace(jobs={"j1": 1},
                                   output=lambda id, offset=0, limit=0: HOSTILE),
        checkpoints=types.SimpleNamespace(get_diff=lambda cp: "+" + HOSTILE))
    for text in (commands._logs(repl, "j1"), commands._diff(repl, "")):
        _clean(text)
        assert "evil" in text


# --- 2. `Bash(cmd:*)` -------------------------------------------------------

@pytest.mark.parametrize("command,hit", [
    ("git push", True), ("git push --force origin", True),
    ("git push\t-f", True), ("git pushx", False), ("git push:x", False),
    ("git status", False)])
def test_colon_star_is_a_word_prefix(command, hit):
    assert permissions.Rule.parse("Bash(git push:*)")._command_matches(command) is hit


def test_colon_star_deny_and_allow_through_the_cli_flags(tmp_path):
    args = cli.build_parser().parse_args(
        ["-w", str(tmp_path), "--disallowed-tools", "Bash(rm:*)",
         "--allowed-tools", "Bash(git status:*),Read", "x"])
    guard = next(g for g in permissions.path_guards(cli._guards(args, "default"))
                 if isinstance(g, permissions.PermissionGuard))

    def allowed(command):
        return guard.check("bash", {"command": command}, True).allowed

    assert not allowed("rm -rf /"), "Bash(rm:*) must deny rm with arguments"
    assert not allowed("rm")
    assert allowed("git status -s")
    assert allowed("git status")
    assert not allowed("git statusx")
    assert not allowed("rmdir x"), "not on the allow list"


# --- 3. `@path` and Read deny rules -----------------------------------------

def test_a_denied_mention_stays_plain_text(tmp_path):
    (tmp_path / ".env").write_text("SECRET_TOKEN=hunter2\n")
    (tmp_path / "ok.txt").write_text("FINE\n")
    guards = permissions.build(str(tmp_path), disallowed=["Read(.env)"])
    text, sent = mentions.expand("read @.env and @ok.txt", str(tmp_path),
                                 path_guards=guards)
    assert "hunter2" not in text
    assert sent == ["ok.txt"] and "FINE" in text
    assert "@.env" in text


def test_the_one_shot_prompt_honours_read_deny(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("SECRET_TOKEN=hunter2\n")
    _stub_run(monkeypatch)
    _ErrAgent.stopped = "answered"
    backend = Backend(name="fake", model="m", base_url="http://127.0.0.1:1/v1")
    args = cli.build_parser().parse_args(
        ["-w", str(tmp_path), "-q", "--no-stream", "-m", "fake",
         "--disallowed-tools", "Read(.env)", "look at @.env"])
    cli.run_once(args, backend, [], "sid-env")
    assert "@.env" in _ErrAgent.seen
    assert "hunter2" not in _ErrAgent.seen


def test_the_repl_honours_read_deny(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("SECRET_TOKEN=hunter2\n")
    _stub_run(monkeypatch)
    _ErrAgent.stopped = "answered"
    monkeypatch.setattr(cli, "_banner", lambda *a, **k: "")
    monkeypatch.setattr(cli.joblib, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cli.spilllib, "SPILL_DIR", tmp_path / "spill")
    keys = ["look at @.env"]

    def fake_input(_prompt=""):
        if not keys:
            raise EOFError
        return keys.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    args = cli.build_parser().parse_args(
        ["-w", str(tmp_path), "--disallowed-tools", "Read(.env)"])
    backend = types.SimpleNamespace(name="fake", model="m", display="fake",
                                    base_url="http://x", source="default")
    cli.interactive(args, backend, None, "sid-repl")
    assert "@.env" in _ErrAgent.seen
    assert "hunter2" not in _ErrAgent.seen


# --- 6. mask files fail closed ----------------------------------------------

@pytest.mark.parametrize("body,needle", [
    ({"audit": {"preamble": "p", "execution_polcy": {"kind": "review"}}}, "execution_polcy"),
    ({"audit": "just a string"}, "must be an object"),
])
def test_a_malformed_mask_file_is_refused(body, needle, tmp_path, monkeypatch):
    path = tmp_path / "masks.json"
    path.write_text(json.dumps(body))
    monkeypatch.setenv("ONE_MASKS_JSON", str(path))
    masks, error = corpus.load_masks()
    assert masks == {} and error and needle in error
    from onecli import masks as masklib
    with pytest.raises(ValueError, match=needle):
        masklib.catalog()


# --- 8. arena probe and judge under --budget --------------------------------

def _priced_lane(monkeypatch):
    from onecli import config as cfg
    from onecli.evals import arena
    monkeypatch.setattr(cfg, "load_pricing", lambda: {"m-priced": (1.0, 2.0)})
    lane = Backend(name="priced", model="m-priced", wire=WIRE_OPENAI,
                   base_url="https://example.test/v1", api_key="k")
    monkeypatch.setattr(arena.bench, "select_backends", lambda *a, **k: [lane])
    return arena, lane


def test_budget_zero_sends_no_request(monkeypatch, tmp_path):
    arena, lane = _priced_lane(monkeypatch)
    from onecli.evals import arena_parity
    sent = []

    def no_wire(*a, **k):
        sent.append(a)
        raise AssertionError("a request went out under --budget 0")

    monkeypatch.setattr(arena_parity.subprocess, "run", no_wire)
    monkeypatch.setattr(arena.subprocess, "run", no_wire)
    monkeypatch.setattr(arena.providers, "build", no_wire)
    monkeypatch.setattr(arena, "judge_backend", lambda: lane)
    monkeypatch.setattr(arena.harnesses.HARNESSES["one"], "available", lambda: True)
    rc = arena.main(["priced", "--harness", "one", "--budget", "0",
                     "--suite", "basic", "--out", str(tmp_path / "out")])
    assert sent == []
    assert rc != 0


def test_probe_attempts_are_charged(monkeypatch, tmp_path):
    from onecli.evals import arena, arena_parity
    budget = arena.bench.Budget(0.05)
    calls = []

    def once(harness, backend, root, timeout, attempt=1):
        calls.append(attempt)
        return arena_parity.ProbeResult(harness.name, False, reason="no", usd=0.03)

    monkeypatch.setattr(arena_parity, "_probe_once", once)
    lane = Backend(name="priced", model="m", wire=WIRE_OPENAI,
                   base_url="https://example.test/v1", api_key="k")
    res = arena_parity.run_probe(arena.harnesses.HARNESSES["one"], lane, tmp_path,
                                 attempts=3, budget=budget)
    assert calls == [1, 2], "the third attempt must not run past the ceiling"
    assert round(budget.spent, 4) == 0.06
    assert not res.ok and "budget" in res.reason


def test_an_unpriced_judge_is_refused_under_budget(monkeypatch, tmp_path, capsys):
    arena, _lane = _priced_lane(monkeypatch)
    judge = Backend(name="mystery", model="m-unpriced", wire=WIRE_OPENAI,
                    base_url="https://judge.example.test/v1", api_key="k")
    monkeypatch.setattr(arena, "judge_backend", lambda: judge)
    rc = arena.main(["priced", "--harness", "one", "--budget", "1",
                     "--out", str(tmp_path / "out")])
    assert rc == 2
    assert "mystery" in capsys.readouterr().out


def test_the_judge_is_charged_and_stops_at_the_ceiling(monkeypatch):
    from onecli import config as cfg
    from onecli.evals import arena
    monkeypatch.setattr(cfg, "load_pricing", lambda: {"m-judge": (1_000.0, 1_000.0)})
    judge = Backend(name="judge", model="m-judge", wire=WIRE_OPENAI,
                    base_url="https://judge.example.test/v1", api_key="k")
    monkeypatch.setattr(arena, "judge_backend", lambda: judge)
    builds = []

    class Wire:
        def chat(self, *a, **k):
            return types.SimpleNamespace(
                text='{"correctness": 5, "minimality": 5, "fit": 5}',
                usage={"prompt_tokens": 100, "completion_tokens": 100})

    monkeypatch.setattr(arena.providers, "build",
                        lambda *a, **k: builds.append(1) or Wire())
    budget = arena.bench.Budget(0.1)
    assert arena.judge("p", "+x", budget=budget)["judge_total"] == 15
    assert round(budget.spent, 4) == 0.2
    assert "budget" in arena.judge("p", "+x", budget=budget)["judge_error"]
    assert builds == [1]


# --- 4. the attachment metadata row names every automatic GET ---------------

def test_attachment_metadata_row_names_vision_and_audio_probes():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    # The private release procedure carries a third copy; it is not in the
    # public tree, so it is compared only where it exists.
    copies = [root / p for p in ("SECURITY.md", "docs/reproducible-releases.md",
                                 "docs/internal/release-procedure.md")]
    rows = {next(line for line in p.read_text().splitlines()
                 if line.startswith("| Attachment capability metadata |"))
            for p in copies if p.exists() or p.name != "release-procedure.md"}
    assert len(rows) == 1, "the inventory copies drifted"
    row = rows.pop()
    for fragment in ("ONE_CLI_VISION_BACKENDS", "vision_backends()", "declares()",
                     "--inline-audio", "perceive()", "declares(backend, 'audio')",
                     "60 seconds", "--media-route off", "no credentials"):
        assert fragment in row, fragment


# --- 5. one-chat --help -----------------------------------------------------

def test_one_chat_help_describes_the_tool_loop_it_runs():
    from onecli import chat
    text = chat.build_parser().description
    assert chat.DEFAULT_TOOL_MODE == "auto"
    assert "No tool loop" not in " ".join(text.split())
    assert "--tools off" in text

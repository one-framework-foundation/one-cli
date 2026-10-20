"""Under `--json`, a run must say what it is DOING on stderr, live.

Before this, a `--json` run wrote NOTHING to stderr for its whole duration and
one JSON object to stdout at the end. A parent process therefore had no live
channel at all and could only paint "waiting on
<model> (Ns)". The session transcript is no help: it is flushed when the run
ENDS.

Two promises are tested here, and they pull against each other:
  1. stderr carries one parseable NDJSON object per event, flushed as it goes.
  2. stdout is STILL exactly one JSON object. A second object there breaks
     every caller that parses it, so the events must never leak onto it.
"""

from __future__ import annotations

import io
import json

from onecli import cli, events
from onecli.providers.base import ToolCall


class _Step:
    def __init__(self, index):
        self.index = index


def _args(argv):
    return cli.build_parser().parse_args(argv)


def _drive(tracer):
    """Replay the event sequence a two-tool run produces."""
    tracer("waiting", {"step": 1})
    tracer("turn", {"step": _Step(1)})
    tracer("tool_start", {"call": ToolCall(id="a", name="Read",
                                           arguments={"path": "host.py"}),
                          "mutating": False})
    tracer("tool_end", {"call": ToolCall(id="a", name="Read",
                                         arguments={"path": "host.py"}),
                        "output": "line one\nline two"})
    tracer("waiting", {"step": 2})
    tracer("tool_start", {"call": ToolCall(id="b", name="Bash",
                                           arguments={"command": "pytest -q " + "x" * 80}),
                          "mutating": True})
    tracer("tool_end", {"call": ToolCall(id="b", name="Bash",
                                         arguments={"command": "pytest"}),
                        "output": "error: boom"})
    tracer("compact", {"ok": True, "messages_before": 40, "messages_after": 8,
                       "before_tokens": 9, "after_tokens": 2})
    tracer("fallback", {"from": "kimi", "to": "local-a", "reason": "429"})


def test_json_run_emits_parseable_ndjson_on_stderr(monkeypatch):
    err = io.StringIO()
    args = _args(["--json", "hi"])
    state = {"step": 0, "stream": False, "streamed": False}
    _real = events.emitter
    monkeypatch.setattr(cli.events, "emitter",
                        lambda stream=None, step=None: _real(err, step))
    tracer = cli._tracer(True, state, "ollama-cloud",
                         emit=cli._run_emitter(args, state))
    _drive(tracer)

    lines = [json.loads(x) for x in err.getvalue().splitlines() if x.strip()]
    kinds = [e["e"] for e in lines]
    assert "waiting" in kinds and "tool_start" in kinds and "tool_end" in kinds
    assert "compact" in kinds and "fallback" in kinds

    reads = [e for e in lines if e.get("tool") == "Read" and e["e"] == "tool_start"]
    assert reads and reads[0]["arg"] == ""
    assert reads[0]["step"] == 1

    ends = [e for e in lines if e["e"] == "tool_end"]
    assert ends[0]["ok"] is True and ends[1]["ok"] is False

    bash = [e for e in lines if e.get("tool") == "Bash" and e["e"] == "tool_start"][0]
    # step is stamped from the run's own counter, not from the event payload
    assert bash["step"] == 2 and bash["mutating"] is True
    # argument content is withheld, including short secrets
    assert len(bash["arg"]) <= 40

    # never one line per token
    assert "text_delta" not in kinds


def test_events_never_reach_stdout(monkeypatch, capsys):
    err = io.StringIO()
    args = _args(["--json", "hi"])
    state = {"step": 0, "stream": False, "streamed": False}
    _real = events.emitter
    monkeypatch.setattr(cli.events, "emitter",
                        lambda stream=None, step=None: _real(err, step))
    tracer = cli._tracer(True, state, "m", emit=cli._run_emitter(args, state))
    _drive(tracer)
    tracer("text_delta", {"kind": "text", "chunk": "the answer"})
    assert capsys.readouterr().out == ""
    assert err.getvalue().strip() != ""


def test_no_json_means_no_event_stream():
    """The terminal trace is the live signal already; a second copy would
    corrupt the same stream it is printed on."""
    assert cli._run_emitter(_args(["hi"]), {"step": 0}) is None
    assert cli._run_emitter(_args(["--json", "hi"]), {"step": 0}) is not None


def test_a_broken_pipe_does_not_take_the_run_down():
    class Dead(io.StringIO):
        def write(self, _):
            raise OSError("broken pipe")

    emit = events.emitter(Dead(), step=lambda: 3)
    emit("tool_start", {"call": ToolCall(id="a", name="Read",
                                         arguments={"path": "x"}),
                        "mutating": False})


def test_a_failed_compaction_says_why_on_the_wire():
    """"context compaction failed" forty times tells a reader nothing.

    The `error` and the failure class were already on the event and were being
    dropped at the emitter, so the only thing a parent process could render was
    the word "failed". Both are stable tokens, never runner payload text.
    """
    import io
    from onecli import events

    out = io.StringIO()
    emit = events.emitter(stream=out, step=lambda: 7)
    emit("compact", {"ok": False, "failure": "starved",
                     "error": "summarizer returned nothing"})
    emit("compact", {"ok": True, "fallback": True, "failure": "",
                     "error": "summarizer returned nothing"})

    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert lines[0]["failure"] == "starved" and lines[0]["ok"] is False
    assert lines[0]["step"] == 7
    assert lines[1]["ok"] is True and lines[1]["fallback"] is True


def test_an_empty_retry_carries_its_step_and_reason():
    """A bare {"e":"empty_retry"} cannot be told from the one before it."""
    import io
    from onecli import events

    out = io.StringIO()
    emit = events.emitter(stream=out, step=lambda: 0)
    emit("empty_retry", {"step": 12, "finish_reason": "length",
                         "max_tokens": 16384})

    line = json.loads(out.getvalue().strip())
    assert line["step"] == 12 and line["finish_reason"] == "length"


def test_progress_never_copies_tool_arguments():
    for include_text in (False, True):
        out = io.StringIO()
        emit = events.emitter(out, include_text=include_text)
        for name, arguments in [("Bash", {"command": "TOKEN=synthetic-secret echo ok"}),
                                ("Read", {"path": "synthetic-secret.txt"}),
                                ("mcp__test__echo", {"text": "synthetic-secret"})]:
            emit("tool_start", {"call": ToolCall(id="a", name=name, arguments=arguments)})
        assert "synthetic-secret" not in out.getvalue()
        assert len(out.getvalue().splitlines()) == 3

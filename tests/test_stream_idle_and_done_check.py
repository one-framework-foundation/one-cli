"""A stalled stream is abandoned, and "done" needs the declared check to pass.

Two regressions: a request that sits open with no reply until a human
interrupts it, and a model that says "done" with the deliverable never copied
to its target.
"""

from __future__ import annotations

import json

import pytest

from onecli.agent import Agent
from onecli.providers import base
from onecli.providers.base import ProviderError, StreamIdle, ToolCall, Turn, is_transient
from onecli.providers.openai_wire import OpenAIWire


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _lines(clock, script):
    """Yield SSE lines, advancing the fake clock before each one."""
    for gap, line in script:
        clock.now += gap
        yield line.encode()


def _chunk(**delta) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]})


# -- stream idle ----------------------------------------------------------------

def test_a_keepalive_only_stream_is_abandoned_at_the_content_limit():
    clock = Clock()
    script = [(1, _chunk(content="a"))] + [(30, ": keepalive")] * 30
    with pytest.raises(StreamIdle) as err:
        list(base.sse_payloads(_lines(clock, script), idle_seconds=120,
                               content_seconds=600, clock=clock))
    assert is_transient(err.value)


def test_thinking_frames_and_pings_keep_a_long_stream_alive():
    """A reasoning model may think for minutes without a text frame."""
    clock = Clock()
    script = ([(60, _chunk(reasoning_content="hmm"))] * 15
              + [(60, 'data: {"type": "keepalive"}')] * 5
              + [(1, _chunk(content="answer")), (0, "data: [DONE]")])
    frames = list(base.sse_payloads(_lines(clock, script), idle_seconds=120,
                                    content_seconds=600, clock=clock))
    assert len(frames) == 21


def test_a_silent_socket_becomes_stream_idle():
    def lines():
        yield _chunk(content="a").encode()
        raise TimeoutError("timed out")
    with pytest.raises(StreamIdle):
        list(base.sse_payloads(lines(), idle_seconds=120))


def test_a_stall_before_any_answer_text_is_retried_then_raised():
    calls = []

    def run(on_delta):
        calls.append(1)
        on_delta("thinking", "x")
        raise StreamIdle("stream idle: timed out")
    with pytest.raises(StreamIdle, match="gave up after 3"):
        base.stream_with_idle_retry(run, lambda k, c: None)
    assert len(calls) == base.MAX_IDLE_RETRIES + 1


def test_a_stall_after_visible_text_is_raised_not_replayed():
    calls = []

    def run(on_delta):
        calls.append(1)
        on_delta("text", "half an ans")
        raise StreamIdle("stream idle: timed out")
    with pytest.raises(StreamIdle):
        base.stream_with_idle_retry(run, lambda k, c: None)
    assert len(calls) == 1


class _Resp:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_the_openai_wire_resends_a_stalled_stream_and_bounds_the_socket(monkeypatch):
    from onecli.providers import openai_wire
    monkeypatch.setenv("ONECLI_STREAM_IDLE_TIMEOUT", "90")
    timeouts = []

    def stalled():
        raise TimeoutError("timed out")
        yield b""

    responses = [_Resp(stalled()),
                 _Resp([_chunk(content="ok").encode(), b"data: [DONE]\n"])]

    def fake_urlopen(req, timeout=None):
        timeouts.append(timeout)
        return responses.pop(0)
    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    wire = OpenAIWire("http://h:1", "m", timeout=600)
    turn = wire.chat([{"role": "user", "content": "hi"}], on_delta=lambda k, c: None)
    assert turn.text == "ok"
    assert timeouts == [90, 90]
    assert wire.retry_reasons == {"stream_idle": 1}


def test_a_streamed_tool_turn_echoes_its_reasoning_unchanged(monkeypatch):
    """Z.ai (GLM) and DeepSeek both require reasoning_content back on tool turns."""
    from onecli.providers import openai_wire
    body = [_chunk(reasoning_content="step one, "), _chunk(reasoning_content="then two"),
            _chunk(tool_calls=[{"index": 0, "id": "c1",
                                "function": {"name": "ls", "arguments": "{}"}}]),
            "data: [DONE]"]
    monkeypatch.setattr(openai_wire.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp([b.encode() for b in body]))
    turn = OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    assert turn.echo == {"reasoning_content": "step one, then two"}


# -- done check -----------------------------------------------------------------

class Scripted:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.seen.append([dict(m) for m in messages])
        if not self.turns:
            raise ProviderError("script exhausted")
        return self.turns.pop(0)


def test_a_failing_done_check_refuses_done_then_ends_unverified(tmp_path):
    provider = Scripted([Turn(text="done")] * 4)
    result = Agent(provider, str(tmp_path),
                   done_check="test -f deliver/out.txt").run("make it")
    assert result.stopped == "unverified"
    assert "deliver/out.txt" in result.error
    assert len(provider.seen) == 4
    assert "done-check" in provider.seen[-1][-1]["content"]


def test_the_done_check_passes_once_the_deliverable_exists(tmp_path):
    provider = Scripted([
        Turn(text="done"),
        Turn(tool_calls=[ToolCall(id="w", name="write", arguments={
            "path": "deliver/out.txt", "content": "x"})]),
        Turn(text="done, really"),
    ])
    result = Agent(provider, str(tmp_path),
                   done_check="test -f deliver/out.txt").run("make it")
    assert result.stopped == "answered"
    assert result.text == "done, really"


def test_a_tool_call_resets_the_done_check_allowance(tmp_path):
    ls = Turn(tool_calls=[ToolCall(id="l", name="ls", arguments={"path": "."})])
    provider = Scripted([Turn(text="done")] * 3 + [ls] + [Turn(text="done")] * 4)
    result = Agent(provider, str(tmp_path), done_check="false").run("go")
    # Without the reset the first claim after the tool call ends the run.
    assert result.stopped == "unverified"
    assert len(provider.seen) == 8

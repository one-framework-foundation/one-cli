"""Release audit of the earlier implementation: truncated streams, inherited stdin, job file
modes and MCP argument names.

- A stream that ended without `[DONE]` and without a finish_reason came back as
  a normal answer ("I will now delete", finish=''). Anthropic never checked
  for message_stop either.
- The bash tool inherited the CLI's stdin, so a model command could swallow
  keystrokes or hang waiting on the terminal.
- Job logs and records were created with the default umask (664/775) while
  sessions and spills are private.
- `tools.run` always passed `workspace=`, so an MCP tool with its own
  `workspace` parameter got a TypeError.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from onecli import tools as toolkit
from onecli.jobs import JobStore
from onecli.providers import anthropic_wire, openai_wire
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.base import ProviderError, is_transient
from onecli.providers.openai_wire import OpenAIWire
from onecli.tools import shell


class _Resp:
    def __init__(self, body: bytes):
        self._lines = body.splitlines(keepends=True)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, module, *bodies: bytes) -> list:
    calls: list = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return _Resp(bodies[min(len(calls), len(bodies)) - 1])
    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return calls


def _oa(**choice) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, **choice}]}) + "\n\n"


def _chat(wire, seen=None):
    return wire.chat([{"role": "user", "content": "hi"}], max_tokens=64,
                     on_delta=lambda k, c: (seen or []).append((k, c)))


# -- 3. truncated streams --------------------------------------------------------

def test_openai_stream_cut_mid_answer_is_an_error_not_an_answer(monkeypatch):
    _serve(monkeypatch, openai_wire, _oa(delta={"content": "I will now delete"}).encode())
    with pytest.raises(ProviderError) as err:
        _chat(OpenAIWire("http://h:1", "m"))
    assert "truncated" in str(err.value)
    assert is_transient(err.value)  # the fallback chain may carry the turn


def test_openai_truncation_before_any_answer_text_is_resent(monkeypatch):
    calls = _serve(monkeypatch, openai_wire,
                   _oa(delta={"reasoning_content": "hmm"}).encode(),
                   (_oa(delta={"content": "ok"}) + "data: [DONE]\n\n").encode())
    turn = _chat(OpenAIWire("http://h:1", "m"))
    assert turn.text == "ok" and len(calls) == 2


def test_openai_finish_reason_without_done_is_a_complete_stream(monkeypatch):
    _serve(monkeypatch, openai_wire,
           (_oa(delta={"content": "ok"}) + _oa(delta={}, finish_reason="stop")).encode())
    turn = _chat(OpenAIWire("http://h:1", "m"))
    assert turn.text == "ok" and turn.finish_reason == "stop"


def test_openai_done_without_finish_reason_is_a_complete_stream(monkeypatch):
    _serve(monkeypatch, openai_wire,
           (_oa(delta={"content": "ok"}) + "data: [DONE]\n\n").encode())
    assert _chat(OpenAIWire("http://h:1", "m")).text == "ok"


def _an(event: dict) -> str:
    return "data: " + json.dumps(event) + "\n\n"


_AN_TEXT = (_an({"type": "message_start", "message": {"usage": {}}})
            + _an({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}})
            + _an({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "I will now delete"}}))


def test_anthropic_stream_without_message_stop_is_an_error(monkeypatch):
    _serve(monkeypatch, anthropic_wire, _AN_TEXT.encode())
    with pytest.raises(ProviderError) as err:
        _chat(AnthropicWire("http://h:1", "m"))
    assert "truncated" in str(err.value) and is_transient(err.value)


def test_anthropic_stop_reason_counts_as_terminal(monkeypatch):
    body = _AN_TEXT + _an({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
    _serve(monkeypatch, anthropic_wire, body.encode())
    assert _chat(AnthropicWire("http://h:1", "m")).text == "I will now delete"


# -- 4. bash tool stdin ----------------------------------------------------------

def test_bash_tool_never_reads_the_cli_stdin(tmp_path):
    read_end, write_end = os.pipe()
    os.write(write_end, b"KEYSTROKES\n")
    os.close(write_end)
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        out = shell.bash(str(tmp_path), "cat; echo end", timeout=5)
        leftover = os.read(0, 100)
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_end)
    assert "KEYSTROKES" not in out and "end" in out
    assert leftover == b"KEYSTROKES\n"


# -- 5. job file modes -----------------------------------------------------------

def test_job_logs_and_records_are_private(tmp_path):
    old = os.umask(0o002)
    try:
        store = JobStore("sid", root=tmp_path / "jobs")
        job = store.start("true", str(tmp_path))
        job.proc.wait()
        store.poll()
    finally:
        os.umask(old)
    mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)  # noqa: E731
    assert mode(store.root) == 0o700
    assert mode(job.log) == 0o600
    assert mode(store.root / f"{job.id}.json") == 0o600


# -- 6. MCP tool with a `workspace` parameter ------------------------------------

def test_a_tool_that_does_not_take_workspace_gets_its_own_argument(monkeypatch):
    seen = {}

    def call(**arguments):
        seen.update(arguments)
        return "ok"
    monkeypatch.setitem(toolkit.REGISTRY, "mcp__x__open", toolkit.Tool(
        name="mcp__x__open", description="", fn=call, takes_workspace=False,
        parameters={"type": "object", "properties": {"workspace": {"type": "string"}}}))
    out = toolkit.run("mcp__x__open", {"workspace": "projA"}, workspace="/ws")
    assert out == "ok" and seen == {"workspace": "projA"}


def test_registered_mcp_tools_do_not_receive_the_internal_workspace():
    from onecli import mcp

    class Server:
        spec = mcp.ServerSpec(name="x", command="true")
        tools = [{"name": "open", "inputSchema": {
            "type": "object", "properties": {"workspace": {"type": "string"}}}}]

        def call(self, tool, arguments):
            return json.dumps(arguments)
    client = mcp.MCPClient([])
    client.servers = [Server()]
    registry: dict = {}
    client.register(registry, toolkit.Tool)
    real = toolkit.REGISTRY
    try:
        toolkit.REGISTRY = {**real, **registry}
        out = toolkit.run("mcp__x__open", {"workspace": "projA"}, workspace="/ws")
    finally:
        toolkit.REGISTRY = real
    assert json.loads(out) == {"workspace": "projA"}

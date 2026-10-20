"""one-cli wire conversion and provider parsing.

No network: every test drives the pure conversion functions or a canned reply
body, because the thing that breaks a tool loop is a shape mistake, not an
endpoint being down.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from onecli.config import WIRE_ANTHROPIC, WIRE_OPENAI, Backend, _parse_env_file, resolve
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.base import ProviderError
from onecli.providers.openai_wire import OpenAIWire

@pytest.fixture(autouse=True)
def _declared_output_maximum(monkeypatch):
    """These fakes serve /messages only; the Models API lookup is tested in
    test_max_tokens.py."""
    monkeypatch.setattr(AnthropicWire, "declared_max_output", lambda self: 4096)



def test_openai_reply_with_string_arguments_parses_to_dict():
    turn = OpenAIWire._to_turn({
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    })
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {"path": "a.py"}
    assert turn.finish_reason == "tool_calls"


def test_openai_unparsable_arguments_surface_rather_than_vanish():
    """A dropped call looks like a no-op turn; the loop must see the failure."""
    turn = OpenAIWire._to_turn({
        "choices": [{"message": {"tool_calls": [
            {"id": "c1", "function": {"name": "read", "arguments": "{not json"}},
        ]}}],
    })
    assert turn.tool_calls[0].arguments == {"__unparsed__": "{not json"}


def test_openai_reply_without_choices_raises():
    with pytest.raises(ProviderError):
        OpenAIWire._to_turn({"object": "chat.completion"})


def test_openai_separates_reasoning_from_text():
    turn = OpenAIWire._to_turn({
        "choices": [{"message": {"content": "answer",
                                 "reasoning_content": "scratchpad"}}],
    })
    assert turn.text == "answer"
    assert turn.thinking == "scratchpad"


def test_anthropic_splits_system_out_of_the_message_list():
    system, convo = AnthropicWire._split_system([
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "S2"},
    ])
    assert system == "S1\n\nS2"
    assert convo == [{"role": "user", "content": "hello"}]


def test_anthropic_converts_openai_tool_call_to_tool_use_block():
    _, convo = AnthropicWire._split_system([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "read",
                                      "arguments": json.dumps({"path": "x"})}},
        ]},
    ])
    assert convo[0]["content"][0] == {
        "type": "tool_use", "id": "t1", "name": "read", "input": {"path": "x"},
    }


def test_anthropic_merges_consecutive_tool_results_into_one_user_turn():
    """Two user turns in a row are rejected by the API, so results must merge."""
    _, convo = AnthropicWire._split_system([
        {"role": "tool", "tool_call_id": "t1", "content": "one"},
        {"role": "tool", "tool_call_id": "t2", "content": "two"},
    ])
    assert len(convo) == 1
    assert [b["tool_use_id"] for b in convo[0]["content"]] == ["t1", "t2"]


def test_anthropic_tool_schema_accepts_the_openai_shape():
    got = AnthropicWire._tool_to_anthropic({
        "type": "function",
        "function": {"name": "read", "description": "d",
                     "parameters": {"type": "object", "properties": {}}},
    })
    assert got["name"] == "read"
    assert got["input_schema"] == {"type": "object", "properties": {}}


def test_anthropic_error_body_raises_instead_of_returning_empty_turn():
    with pytest.raises(ProviderError):
        AnthropicWire._to_turn({"type": "error", "error": {"message": "boom"}})


def test_env_file_tolerates_quotes_and_export():
    path = pytest.importorskip("pathlib").Path
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
        fh.write("# comment\nexport A='one'\nB=\"two\"\n\nC=three\n")
        name = fh.name
    assert _parse_env_file(path(name)) == {"A": "one", "B": "two", "C": "three"}


def test_resolve_with_explicit_base_url_never_touches_discovery():
    b = resolve(base_url="http://127.0.0.1:9/v1/", model="m", wire=WIRE_OPENAI)
    assert b.base_url == "http://127.0.0.1:9/v1"
    assert b.wire == WIRE_OPENAI
    assert b.model == "m"


def test_explicit_base_url_takes_the_model_from_dash_m():
    # `one -m model-x --base-url ...` resolves name=, not model=.
    b = resolve(name="model-x", base_url="http://127.0.0.1:9/v1")
    assert b.model == "model-x"
    assert resolve(name="n", model="m", base_url="http://127.0.0.1:9/v1").model == "m"


def test_backend_display_prefers_label():
    assert Backend(name="n", model="m", base_url="u", label="L").display == "L"
    assert Backend(name="n", model="m", base_url="u").display == "n (m)"


def test_anthropic_url_does_not_double_the_version_segment():
    assert AnthropicWire("http://h:1/v1", "m")._url() == "http://h:1/v1/messages"
    assert AnthropicWire("http://h:1", "m")._url() == "http://h:1/v1/messages"


def test_openai_url_adds_the_version_segment_once():
    assert OpenAIWire("http://h:1/v1", "m")._url() == "http://h:1/v1/chat/completions"
    assert OpenAIWire("http://h:1", "m")._url() == "http://h:1/v1/chat/completions"


def test_wire_constants_are_the_two_we_support():
    assert {WIRE_OPENAI, WIRE_ANTHROPIC} == {"openai", "anthropic"}


# --- streaming -------------------------------------------------------------
#
# The transport is stubbed with the raw SSE bytes an endpoint would write, so
# these exercise the frame folding rather than a mock's idea of it.

class _FakeResponse:
    def __init__(self, body: bytes):
        self._lines = body.splitlines(keepends=True)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_transport(monkeypatch, module, body: bytes, seen: dict | None = None):
    def fake_urlopen(req, timeout=None):
        if seen is not None:
            seen["body"] = json.loads(req.data.decode())
        return _FakeResponse(body)
    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)


def _sse(*frames: str) -> bytes:
    return "".join(f"data: {f}\n\n" for f in frames).encode() + b"data: [DONE]\n\n"


def _chunk(**delta) -> str:
    return json.dumps({"choices": [{"index": 0, "delta": delta}]})


def test_openai_stream_emits_text_deltas_and_accumulates_them(monkeypatch):
    from onecli.providers import openai_wire

    sent = {}
    _stub_transport(monkeypatch, openai_wire,
                    _sse(_chunk(content="he"), _chunk(content="llo")), sent)
    seen = []
    turn = OpenAIWire("http://h:1", "m").chat(
        [{"role": "user", "content": "hi"}], on_delta=lambda k, c: seen.append((k, c)))

    assert sent["body"]["stream"] is True
    assert seen == [("text", "he"), ("text", "llo")]
    assert turn.text == "hello"


def test_openai_stream_keeps_reasoning_out_of_the_text(monkeypatch):
    from onecli.providers import openai_wire

    _stub_transport(monkeypatch, openai_wire,
                    _sse(_chunk(reasoning="think"), _chunk(content="answer")))
    seen = []
    turn = OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: seen.append(k))
    assert turn.text == "answer"
    assert turn.thinking == "think"
    assert seen == ["thinking", "text"]


def test_openai_stream_assembles_a_tool_call_from_three_fragments(monkeypatch):
    """Arguments arrive as partial JSON; only the whole is parseable."""
    from onecli.providers import openai_wire

    _stub_transport(monkeypatch, openai_wire, _sse(
        _chunk(tool_calls=[{"index": 0, "id": "c1",
                            "function": {"name": "read", "arguments": '{"pa'}}]),
        _chunk(tool_calls=[{"index": 0, "function": {"arguments": 'th": "a'}}]),
        _chunk(tool_calls=[{"index": 0, "function": {"arguments": '.py"}'}}]),
    ))
    turn = OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "c1"
    assert turn.tool_calls[0].name == "read"
    assert turn.tool_calls[0].arguments == {"path": "a.py"}


def test_openai_stream_captures_finish_reason_and_usage(monkeypatch):
    from onecli.providers import openai_wire

    _stub_transport(monkeypatch, openai_wire, _sse(
        _chunk(content="x"),
        json.dumps({"choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2}}),
    ))
    turn = OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    assert turn.finish_reason == "stop"
    assert turn.usage == {"prompt_tokens": 7, "completion_tokens": 2}


def test_openai_streamed_turn_equals_the_non_streamed_turn(monkeypatch):
    """Same logical reply over both transports must reach the loop identically."""
    from onecli.providers import openai_wire

    _stub_transport(monkeypatch, openai_wire, _sse(
        _chunk(content="hel"), _chunk(content="lo", reasoning="r"),
        _chunk(tool_calls=[{"index": 0, "id": "c1",
                            "function": {"name": "ls", "arguments": '{"path": "."}'}}]),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {"total_tokens": 3}}),
    ))
    streamed = OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    whole = OpenAIWire._to_turn({
        "choices": [{"message": {
            "content": "hello", "reasoning_content": "r",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "ls", "arguments": '{"path": "."}'}}],
        }, "finish_reason": "tool_calls"}],
        "usage": {"total_tokens": 3},
    })
    for field in ("text", "thinking", "tool_calls", "finish_reason", "usage"):
        assert getattr(streamed, field) == getattr(whole, field), field


def test_openai_malformed_sse_line_raises_provider_error_with_the_payload(monkeypatch):
    from onecli.providers import openai_wire

    _stub_transport(monkeypatch, openai_wire, b'data: {"choices": [\n\n')
    with pytest.raises(ProviderError) as exc:
        OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    assert '{"choices": [' in str(exc.value)


def _anthropic_stream() -> bytes:
    return (
        b'event: message_start\n'
        b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}\n\n'
        b'data: {"type": "content_block_start", "index": 0,'
        b' "content_block": {"type": "thinking", "thinking": ""}}\n\n'
        b'data: {"type": "content_block_delta", "index": 0,'
        b' "delta": {"type": "thinking_delta", "thinking": "scratch"}}\n\n'
        b'data: {"type": "content_block_start", "index": 1,'
        b' "content_block": {"type": "text", "text": ""}}\n\n'
        b'data: {"type": "content_block_delta", "index": 1,'
        b' "delta": {"type": "text_delta", "text": "he"}}\n\n'
        b'data: {"type": "content_block_delta", "index": 1,'
        b' "delta": {"type": "text_delta", "text": "llo"}}\n\n'
        b'data: {"type": "content_block_start", "index": 2, "content_block":'
        b' {"type": "tool_use", "id": "t1", "name": "read", "input": {}}}\n\n'
        b'data: {"type": "content_block_delta", "index": 2,'
        b' "delta": {"type": "input_json_delta", "partial_json": "{\\"pa"}}\n\n'
        b'data: {"type": "content_block_delta", "index": 2,'
        b' "delta": {"type": "input_json_delta", "partial_json": "th\\": \\"a"}}\n\n'
        b'data: {"type": "content_block_delta", "index": 2,'
        b' "delta": {"type": "input_json_delta", "partial_json": ".py\\"}"}}\n\n'
        b'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"},'
        b' "usage": {"output_tokens": 9}}\n\n'
        b'data: {"type": "message_stop"}\n\n'
    )


def test_anthropic_stream_folds_the_event_stream_into_one_turn(monkeypatch):
    from onecli.providers import anthropic_wire

    sent = {}
    _stub_transport(monkeypatch, anthropic_wire, _anthropic_stream(), sent)
    seen = []
    turn = AnthropicWire("http://h:1", "m").chat(
        [{"role": "user", "content": "hi"}],
        on_delta=lambda k, c: seen.append((k, c)))

    assert sent["body"]["stream"] is True
    assert [(k, c) for k, c in seen if k != "tool"] == [
        ("thinking", "scratch"), ("text", "he"), ("text", "llo")]
    assert ''.join(c for k, c in seen if k == 'tool') == '{"path": "a.py"}'
    assert turn.text == "hello"
    assert turn.thinking == "scratch"
    assert turn.tool_calls[0].arguments == {"path": "a.py"}
    assert turn.finish_reason == "tool_use"
    # Plus the canonical fields every wire adds (providers.base.Turn).
    assert turn.usage == {"input_tokens": 5, "output_tokens": 9,
                          "prompt_tokens": 5, "cache_read_tokens": 0}


def test_anthropic_streamed_turn_equals_the_non_streamed_turn(monkeypatch):
    from onecli.providers import anthropic_wire

    _stub_transport(monkeypatch, anthropic_wire, _anthropic_stream())
    streamed = AnthropicWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    whole = AnthropicWire._to_turn({
        "content": [
            {"type": "thinking", "thinking": "scratch"},
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "t1", "name": "read",
             "input": {"path": "a.py"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 9},
    })
    for field in ("text", "thinking", "tool_calls", "finish_reason", "usage"):
        assert getattr(streamed, field) == getattr(whole, field), field


def test_anthropic_half_written_sse_line_raises_provider_error(monkeypatch):
    from onecli.providers import anthropic_wire

    _stub_transport(monkeypatch, anthropic_wire,
                    b'data: {"type": "content_block_del\n\n')
    with pytest.raises(ProviderError):
        AnthropicWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)


# --- the server failing to read its own model ------------------------------

class _ParserRefusal(urllib.error.HTTPError):
    """What llama.cpp answers when its tool-call parser rejects the model."""

    BODY = (b'{"error":{"code":500,"message":"The model produced output that '
            b'does not match the expected peg-native format",'
            b'"type":"server_error"}}')

    def __init__(self):
        super().__init__("http://h:1/v1/chat/completions", 500, "Internal",
                         {}, None)

    def read(self):
        return self.BODY


def _answer(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text},
                                    "finish_reason": "stop"}]}).encode()


class _RetiredSampling(urllib.error.HTTPError):
    """What a model that retired sampling parameters answers.

    The error names ONE parameter - and deliberately one the body does NOT
    carry. A retry that drops only the named field finds nothing to drop and
    gives up, even though removing the sampling parameters it DID send would
    have been accepted. That is the case this fixture exists to catch.
    """

    BODY = (b'{"error":{"code":400,"message":"INVALID_ARGUMENT: top_k is no '
            b'longer supported for this model","status":"INVALID_ARGUMENT"}}')

    def __init__(self):
        super().__init__("http://h:1/v1/chat/completions", 400, "Bad Request",
                         {}, None)

    def read(self):
        return self.BODY


class _JsonResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_parser_refusal_is_resampled_once(monkeypatch):
    """The request was fine - the server could not read its own model's reply.

    llama.cpp 500s when gpt-oss emits a harmony block its tool-call parser
    rejects, and the next sample of the same request usually parses. Failing
    the turn here would end an agent loop mid-task, and failing over to
    another backend would blame a different model for a local parser's bad
    minute.
    """
    from onecli.providers import openai_wire

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode()))
        if len(calls) == 1:
            raise _ParserRefusal()
        return _JsonResponse(_answer("second sample parsed"))

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    wire = OpenAIWire("http://h:1", "m")
    turn = wire.chat([{"role": "user", "content": "hi"}])

    assert turn.text == "second sample parsed"
    assert len(calls) == 2
    assert calls[0] == calls[1]          # the same request, resampled
    assert wire.output_retries == 1      # and counted, never hidden


def test_a_parser_refusal_is_resampled_only_once(monkeypatch):
    from onecli.providers import openai_wire

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise _ParserRefusal()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as caught:
        OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2
    assert "500" in str(caught.value)


def test_an_ordinary_500_is_not_resampled(monkeypatch):
    """A busy or broken endpoint is the fallback chain's job, not ours."""
    from onecli.providers import openai_wire

    class _Overloaded(_ParserRefusal):
        BODY = b'{"error":{"message":"model is overloaded"}}'

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise _Overloaded()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError):
        OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_a_parser_refusal_mid_stream_is_resampled_before_anything_is_printed(monkeypatch):
    from onecli.providers import openai_wire

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise _ParserRefusal()
        return _FakeResponse(_sse(_chunk(content="clean")))

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    seen = []
    wire = OpenAIWire("http://h:1", "m")
    turn = wire.chat([{"role": "user", "content": "hi"}],
                     on_delta=lambda k, c: seen.append(c))

    assert turn.text == "clean"
    assert seen == ["clean"]             # printed once, not twice
    assert wire.output_retries == 1


def test_a_stream_that_sends_nothing_is_a_transport_failure(monkeypatch):
    """llama.cpp closes the stream mid-run when a request overruns its n_ctx:
    no frames, no finish_reason. Handed back as an ordinary empty turn it ends
    the run; raised as a ProviderError the fallback chain carries it.
    """
    from onecli.providers import openai_wire

    _stub_transport(monkeypatch, openai_wire, b"data: [DONE]\n\n")
    with pytest.raises(ProviderError) as exc:
        OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    assert "context window" in str(exc.value)


def test_a_stream_cut_off_mid_answer_keeps_what_arrived(monkeypatch):
    """Only a stream that said NOTHING is a failure. Half a sentence and no
    finish_reason is a short answer, and throwing it away would lose work the
    model already did.
    """
    from onecli.providers import openai_wire

    _stub_transport(monkeypatch, openai_wire, _sse(_chunk(content="par")))
    turn = OpenAIWire("http://h:1", "m").chat([], on_delta=lambda k, c: None)
    assert turn.text == "par"


# --- transient transport failures ------------------------------------------

class _Status(urllib.error.HTTPError):
    """An HTTP failure with a chosen status and headers."""

    CODE = 500
    BODY = b'{"error":{"message":"boom"}}'
    HEADERS: dict = {}

    def __init__(self):
        super().__init__("http://h:1/v1/chat/completions", self.CODE,
                         "Failure", self.HEADERS, None)

    def read(self):
        return self.BODY


class _RateLimited(_Status):
    CODE = 429
    BODY = b'{"error":{"message":"rate limited"}}'


class _Unavailable(_Status):
    CODE = 503
    BODY = b'{"error":{"message":"service unavailable"}}'


def test_a_429_is_waited_out_rather_than_failed_over(monkeypatch):
    """Failing over on a rate limit changes the model AND can walk a rotation
    through every credential we own. One short local wait is cheaper."""
    from onecli.providers import openai_wire

    calls, slept = [], []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise _RateLimited()
        return _JsonResponse(_answer("after the wait"))

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_wire.time, "sleep", slept.append)
    wire = OpenAIWire("http://h:1", "m")
    turn = wire.chat([{"role": "user", "content": "hi"}])

    assert turn.text == "after the wait"
    assert len(calls) == 2
    assert wire.transient_retries == 1
    assert slept and slept[0] > 0


def test_a_503_is_retried_too(monkeypatch):
    from onecli.providers import openai_wire

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise _Unavailable()
        return _JsonResponse(_answer("ok"))

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_wire.time, "sleep", lambda _s: None)
    assert OpenAIWire("http://h:1", "m").chat(
        [{"role": "user", "content": "hi"}]).text == "ok"
    assert len(calls) == 2


def test_a_persistent_429_gives_up_and_names_the_status(monkeypatch):
    from onecli.providers import openai_wire

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise _RateLimited()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_wire.time, "sleep", lambda _s: None)
    with pytest.raises(ProviderError) as caught:
        OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1 + OpenAIWire.MAX_TRANSIENT_RETRIES
    assert "429" in str(caught.value)


def test_retry_after_is_honoured_but_capped(monkeypatch):
    """A gateway asking for an hour is asking us to hang."""
    from onecli.providers import openai_wire

    class _Patient(_RateLimited):
        HEADERS = {"Retry-After": "3600"}

    slept = []

    def fake_urlopen(req, timeout=None):
        raise _Patient()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_wire.time, "sleep", slept.append)
    with pytest.raises(ProviderError):
        OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    assert slept and max(slept) <= OpenAIWire.MAX_RETRY_AFTER


def test_a_404_is_never_retried(monkeypatch):
    """A 4xx other than 429 is the server saying the request itself is wrong."""
    from onecli.providers import openai_wire

    class _NotFound(_Status):
        CODE = 404

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise _NotFound()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError):
        OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_a_stream_is_not_replayed_after_a_token_reached_the_caller(monkeypatch):
    """Retrying a stream that already printed half a sentence shows the reader
    the answer twice - worse than the failure it is recovering from."""
    from onecli.providers import openai_wire

    calls = []

    class _DiesMidStream:
        def __init__(self):
            # No `data: [DONE]`: the reader breaks on it, and the point of
            # this stub is that the connection dies BEFORE the terminator.
            self._frames = [f"data: {_chunk(content='partial')}\n".encode(),
                            b"\n"]

        def __iter__(self):
            yield from self._frames
            raise _Unavailable()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        return _DiesMidStream()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_wire.time, "sleep", lambda _s: None)
    seen = []
    with pytest.raises(ProviderError):
        OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}],
                                           on_delta=lambda k, c: seen.append(c))
    assert len(calls) == 1
    assert seen == ["partial"]


def test_a_retired_sampling_param_drops_all_three_and_retries(monkeypatch):
    """A model that retired sampling parameters refuses the whole request.

    The error names one parameter, but the request is refused for carrying any
    of them, so dropping only the named field would spend a second 400 on the
    next one. All three go together.
    """
    from onecli.providers import openai_wire

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode()))
        if len(calls) == 1:
            raise _RetiredSampling()
        return _JsonResponse(_answer("answered without sampling params"))

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    turn = OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}], temperature=0.0)

    assert turn.text == "answered without sampling params"
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.0
    assert not {"temperature", "top_p", "top_k"} & set(calls[1])


def test_a_retired_sampling_param_is_not_dropped_when_the_dropin_pinned_it(monkeypatch):
    """An operator's ONE_PROVIDER_PARAMS entry is a claim about the endpoint.

    Silently overriding it would turn a deliberate setting into a mystery.
    """
    from onecli.providers import openai_wire

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise _RetiredSampling()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError):
        OpenAIWire("http://h:1", "m", extra_body={"temperature": 1}).chat(
            [{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_an_unrelated_400_is_not_retried_as_sampling(monkeypatch):
    from onecli.providers import openai_wire

    calls = []

    class _BadRequest(urllib.error.HTTPError):
        BODY = b'{"error":{"message":"messages: field required"}}'

        def __init__(self):
            super().__init__("http://h:1/v1/chat/completions", 400,
                             "Bad Request", {}, None)

        def read(self):
            return self.BODY

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise _BadRequest()

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError):
        OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_both_wires_send_a_user_agent():
    """urllib's default UA is a Cloudflare-blocked signature.

    Sending no User-Agent means urllib supplies `Python-urllib/3.x`, which
    Cloudflare answers with `403 error code: 1010` and no JSON body - measured
    against OpenCode Zen, where the identical request from curl
    returned 200. The header is therefore load-bearing on any endpoint behind
    Cloudflare, and easy to drop by accident because nothing else reads it.
    """
    oai = OpenAIWire("https://h/v1", "m", api_key="k")
    ant = AnthropicWire("https://h", "m", api_key="k")
    for headers in (oai._headers(), ant._headers()):
        ua = headers.get("User-Agent", "")
        assert ua, "no User-Agent: urllib will send its blocked default"
        assert not ua.lower().startswith("python-urllib")


@pytest.mark.parametrize("wire_class", [OpenAIWire, AnthropicWire])
@pytest.mark.parametrize("stream", [False, True])
def test_html_access_denial_is_compact_and_never_retried(monkeypatch, wire_class, stream):
    calls = []
    class Blocked(_Status):
        CODE = 403
        BODY = b'<!DOCTYPE html><html><head><style>PRIVATE_FONT_PAYLOAD</style></head><body>Blocked</body></html>'
    def denied(*args, **kwargs):
        calls.append(1)
        raise Blocked()
    monkeypatch.setattr("urllib.request.urlopen", denied)
    wire = wire_class("https://example.invalid/v1", "test")
    with pytest.raises(ProviderError) as caught:
        wire.chat([{"role": "user", "content": "hi"}],
                  on_delta=(lambda *args: None) if stream else None)
    message = str(caught.value)
    assert "HTTP 403" in message
    assert "endpoint or gateway" in message
    assert "PRIVATE_FONT_PAYLOAD" not in message
    assert "<html>" not in message
    assert len(calls) == 1


@pytest.mark.parametrize("detail", [
    '{"error":{"message":"response_format unsupported"}}',
    'Key limit exceeded (total limit)',
])
def test_http_error_display_preserves_api_error_semantics(detail):
    from onecli.providers.base import http_error_detail
    assert http_error_detail(detail) == detail


def test_json_quota_error_containing_html_retains_failover_semantics():
    from onecli.providers.base import http_error_detail, is_transient
    detail = '{"error":{"message":"Key limit exceeded (total limit); upstream returned <html>"}}'
    assert http_error_detail(detail) == detail
    assert is_transient("HTTP 403: " + http_error_detail(detail))


def test_no_temperature_is_sent_unless_configured(monkeypatch):
    """A hardcoded 0.0 is greedy decoding on every model; the provider's own
    default should apply unless the operator chose one."""
    from onecli import cli
    from onecli.providers import openai_wire
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode()))
        return _JsonResponse(_answer("ok"))

    monkeypatch.setattr(openai_wire.urllib.request, "urlopen", fake_urlopen)
    OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    OpenAIWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}], temperature=0.3)
    assert "temperature" not in calls[0]
    assert calls[1]["temperature"] == 0.3
    assert cli.build_parser().parse_args([]).temperature is None


def test_vertex_openapi_root_is_not_double_versioned():
    """Vertex's OpenAI-compatible root ends in `/openapi`, a different string
    from Google AI Studio's `/openai`. Missing it made the wire post to
    `.../endpoints/openapi/v1/chat/completions`, a 404 on every call."""
    from onecli.providers.openai_wire import OpenAIWire
    base = ("https://us-central1-aiplatform.googleapis.com/v1/projects/p/"
            "locations/us-central1/endpoints/openapi")
    assert OpenAIWire(base, "google/gemini-2.5-pro")._url() == base + "/chat/completions"


@pytest.mark.parametrize('content,reasoning,answer,thought', [
    ('5\n</think>\n\n', 'working', '5', 'working'),
    ('<think>working</think>OK', '', 'OK', 'working'),
    ('OK', '', 'OK', ''),
    ('Example: `<think>x</think>`', 'working', 'Example: `<think>x</think>`', 'working'),
    ('```xml\n</think>\n```', 'working', '```xml\n</think>\n```', 'working'),
    # Orphan-tag regression: a Markdown answer still loses the orphan tag that trails it, and an
    # orphan close tag that leads the content never reaches the user.
    ('Run `ls` first.\n</think>\n\n', 'working', 'Run `ls` first.', 'working'),
    ('</think>\n\nOK', 'working', 'OK', 'working'),
    ('```sh\nls\n```\n</think>\n\n', 'working', '```sh\nls\n```', 'working'),
    ('```\n</think>', 'working', '```\n</think>', 'working'),
])
def test_reasoning_tag_normalization_stream_and_nonstream(monkeypatch, content, reasoning, answer, thought):
    from onecli.providers import openai_wire
    data = {'choices': [{'message': {'content': content, 'reasoning_content': reasoning}}]}
    turn = OpenAIWire._to_turn(data)
    assert (turn.text, turn.thinking) == (answer, thought)
    # Every character is its own SSE frame: includes split tag boundaries.
    frames = ([_chunk(reasoning_content=reasoning)] if reasoning else [])
    frames += [_chunk(content=c) for c in content]
    _stub_transport(monkeypatch, openai_wire, _sse(*frames))
    seen = []
    streamed = OpenAIWire('http://h:1', 'm').chat([], on_delta=lambda k,c: seen.append((k,c)))
    assert (streamed.text, streamed.thinking) == (answer, thought)
    assert ''.join(c for k,c in seen if k == 'text') == answer
    assert ''.join(c for k,c in seen if k == 'thinking') == thought


def test_leading_close_tag_kept_without_a_reasoning_channel():
    from onecli.providers.openai_wire import _reasoning_content
    text = "</think> is the closing tag."
    assert _reasoning_content(text, False) == (text, "")
    assert _reasoning_content(text, True)[0] == "is the closing tag."

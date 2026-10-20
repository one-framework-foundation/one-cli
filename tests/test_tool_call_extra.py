"""Opaque per-call passthrough on tool calls.

Gemini 3 signs its reasoning and puts the signature beside the tool call as
`extra_content.google.thought_signature`. A follow-up request whose assistant
message dropped it is refused with HTTP 400, "Function call is missing a
thought_signature in functionCall parts", so the round trip is what makes a
multi-step loop possible on that backend at all. Nothing here is google-shaped:
the wire carries whatever extra keys arrived, and a wire that sent none must
produce byte-identical requests to before the field existed.
"""

import json

from onecli.agent import Agent
from onecli.providers.openai_wire import OpenAIWire

SIGNATURE = "EpoGCpcGAXLI2nx/exampleBase64Signature=="


def _response(tool_call: dict) -> dict:
    return {"choices": [{"message": {"content": "",
                                     "tool_calls": [tool_call]},
                         "finish_reason": "tool_calls"}]}


def _call(**extra) -> dict:
    return {"id": "call_1", "type": "function",
            "function": {"name": "read", "arguments": '{"path": "pyproject.toml"}'},
            **extra}


def test_extra_content_round_trips_onto_the_next_request():
    turn = OpenAIWire._to_turn(_response(
        _call(extra_content={"google": {"thought_signature": SIGNATURE}})))

    assert turn.tool_calls[0].extra == {
        "extra_content": {"google": {"thought_signature": SIGNATURE}}}

    message = Agent._assistant_message(turn)
    sent = message["tool_calls"][0]
    assert sent["extra_content"] == {"google": {"thought_signature": SIGNATURE}}
    # and the ordinary fields are untouched by carrying it
    assert sent["id"] == "call_1"
    assert sent["function"]["name"] == "read"
    assert json.loads(sent["function"]["arguments"]) == {"path": "pyproject.toml"}


def test_a_wire_that_sent_nothing_extra_gets_no_extra_key():
    turn = OpenAIWire._to_turn(_response(_call()))

    assert turn.tool_calls[0].extra == {}

    sent = Agent._assistant_message(turn)["tool_calls"][0]
    assert set(sent) == {"id", "type", "function"}, sent
    # byte-identical to the request shape kimi/dashscope/deepseek got before
    assert json.dumps(sent) == json.dumps({
        "id": "call_1", "type": "function",
        "function": {"name": "read",
                     "arguments": json.dumps({"path": "pyproject.toml"})}})


def test_streaming_fragments_keep_the_extra_field():
    fragments: dict[int, dict] = {}
    OpenAIWire._merge_tool_fragment(fragments, {
        "index": 0, "id": "call_1", "type": "function",
        "function": {"name": "read", "arguments": '{"path": '},
        "extra_content": {"google": {"thought_signature": SIGNATURE}}})
    OpenAIWire._merge_tool_fragment(fragments, {
        "index": 0, "function": {"arguments": '"pyproject.toml"}'}})

    turn = OpenAIWire._to_turn(_response(fragments[0]))
    call = turn.tool_calls[0]
    assert call.arguments == {"path": "pyproject.toml"}
    assert call.extra == {
        "extra_content": {"google": {"thought_signature": SIGNATURE}}}
    # `index` is streaming bookkeeping, not something to echo back
    assert "index" not in Agent._assistant_message(turn)["tool_calls"][0]


def test_a_streamed_call_without_extras_is_unchanged():
    fragments: dict[int, dict] = {}
    OpenAIWire._merge_tool_fragment(fragments, {
        "index": 0, "id": "call_1",
        "function": {"name": "read", "arguments": '{}'}})
    assert set(fragments[0]) == {"id", "type", "function"}


def test_openai_api_root_is_not_given_a_second_version_segment():
    wire = OpenAIWire("https://generativelanguage.googleapis.com/v1beta/openai",
                      "gemini-3.8-flash")
    assert wire._url() == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    # the ordinary hosts keep the behaviour they had
    assert OpenAIWire("http://localhost:8080", "m")._url() == \
        "http://localhost:8080/v1/chat/completions"
    assert OpenAIWire("http://localhost:8080/v1", "m")._url() == \
        "http://localhost:8080/v1/chat/completions"

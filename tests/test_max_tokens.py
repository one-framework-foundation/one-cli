"""No output ceiling unless one is configured.

A 4096-token default per call starved reasoning models before they wrote
anything visible. The OpenAI wire now
sends no max_tokens at all; the Anthropic wire, which requires one, sends the
model's own maximum as its Models API declares it.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from onecli.agent import Agent
from onecli.cli import build_parser
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.base import ProviderError, Turn
from onecli.providers.openai_wire import OpenAIWire


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(monkeypatch, replies):
    sent = []

    def urlopen(req, timeout=None):
        sent.append((req.get_method(), req.full_url,
                     json.loads(req.data) if req.data else None))
        return _Reply(json.dumps(replies[req.full_url.rsplit("/", 1)[-1]]).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return sent


def test_the_cli_sets_no_default_output_ceiling():
    assert build_parser().parse_args([]).max_tokens is None


def test_the_openai_wire_sends_no_max_tokens_unless_asked(monkeypatch):
    sent = _capture(monkeypatch, {"completions": {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}})
    OpenAIWire("http://h:1/v1", "m").chat([{"role": "user", "content": "hi"}])
    OpenAIWire("http://h:1/v1", "m").chat([{"role": "user", "content": "hi"}], max_tokens=50)
    assert "max_tokens" not in sent[0][2]
    assert sent[1][2]["max_tokens"] == 50


def test_the_anthropic_wire_sends_the_models_declared_maximum(monkeypatch):
    sent = _capture(monkeypatch, {
        "m": {"id": "m", "max_tokens": 128000},
        "messages": {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
    })
    wire = AnthropicWire("http://h:1", "m")
    wire.chat([{"role": "user", "content": "hi"}])
    wire.chat([{"role": "user", "content": "again"}])
    assert [(m, u) for m, u, _ in sent] == [
        ("GET", "http://h:1/v1/models/m"),
        ("POST", "http://h:1/v1/messages"),
        ("POST", "http://h:1/v1/messages"),
    ]
    assert sent[1][2]["max_tokens"] == 128000


def test_an_undeclared_maximum_is_named_not_guessed(monkeypatch):
    _capture(monkeypatch, {"m": {"id": "m"}, "models": {"data": [{"id": "m"}]}})
    with pytest.raises(ProviderError, match="set --max-tokens"):
        AnthropicWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])


def test_the_agent_passes_no_ceiling_by_default(tmp_path):
    seen = []

    class Provider:
        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            seen.append(max_tokens)
            return Turn(text="done")

    Agent(Provider(), str(tmp_path)).run("hi")
    assert seen == [None]


def test_llama_server_404_falls_back_to_the_listed_context(monkeypatch, capsys):
    """llama-server 404s /v1/models/<id> and declares no max_tokens; the list
    carries meta.n_ctx."""
    sent = []

    def urlopen(req, timeout=None):
        sent.append((req.get_method(), req.full_url,
                     json.loads(req.data) if req.data else None))
        tail = req.full_url.rsplit("/", 1)[-1]
        if tail == "m":
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        body = ({"object": "list", "data": [
                    {"id": "other", "meta": {"n_ctx": 4096}},
                    {"id": "m", "object": "model",
                     "meta": {"n_ctx": 131072, "n_ctx_train": 262144}}]}
                if tail == "models" else
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})
        return _Reply(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    AnthropicWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])
    assert [(m, u) for m, u, _ in sent] == [
        ("GET", "http://h:1/v1/models/m"),
        ("GET", "http://h:1/v1/models"),
        ("POST", "http://h:1/v1/messages"),
    ]
    assert sent[2][2]["max_tokens"] == 131072
    assert "meta.n_ctx = 131072" in capsys.readouterr().err


def test_an_unlisted_model_still_asks_for_max_tokens(monkeypatch):
    _capture(monkeypatch, {"m": {"id": "m"}, "models": {"data": [{"id": "x", "meta": {"n_ctx": 8}}]}})
    with pytest.raises(ProviderError, match="does not list m.*set --max-tokens"):
        AnthropicWire("http://h:1", "m").chat([{"role": "user", "content": "hi"}])

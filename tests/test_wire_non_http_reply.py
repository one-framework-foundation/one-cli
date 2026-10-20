"""A model request answered by something that is not HTTP is a transport error.

A loopback listener that passes /health and /v1/models but answers the chat
request with a non-HTTP reply raised http.client.BadStatusLine out of the
wire; a reply cut mid-body raised IncompleteRead, streaming or not. Neither is
an OSError, so the fallback chain never advanced and the one-chat REPL died.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from onecli import cli, config
from onecli._vendor import dropins
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.base import ProviderError, StreamTruncated, is_transient
from onecli.providers.openai_wire import OpenAIWire
from onecli.serving import pull, tune

MODELS = json.dumps({"data": [{"id": "m"}]}).encode()

GARBAGE = b"SSH-2.0-not-http\r\n\r\n"
# Promises 1000 bytes, sends a few, hangs up.
CUT = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
       b"Content-Length: 1000\r\n\r\n{\"choi")
# A chunked event stream whose first chunk is cut short.
CUT_STREAM = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
              b"Transfer-Encoding: chunked\r\n\r\n"
              b"200\r\ndata: {\"choices\": [{\"delta\": {\"content\": \"hal")


def _ok(body: bytes) -> bytes:
    return (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n" + body)


class Server:
    """Answers /health and /v1/models as HTTP; every other path with `reply`."""

    def __init__(self, reply: bytes, models: bytes | None = MODELS):
        self.reply = reply
        self.models = models
        self.paths: list[str] = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    path = self._read_request(conn)
                    self.paths.append(path)
                    if path == "/health":
                        conn.sendall(_ok(b'{"status": "ok"}'))
                    elif path == "/v1/models" and self.models is not None:
                        conn.sendall(_ok(self.models))
                    else:
                        conn.sendall(self.reply)
                except OSError:
                    pass

    @staticmethod
    def _read_request(conn) -> str:
        # Read the whole request, so the client's send never meets a reset
        # (which is an OSError and would pass without the fix).
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return ""
            data += chunk
        head, _, body = data.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value.strip())
        while len(body) < length:
            chunk = conn.recv(65536)
            if not chunk:
                break
            body += chunk
        return head.split(b" ")[1].decode()

    def close(self):
        self.sock.close()


@pytest.fixture
def serve():
    servers = []

    def start(reply, **kw):
        servers.append(Server(reply, **kw))
        return servers[-1]
    yield start
    for s in servers:
        s.close()


def _unvalidated(monkeypatch):
    from onecli import config
    monkeypatch.setattr(config, "validate_model", lambda *a, **k: None)


MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("reply", [GARBAGE, CUT], ids=["bad-status", "cut-body"])
def test_openai_wire_reports_a_non_http_reply_as_transient(serve, reply, monkeypatch):
    _unvalidated(monkeypatch)
    wire = OpenAIWire(serve(reply).url + "/v1", "m", timeout=5)
    with pytest.raises(ProviderError) as info:
        wire.chat(MESSAGES)
    assert is_transient(info.value)


@pytest.mark.parametrize("reply", [GARBAGE, CUT_STREAM], ids=["bad-status", "cut-stream"])
def test_openai_wire_stream_reports_a_non_http_reply_as_transient(serve, reply, monkeypatch):
    _unvalidated(monkeypatch)
    wire = OpenAIWire(serve(reply).url + "/v1", "m", timeout=5)
    with pytest.raises(ProviderError) as info:
        wire.chat(MESSAGES, on_delta=lambda kind, chunk: None)
    assert is_transient(info.value)


@pytest.mark.parametrize("reply", [GARBAGE, CUT], ids=["bad-status", "cut-body"])
def test_anthropic_wire_reports_a_non_http_reply_as_transient(serve, reply, monkeypatch):
    _unvalidated(monkeypatch)
    wire = AnthropicWire(serve(reply).url, "m", timeout=5)
    with pytest.raises(ProviderError) as info:
        wire.chat(MESSAGES, max_tokens=16)
    assert is_transient(info.value)


@pytest.mark.parametrize("reply", [GARBAGE, CUT_STREAM], ids=["bad-status", "cut-stream"])
def test_anthropic_wire_stream_reports_a_non_http_reply_as_transient(serve, reply, monkeypatch):
    _unvalidated(monkeypatch)
    wire = AnthropicWire(serve(reply).url, "m", timeout=5)
    with pytest.raises(ProviderError) as info:
        wire.chat(MESSAGES, max_tokens=16, on_delta=lambda kind, chunk: None)
    assert is_transient(info.value)


def test_anthropic_output_maximum_probes_survive_a_non_http_reply(serve):
    # Both the per-model route and the list answer garbage: the probe reports
    # that it cannot read the maximum instead of crashing.
    wire = AnthropicWire(serve(GARBAGE, models=None).url, "m", timeout=5)
    with pytest.raises(ProviderError, match="cannot read the output maximum"):
        wire.declared_max_output()


def test_tune_measure_records_a_non_http_reply_as_a_sample_error(serve):
    result = tune.measure(serve(GARBAGE).url, "m", warmup=True,
                          tasks=[{"name": "t", "prompt": "p"}])
    assert result.samples[0].error


def test_pull_reports_a_non_http_reply_as_a_pull_error(serve, tmp_path):
    with pytest.raises(pull.PullError):
        pull.fetch(serve(GARBAGE).url + "/model.gguf", tmp_path / "model.gguf",
                   resume=False)


@pytest.mark.parametrize("make", [
    lambda url: (OpenAIWire(url + "/v1", "m", timeout=5), {}),
    lambda url: (AnthropicWire(url, "m", timeout=5), {"max_tokens": 16}),
], ids=["openai", "anthropic"])
def test_a_stream_cut_mid_body_takes_the_truncation_path(serve, make, monkeypatch):
    _unvalidated(monkeypatch)
    server = serve(CUT_STREAM)
    wire, extra = make(server.url)
    with pytest.raises(StreamTruncated) as info:
        wire.chat(MESSAGES, on_delta=lambda kind, chunk: None, **extra)
    assert is_transient(info.value)
    # Re-sent while nothing had been shown, as any truncated stream is.
    assert len([p for p in server.paths if p.endswith(("/chat/completions", "/messages"))]) == 3


@pytest.fixture
def loopback_only(monkeypatch):
    real = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        if host not in ("127.0.0.1", "localhost"):
            raise OSError(f"test refused a network lookup of {host}")
        return real(host, *args, **kwargs)
    monkeypatch.setattr(socket, "getaddrinfo", guarded)


ANSWER = _ok(json.dumps({"choices": [{"message": {"content": "fine"},
                                      "finish_reason": "stop"}]}).encode())


def test_one_falls_back_past_a_non_http_model_reply(serve, loopback_only, monkeypatch,
                                                    tmp_path, capsys):
    bad, good = serve(GARBAGE), serve(ANSWER)
    (tmp_path / "bad.env").write_text(f"ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL={bad.url}/v1\n")
    (tmp_path / "good.env").write_text(f"ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL={good.url}/v1\n")
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path)
    monkeypatch.delenv(config.PROVIDER_DIR_ENV, raising=False)
    monkeypatch.setattr(config, "llamacpp_backends", lambda *a, **k: [])
    monkeypatch.chdir(tmp_path)
    dropins._PRESENCE_CACHE.clear()
    try:
        code = cli.main(["-m", "bad", "--fallback-model", "good",
                         "--response-gate", "off", "-p", "hi"])
    finally:
        dropins.await_presence_priming(5)
        dropins._PRESENCE_CACHE.clear()
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "fine" in captured.out
    assert "/v1/chat/completions" in bad.paths and "/v1/chat/completions" in good.paths

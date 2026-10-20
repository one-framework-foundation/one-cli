"""Strict admissions cannot silently follow HTTP redirects to another route."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from onecli.providers.base import ProviderError
from onecli.providers.openai_wire import OpenAIWire


@pytest.fixture
def redirect_server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.do_GET()

        def do_GET(self):
            calls.append((self.command, self.path))
            if self.path.startswith("/redirect/"):
                self.send_response(int(self.path.split("/")[2]))
                self.send_header("Location", "/target")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("stream", [False, True])
def test_strict_redirect_never_reaches_target(redirect_server, status, stream):
    base, calls = redirect_server
    wire = OpenAIWire(f"{base}/redirect/{status}", "model", timeout=2)
    wire.reject_redirects = True
    admitted = []
    wire.request_gate = lambda body: admitted.append(body)
    with pytest.raises(ProviderError, match="redirect denied"):
        wire.chat([{"role": "user", "content": "test"}],
                  on_delta=(lambda *args: None) if stream else None)
    assert len(admitted) == 1
    assert calls == [("POST", f"/redirect/{status}/v1/chat/completions")]


def test_default_transport_retains_redirect_behavior(redirect_server):
    base, calls = redirect_server
    wire = OpenAIWire(f"{base}/redirect/302", "model", timeout=2)
    assert wire.chat([{"role": "user", "content": "test"}]).text == "ok"
    assert calls == [("POST", "/redirect/302/v1/chat/completions"),
                     ("GET", "/target")]

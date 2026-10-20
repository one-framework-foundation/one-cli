"""Credentials never follow a redirect to another origin (v0.1.1 safety review).

Receiver-verified: the second server records what actually arrived.
"""
from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from onecli import config, netio, providers
from onecli.providers.anthropic_wire import AnthropicWire

# Configured extras: some named like credentials, one (`x-opencode-tenant`)
# not, which no name list could have caught.
CONFIGURED = {"Cookie": "session=secret", "Api-Key": "secret",
              "X-Auth-Token": "secret", "X-Opencode-Tenant": "secret"}
SENT = {"authorization", "proxy-authorization", "x-api-key", "anthropic-version",
        *(name.lower() for name in CONFIGURED)}


def _servers():
    seen: dict = {}

    class Target(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.update({k.lower(): v for k, v in self.headers.items()})
            body = json.dumps({"max_tokens": 4096}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    target = http.server.HTTPServer(("127.0.0.1", 0), Target)

    class Redirector(Target):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}{self.path}")
            self.end_headers()

    start = http.server.HTTPServer(("127.0.0.1", 0), Redirector)
    for server in (target, start):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return start, target, seen


@pytest.fixture
def redirect_pair():
    start, target, seen = _servers()
    yield f"http://127.0.0.1:{start.server_port}", seen
    for server in (start, target):
        server.shutdown()
        server.server_close()


def _assert_no_credentials(seen):
    assert seen, "the second origin was never reached"
    assert seen.get("user-agent") == "probe"
    leaked = SENT & set(seen)
    assert not leaked, f"second origin received {sorted(leaked)}"


def test_netio_strips_configured_credentials_cross_origin(redirect_pair):
    base, seen = redirect_pair
    req = urllib.request.Request(base + "/x", headers={
        **CONFIGURED, "User-Agent": "probe", "Authorization": "Bearer secret",
        "Proxy-Authorization": "Basic secret", "x-api-key": "secret"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()
    _assert_no_credentials(seen)


def test_anthropic_metadata_request_strips_credentials_cross_origin(redirect_pair):
    base, seen = redirect_pair
    wire = AnthropicWire(base, "m", api_key="secret", timeout=5,
                         extra_headers={**CONFIGURED, "User-Agent": "probe"})
    assert wire.declared_max_output() == 4096
    _assert_no_credentials(seen)


def test_drop_in_headers_do_not_follow_cross_origin(redirect_pair, tmp_path):
    """The real config path: a drop-in's ONE_PROVIDER_HEADERS on the metadata GET."""
    base, seen = redirect_pair
    headers = json.dumps({**CONFIGURED, "User-Agent": "probe"})
    dropin = tmp_path / "probe.env"
    dropin.write_text(f"ANTHROPIC_BASE_URL={base}\nONE_PROVIDER_MODEL=m\n"
                      f"ANTHROPIC_API_KEY=secret\nONE_PROVIDER_HEADERS='{headers}'\n")
    backend = config._backend_from_env(dropin)
    assert backend is not None and backend.headers["X-Opencode-Tenant"] == "secret"
    wire = providers.build(backend, timeout=5)
    assert wire.declared_max_output() == 4096
    _assert_no_credentials(seen)


def test_same_origin_redirect_keeps_headers():
    req = urllib.request.Request("http://h/a", headers=CONFIGURED)
    new = netio.StripCredentialsOnRedirect().redirect_request(
        req, None, 302, "Found", {}, "http://h:80/b")
    assert {k.lower() for k in new.headers} == {k.lower() for k in CONFIGURED}


def test_https_to_http_redirect_is_refused():
    req = urllib.request.Request("https://example.test/a", headers={"X-Keep": "1"})
    with pytest.raises(urllib.error.HTTPError, match="https to http"):
        netio.StripCredentialsOnRedirect().redirect_request(
            req, None, 302, "Found", {}, "http://example.test/b")

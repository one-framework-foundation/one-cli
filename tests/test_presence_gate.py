"""An offline profile is shown as offline and cannot be chosen; one that comes
back can be chosen again without a restart.

The endpoint is a real HTTP server on 127.0.0.1 that is stopped and started on
the same port, so the check runs through runner's reader exactly as it does
against a remote machine.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading

import pytest

from onecli import cli, config, presence
from onecli._vendor import dropins
from onecli.model_picker import ModelPicker


class _Models(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        body = json.dumps({"data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Rig:
    """A model server that can be switched off and on at one address."""

    def __init__(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.srv = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def up(self):
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), _Models)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def down(self):
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()
            self.srv = None


@pytest.fixture
def rig(monkeypatch, tmp_path):
    dropins._PRESENCE_CACHE.clear()
    monkeypatch.setattr(presence, "SELECT_MAX_AGE", 0.0)
    r = Rig()
    (tmp_path / "rig.env").write_text(
        f"ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL={r.base}\n")
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path)
    monkeypatch.delenv(config.PROVIDER_DIR_ENV, raising=False)
    monkeypatch.setattr(config, "llamacpp_backends", lambda *a, **k: [])
    yield r
    r.down()
    dropins.await_presence_priming(5)
    dropins._PRESENCE_CACHE.clear()


def _backend():
    return config.resolve(name="rig")


class Args:
    json = True


def _listed(capsys):
    assert cli.cmd_list(Args()) == 0
    return {r["name"]: r["status"] for r in json.loads(capsys.readouterr().out)}


def test_list_reports_the_live_status(rig, capsys):
    assert _listed(capsys) == {"rig": presence.OFFLINE}
    rig.up()
    dropins._PRESENCE_CACHE.clear()     # a new `one --list` is a new process
    assert _listed(capsys) == {"rig": presence.ONLINE}


def test_an_offline_profile_is_refused_and_selectable_once_back(rig):
    b = _backend()
    assert "is offline" in presence.refusal(b)
    rig.up()
    # Same process, no restart: the down verdict is still in runner's cache
    # (PRESENCE_DOWN_TTL), and the selection must look again anyway.
    assert presence.refusal(b) == ""


def test_the_picker_refuses_an_offline_model_until_it_returns(rig):
    b = _backend()
    picker = ModelPicker([b], b)
    picker.choose("1")
    choice, done, text = picker.choose("1")
    assert choice is None and not done and "is offline" in text
    assert "[offline]" in picker.render()
    rig.up()
    choice, done, _ = picker.choose("1")
    assert choice is b and done


def test_dash_m_fails_fast_on_an_offline_profile(rig, capsys):
    assert cli.main(["-m", "rig", "hello"]) == 2
    assert "rig is offline" in capsys.readouterr().err


def test_a_paid_lane_is_never_dialled_and_stays_selectable():
    paid = config.Backend(name="v", model="m", base_url="https://api.example.com/v1")
    assert presence.status(paid, fresh=True) == presence.UNCHECKED
    assert presence.refusal(paid) == ""


def test_a_free_anthropic_lane_is_checked_too(rig):
    """An Anthropic-wire endpoint is checked the same way; down is down."""
    b = config.Backend(name="a", model="m", base_url=f"http://127.0.0.1:{rig.port}",
                       wire="anthropic")
    assert "is offline" in presence.refusal(b)
    rig.up()                      # answers /health with 404: the host is serving
    assert presence.refusal(b) == ""


def test_no_reader_means_unknown_not_hidden(monkeypatch, rig):
    monkeypatch.setattr(presence, "_reader", lambda: None)
    b = _backend()
    assert presence.status(b, fresh=True) == presence.UNKNOWN
    assert presence.refusal(b) == ""

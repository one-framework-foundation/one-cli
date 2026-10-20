"""A listener that does not speak HTTP is an unreachable backend, not a crash.

http.client raises HTTPException subclasses (BadStatusLine, IncompleteRead)
that are not OSError, so probes catching only OSError/URLError let them escape:
`one --list` with ONE_LLAMACPP_HOSTS pointing at such a port exited 1 with a
traceback.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading

import pytest

from onecli import cli, config, lanehealth, onboard, presence, update
from onecli._vendor import dropins


class Garbage:
    """Accepts a connection, answers with a non-HTTP banner, hangs up."""

    def __init__(self, reply: bytes = b"SSH-2.0-not-http\r\n\r\n"):
        self.reply = reply
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.recv(65536)
                    conn.sendall(self.reply)
                except OSError:
                    pass

    def close(self):
        self.sock.close()


@pytest.fixture
def garbage():
    g = Garbage()
    yield g
    g.close()


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    dropins._PRESENCE_CACHE.clear()
    dropins._RELAY_CACHE.clear()
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path / "providers")
    monkeypatch.delenv(config.PROVIDER_DIR_ENV, raising=False)
    monkeypatch.delenv("ONECLI_NO_PRESENCE", raising=False)
    real = config.llamacpp_backends
    # The default local port belongs to the machine running the tests.
    monkeypatch.setattr(config, "llamacpp_backends",
                        lambda base_url=config.LLAMACPP_DEFAULT, prefix="llamacpp":
                        [] if base_url == config.LLAMACPP_DEFAULT else real(base_url, prefix))
    yield tmp_path
    dropins.await_presence_priming(5)
    dropins._PRESENCE_CACHE.clear()
    dropins._RELAY_CACHE.clear()


def test_list_survives_a_non_http_llamacpp_host(garbage, isolated, monkeypatch, capsys):
    providers = isolated / "providers"
    providers.mkdir()
    (providers / "odd.env").write_text(
        f"ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL={garbage.url}/v1\n")
    monkeypatch.setenv(config.LLAMACPP_HOSTS_ENV, garbage.url)
    assert cli.main(["--list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in rows] == ["odd"]  # the llama host contributed none
    assert rows[0]["status"] == presence.OFFLINE
    assert "not answering" in rows[0]["reason"]


def test_every_probe_treats_a_non_http_answer_as_unreachable(garbage, isolated):
    assert config._http_json(garbage.url + "/v1/models", timeout=2) is None
    assert config.llamacpp_backends(garbage.url) == []
    assert dropins._http_json(garbage.url + "/models", 2) == (None, "conn")
    assert dropins.relay_served(garbage.url + "/v1") is None
    assert onboard._get_json(garbage.url + "/v1/models") is None
    assert lanehealth._fetch_openrouter_key(garbage.url, "k") is None
    backend = config.Backend(name="odd", model="m", base_url=garbage.url + "/v1")
    assert onboard.probe(backend, timeout=2).startswith("unreachable")


def test_update_check_reports_a_non_http_answer_as_could_not_ask(garbage, monkeypatch):
    monkeypatch.setattr(update, "PYPI_URL", garbage.url + "/pypi/one-core-cli/json")
    with pytest.raises(OSError):
        update.fetch_latest(timeout=2)


def test_bad_status_line_is_not_an_oserror():
    # The premise: without an explicit HTTPException clause these escape.
    assert not issubclass(http.client.BadStatusLine, OSError)
    assert not issubclass(http.client.IncompleteRead, OSError)

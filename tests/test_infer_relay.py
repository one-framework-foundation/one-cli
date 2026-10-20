"""The relay's /v1/infer route: auth, disabled-by-default, existing routes intact."""

import json

import pytest
from fastapi.testclient import TestClient

from onecli import infer
from onecli.providers.base import Turn
from onecli.relay import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_TOKENS", {"tok-host-a": "host:host-a"})
    monkeypatch.setattr(infer, "_ENGINE", None)
    return TestClient(server.app)


def _body():
    return {"kind": "classify.mask", "caller": "host:host-a",
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "budget": {"max_input_tokens": 100, "max_output_tokens": 10, "deadline_ms": 2000}}


def test_infer_requires_bearer(client):
    assert client.post("/v1/infer", json=_body()).status_code == 401
    r = client.post("/v1/infer", json=_body(), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 403


def test_infer_disabled_without_config(client, tmp_path, monkeypatch):
    monkeypatch.setenv(infer.CONFIG_ENV, str(tmp_path / "missing.json"))
    r = client.post("/v1/infer", json=_body(), headers={"Authorization": "Bearer tok-host-a"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "not_configured"
    h = client.get("/v1/infer/health", headers={"Authorization": "Bearer tok-host-a"})
    assert h.json() == {"enabled": False}


def test_invalid_config_is_503_not_crash(client, tmp_path, monkeypatch):
    bad = tmp_path / "infer.json"
    bad.write_text(json.dumps({"version": 1, "lanes": [], "callers": {}}))
    monkeypatch.setenv(infer.CONFIG_ENV, str(bad))
    r = client.post("/v1/infer", json=_body(), headers={"Authorization": "Bearer tok-host-a"})
    assert r.status_code == 503
    assert client.get("/health").status_code == 200


def test_infer_end_to_end_through_route(client, tmp_path, monkeypatch):
    cfg = infer.parse_config({
        "version": 1,
        "lanes": [{"name": "l1", "slots": 1, "locality": "private", "cost": "free"}],
        "callers": {"host:host-a": {"kinds": ["classify.mask"]}}})

    class Wire:
        def chat(self, messages, tools=None, **kw):
            assert tools is None
            return Turn(text="code", usage={"prompt_tokens": 3, "completion_tokens": 1})

    eng = infer.Engine(cfg, wire_factory=lambda n, t: (Wire(), "m1"),
                       ledger_path=tmp_path / "ledger.jsonl")
    monkeypatch.setattr(infer, "_ENGINE", eng)
    r = client.post("/v1/infer", json=_body(), headers={"Authorization": "Bearer tok-host-a"})
    assert r.status_code == 200, r.text
    assert r.headers["x-onecli-relay-served-by"] == "l1"
    assert r.json()["output"] == "code"
    # The token owner is the identity; the body cannot claim another host.
    spoof = dict(_body(), caller="host:host-b")
    r = client.post("/v1/infer", json=spoof, headers={"Authorization": "Bearer tok-host-a"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "caller_mismatch"


def test_existing_chat_route_unchanged(client, monkeypatch):
    class FakeRelay:
        rungs = []
        warnings = []

        def serve(self, messages, max_tokens, temperature):
            class B:
                model = "m"
            return "rung1", B(), Turn(text="hello"), []

    monkeypatch.setattr(server, "RELAY", FakeRelay())
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer tok-host-a"})
    assert r.status_code == 200
    assert r.headers["x-onecli-relay-served-by"] == "rung1"
    assert r.json()["choices"][0]["message"]["content"] == "hello"


@pytest.mark.parametrize("path", ["/v1/infer", "/v1/messages", "/v1/chat/completions"])
@pytest.mark.parametrize("authorization", ["Bearer ", "Bearer   ", "Bearer \t"])
def test_empty_bearer_is_rejected_without_crash(client, monkeypatch, path, authorization):
    def forbidden(*args, **kwargs):
        pytest.fail("unauthenticated request reached a provider")
    monkeypatch.setattr(server.RELAY, "serve", forbidden)
    response = client.post(path, json={}, headers={"Authorization": authorization})
    assert response.status_code == 401

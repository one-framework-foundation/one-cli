"""The gate that keeps a turn off a lane whose key is spent.

A fallback onto an OpenRouter drop-in whose key reports `limit: 0` must not
hand the user `HTTP 403 ... Key limit exceeded (total limit)` raw: a 403 of
that kind fails over, and a known-dead lane is left out of the chain.
"""

from __future__ import annotations

import json
import time

import pytest

from onecli import config, fallback, lanehealth
from onecli.providers.base import ProviderError, is_transient

OPENROUTER_403 = ('HTTP 403 from https://openrouter.ai/api/v1/chat/completions: '
                  '{"error":{"message":"Key limit exceeded (total limit)","code":403}}')


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv(lanehealth.STATE_ENV, str(tmp_path / "lane_health.json"))


def be(name, url, profile="default", model="m", api_key=""):
    return config.Backend(name=name, model=model, base_url=url, profile=profile,
                          api_key=api_key)


def _build(b, timeout=1.0):
    return f"provider:{b.name}"


# --- the classification half --------------------------------------------------

def test_a_spent_key_is_a_failover_reason_not_a_bad_request():
    """A key that reports exhausted quota is a failover reason, not a bad request."""
    assert lanehealth.is_quota_exhausted(OPENROUTER_403)
    assert is_transient(ProviderError(OPENROUTER_403))


def test_an_ordinary_bad_request_still_is_not():
    assert not is_transient(ProviderError("HTTP 400: unknown parameter 'foo'"))
    assert not lanehealth.is_quota_exhausted("HTTP 401: invalid api key")


# --- the reactive half --------------------------------------------------------

def test_a_refusal_opens_a_window_and_an_answer_closes_it():
    assert not lanehealth.is_cooling("router-lane-a")
    lanehealth.mark_down("router-lane-a", OPENROUTER_403)
    assert lanehealth.is_cooling("router-lane-a")
    lanehealth.mark_up("router-lane-a")
    assert not lanehealth.is_cooling("router-lane-a")


def test_the_backoff_doubles_and_a_spent_key_starts_at_the_hour_floor():
    now = time.time()
    first = lanehealth.mark_down("busy", "HTTP 503 service unavailable")
    assert lanehealth.BACKOFF_BASE_S - 5 <= first - now <= lanehealth.BACKOFF_BASE_S + 5
    second = lanehealth.mark_down("busy", "HTTP 503 service unavailable")
    assert second - now >= 2 * lanehealth.BACKOFF_BASE_S - 5

    spent = lanehealth.mark_down("dead-key", OPENROUTER_403)
    assert spent - now >= lanehealth.QUOTA_FLOOR_S - 5


def test_the_window_expires_on_its_own_with_no_polling():
    lanehealth.mark_down("blip", "HTTP 503 service unavailable")
    data = json.loads(lanehealth.state_path().read_text())
    data["lanes"]["blip"]["until"] = time.time() - 1
    lanehealth.state_path().write_text(json.dumps(data))
    assert not lanehealth.is_cooling("blip")


# --- the gate on chain construction -------------------------------------------

def test_a_cooling_lane_is_left_off_the_chain(monkeypatch):
    """A spare whose key 403s must not be left on the ladder."""
    primary = be("primary-lane", "https://ollama.com/v1")
    spare = be("router-lane-a", "https://openrouter.ai/api/v1")
    healthy = be("kimi", "https://api.moonshot.ai/v1")
    monkeypatch.setattr(fallback, "resolve_backends",
                        lambda names: ([spare, healthy], []))
    lanehealth.mark_down("router-lane-a", OPENROUTER_403)

    chain = fallback.build(primary, ["router-lane-a", "kimi"],
                           build_provider=_build)
    assert chain.rungs[0].name == "primary-lane"
    assert chain.spares == ["kimi"]
    assert any("router-lane-a" in w for w in chain.warnings)


def test_the_primary_is_never_filtered_out_from_under_the_operator():
    """`-m dead-lane` must still go there: naming a lane is an instruction."""
    primary = be("router-lane-a", "https://openrouter.ai/api/v1")
    lanehealth.mark_down("router-lane-a", OPENROUTER_403)
    chain = fallback.build(primary, [], build_provider=_build)
    assert chain.current.name == "router-lane-a"


# --- the proactive half -------------------------------------------------------

def test_an_openrouter_key_with_no_limit_left_is_excluded_before_it_is_tried(monkeypatch):
    """A key that reports limit 0 and limit_remaining 0 has nothing left."""
    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key",
                        lambda url, key: {"limit": 0, "limit_remaining": 0, "usage": 0})
    b = be("router-lane-a", "https://openrouter.ai/api/v1", api_key="sk-or-x")
    assert lanehealth.backend_unavailable(b) == "quota exhausted"
    monkeypatch.setattr(fallback, "resolve_backends", lambda names: ([b], []))
    chain = fallback.build(be("primary-lane", "https://ollama.com/v1"),
                           ["router-lane-a"], build_provider=_build)
    assert chain.spares == []


def test_an_unmetered_key_and_one_with_budget_are_both_fine(monkeypatch):
    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key",
                        lambda url, key: {"limit": None, "usage": 12.0})
    assert not lanehealth.openrouter_exhausted("https://openrouter.ai/api/v1",
                                               "sk-or-x", "unmetered")
    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key",
                        lambda url, key: {"limit": 10, "limit_remaining": 4.0})
    assert not lanehealth.openrouter_exhausted("https://openrouter.ai/api/v1",
                                               "hasbudget", "hasbudget")


def test_an_unreachable_key_endpoint_does_not_condemn_the_lane(monkeypatch):
    """Exclude on evidence, never on a failure to gather it."""
    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key", lambda url, key: None)
    b = be("router-lane-a", "https://openrouter.ai/api/v1", api_key="sk-or-x")
    assert lanehealth.backend_unavailable(b) == ""


def test_the_precheck_is_asked_once_per_ttl_not_once_per_call(monkeypatch):
    calls = []

    def fake(url, key):
        calls.append(url)
        return {"limit": 10, "limit_remaining": 5.0}

    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key", fake)
    for _ in range(4):
        lanehealth.openrouter_exhausted("https://openrouter.ai/api/v1", "k", "cached")
    assert len(calls) == 1


def test_only_openrouter_is_prechecked(monkeypatch):
    def boom(url, key):
        raise AssertionError("no other vendor should be prechecked")

    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key", boom)
    assert lanehealth.backend_unavailable(
        be("kimi", "https://api.moonshot.ai/v1", api_key="sk-x")) == ""

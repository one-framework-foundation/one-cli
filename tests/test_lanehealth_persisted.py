"""Malformed local ledgers must not discard a neighboring valid cooldown."""

import json

import pytest

from onecli import lanehealth


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    monkeypatch.setenv(lanehealth.STATE_ENV, str(path))
    monkeypatch.setattr(lanehealth.time, "time", lambda: 1000.0)

    def no_network(*args, **kwargs):
        pytest.fail("persisted health checks must not contact a provider")

    monkeypatch.setattr(lanehealth.urllib.request, "urlopen", no_network)
    return path


def write_ledger(path, lanes):
    path.write_text(json.dumps({"lanes": lanes, "extra": {"preserve": True}}))


@pytest.mark.parametrize("lanes", [None, [], ["lane"], "lane", 1, True])
def test_invalid_lane_container_can_be_read_and_repaired(ledger, lanes):
    write_ledger(ledger, lanes)
    assert lanehealth.cooling_until("bad") == 0.0
    assert lanehealth.status() == {}
    assert lanehealth.mark_down("bad") == 1300.0
    assert json.loads(ledger.read_text())["lanes"]["bad"]["strikes"] == 1


@pytest.mark.parametrize("until", [None, [], {}, "bad", True, False,
                                    float("nan"), float("inf"), float("-inf"),
                                    "NaN", "Infinity", "1e9999", 10 ** 400, -1, 999])
def test_bad_until_does_not_hide_good_neighbor(ledger, until):
    good = {"until": "2000", "strikes": 2, "reason": "busy"}
    write_ledger(ledger, {"good": good, "bad": {"until": until}})
    assert lanehealth.cooling_until("bad") == 0.0
    assert lanehealth.cooling_until("good") == 2000.0
    assert lanehealth.status() == {"good": good}


@pytest.mark.parametrize("entry", [None, [], "bad", 1, True])
def test_bad_row_is_replaced_without_altering_neighbor(ledger, entry):
    good = {"until": 2000, "strikes": 2}
    write_ledger(ledger, {"good": good, "bad": entry})
    assert lanehealth.cooling_until("bad") == 0.0
    assert lanehealth.status() == {"good": good}
    lanehealth.mark_down("bad")
    data = json.loads(ledger.read_text())
    assert data["lanes"]["good"] == good
    assert data["extra"] == {"preserve": True}


@pytest.mark.parametrize("strikes", [None, [], {}, "bad", True, False, -1,
                                      float("nan"), float("inf"), float("-inf"),
                                      "Infinity", "NaN"])
def test_invalid_strikes_restart_backoff_and_preserve_neighbor(ledger, strikes):
    good = {"until": 2000, "strikes": 2}
    write_ledger(ledger, {"good": good, "bad": {"strikes": strikes}})
    assert lanehealth.mark_down("bad") == 1300.0
    data = json.loads(ledger.read_text())
    assert data["lanes"]["bad"]["strikes"] == 1
    assert data["lanes"]["good"] == good
    assert lanehealth.cooling_until("good") == 2000.0


@pytest.mark.parametrize("strikes", [8, 1000000, 10 ** 400, "1000000000"])
def test_huge_strikes_stay_capped_on_repeated_failures(ledger, strikes):
    write_ledger(ledger, {"busy": {"strikes": strikes}})
    for _ in range(3):
        assert lanehealth.mark_down("busy") == 1000 + lanehealth.BACKOFF_MAX_S
        saved = json.loads(ledger.read_text())["lanes"]["busy"]
        assert 1 <= saved["strikes"] <= lanehealth.MAX_STRIKES


def test_valid_numeric_strings_and_quota_floor_remain_compatible(ledger):
    write_ledger(ledger, {"busy": {"strikes": "2", "until": "2000"}})
    assert lanehealth.cooling_until("busy") == 2000.0
    assert lanehealth.mark_down("busy") == 2200.0
    assert lanehealth.mark_down("quota", quota=True) == 4600.0
    lanehealth.mark_up("busy")
    assert lanehealth.cooling_until("busy") == 0.0
    assert lanehealth.cooling_until("quota") == 4600.0


@pytest.mark.parametrize("precheck", [
    {"lane": {"at": "yesterday", "exhausted": True}},
    {"lane": {"at": [1], "exhausted": True}},
    {"lane": {"at": float("inf"), "exhausted": True}},
    ["lane"], "lane",
])
def test_malformed_precheck_cache_is_absent_not_a_crash(ledger, monkeypatch, precheck):
    ledger.write_text(json.dumps({"lanes": {}, "precheck": precheck}))
    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key",
                        lambda url, key: {"limit": 10, "limit_remaining": 4.0})
    assert lanehealth.openrouter_exhausted("https://openrouter.ai/api/v1",
                                           "sk-or-x", "lane") is False
    stored = json.loads(ledger.read_text())["precheck"]["lane"]
    assert stored == {"at": 1000.0, "exhausted": False}

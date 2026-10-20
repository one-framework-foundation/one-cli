"""A live server that does not list the profile's model is `model-missing`
and is treated exactly like offline.

A server can be up and list only a different model; the rig below lists only
`m`, while a profile names a model it does not serve.
"""

from __future__ import annotations

import json

import pytest

from onecli import cli, config, presence, unavailable, fallback
from onecli._vendor import dropins
from onecli.model_picker import ModelPicker
from test_presence_gate import Rig


@pytest.fixture
def lanes(monkeypatch, tmp_path):
    """`gone` names a model its live server does not list; `alt` is fine."""
    dropins._PRESENCE_CACHE.clear()
    dropins._LISTING_CACHE.clear()
    monkeypatch.setattr(presence, "SELECT_MAX_AGE", 0.0)
    for var in (unavailable.MODE_ENV, unavailable.ALLOW_PAID_ENV,
                fallback.FALLBACK_ENV, config.PROVIDER_DIR_ENV):
        monkeypatch.delenv(var, raising=False)
    rig = Rig()
    rig.up()
    (tmp_path / "gone.env").write_text(
        f"ONE_PROVIDER_MODEL=absent-model\nOPENAI_BASE_URL={rig.base}\n")
    (tmp_path / "alt.env").write_text(
        f"ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL={rig.base}\n")
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path)
    monkeypatch.setattr(config, "llamacpp_backends", lambda *a, **k: [])
    yield rig
    rig.down()
    dropins.await_presence_priming(5)
    dropins._PRESENCE_CACHE.clear()
    dropins._LISTING_CACHE.clear()


class Args:
    json = True


def test_list_shows_model_missing_with_its_reason(lanes, capsys):
    assert cli.cmd_list(Args()) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["gone"]["status"] == presence.MODEL_MISSING
    assert "does not list absent-model" in rows["gone"]["reason"]
    assert rows["alt"]["status"] == presence.ONLINE and rows["alt"]["reason"] == ""


def test_refusal_names_the_missing_model(lanes):
    refused = presence.refusal(config.resolve(name="gone"))
    assert "gone is model-missing" in refused and "absent-model" in refused
    assert presence.refusal(config.resolve(name="alt")) == ""


def test_the_picker_refuses_and_tags_it(lanes):
    gone = config.resolve(name="gone")
    picker = ModelPicker([gone], gone)
    picker.choose("1")
    choice, done, text = picker.choose("1")
    assert choice is None and not done and "model-missing" in text
    assert "[model-missing]" in picker.render()


def test_dash_m_takes_the_unavailable_path_with_suggestions(lanes, capsys):
    assert cli.main(["-m", "gone", "--on-unavailable-model", "error", "hi"]) == 2
    err = capsys.readouterr().err
    assert "gone is model-missing" in err and "alt (m)" in err


def test_runner_unreachable_stays_unknown_and_visible(monkeypatch, lanes):
    monkeypatch.setattr(presence, "_reader", lambda: None)
    gone = config.resolve(name="gone")
    assert presence.status(gone, fresh=True) == presence.UNKNOWN
    assert presence.refusal(gone) == ""

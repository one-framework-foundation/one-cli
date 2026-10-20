"""`one --list --json`: the machine-readable backend listing.

The text listing has always printed name, wire and url and nothing else, so a
caller that needed to know WHICH MODEL a backend actually runs had nowhere to
ask. one-cli is the only resolver for `~/.one-providers`, so callers that
needed it guessed and wrote the backend name into a model field. These tests
pin the two halves of the fix: the JSON carries the model id, and the text
listing does not move.
"""

from __future__ import annotations

import json

import pytest

from onecli import cli, config, presence


@pytest.fixture(autouse=True)
def _no_dial(monkeypatch):
    """The fixture rows point at a remote machine; the listing shape is the test."""
    monkeypatch.setattr(presence, "survey",
                        lambda found: {b.name: presence.UNKNOWN for b in found})


class Args:
    def __init__(self, json=False):
        self.json = json


def _backends():
    return [
        config.Backend(name="ollama-cloud", model="deepseek-v4-flash:0731",
                       base_url="https://ollama.com/v1", wire="openai",
                       api_key="sk-secret-never-print-me", source="/x/oc.env"),
        config.Backend(name="local-a", model="example-model-27b",
                       base_url="http://192.0.2.13:8137", wire="anthropic",
                       api_key="sk-also-secret", source="/x/q.env"),
    ]


def test_json_listing_carries_the_model_id(monkeypatch, capsys):
    monkeypatch.setattr(config, "discover", lambda: _backends())
    assert cli.cmd_list(Args(json=True)) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in rows] == ["ollama-cloud", "local-a"]
    # The whole point: the model is the resolved id, never the backend name.
    assert rows[0]["model"] == "deepseek-v4-flash:0731"
    assert rows[0]["model"] != rows[0]["name"]
    assert rows[1]["model"] == "example-model-27b"
    assert rows[0]["base_url"] == "https://ollama.com/v1"
    assert rows[0]["wire"] == "openai"
    assert rows[0]["source"] == "/x/oc.env"


def test_json_listing_never_leaks_the_key(monkeypatch, capsys):
    monkeypatch.setattr(config, "discover", lambda: _backends())
    cli.cmd_list(Args(json=True))
    out = capsys.readouterr().out
    assert "api_key" not in out
    assert "secret" not in out


def test_text_listing_is_unchanged(monkeypatch, capsys):
    """Anything parsing the text listing reads the url as the
    LAST field. A fourth column would break every one of them at once, on hosts
    that update on their own schedule, so the text output must not move."""
    monkeypatch.setattr(config, "discover", lambda: _backends())
    assert cli.cmd_list(Args(json=False)) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert len(parts) == 3
        assert "://" in parts[-1]


def test_no_backends_still_fails(monkeypatch, capsys):
    monkeypatch.setattr(config, "discover", lambda: [])
    assert cli.cmd_list(Args(json=True)) == 1
    assert capsys.readouterr().out == ""


def test_list_without_args_keeps_working(monkeypatch, capsys):
    """cmd_list() is called with no argument in older paths; that must not
    become a TypeError."""
    monkeypatch.setattr(config, "discover", lambda: _backends())
    assert cli.cmd_list() == 0
    assert len(capsys.readouterr().out.splitlines()) == 2

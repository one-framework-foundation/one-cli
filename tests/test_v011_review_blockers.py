"""Regressions for the v0.1.1 review: key choice by wire, MCP permission
clauses, non-UTF-8 `!command` output, and malformed relay metadata."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from onecli import chat as chatlib
from onecli import cli, config, permissions, presence, session
from onecli._vendor import dropins


# --- 1. the bearer belongs to the wire ---------------------------------------

@pytest.fixture
def both_keys(tmp_path):
    src = tmp_path / "secrets.env"
    src.write_text("ANTHROPIC_API_KEY=anthropic-key\nOPENAI_API_KEY=openai-key\n"
                   "ANTHROPIC_AUTH_TOKEN=anthropic-token\n")
    return src


@pytest.mark.parametrize("url_var,expected", [
    ("OPENAI_BASE_URL", "openai-key"),
    ("ANTHROPIC_BASE_URL", "anthropic-token"),
])
def test_both_readers_pick_the_key_of_the_drop_ins_wire(tmp_path, both_keys, url_var, expected):
    path = tmp_path / "lane.env"
    path.write_text(f"ONE_PROVIDER_SOURCE={both_keys}\nONE_PROVIDER_MODEL=m\n"
                    f"{url_var}=https://api.example.com/v1\n")
    assert config._backend_from_env(path).api_key == expected
    assert dropins.read(path)["key"] == expected


def test_an_openai_wire_never_sends_an_anthropic_key(tmp_path):
    path = tmp_path / "lane.env"
    path.write_text("ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL=https://api.example.com/v1\n"
                    "ANTHROPIC_API_KEY=anthropic-key\n")
    assert config._backend_from_env(path).api_key == ""
    assert dropins.read(path)["key"] == ""


def test_named_key_variable_wins_and_may_live_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ODD_PROVIDER_TOKEN", "from-env")
    path = tmp_path / "lane.env"
    path.write_text("ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL=https://api.example.com/v1\n"
                    "ONE_PROVIDER_KEY_ENV=ODD_PROVIDER_TOKEN\nOPENAI_API_KEY=openai-key\n")
    assert config._backend_from_env(path).api_key == "from-env"
    assert dropins.read(path)["key"] == "from-env"


def test_one_provider_key_precedence_per_wire():
    env = {"ONE_PROVIDER_KEY": "generic", "OPENAI_API_KEY": "o",
           "ANTHROPIC_API_KEY": "a"}
    assert dropins.key_for(env, "openai") == "generic"
    assert dropins.key_for(env, "anthropic") == "a"


# --- 2. MCP clauses match, and a clause that cannot match is refused ----------

WS = "/ws"


@pytest.mark.parametrize("arguments", [
    {"url": "https://evil.example/page"},
    {"target": "https://evil.example/page"},          # any argument name
    {"opts": {"urls": ["https://ok.example/", "https://evil.example/x"]}},
    {"url": "HTTPS://EVIL.example:443/page"},          # same origin, other spelling
])
def test_mcp_deny_clause_refuses_the_call(arguments):
    guard = permissions.PermissionGuard(
        deny=permissions.parse_rules(["mcp__web__fetch(https://evil.example/*)"]),
        workspace=WS)
    verdict = guard.check("mcp__web__fetch", arguments, mutating=False)
    assert not verdict.allowed


@pytest.mark.parametrize("arguments", [
    {"url": "https://good.example/page"},
    {"url": "https://evil.example.attacker/page"},
    {"https://evil.example/page": "key, not value"},
    {},
])
def test_mcp_deny_clause_leaves_other_calls_alone(arguments):
    rule = permissions.Rule.parse("mcp__web__fetch(https://evil.example/*)")
    assert not rule.matches("mcp__web__fetch", arguments, WS, deny=True)


def test_mcp_allow_clause_and_other_tools():
    rule = permissions.Rule.parse("mcp__db__query(SELECT *)")
    assert rule.matches("mcp__db__query", {"sql": "SELECT 1"}, WS)
    assert not rule.matches("mcp__db__query", {"sql": "DROP TABLE t"}, WS)
    assert not rule.matches("mcp__db__other", {"sql": "SELECT 1"}, WS)


@pytest.mark.parametrize("pattern", ["todo_write(x)", "Task(anything)", "skill(name)"])
def test_a_clause_on_a_tool_without_a_clause_argument_is_refused(pattern):
    with pytest.raises(ValueError, match="takes no clause"):
        permissions.Rule.parse(pattern)
    assert permissions.Rule.parse(pattern.split("(")[0]).clause is None


def test_cli_rejects_an_unmatchable_clause_before_any_request(capsys, monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError("backend resolution reached: the rule was not refused")
    monkeypatch.setattr(config, "discover", reached)
    monkeypatch.setattr(config, "resolve", reached)
    assert cli.main(["--disallowed-tools", "todo_write(x)", "hello"]) == 2
    assert "takes no clause" in capsys.readouterr().err


# --- 4. `!command` output that is not UTF-8 ----------------------------------

BYTES = "printf '\\377\\376ok'"


def test_chat_shell_survives_non_utf8_output(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(presence, "refusal", lambda b: "")
    c = chatlib.Chat(config.Backend(name="f", model="f", base_url="http://127.0.0.1:1"),
                     out=io.StringIO(), err=io.StringIO(), stream=False)
    c.workspace = str(tmp_path)
    chatlib.shell(c, "!" + BYTES)
    assert c.staged and "��ok" in c.staged[-1]


def test_cli_shell_survives_non_utf8_output(tmp_path):
    seen = []
    repl = SimpleNamespace(remember_shell=lambda cmd, out, code: seen.append((out, code)))
    shown = cli._shell(SimpleNamespace(workspace=str(tmp_path)), BYTES, repl)
    assert seen == [("��ok", 0)]
    assert "ok" in shown


# --- 7. malformed relay metadata is absent, not a crash -----------------------

@pytest.mark.parametrize("listing,window", [
    ({"data": [{"id": "m", "meta": {"n_ctx": "lots"}}]}, None),
    ({"data": [{"id": "m", "meta": {"n_ctx": [4096]}}]}, None),
    ({"data": [{"id": "m", "context_window": {"n": 1}}]}, None),
    ({"data": [{"id": "m", "meta": "flat"}]}, None),
    ({"data": [{"id": "m", "meta": {"n_ctx": 8192, "ftype": 3}}]}, 8192),
])
def test_relay_served_tolerates_malformed_metadata(monkeypatch, listing, window):
    dropins._RELAY_CACHE.clear()
    monkeypatch.setattr(dropins, "_http_json", lambda url, timeout: (listing, None))
    served = dropins.relay_served("http://relay.example/v1")
    dropins._RELAY_CACHE.clear()
    assert served is not None and served["model"] == "m"
    assert served["context_window"] == window


@pytest.mark.parametrize("listing", [{"data": ["m"]}, {"data": [{"id": 7}]}, {"data": [None]}])
def test_relay_served_refuses_a_malformed_model_entry(monkeypatch, listing):
    dropins._RELAY_CACHE.clear()
    monkeypatch.setattr(dropins, "_http_json", lambda url, timeout: (listing, None))
    assert dropins.relay_served("http://relay.example/v1") is None
    dropins._RELAY_CACHE.clear()


def test_mcp_allow_clause_needs_every_value_to_match():
    from onecli.permissions import Rule
    rule = Rule.parse("mcp__web__fetch(https://ok.example/*)")
    ok = {"url": "https://ok.example/a"}
    smuggled = {"url": "https://evil.example/x", "note": "https://ok.example/a"}
    assert rule.matches("mcp__web__fetch", ok, "/tmp")
    assert not rule.matches("mcp__web__fetch", smuggled, "/tmp")
    assert rule.matches("mcp__web__fetch", smuggled, "/tmp", deny=True)
    assert not rule.matches("mcp__web__fetch", {}, "/tmp")

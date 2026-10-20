"""First-run onboarding and `one doctor`'s backend section."""

from __future__ import annotations

import http.server
import stat
import threading

import pytest

from onecli import cli, config, onboard
from onecli.serving import cli as serving_cli


@pytest.fixture
def empty_home(tmp_path, monkeypatch):
    """No drop-ins, no local servers: exactly what a newcomer has."""
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path / ".one-providers")
    monkeypatch.delenv(config.PROVIDER_DIR_ENV, raising=False)
    monkeypatch.delenv(config.LLAMACPP_HOSTS_ENV, raising=False)
    monkeypatch.setattr(config, "llamacpp_backends", lambda *a, **k: [])
    return tmp_path / ".one-providers"


def test_no_backend_non_tty_is_actionable_not_a_traceback(empty_home, monkeypatch, capsys):
    monkeypatch.setattr(onboard, "interactive", lambda: False)
    code = cli.main(["hello"])
    err = capsys.readouterr().err
    assert code == 2
    assert "no model backend is configured" in err
    assert "ONE_PROVIDER_BASE_URL" in err and "one doctor" in err
    assert "Traceback" not in err


def test_no_backend_in_a_tty_runs_the_wizard(empty_home, monkeypatch):
    calls = []
    monkeypatch.setattr(onboard, "interactive", lambda: True)
    monkeypatch.setattr(onboard, "run", lambda: calls.append(1))
    assert cli.main(["hello"]) == 2  # wizard cancelled -> no backend -> 2
    assert calls == [1]


def test_detect_local_probes_ollama_and_lm_studio(monkeypatch):
    seen = []

    def fake(url, headers=None, timeout=2.0):
        seen.append(url)
        if ":11434" in url:
            return {"data": [{"id": "qwen3:8b"}]}
        return None

    monkeypatch.setattr(onboard, "_get_json", fake)
    found = onboard.detect_local()
    assert found == [("ollama", "http://127.0.0.1:11434/v1", ["qwen3:8b"])]
    assert any(":1234/" in u for u in seen) and any(":8080/" in u for u in seen)


def test_dropin_is_private_readable_and_never_overwritten(empty_home):
    path = onboard.write_dropin("openai", onboard.dropin_values(
        "https://api.openai.com/v1", config.WIRE_OPENAI, "m1", key="sk-secret"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    b = config._backend_from_env(path)
    assert (b.base_url, b.model, b.api_key) == ("https://api.openai.com/v1", "m1", "sk-secret")
    with pytest.raises(FileExistsError):
        onboard.write_dropin("openai", {"ONE_PROVIDER_MODEL": "x"})


def test_key_env_resolves_from_process_environment(empty_home, monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "sk-from-env")
    path = onboard.write_dropin("anthropic", onboard.dropin_values(
        "https://api.anthropic.com", config.WIRE_ANTHROPIC, "claude-sonnet-5",
        key_env="MY_TEST_KEY"))
    assert "sk-from-env" not in path.read_text()
    b = config._backend_from_env(path)
    assert b.wire == config.WIRE_ANTHROPIC and b.api_key == "sk-from-env"


def test_wizard_hidden_key_is_never_echoed(empty_home, monkeypatch, capsys):
    monkeypatch.setattr(onboard, "detect_local", lambda: [])
    picks = iter([2, 1])  # OpenAI, then "type it now"
    monkeypatch.setattr(onboard.menu, "choose", lambda *a, **k: next(picks))
    monkeypatch.setattr(onboard.menu, "ask", lambda q, d="": "gpt-test")
    monkeypatch.setattr(onboard.getpass, "getpass", lambda prompt: "sk-hidden-123")
    monkeypatch.setattr(onboard, "list_models", lambda *a: [])
    tested = []
    monkeypatch.setattr(onboard, "test_request", lambda b: tested.append(b) or None)
    backend = onboard.run()
    out = capsys.readouterr()
    assert backend is not None and backend.api_key == "sk-hidden-123"
    assert tested and tested[0].model == "gpt-test"
    assert "sk-hidden-123" not in out.out + out.err
    assert (empty_home / "openai.env").exists()


def test_wizard_reports_failed_test_request(empty_home, monkeypatch, capsys):
    monkeypatch.setattr(onboard, "detect_local",
                        lambda: [("ollama", "http://127.0.0.1:11434/v1", ["m"])])
    monkeypatch.setattr(onboard.menu, "choose", lambda *a, **k: 0)
    monkeypatch.setattr(onboard, "test_request", lambda b: "ProviderError: boom")
    assert onboard.run() is None
    assert "test request failed: ProviderError: boom" in capsys.readouterr().err
    assert list(empty_home.glob("*.env")) == []  # next `one` offers setup again


class _Deny(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(401)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_doctor_reports_rejected_key_and_sandbox(empty_home):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Deny)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        onboard.write_dropin("local", onboard.dropin_values(
            f"http://127.0.0.1:{srv.server_port}/v1", config.WIRE_OPENAI, "m",
            key="bad"))
        lines, usable = onboard.doctor_report()
    finally:
        srv.shutdown()
    text = "\n".join(lines)
    assert not usable
    assert "key rejected (HTTP 401)" in text
    assert "sandbox" in text
    assert "bad" not in text.replace("bad key", "")


def test_doctor_with_nothing_configured_points_at_setup(empty_home):
    lines, usable = onboard.doctor_report()
    assert not usable
    assert any("guided setup" in line for line in lines)


def test_doctor_skips_slow_spec_type_probe_by_default(empty_home, monkeypatch):
    probed = []
    monkeypatch.setattr(serving_cli.recipes, "supported_spec_types",
                        lambda facts: probed.append(1) or [])
    assert serving_cli.cmd_doctor([]) == 1  # no usable backend
    assert probed == []


def test_wizard_keeps_dropin_when_key_variable_is_not_exported_yet(empty_home, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(onboard, "detect_local", lambda: [])
    picks = iter([2, 0])  # OpenAI, then "read it from an environment variable"
    monkeypatch.setattr(onboard.menu, "choose", lambda *a, **k: next(picks))
    monkeypatch.setattr(onboard.menu, "ask", lambda q, d="": d or "gpt-test")
    monkeypatch.setattr(onboard, "list_models", lambda *a: [])
    tested = []
    monkeypatch.setattr(onboard, "test_request", lambda b: tested.append(b) or "401")
    assert onboard.run() is None
    assert tested == []  # no request that is certain to fail
    assert "ONE_PROVIDER_KEY_ENV=OPENAI_API_KEY" in (empty_home / "openai.env").read_text()
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err

"""A relay-served role drop-in becomes a backend with the relay's model."""
from onecli import config
from onecli._vendor import dropins


def test_role_backend_takes_relay_model(tmp_path, monkeypatch):
    p = tmp_path / "relay.env"
    p.write_text("ONE_PROVIDER_ROLE='relay'\nOPENAI_BASE_URL='http://relay:9999/v1'\n")
    monkeypatch.setattr(dropins, "relay_served", lambda base: {
        "model": "qwen3.6-35b-a3b", "context_window": 32768, "label": "q (relay)"})
    b = config._backend_from_env(p)
    assert (b.model, b.label) == ("qwen3.6-35b-a3b", "q (relay)")


def test_role_backend_refuses_pinned_model(tmp_path, monkeypatch):
    p = tmp_path / "relay.env"
    p.write_text("ONE_PROVIDER_ROLE='relay'\nOPENAI_BASE_URL='http://relay:9999/v1'\n"
                 "ONE_PROVIDER_MODEL='alt'\n")
    monkeypatch.setattr(dropins, "relay_served", lambda base: {
        "model": "qwen3.6-35b-a3b", "context_window": 32768, "label": "q"})
    assert config._backend_from_env(p) is None


def test_role_backend_off_when_relay_down(tmp_path, monkeypatch):
    p = tmp_path / "relay.env"
    p.write_text("ONE_PROVIDER_ROLE='relay'\nOPENAI_BASE_URL='http://relay:9999/v1'\n")
    monkeypatch.setattr(dropins, "relay_served", lambda base: None)
    assert config._backend_from_env(p) is None

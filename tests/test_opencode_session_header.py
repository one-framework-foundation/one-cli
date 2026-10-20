"""OpenCode Go endpoints carry the conversation id as a session header."""

import pytest

from onecli import config


def test_go_provider_uses_conversation_session_header():
    from types import SimpleNamespace
    from onecli import cli

    backend = config.Backend(
        name="opencode-go-glm53-flash", model="glm-5.3-flash",
        base_url="https://opencode.ai/zen/go/v1",
    )
    args = SimpleNamespace(effort="auto", task_class="test", _session_id="session_123")
    first = cli._safe_build_provider(backend, 10, args)
    second = cli._safe_build_provider(backend, 10, args)
    assert first._headers()["x-opencode-session"] == "session_123"
    assert second._headers()["x-opencode-session"] == "session_123"
    args._session_id = "session_456"
    third = cli._safe_build_provider(backend, 10, args)
    assert third._headers()["x-opencode-session"] == "session_456"


def test_chat_session_header_tracks_reset_and_model_switch(tmp_path, monkeypatch):
    from onecli.chat import Chat
    monkeypatch.chdir(tmp_path)
    backend = config.Backend(name="go", model="flash", base_url="https://opencode.ai/zen/go/v1", headers={"X-Test": "keep"})
    chat = Chat(backend, sid="conversation_1")
    assert chat.wire._headers()["x-opencode-session"] == "conversation_1"
    assert chat.wire._headers()["X-Test"] == "keep"
    chat.reset()
    assert chat.wire._headers()["x-opencode-session"] == chat.sid
    assert chat.sid != "conversation_1"
    chat.use(backend)
    assert chat.wire._headers()["x-opencode-session"] == chat.sid
    assert backend.headers == {"X-Test": "keep"}


@pytest.mark.parametrize("url", [
    "https://example.com/opencode.ai/zen/go/v1",
    "https://opencode.ai/zen/gopher/v1",
    "https://opencode.ai/zen/v1",
    "https://opencode.ai.evil.example/zen/go/v1",
])
def test_session_header_is_not_sent_to_unrelated_endpoint(url):
    from types import SimpleNamespace
    from onecli import cli
    backend = config.Backend(name="other", model="flash", base_url=url)
    wire = cli._safe_build_provider(backend, 10, SimpleNamespace(_session_id="private_session"))
    assert "x-opencode-session" not in wire._headers()

"""A provider error after a tool call keeps the turn, records it, and /undo
undoes exactly that turn.

Before: agent_turn reset the conversation to before the turn, so turn() found
no exchange and forgot its checkpoint. The file the tool wrote stayed on disk,
nothing said the turn failed, and the next /undo rolled back the PREVIOUS
turn - deleting its file too.
"""

from __future__ import annotations

import io
import json

import pytest

from onecli import chat as chatlib
from onecli import checkpoints, config, presence, session
from onecli.providers.base import ProviderError, ToolCall, Turn


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "ROOT", str(tmp_path / "checkpoints"))
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(presence, "refusal", lambda b: "")
    backend = config.Backend(name="fixture", model="fixture",
                             base_url="http://127.0.0.1:1", wire="openai")
    c = chatlib.Chat(backend, out=io.StringIO(), err=io.StringIO(), stream=False)
    c.tools = "all"
    c.mode = "bypassPermissions"
    c.ws = ws
    return c


class Script:
    """Each turn: a write tool call, then either an answer or an HTTP 429."""

    def __init__(self):
        self.plan: list = []

    def chat(self, *args, **kwargs):
        step = self.plan.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _write(call_id, path):
    return Turn(tool_calls=[ToolCall(id=call_id, name="write",
                                     arguments={"path": path, "content": "x\n"})])


def test_provider_error_after_a_tool_call_keeps_the_turn_for_undo(chat):
    wire = Script()
    chat._wire = wire
    wire.plan = [_write("w1", "first.txt"), Turn(text="wrote first")]
    chatlib.turn(chat, "write first", echo=False)
    assert (chat.ws / "first.txt").exists()

    wire.plan = [_write("w2", "second.txt"),
                 ProviderError("HTTP 429: rate limited")]
    stat = chatlib.turn(chat, "write second", echo=False)
    assert "429" in stat["error"]
    assert (chat.ws / "second.txt").exists()

    # History keeps the failed turn's request, call and result, paired.
    roles = [m["role"] for m in chat.messages]
    assert roles[-3:] == ["user", "assistant", "tool"]
    assert chat.messages[-3]["content"] == "write second"
    assert chat.messages[-2]["tool_calls"][0]["id"] == "w2"
    assert chat.messages[-1]["tool_call_id"] == "w2"
    assert session.load_messages(chat.sid) == chat.messages

    # The error is in the session record.
    records = [json.loads(line) for line in
               session.path_for(chat.sid).read_text().splitlines() if line.strip()]
    assert any(r.get("kind") == "mask_run" and "429" in (r.get("error") or "")
               for r in records)

    chatlib.dispatch(chat, "/undo")
    assert not (chat.ws / "second.txt").exists()
    assert (chat.ws / "first.txt").exists()
    assert [m["content"] for m in chat.messages if m["role"] == "user"] == ["write first"]

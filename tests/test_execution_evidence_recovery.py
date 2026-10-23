"""Trusted built-in receipts and durable recovery across real process death."""

import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import pytest
from onecli import session, tools
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall


class Provider:
    def __init__(self, *turns):
        self.turns, self.requests = iter(turns), []

    def chat(self, messages, tools=None, **kwargs):
        self.requests.append(messages)
        return next(self.turns)


def test_exact_utf8_receipt_is_private_and_persisted(tmp_path, monkeypatch):
    from onecli import execution_evidence as evidence

    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    r, w = os.pipe()
    monkeypatch.setenv("ONECLI_EXECUTOR_EVIDENCE_FD", str(w))
    monkeypatch.setenv("ONECLI_BRIDGE_TURN_ID", "bridge-turn")
    events, text = [], "évidence\n"
    provider = Provider(
        Turn(
            tool_calls=[
                ToolCall(
                    "write-unique", "write", {"path": "proof.txt", "content": text}
                )
            ]
        ),
        Turn(text="done"),
    )
    agent = Agent(
        provider,
        str(tmp_path),
        max_steps=3,
        session_id="proof",
        on_event=lambda name, data: events.append(name),
    )
    try:
        assert not os.get_inheritable(w)
        assert "ONECLI_EXECUTOR_EVIDENCE_FD" not in os.environ
        agent.run("write proof")
        command = shlex.join([
            sys.executable, "-c",
            "import os,sys\ntry: os.fstat(int(sys.argv[1]))\nexcept OSError: print('private-fd-closed'); sys.exit(0)\nsys.exit(1)",
            str(w),
        ])
        assert "private-fd-closed" in tools.shell.bash(str(tmp_path), command)

    finally:
        os.close(w)
        evidence._channel = None
    with os.fdopen(r) as fh:
        records = [json.loads(line) for line in fh]
    receipts = [r for r in records if r["type"] == "executor_write_receipt"]
    assert receipts == [
        {
            "type": "executor_write_receipt",
            "turn_id": "bridge-turn",
            "call_id": "write-unique",
            "path": str(tmp_path / "proof.txt"),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "byte_count": len(text.encode()),
            "write_ok": True,
            "readback_ok": True,
        }
    ]
    assert (
        session.load_records("proof", "execution_write_receipt")[-1]["sha256"]
        == receipts[0]["sha256"]
    )
    assert all(not name.startswith("executor_") for name in events)


@pytest.mark.parametrize("failure", ["write", "readback", "mismatch"])
def test_no_success_for_failed_write_or_readback(tmp_path, monkeypatch, failure):
    from onecli import execution_evidence as evidence

    r, w = os.pipe()
    channel = evidence.EvidenceChannel(w, "turn")
    original = evidence.secureio.replace_text

    def replace(path, text, **kwargs):
        if failure == "write":
            raise OSError("full disk")
        result = original(path, text, **kwargs)
        if failure == "mismatch":
            Path(path).write_bytes(b"changed")
        elif failure == "readback":
            Path(path).unlink()
        return result

    monkeypatch.setattr(evidence.secureio, "replace_text", replace)
    try:
        with evidence.call_scope("exact-call", "write", channel):
            if failure in ("write", "readback"):
                with pytest.raises(OSError):
                    tools.fs.write(str(tmp_path), "proof", "expected")
            else:
                tools.fs.write(str(tmp_path), "proof", "expected")
    finally:
        os.close(w)
    with os.fdopen(r) as fh:
        records = [json.loads(line) for line in fh]
    assert records[-1]["readback_ok"] is False
    assert records[-1]["write_ok"] is (failure != "write")


def test_edit_receipt_preserves_crlf_and_non_utf8_bytes(tmp_path):
    from onecli import execution_evidence as evidence

    r, w = os.pipe()
    channel = evidence.EvidenceChannel(w, "edit-turn")
    path = tmp_path / "proof"
    path.write_bytes(b"old\r\n\xff\r\n")
    try:
        with evidence.call_scope("edit-call", "edit", channel):
            tools.fs.edit(str(tmp_path), "proof", "old\n", "new\n")
    finally:
        os.close(w)
    with os.fdopen(r) as fh:
        record = [json.loads(line) for line in fh][-1]
    assert path.read_bytes() == b"new\r\n\xff\r\n"
    assert record["readback_ok"] is True
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_predispatch_checkpoint_survives_process_death_and_resume(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    script = """
import os,sys,inspect
from pathlib import Path
from onecli import session
from onecli.agent import Agent
from onecli.providers.base import Turn,ToolCall
session.SESSION_DIR=Path(sys.argv[1])/'sessions'
class Provider:
    def chat(self,*args,**kwargs):
        return Turn(tool_calls=[ToolCall('write-once','write',{'path':'proof','content':'unique'})])
def die(kind,data):
    if kind=='tool_end': os._exit(23)
kwargs={'session_id':'recover'} if 'session_id' in inspect.signature(Agent).parameters else {}
Agent(Provider(),sys.argv[1],max_steps=2,on_event=die,**kwargs).run('write once')
"""
    r, w = os.pipe()
    env = dict(
        os.environ,
        ONECLI_EXECUTOR_EVIDENCE_FD=str(w),
        ONECLI_BRIDGE_TURN_ID="first-process",
        PYTHONPATH=str(Path(__file__).parents[1]),
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        pass_fds=(w,),
        capture_output=True,
        timeout=20,
    )
    os.close(w)
    with os.fdopen(r) as fh:
        receipts = [json.loads(line) for line in fh]
    assert child.returncode == 23, child.stderr.decode()
    assert (tmp_path / "proof").read_bytes() == b"unique"
    assert any(r.get("readback_ok") for r in receipts)
    checkpoint = session.load_records("recover", "execution_checkpoint")[-1]
    assert checkpoint["pending_actions"] == [
        {
            "call_id": "write-once",
            "name": "write",
            "input": {"path": "proof", "content": "unique"},
        }
    ]
    assert session.load_records("recover", "execution_write_receipt")[-1]["readback_ok"]
    before = (tmp_path / "proof").stat().st_mtime_ns
    provider = Provider(
        Turn(tool_calls=[ToolCall("read-after-resume", "read", {"path": "proof"})]),
        Turn(text="verified"),
    )
    result = Agent(provider, str(tmp_path), max_steps=3, session_id="recover").run(
        "inspect and continue", messages=session.load_messages("recover")
    )
    assert result.text == "verified" and result.tool_calls == 1
    assert (tmp_path / "proof").stat().st_mtime_ns == before
    assert any(
        m.get("role") == "tool"
        and m.get("tool_call_id") == "write-once"
        and "Outcome unknown" in m["content"]
        for m in provider.requests[0]
    )
    assert (
        session.load_records("recover", "execution_checkpoint")[-1]["pending_actions"]
        == []
    )


@pytest.mark.parametrize(
    "state,turn,expected",
    [
        ({"empty_streak": 1}, Turn(), "empty"),
        (
            {"cutoff_retries": 2},
            Turn(text="unfinished", finish_reason="length"),
            "truncated",
        ),
    ],
)
def test_retry_budget_does_not_reset_on_resume(
    tmp_path, monkeypatch, state, turn, expected
):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    session.append(
        "retry",
        {
            "kind": "execution_checkpoint",
            "retry_state": state,
            "pending_actions": [],
            "status": "running",
        },
    )
    provider = Provider(turn)
    result = Agent(
        provider,
        str(tmp_path),
        max_steps=5,
        session_id="retry",
        loop_policy={"empty_retries": 1},
    ).run("continue")
    assert result.stopped == expected
    assert len(provider.requests) == 1


def test_persistent_empty_retry_default_is_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    provider = Provider(Turn(), Turn(), Turn(text="must not reach unbounded retry"))
    result = Agent(provider, str(tmp_path), session_id="finite", max_steps=6).run(
        "continue"
    )
    assert result.stopped == "empty"
    assert len(provider.requests) == 2


def test_observed_retry_budget_survives_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path / 'sessions')

    class InterruptAfterRetry:
        calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return Turn()

    first = InterruptAfterRetry()
    with pytest.raises(KeyboardInterrupt):
        Agent(first, str(tmp_path), session_id='retry', loop_policy={'empty_retries': 1}).run('continue')
    checkpoint = session.load_records('retry', 'execution_checkpoint')[-1]
    assert checkpoint['retry_state']['empty_streak'] == 1
    second = Provider(Turn())
    result = Agent(second, str(tmp_path), session_id='retry', loop_policy={'empty_retries': 1}).run(
        'resume', session.load_messages('retry'))
    assert len(second.requests) == 1
    assert result.stopped == 'empty'

"""Opt-in executor pipe. Model output and ordinary progress are never evidence.

This channel trusts the executor process, not arbitrary hostile processes with
same-user /proc or debugger access. The pipe is not inherited by tool children.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path

from onecli import secureio

MAX_READBACK_BYTES = 8 * 1024 * 1024
_active_call: ContextVar[tuple[str, str, EvidenceChannel | None] | None] = ContextVar(
    "executor_evidence_call", default=None
)
_channel: EvidenceChannel | None = None


class EvidenceChannel:
    def __init__(self, fd: int, turn_id: str):
        if not turn_id or not isinstance(turn_id, str):
            raise ValueError("executor evidence requires bridge turn identity")
        os.set_inheritable(fd, False)
        self.fd, self.turn_id = fd, turn_id
        self.session_id: str | None = None
        self.emit(
            {
                "type": "executor_evidence_ready",
                "version": 1,
                "write_completion": 1,
                "checkpoint": 1,
            }
        )

    def emit(self, record):
        if record.get("type") == "executor_write_receipt" and getattr(
            self, "session_id", None
        ):
            from onecli import session

            session.append(
                self.session_id,
                {"kind": "execution_write_receipt", **record, "turn_id": self.turn_id},
                durable=True,
            )
        payload = (
            json.dumps({**record, "turn_id": self.turn_id}, ensure_ascii=True) + "\n"
        ).encode()
        view = memoryview(payload)
        while view:
            written = os.write(self.fd, view)
            if not written:
                raise BrokenPipeError("executor evidence pipe closed")
            view = view[written:]


def initialize():
    """Consume private env exactly once, before launching any tool child."""
    global _channel
    raw = os.environ.pop("ONECLI_EXECUTOR_EVIDENCE_FD", None)
    turn_id = os.environ.pop("ONECLI_BRIDGE_TURN_ID", None)
    if raw is not None:
        _channel = EvidenceChannel(int(raw), turn_id)
    elif turn_id is not None:
        raise ValueError("executor turn identity supplied without evidence pipe")
    return _channel


@contextmanager
def call_scope(call_id, name, channel):
    token = _active_call.set((call_id, name, channel))
    try:
        yield
    finally:
        _active_call.reset(token)


def replace_text(path, text, *, errors="strict", newline=None):
    """The owning write/edit operation supplies its exact expected bytes."""
    active = _active_call.get()
    if active is None or active[2] is None or active[1] not in ("write", "edit"):
        return secureio.replace_text(path, text, errors=errors, newline=newline)
    record = {
        "type": "executor_write_receipt",
        "call_id": active[0],
        "path": str(Path(path).absolute()),
        "sha256": "",
        "byte_count": 0,
        "write_ok": False,
        "readback_ok": False,
    }
    try:
        result = secureio.replace_text(path, text, errors=errors, newline=newline)
        record["write_ok"] = True
        expected = text.encode("utf-8", errors=errors)
        # On platforms translating newlines, mirror the actual writer mode.
        if newline is None and os.linesep != "\n":
            expected = expected.replace(b"\n", os.linesep.encode())
        if len(expected) <= MAX_READBACK_BYTES:
            with secureio.text_file(path, errors="surrogateescape", newline="") as fh:
                actual = fh.read(MAX_READBACK_BYTES + 1).encode(
                    "utf-8", "surrogateescape"
                )
            if len(actual) <= MAX_READBACK_BYTES:
                record.update(
                    sha256=hashlib.sha256(actual).hexdigest(),
                    byte_count=len(actual),
                    readback_ok=actual == expected,
                )
        return result
    finally:
        # A failed operation or failed independent readback is explicit, even
        # if the tool dispatcher later catches the exception as ordinary text.
        active[2].emit(record)

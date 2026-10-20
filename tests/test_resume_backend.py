"""A resume keeps the conversation and NOT the backend. It should say so.

`-r <id>` with no `-m` continues yesterday's session on whatever backend
resolves today, which is a different model answering a conversation it never
had. The reply that comes back is then indistinguishable from a resume that
silently lost the history - a persona spent four commands and a `--json`
message count proving to itself that the history was actually intact.

Not changed, disclosed: moving a conversation to another model on purpose is a
legitimate thing to do with `-m`, and the silence was the bug.
"""

from __future__ import annotations

import pytest

from onecli import session


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    return tmp_path


def test_backend_of_reads_the_run_record():
    session.append("s1", {"kind": "run", "backend": "local-a", "model": "q"})
    assert session.backend_of("s1") == "local-a"


def test_backend_of_takes_the_last_run_not_the_first():
    """A session resumed twice ran on two backends; the latest is the truth."""
    session.append("s1", {"kind": "run", "backend": "first", "model": "a"})
    session.append("s1", {"kind": "messages", "messages": []})
    session.append("s1", {"kind": "run", "backend": "second", "model": "b"})
    assert session.backend_of("s1") == "second"


def test_backend_of_is_empty_when_nothing_recorded_one():
    session.append("s1", {"kind": "messages", "messages": []})
    assert session.backend_of("s1") == ""


def test_backend_of_is_empty_for_a_session_that_is_not_there():
    """An answer, not an exception - it is called on the resume path."""
    assert session.backend_of("nope") == ""


def test_a_run_record_with_no_backend_does_not_win():
    session.append("s1", {"kind": "run", "backend": "real", "model": "a"})
    session.append("s1", {"kind": "run", "model": "b"})
    assert session.backend_of("s1") == "real"

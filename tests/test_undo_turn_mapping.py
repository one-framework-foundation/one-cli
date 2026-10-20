"""/undo reverts exactly the turn it undoes, and checkpoints stay bounded.

Audit of the earlier implementation: a turn whose checkpoint failed (a FIFO in the
workspace was enough) left no record, so /undo rolled the workspace back to the
turn BEFORE it while dropping one exchange - deleting the earlier turn's work.
Non-git checkpoints also copied the whole workspace every turn, with no ceiling
and no cleanup.
"""

from __future__ import annotations

import io
import os
import subprocess
import time

import pytest

from onecli import chat as chatlib
from onecli import checkpoints, config, presence, session
from onecli.checkpoints import CheckpointManager


@pytest.fixture(autouse=True)
def _isolated_checkpoint_root(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "ROOT", str(tmp_path / "checkpoints"))


@pytest.fixture
def chat(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(presence, "refusal", lambda b: "")
    backend = config.Backend(name="fixture", model="fixture",
                             base_url="http://127.0.0.1:1", wire="openai")
    c = chatlib.Chat(backend, out=io.StringIO(), err=io.StringIO(), stream=False)
    c.tools = "auto"
    actions: dict = {}

    def fake_turn(chat, prompt, echo=True):
        chat.messages += [{"role": "user", "content": prompt},
                          {"role": "assistant", "content": "done"}]
        chat.turns.append({})
        actions.get(prompt, lambda: None)()
        return {}

    monkeypatch.setattr(chatlib, "agent_turn", fake_turn)
    monkeypatch.setattr(chatlib, "_report", lambda chat, stat: None)
    monkeypatch.setattr(chatlib.Chat, "ask", lambda self, p, echo=True: fake_turn(self, p))
    c.actions = actions
    c.ws = ws
    return c


def _undo(chat) -> str:
    chat.err.seek(0)
    chat.err.truncate()
    chatlib.dispatch(chat, "/undo")
    return chat.err.getvalue()


def test_a_fifo_does_not_break_the_checkpoint_and_undo_keeps_the_earlier_turn(chat):
    ws = chat.ws

    def one():
        (ws / "keep.txt").write_text("turn one\n")
        os.mkfifo(ws / "pipe")
    chat.actions["one"] = one
    chat.actions["two"] = lambda: (ws / "two.txt").write_text("turn two\n")
    chatlib.turn(chat, "one")
    chatlib.turn(chat, "two")

    _undo(chat)
    assert (ws / "keep.txt").read_text() == "turn one\n"
    assert not (ws / "two.txt").exists()
    assert (ws / "pipe").exists()  # special files are left alone, not deleted
    assert [m["content"] for m in chat.messages] == ["one", "done"]


def test_undo_refuses_when_the_turn_has_no_checkpoint(chat):
    ws = chat.ws
    locked = ws / "locked"

    def one():
        (ws / "keep.txt").write_text("turn one\n")
        locked.mkdir()
        locked.chmod(0)
    chat.actions["one"] = one
    chatlib.turn(chat, "one")
    try:
        chat.err.seek(0)
        chat.err.truncate()
        chatlib.turn(chat, "two")
        assert "checkpoint" in chat.err.getvalue()  # the failure is surfaced
        said = _undo(chat)
    finally:
        locked.chmod(0o700)

    assert "no checkpoint" in said
    assert (ws / "keep.txt").read_text() == "turn one\n"
    assert len(chat.messages) == 4  # nothing dropped either


def test_a_tools_off_turn_is_undone_without_touching_the_workspace(chat):
    ws = chat.ws
    chat.actions["one"] = lambda: (ws / "keep.txt").write_text("turn one\n")
    chatlib.turn(chat, "one")
    chat.tools = "off"
    chatlib.turn(chat, "just talk")

    _undo(chat)
    assert (ws / "keep.txt").read_text() == "turn one\n"
    assert [m["content"] for m in chat.messages] == ["one", "done"]


def test_retry_does_not_leave_a_stale_checkpoint_for_the_next_undo(chat):
    ws = chat.ws
    chat.actions["one"] = lambda: (ws / "keep.txt").write_text("turn one\n")
    chat.actions["two"] = lambda: (ws / "two.txt").write_text("x\n")
    chatlib.turn(chat, "one")
    chatlib.turn(chat, "two")
    chatlib.dispatch(chat, "/retry")
    _undo(chat)
    _undo(chat)
    assert not (ws / "keep.txt").exists()
    assert chat.messages == []


# -- non-git checkpoint bounds --------------------------------------------------


def test_a_workspace_over_the_ceiling_is_refused_with_a_note(tmp_path, monkeypatch):
    ws = tmp_path / "big"
    ws.mkdir()
    for i in range(5):
        (ws / f"f{i}").write_text("x")
    monkeypatch.setattr(checkpoints, "MAX_FILES", 3)
    mgr = CheckpointManager(str(ws), sid="big")
    note = mgr.begin_turn("p", [])
    assert note and "too large" in note
    assert mgr.checkpoints == []
    assert not os.path.exists(os.path.join(checkpoints.backup_root(), "big"))


def test_old_checkpoints_are_pruned_and_close_removes_the_rest(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a").write_text("a")
    monkeypatch.setattr(checkpoints, "KEEP_LAST", 2)
    mgr = CheckpointManager(str(ws), sid="prune")
    for i in range(4):
        assert mgr.begin_turn(f"t{i}", []) is None
    dirs = [cp.backup_dir for cp in mgr.checkpoints]
    assert len(dirs) == 2 and all(os.path.isdir(d) for d in dirs)
    session_dir = os.path.join(checkpoints.backup_root(), "prune")
    snaps = lambda: sorted(n for n in os.listdir(session_dir) if n.startswith("snapshot-"))  # noqa: E731
    assert snaps() == sorted(os.path.basename(d) for d in dirs)

    # The two turns whose checkpoints were pruned refuse rather than roll
    # back further than they should.
    assert mgr.undo_turn([]).success and mgr.undo_turn([]).success
    refused = mgr.undo_turn([])
    assert not refused.success and "no checkpoint" in refused.error
    assert snaps() == []

    mgr.begin_turn("again", [])
    mgr.close()
    assert not os.path.exists(session_dir)


def test_rollback_deletes_the_checkpoints_it_rolled_past(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    mgr = CheckpointManager(str(ws), sid="past")
    mgr.begin_turn("one", [])
    mgr.begin_turn("two", [])
    later = mgr.checkpoints[1].backup_dir
    res = mgr.undo_turn([], mgr.checkpoints[0].id)
    assert res.success and res.turns == 2
    assert not os.path.exists(later)


def test_git_checkpoint_skips_a_fifo_and_needs_no_git_identity(tmp_path, monkeypatch):
    # No user.name/email anywhere: commit-tree refused, so every git
    # checkpoint failed on a host (or CI runner) without a git identity.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("EMAIL", "")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.delenv(f"GIT_{name}_NAME", raising=False)
        monkeypatch.delenv(f"GIT_{name}_EMAIL", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "a").write_text("a")
    os.mkfifo(repo / "pipe")
    mgr = CheckpointManager(str(repo), sid="gitfifo")
    assert mgr.begin_turn("t", []) is None
    (repo / "b").write_text("b")
    assert mgr.undo_turn([]).success
    assert not (repo / "b").exists() and (repo / "a").exists()


# -- undo boundary review ------------------------------------------------------------


def test_chat_undo_takes_a_turns_harness_messages_with_it(chat):
    """One turn can hold several user-role messages; /undo must cut where the
    turn began, not at its last user-role message."""
    ws = chat.ws
    chat.actions["one"] = lambda: (ws / "keep.txt").write_text("turn one\n")

    def two():
        (ws / "two.txt").write_text("x\n")
        chat.messages += [{"role": "user", "content": "[harness] nudge"},
                          {"role": "assistant", "content": "more"}]
    chat.actions["two"] = two
    chatlib.turn(chat, "one")
    chatlib.turn(chat, "two")
    _undo(chat)
    assert (ws / "keep.txt").exists() and not (ws / "two.txt").exists()
    assert [m["content"] for m in chat.messages] == ["one", "done"]


def test_undo_by_id_refuses_a_checkpoint_from_before_a_reset(chat):
    ws = chat.ws
    chat.actions["one"] = lambda: (ws / "keep.txt").write_text("turn one\n")
    chatlib.turn(chat, "one")
    chatlib.dispatch(chat, "/new")
    chatlib.turn(chat, "two")
    said = _undo_arg(chat, "1")
    assert (ws / "keep.txt").exists(), "a reset conversation cannot roll back its old turns"
    assert "not a turn of this conversation" in said or "not found" in said
    assert chat.checkpoints.list_checkpoints()[0].id != 1


def _undo_arg(chat, arg) -> str:
    chat.err.seek(0)
    chat.err.truncate()
    chatlib.dispatch(chat, f"/undo {arg}")
    return chat.err.getvalue()


def test_startup_sweep_removes_backups_whose_session_is_gone(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a").write_text("a")
    root = checkpoints.backup_root()

    def session_with_backup(sid):
        mgr = CheckpointManager(str(ws), sid=sid)
        assert mgr.create_checkpoint("t") is not None
        return os.path.join(root, sid)
    dead = session_with_backup("crashed")
    dead_pid = subprocess.Popen(["true"])
    dead_pid.wait()
    with open(os.path.join(dead, "owner.pid"), "w") as fh:
        fh.write(str(dead_pid.pid))
    live = session_with_backup("running")
    old = session_with_backup("stale")
    week = time.time() - 8 * 24 * 3600
    os.utime(old, (week, week))
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "snapshot-keep").mkdir(parents=True)
    os.symlink(elsewhere, os.path.join(root, "linked"))

    monkeypatch.setattr(checkpoints, "_swept", set(), raising=False)
    CheckpointManager(str(ws), sid="new")
    assert not os.path.exists(dead) and not os.path.exists(old)
    assert os.listdir(live)
    assert (elsewhere / "snapshot-keep").is_dir()


def test_backups_are_removed_at_exit_without_a_clean_close(tmp_path, monkeypatch):
    import atexit
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a: registered.append((fn, a)))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a").write_text("a")
    mgr = CheckpointManager(str(ws), sid="exits")
    assert mgr.create_checkpoint("t") is not None
    session_dir = os.path.join(checkpoints.backup_root(), "exits")
    assert os.path.isdir(session_dir)
    for fn, args in registered:
        fn(*args)
    assert not os.path.exists(session_dir)

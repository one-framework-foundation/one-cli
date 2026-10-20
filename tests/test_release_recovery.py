"""Offline release recovery exercises using only disposable user state."""
import importlib
import json
import multiprocessing
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from onecli import session
from onecli.budget import BudgetExceeded, RequestBudget


def _append_records(root, worker):
    session.SESSION_DIR = Path(root)
    for sequence in range(100):
        session.append("parallel", {"kind": "event", "worker": worker,
                                    "sequence": sequence, "content": "x" * 2048})


def test_concurrent_session_records_remain_complete(tmp_path):
    context = multiprocessing.get_context("spawn")
    children = [context.Process(target=_append_records, args=(str(tmp_path), i))
                for i in range(4)]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(20)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()
    records = [json.loads(line) for line in (tmp_path / "parallel.jsonl").read_text().splitlines()]
    assert {(r["worker"], r["sequence"]) for r in records} == {
        (i, j) for i in range(4) for j in range(100)}
    assert len(records) == 400


def test_long_history_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    messages = [{"role": "user", "content": f"{i}:" + "x" * 1024}
                for i in range(10_000)]
    started = time.monotonic()
    session.append("large", {"kind": "messages", "messages": messages})
    assert session.load_messages("large") == messages
    # A deliberately generous release envelope, not a microbenchmark.
    assert time.monotonic() - started < 20
    assert session.path_for("large").stat().st_size < 12_000_000


def test_offline_restore_preserves_state_and_budget(tmp_path, monkeypatch):
    source = tmp_path / "source"
    restored = tmp_path / "restored"
    monkeypatch.setattr(session, "SESSION_DIR", source / "sessions")
    attachment = source / "attachment.txt"
    source.mkdir(mode=0o700)
    attachment.write_text("synthetic attachment")
    attachment.chmod(0o600)
    messages = [{"role": "user", "content": "synthetic conversation"}]
    session.append("restore", {"kind": "messages", "messages": messages})
    session.append("restore", {"kind": "metadata", "attachment": str(attachment)})
    session.save_draft("restore", "unsent text")
    config = source / "config.json"
    config.write_text('{"backend":"offline-fixture","api_key":"SYNTHETIC"}')
    config.chmod(0o600)
    budget = RequestBudget(source / "budget.sqlite", "1")
    budget.reserve("before-backup", ".75")
    # Writers are quiescent. SQLite backup also supports consistent DB copying.
    shutil.copytree(source, restored)
    with sqlite3.connect(source / "budget.sqlite") as src:
        with sqlite3.connect(restored / "budget.sqlite") as dst:
            src.backup(dst)
    monkeypatch.setattr(session, "SESSION_DIR", restored / "sessions")
    assert session.load_messages("restore") == messages
    assert session.load_draft("restore") == "unsent text"
    assert (restored / "config.json").read_bytes() == config.read_bytes()
    assert (restored / "attachment.txt").read_bytes() == attachment.read_bytes()
    assert (restored / "config.json").stat().st_mode & 0o777 == 0o600
    assert session.path_for("restore").stat().st_mode & 0o777 == 0o600
    recovered = RequestBudget(restored / "budget.sqlite", "1", create=False)
    with pytest.raises(BudgetExceeded):
        recovered.reserve("after-restore", ".26")
    session.append("restore", {"kind": "metadata", "restored": True})
    assert session.metadata("restore")["restored"] is True
    assert "restored" not in json.dumps(session.load_records("restore", "messages"))
    # Absolute attachment references require preserving paths or explicit remapping.
    assert session.metadata("restore")["attachment"] == str(attachment)


def test_draft_reader_reload_preserves_existing_consumer(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    session.save_draft("reload", "retained")
    def consumer():
        return session.load_draft("reload")
    original = session.load_draft
    monkeypatch.setattr(session, "load_draft", lambda _: "old behavior")
    assert consumer() == "old behavior"
    importlib.reload(session)
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    assert session.load_draft is not original
    assert consumer() == "retained"
    session.path_for("reload").with_suffix(".draft.json").write_text("null")
    assert consumer() == ""


def test_restored_job_log_survives_new_job(tmp_path):
    from onecli.jobs import JobStore
    root = tmp_path / "jobs"
    (root / "restored").mkdir(parents=True)
    old = root / "restored" / "j1.log"
    old.write_text("retained output")
    store = JobStore("restored", root)
    try:
        job = store.start("printf new-output", str(tmp_path))
        job.proc.wait(timeout=5)
        store.poll()
        assert old.read_text() == "retained output"
        assert "new-output" in store.output(job.id)
    finally:
        store.close()


def test_restored_task_files_survive_new_task(tmp_path, monkeypatch):
    import sys
    from onecli.tasks import TaskManager
    monkeypatch.delenv("ONE_DETACHED_OWNER", raising=False)
    old = tmp_path / "task_1.out"
    old.write_text("retained task")
    manager = TaskManager(root=tmp_path)
    task = manager.spawn("offline", cmd=[sys.executable, "-c", "print('new task')"])
    task.proc.wait(timeout=5)
    manager.poll()
    assert old.read_text() == "retained task"
    assert "new task" in task.output()


def test_restored_spill_allows_more_output(tmp_path):
    from onecli.spill import SpillStore
    store = SpillStore("restored", tmp_path)
    store.spill("bash", "old output" * 100, 100)
    restored = SpillStore("restored", tmp_path)
    result = restored.spill("bash", "new output" * 100, 100)
    assert "could not spill" not in result
    assert (tmp_path / "restored" / "001-bash.txt").read_text() == "old output" * 100
    assert (tmp_path / "restored" / "002-bash.txt").read_text() == "new output" * 100

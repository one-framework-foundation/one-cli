"""Architecture regressions: local fixtures, no provider or user state."""

import asyncio
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from onecli import checkpoints, config, lanehealth
from onecli.pool import Pool, walk


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    (root / "file").write_text("committed\n")
    git(root, "add", "file")
    git(root, "commit", "-qm", "initial")
    return root


@pytest.fixture(autouse=True)
def private_checkpoint_storage(tmp_path, monkeypatch):
    # Only expand checkpoint storage; never repoint the operator's HOME.
    original = os.path.expanduser
    monkeypatch.setattr(checkpoints.os.path, "expanduser", lambda p:
                        str(tmp_path / "backups") if p.startswith("~/.one-cli/checkpoints")
                        else original(p))


@pytest.fixture
def nongit(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "original").write_text("original content")
    return root


def test_git_undo_restores_index_and_worktree_separately(repo):
    (repo / "file").write_text("staged\n")
    git(repo, "add", "file")
    (repo / "file").write_text("unstaged\n")
    (repo / "untracked").write_text("keep untracked")
    index_before = (repo / ".git/index").read_bytes()
    manager = checkpoints.CheckpointManager(str(repo))
    assert manager.create_checkpoint()
    (repo / "file").write_text("agent changed\n")
    (repo / "new staged\nfile").write_text("remove me")
    git(repo, "add", "-A")
    result = manager.rollback()
    assert result.success, result.error
    assert git(repo, "show", ":file") == b"staged\n"
    assert (repo / "file").read_text() == "unstaged\n"
    assert (repo / "untracked").read_text() == "keep untracked"
    assert "untracked" not in git(repo, "ls-files").decode()
    assert not (repo / "new staged\nfile").exists()
    assert (repo / ".git/index").read_bytes() == index_before


def test_git_checkpoint_failure_does_not_fall_back_to_incomplete_undo(repo, monkeypatch):
    manager = checkpoints.CheckpointManager(str(repo))
    original = checkpoints.subprocess.check_output

    def fail_tree(args, **kwargs):
        if "write-tree" in args:
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)

    monkeypatch.setattr(checkpoints.subprocess, "check_output", fail_tree)
    assert manager.create_checkpoint() is None
    assert manager.list_checkpoints() == []


def test_failed_file_backup_is_not_a_checkpoint(nongit, monkeypatch):
    manager = checkpoints.CheckpointManager(str(nongit))

    def fail_copy(*args, **kwargs):
        raise OSError("injected disk full")

    monkeypatch.setattr(checkpoints.shutil, "copy2", fail_copy)
    assert manager.create_checkpoint() is None
    assert manager.list_checkpoints() == []
    assert not manager.rollback().success
    assert (nongit / "original").read_text() == "original content"


def test_walk_error_cannot_publish_partial_checkpoint(nongit, monkeypatch):
    manager = checkpoints.CheckpointManager(str(nongit))

    def fail_walk(path, **kwargs):
        if kwargs.get("onerror"):
            kwargs["onerror"](PermissionError("unreadable subtree"))
        return iter(())

    monkeypatch.setattr(checkpoints.os, "walk", fail_walk)
    assert manager.create_checkpoint() is None


@pytest.mark.parametrize("damage", ["remove", "corrupt"])
def test_damaged_backup_never_deletes_new_files(nongit, damage):
    manager = checkpoints.CheckpointManager(str(nongit))
    cp = manager.create_checkpoint()
    assert cp
    saved = Path(cp.files_snapshot["original"])
    saved.unlink() if damage == "remove" else saved.write_text("corrupt")
    (nongit / "new").write_text("must survive failed undo")
    (nongit / "original").write_text("current original")
    assert not manager.rollback().success
    assert (nongit / "new").read_text() == "must survive failed undo"
    assert (nongit / "original").read_text() == "current original"


def test_undo_preserves_excluded_cache(nongit):
    cache = nongit / "__pycache__"
    cache.mkdir()
    (cache / "keep").write_text("cache")
    manager = checkpoints.CheckpointManager(str(nongit))
    assert manager.create_checkpoint()
    assert manager.rollback().success
    assert (cache / "keep").read_text() == "cache"


def test_restore_does_not_follow_replaced_file_symlink(nongit, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("outside")
    manager = checkpoints.CheckpointManager(str(nongit))
    assert manager.create_checkpoint()
    (nongit / "original").unlink()
    (nongit / "original").symlink_to(outside)
    assert manager.rollback().success
    assert outside.read_text() == "outside"
    assert not (nongit / "original").is_symlink()


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_pool_cancellation_releases_slot_without_marking_success(exc):
    pool = Pool([("lane", 1)])

    def cancel(*args):
        raise exc()

    with pytest.raises(exc):
        walk(pool, cancel, deadline=pool.clock() + 1)
    assert pool.snapshot()[0]["in_flight"] == 0
    assert pool.snapshot()[0]["ok"] == 0


@pytest.mark.parametrize("url", ["http://localhost:8080", "http://127.0.0.1:8080",
                                  "http://192.0.2.1", "http://100.64.0.1"])
@pytest.mark.parametrize("price, expected", [((10.0, 30.0), config.METERED),
                                           (None, config.UNKNOWN)])
def test_private_address_is_not_billing_evidence(monkeypatch, url, price, expected):
    monkeypatch.setattr(config, "price_for", lambda *args: price)
    assert config.lane_kind("proxy-model", "proxy", url) == expected


def test_unpriced_private_proxy_does_not_report_zero_cost(monkeypatch):
    from onecli.evals.bench import cost_fields
    monkeypatch.setattr(config, "price_for", lambda *args: None)
    assert not cost_fields("proxy", "proxy", "http://127.0.0.1:8080")["price_known"]


def test_backend_name_is_not_billing_evidence(monkeypatch):
    monkeypatch.delenv(config.PRICING_ENV, raising=False)
    assert config.lane_kind("proxy-model", "llamacpp/proxy", "https://proxy.invalid") == config.UNKNOWN


def test_health_updates_are_serialized(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    monkeypatch.setenv(lanehealth.STATE_ENV, str(path))
    original = lanehealth._load
    active = 0
    maximum = 0
    lock = threading.Lock()
    start = threading.Barrier(2)

    def slow_load():
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        data = original()
        time.sleep(0.05)
        with lock:
            active -= 1
        return data

    monkeypatch.setattr(lanehealth, "_load", slow_load)

    def update(name):
        start.wait(timeout=2)
        lanehealth.mark_down(name)

    threads = [threading.Thread(target=update, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert maximum == 1
    assert set(json.loads(path.read_text())["lanes"]) == {"a", "b"}


def _health_process(path, name, start):
    os.environ[lanehealth.STATE_ENV] = path
    original = lanehealth._load

    def slow_load():
        data = original()
        time.sleep(0.1)
        return data

    lanehealth._load = slow_load
    start.wait(timeout=5)
    lanehealth.mark_down(name)


def test_health_process_updates_merge(tmp_path):
    path = tmp_path / "health.json"
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Barrier(3)
    workers = [ctx.Process(target=_health_process, args=(str(path), str(n), start))
               for n in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert set(json.loads(path.read_text())["lanes"]) == {"0", "1", "2"}


@pytest.mark.parametrize("route", ["chat_completions", "messages"])
def test_legacy_relay_runs_provider_off_event_loop(monkeypatch, route):
    # The legacy module initializes at import: prevent discovery and token reads
    # before installing the request stub, including when this test runs alone.
    monkeypatch.setattr(config, "discover", lambda **kwargs: [])
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        "{}" if path.name in {"host_tokens.json", "tokens.json"}
                        else read_text(path, *args, **kwargs))
    from onecli.relay import server
    from onecli.providers.base import Turn
    loop_thread = threading.get_ident()
    worker_threads = []
    monkeypatch.setattr(server, "_TOKENS", {"token": "owner"})

    def run(body):
        worker_threads.append(threading.get_ident())
        return "lane", SimpleNamespace(model="m"), Turn(text="ok"), []

    class Request:
        async def json(self):
            return {"messages": [{"role": "user", "content": "hi"}]}

    monkeypatch.setattr(server, "_run", run)
    response = asyncio.run(getattr(server, route)(Request(), "Bearer token"))
    assert response.status_code == 200
    assert worker_threads and worker_threads[0] != loop_thread


def test_existing_git_index_lock_blocks_undo_without_mutation(repo):
    manager = checkpoints.CheckpointManager(str(repo))
    assert manager.create_checkpoint()
    (repo / "file").write_text("current")
    lock = repo / ".git/index.lock"
    lock.write_text("another git writer")
    result = manager.rollback()
    assert not result.success
    assert (repo / "file").read_text() == "current"
    assert lock.read_text() == "another git writer"


def test_linked_worktree_restores_its_own_index(repo, tmp_path):
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-qb", "linked-test", str(linked))
    (linked / "file").write_text("staged")
    git(linked, "add", "file")
    (linked / "file").write_text("unstaged")
    manager = checkpoints.CheckpointManager(str(linked))
    assert manager.create_checkpoint()
    (linked / "file").write_text("agent")
    assert manager.rollback().success
    assert git(linked, "show", ":file") == b"staged"
    assert (linked / "file").read_text() == "unstaged"
    assert (repo / "file").read_text() == "committed\n"


def test_git_subdirectory_refuses_undo_without_touching_siblings(repo):
    sub = repo / "sub"
    sub.mkdir()
    (sub / "file").write_text("original")
    manager = checkpoints.CheckpointManager(str(sub))
    assert manager.create_checkpoint() is None
    (repo / "file").write_text("sibling edit")
    (sub / "file").write_text("agent edit")
    assert not manager.rollback().success
    assert (sub / "file").read_text() == "agent edit"
    assert (repo / "file").read_text() == "sibling edit"


def test_non_git_parent_symlink_refuses_undo(nongit, tmp_path):
    sub = nongit / "sub"
    sub.mkdir()
    (sub / "file").write_text("original")
    manager = checkpoints.CheckpointManager(str(nongit))
    assert manager.create_checkpoint()
    (sub / "file").unlink()
    sub.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside")
    sub.symlink_to(outside, target_is_directory=True)
    (nongit / "new").write_text("preserve on error")
    assert not manager.rollback().success
    assert (outside / "file").read_text() == "outside"
    assert (nongit / "new").exists()


def test_non_git_symlinks_are_captured_without_reading_target(nongit, tmp_path):
    missing = tmp_path / "not-there"
    link = nongit / "link"
    link.symlink_to(missing)
    manager = checkpoints.CheckpointManager(str(nongit))
    assert manager.create_checkpoint()
    link.unlink()
    link.write_text("replaced")
    assert manager.rollback().success
    assert link.is_symlink()
    assert os.readlink(link) == str(missing)
    assert not missing.exists()


def test_non_git_restore_error_never_deletes_new_files(nongit, monkeypatch):
    manager = checkpoints.CheckpointManager(str(nongit))
    assert manager.create_checkpoint()
    (nongit / "new").write_text("preserve on error")

    def fail_copy(*args, **kwargs):
        raise OSError("injected restore error")

    monkeypatch.setattr(checkpoints.shutil, "copy2", fail_copy)
    assert not manager.rollback().success
    assert (nongit / "new").exists()


def test_precheck_preserves_updates_during_network_call(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    monkeypatch.setenv(lanehealth.STATE_ENV, str(path))

    def fetch(*args):
        lanehealth.mark_down("other")
        return {"limit": 10, "limit_remaining": 5}

    monkeypatch.setattr(lanehealth, "_fetch_openrouter_key", fetch)
    assert not lanehealth.openrouter_exhausted("https://openrouter.ai/api/v1", "stub", "one")
    assert lanehealth.is_cooling("other")
    assert "one" in json.loads(path.read_text())["precheck"]


def test_failed_atomic_health_replace_keeps_previous_json(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    monkeypatch.setenv(lanehealth.STATE_ENV, str(path))
    lanehealth.mark_down("before")
    previous = path.read_bytes()

    def fail_replace(*args):
        raise OSError("injected replace error")

    monkeypatch.setattr(lanehealth.os, "replace", fail_replace)
    lanehealth.mark_down("after")
    assert path.read_bytes() == previous
    assert set(p.name for p in tmp_path.iterdir()) == {"health.json", "health.json.lock"}

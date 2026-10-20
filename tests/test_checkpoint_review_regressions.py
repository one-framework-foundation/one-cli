"""Independent-review regressions against the checkpoint architecture."""

import os
import subprocess

import pytest

from onecli.checkpoints import CheckpointManager


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo)


@pytest.fixture(autouse=True)
def private_storage(tmp_path, monkeypatch):
    original = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path / "backups") if path.startswith("~/.one-cli/checkpoints")
                        else original(path))


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    (root / "file").write_text("committed")
    git(root, "add", "file")
    git(root, "commit", "-qm", "initial")
    return root


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_capture_reads_flagged_worktree_content_and_preserves_index(repo, flag):
    git(repo, "update-index", flag, "file")
    (repo / "file").write_text("preexisting unstaged work")
    original_index = (repo / ".git/index").read_bytes()
    original_flags = git(repo, "ls-files", "-v", "--", "file")
    manager = CheckpointManager(str(repo))
    assert manager.create_checkpoint()
    assert (repo / ".git/index").read_bytes() == original_index
    (repo / "file").write_text("agent edit")
    result = manager.rollback()
    assert result.success, result.error
    assert (repo / "file").read_text() == "preexisting unstaged work"
    assert git(repo, "show", ":file") == b"committed"
    assert git(repo, "ls-files", "-v", "--", "file") == original_flags
    assert (repo / ".git/index").read_bytes() == original_index


@pytest.mark.parametrize("target", ["root", "ancestor"])
@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_substituted_workspace_or_ancestor_is_refused_before_mutation(tmp_path, target, kind):
    parent = tmp_path / "parent"
    workspace = parent / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "file").write_text("original")
    manager = CheckpointManager(str(workspace))
    assert manager.create_checkpoint()
    replaced = workspace if target == "root" else parent
    replaced.rename(tmp_path / "original-location")
    if kind == "symlink":
        destination = tmp_path / "outside"
        destination.mkdir()
        replaced.symlink_to(destination, target_is_directory=True)
    else:
        replaced.mkdir()
        destination = replaced
    outside = destination if target == "root" else destination / "workspace"
    outside.mkdir(exist_ok=True)
    (outside / "file").write_text("outside original")
    (outside / "unique").write_text("outside unique")
    result = manager.rollback()
    assert not result.success
    assert (outside / "file").read_text() == "outside original"
    assert (outside / "unique").read_text() == "outside unique"
    assert manager.list_checkpoints(), "refusal must retain the checkpoint"


def test_retargeted_initial_workspace_symlink_is_refused(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "file").write_text("original")
    link = tmp_path / "link"
    link.symlink_to(original, target_is_directory=True)
    manager = CheckpointManager(str(link))
    assert manager.create_checkpoint()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside")
    link.unlink()
    link.symlink_to(outside, target_is_directory=True)
    assert not manager.rollback().success
    assert (outside / "file").read_text() == "outside"


def test_git_subdirectory_refuses_incomplete_index_contract(repo):
    sub = repo / "sub"
    sub.mkdir()
    (sub / "file").write_text("staged before")
    git(repo, "add", "sub/file")
    (sub / "file").write_text("unstaged before")
    original_index = (repo / ".git/index").read_bytes()
    manager = CheckpointManager(str(sub))
    # Until scoped index merging preserves all entry flags/stages, no checkpoint
    # is safer than a filesystem-only checkpoint promising successful Git undo.
    assert manager.create_checkpoint() is None
    assert not manager.rollback().success
    assert (repo / ".git/index").read_bytes() == original_index
    assert (sub / "file").read_text() == "unstaged before"


def test_git_root_replacement_is_refused(repo, tmp_path):
    manager = CheckpointManager(str(repo))
    assert manager.create_checkpoint()
    moved = tmp_path / "original-repository"
    repo.rename(moved)
    outside = tmp_path / "outside"
    outside.mkdir()
    git(outside, "init", "-q")
    (outside / "file").write_text("outside")
    repo.symlink_to(outside, target_is_directory=True)
    assert not manager.rollback().success
    assert (outside / "file").read_text() == "outside"

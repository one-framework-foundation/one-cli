import os
import subprocess
import pytest
from onecli.checkpoints import CheckpointManager


@pytest.fixture(autouse=True)
def private_backups(tmp_path, monkeypatch):
    original = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda p:
                        str(tmp_path / "backups") if p.startswith("~/.one-cli/checkpoints")
                        else original(p))


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
    
    file_a = repo / "a.txt"
    file_a.write_text("initial a\n")
    subprocess.run(["git", "add", "a.txt"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(repo), check=True)
    return repo


def test_checkpoint_creation_and_diff(git_repo):
    mgr = CheckpointManager(workspace=str(git_repo), sid="test-sid")
    assert mgr.is_git is True
    
    cp = mgr.create_checkpoint(prompt="turn 1: add features")
    assert cp is not None
    assert cp.id == 1
    assert cp.commit_sha is not None
    assert cp.prompt == "turn 1: add features"

    # Modify a file and create a new file
    file_a = git_repo / "a.txt"
    file_a.write_text("initial a\nmodified line\n")
    file_b = git_repo / "b.txt"
    file_b.write_text("new b\n")

    diff = mgr.get_diff()
    assert "modified line" in diff
    
    stat = mgr.get_diff_stat()
    assert "a.txt" in stat


def test_checkpoint_rollback(git_repo):
    mgr = CheckpointManager(workspace=str(git_repo), sid="test-sid")
    mgr.create_checkpoint(prompt="before edits")

    # Make modifications
    file_a = git_repo / "a.txt"
    file_a.write_text("modified content\n")
    file_b = git_repo / "sub" / "b.txt"
    file_b.parent.mkdir(parents=True)
    file_b.write_text("new sub file\n")

    assert file_b.exists()

    res = mgr.rollback()
    assert res.success is True
    assert res.checkpoint_id == 1
    assert "a.txt" in res.reverted_files
    assert any("b.txt" in f for f in res.removed_files)

    # File A is reverted
    assert file_a.read_text() == "initial a\n"
    # File B is removed
    assert not file_b.exists()
    # Checkpoints list is updated
    assert len(mgr.list_checkpoints()) == 0


def test_non_git_workspace_checkpoint(tmp_path):
    ws = tmp_path / "nongit"
    ws.mkdir()
    f1 = ws / "file1.txt"
    f1.write_text("v1\n")

    mgr = CheckpointManager(workspace=str(ws), sid="nongit-sid")
    assert mgr.is_git is False

    cp = mgr.create_checkpoint(prompt="non git turn")
    assert cp is not None
    assert cp.id == 1

    # Modify f1, add f2
    f1.write_text("v2\n")
    f2 = ws / "file2.txt"
    f2.write_text("v2\n")

    res = mgr.rollback()
    assert res.success is True
    assert f1.read_text() == "v1\n"
    assert not f2.exists()


def test_rollback_keeps_files_ignored_before_the_turn(git_repo):
    """A turn that deletes .gitignore must not make undo unlink the files it
    ignored: they were never in the snapshot and have no other copy."""
    (git_repo / ".gitignore").write_text("data/\nlocal.cfg\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=str(git_repo), check=True)
    subprocess.run(["git", "commit", "-qm", "ignore"], cwd=str(git_repo), check=True)
    (git_repo / "data").mkdir()
    (git_repo / "data" / "results.csv").write_text("1,2,3\n")
    (git_repo / "local.cfg").write_text("token=x\n")
    mgr = CheckpointManager(workspace=str(git_repo), sid="test-sid")
    assert mgr.create_checkpoint(prompt="turn") is not None

    (git_repo / ".gitignore").unlink()
    (git_repo / "made.txt").write_text("new\n")

    res = mgr.rollback()
    assert res.success is True, res.error
    assert (git_repo / "data" / "results.csv").read_text() == "1,2,3\n"
    assert (git_repo / "local.cfg").read_text() == "token=x\n"
    assert (git_repo / ".gitignore").read_text() == "data/\nlocal.cfg\n"
    assert not (git_repo / "made.txt").exists()
    assert res.removed_files == ["made.txt"]


def test_rollback_keeps_files_ignored_by_rules_outside_the_snapshot(git_repo):
    """.git/info/exclude is not restored by undo, so only the pre-turn file
    list protects what it ignored."""
    exclude = git_repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    exclude.write_text("*.log\n")
    (git_repo / "run.log").write_text("history\n")
    mgr = CheckpointManager(workspace=str(git_repo), sid="test-sid")
    assert mgr.create_checkpoint(prompt="turn") is not None

    exclude.write_text("")

    res = mgr.rollback()
    assert res.success is True, res.error
    assert (git_repo / "run.log").read_text() == "history\n"

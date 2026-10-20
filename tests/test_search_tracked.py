"""Project searches read tracked files only; explicit files and non-repos
are searched directly, and the header always names the backend."""
import shutil
import subprocess

import pytest

from onecli.tools import search
from onecli.tools.search import grep


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # No registered index: these cases cover the shipped backends only.
    monkeypatch.setattr(search, "INDEXES", {})
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in ("src/live.py", "other/live.py", "src/readme.md"):
        p = root / name
        p.parent.mkdir(exist_ok=True)
        p.write_text("unique_tracked_probe\n-option_probe\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    (root / "src" / "untracked.py").write_text("unique_tracked_probe\n")
    return root


def test_project_search_reads_tracked_files_only(repo):
    result = grep(str(repo), "unique_tracked_probe", path="src", glob="*.py")
    assert result.startswith("[python]")
    assert str(repo / "src/live.py") + ":1:" in result
    assert "untracked.py" not in result
    assert "other" not in result
    assert "readme" not in result
    assert "untracked files are not searched" in result


def test_explicit_untracked_file_remains_searchable(repo):
    assert "unique_tracked_probe" in grep(str(repo), "unique_tracked_probe", "src/untracked.py")


def test_ignored_subdir_with_no_tracked_files_falls_back_to_rg(repo):
    ignored = repo / "ignored"
    ignored.mkdir()
    (repo / ".gitignore").write_text("ignored/\n")
    subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True)
    (ignored / "scratch.py").write_text("BEACON_PROBE\n")
    result = grep(str(repo), "BEACON_PROBE", path="ignored")
    assert "tracked files only" not in result
    assert str(ignored / "scratch.py") + ":1:BEACON_PROBE" in result


def test_option_pattern_and_global_limit_on_tracked_files(repo):
    result = grep(str(repo), "-option_probe", max_results=1, context=0)
    matches = [line for line in result.splitlines() if ":2:-option_probe" in line]
    assert len(matches) == 1
    found = int(result.split("] ", 1)[1].split()[0])
    # A cut is announced, never silent.
    assert (found > 1) == ("more matching lines truncated" in result)


def test_registered_index_answers_and_a_missing_one_is_announced(repo, monkeypatch):
    calls = []

    def rows(repo_root, relative, root, p, glob, timeout):
        calls.append((relative, glob))
        return [(repo_root / "src/live.py", "src/live.py", 1, "unique_tracked_probe")]

    monkeypatch.setattr(search, "INDEXES", {"index": lambda: rows})
    result = grep(str(repo), "unique_tracked_probe", glob="*.py")
    assert result.startswith("[index]") and calls == [(".", "*.py")]
    assert search.search(str(repo), "unique_tracked_probe",
                         backend="python").startswith("[python]")
    monkeypatch.setattr(search, "INDEXES", {"index": lambda: "index not installed"})
    result = grep(str(repo), "unique_tracked_probe")
    assert result.startswith("[python: index not installed]")
    assert "untracked.py" not in result   # still tracked files only


def test_non_repo_and_python_explicit_file(tmp_path, monkeypatch):
    target = tmp_path / "log.txt"
    target.write_text("-option_probe\n")
    assert "-option_probe" in grep(str(tmp_path), "-option_probe")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert "-option_probe" in grep(str(tmp_path), "-option_probe", "log.txt")


def test_guard_guidance_names_grep_tool():
    from onecli.tools.shell import bash
    result = bash(".", "grep -r needle .")
    assert "grep tool" in result
    assert "use rg" not in result


def test_failed_tracked_listing_does_not_expand_search_scope(repo, monkeypatch):
    real_run = subprocess.run

    def run(cmd, **kwargs):
        if "ls-files" in cmd:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="index unavailable")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    result = grep(str(repo), "unique_tracked_probe")
    assert result.startswith("error: cannot list tracked files:")
    assert "index unavailable" in result

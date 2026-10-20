"""Loop-guard regressions: checkpoint decisions, shell accounting and safety."""

from __future__ import annotations

import json
import subprocess

from onecli import loopguard as loops
from onecli.providers.base import ToolCall
from onecli.tools import shell





def _search(i):
    return [ToolCall(id=f"s{i}", name="grep", arguments={"pattern": f"garden{i}", "path": "."})]






def test_reading_at_the_checkpoint_is_still_not_a_commit():
    assert not loops.checkpoint_commits("", [("grep", {"pattern": "x"})])
    assert not loops.checkpoint_commits("", [("bash", {"command": "cd /b && ls build && cat x.py | head"})])
    assert not loops.checkpoint_commits("", [("bash", {"command": "sed -n 1,5p x.py"})])
    assert loops.checkpoint_commits("COMMIT: the build", [("grep", {"pattern": "x"})])
















def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def test_stash_pop_after_an_empty_push_is_refused(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "base")
    (tmp_path / "theirs.txt").write_text("another session's work")
    _git(repo, "add", "theirs.txt")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "stash", "push", "-q", "-m", "theirs")
    shell._EMPTY_STASH.discard(repo)
    out = shell.bash(repo, "git stash push -m mine")
    assert "No local changes to save" in out
    refused = shell.bash(repo, "git stash pop")
    # Before: the pop restored the other session's stash into this tree.
    assert refused.startswith("blocked:") and "saved nothing" in refused
    assert not (tmp_path / "theirs.txt").exists()
    assert "theirs" in _git(repo, "stash", "list").stdout
    # A push that saves something makes the pop legitimate again.
    (tmp_path / "mine.txt").write_text("x")
    shell.bash(repo, "git add mine.txt && git stash push -m mine")
    assert not shell.bash(repo, "git stash pop").startswith("blocked:")
    assert (tmp_path / "mine.txt").exists()


def test_a_background_process_ends_with_the_session(tmp_path):
    out = shell.bash(str(tmp_path), "sleep 60 > /dev/null 2>&1 &")
    assert "left a process running in the background" in out
    pgid = int(out.split("process group ")[1].split(")")[0])
    assert shell._group_alive(pgid)
    stopped = shell.reap()
    assert any(f"pgid {pgid}" in s for s in stopped)
    for _ in range(50):
        if not shell._group_alive(pgid):
            break
        subprocess.run(["sleep", "0.05"])
    assert not shell._group_alive(pgid)
    assert "background" not in shell.bash(str(tmp_path), "true")




def test_shell_session_state_survives_a_reload(tmp_path):
    import importlib

    shell._EMPTY_STASH.add(str(tmp_path))
    reloaded = importlib.reload(shell)
    assert str(tmp_path) in reloaded._EMPTY_STASH
    assert reloaded.bash(str(tmp_path), "git stash pop").startswith("blocked:")
    reloaded._EMPTY_STASH.discard(str(tmp_path))


def test_the_analyzer_does_not_count_test_and_ci_shell_as_search():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "session_waste.py"
    spec = importlib.util.spec_from_file_location("session_waste", path)
    waste = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(waste)

    def transcript(commands):
        messages = [{"role": "user", "content": "fix CI"}]
        for i, command in enumerate(commands):
            messages.append({"role": "assistant", "tool_calls": [{"id": f"c{i}", "function": {
                "name": "bash", "arguments": json.dumps({"command": command})}}]})
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "exit 0\nok"})
        return messages

    ci = ["cd /repo && python3 -m pytest -q tests/test_a.py", "ruff check onecli",
          "gh pr checks 240", "git status --short", "mypy onecli"] * 3
    # CI work is not searching, however many rounds it takes.
    assert waste.analyze(transcript(ci)).get("long_search", 0) == 0
    search = [f"ls d{i} && cat f{i}" for i in range(15)]
    assert waste.analyze(transcript(search))["long_search"] == 15 - waste.DISCOVERY_LIMIT
    # An image check with PIL is not a hand-rolled image route.
    assert not waste.analyze(transcript(["python3 -c 'from PIL import Image; Image.open(\"a\")'"]))






def _repo_with_their_stash(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "base")
    (tmp_path / "theirs.txt").write_text("x")
    _git(repo, "add", "theirs.txt")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "stash", "push", "-q", "-m", "theirs")
    shell._EMPTY_STASH.discard(repo)
    return repo


def test_a_quiet_or_short_form_stash_that_saved_work_can_be_popped(tmp_path):
    repo = _repo_with_their_stash(tmp_path)
    shell.bash(repo, "git stash")
    for form in ("git stash -u", "git stash -m mine", "git stash push -q"):
        (tmp_path / "mine.txt").write_text(form)
        shell.bash(repo, "git add mine.txt" if form != "git stash -u" else "true")
        shell.bash(repo, form)
        popped = shell.bash(repo, "git stash pop")
        assert not popped.startswith("blocked:"), form
        assert (tmp_path / "mine.txt").read_text() == form
        _git(repo, "reset", "-q")
        (tmp_path / "mine.txt").unlink()
        shell.bash(repo, "git stash")  # empty again
        assert repo in shell._EMPTY_STASH


def test_stash_and_pop_in_one_command_is_refused(tmp_path):
    repo = _repo_with_their_stash(tmp_path)
    out = shell.bash(repo, "git stash && true; git stash pop")
    assert out.startswith("blocked:")
    assert not (tmp_path / "theirs.txt").exists()


def test_reap_leaves_a_reused_pgid_alone(tmp_path):
    out = shell.bash(str(tmp_path), "sleep 60 > /dev/null 2>&1 &")
    pgid = int(out.split("process group ")[1].split(")")[0])
    command, member = shell._LEFT_RUNNING[pgid]
    shell._LEFT_RUNNING[pgid] = (command, member.split(":")[0] + ":0")  # a different process
    assert shell.reap() == []
    assert shell._group_alive(pgid)
    import os
    import signal
    os.killpg(pgid, signal.SIGTERM)


def test_a_silent_rg_is_stopped_at_its_deadline(tmp_path):
    # An rg that prints nothing must still stop at the deadline, which is
    # enforced by a timer, not only between output lines.
    import time
    from types import SimpleNamespace

    import pytest

    from onecli.tools import search
    fake = tmp_path / "rg"
    fake.write_text("#!/bin/sh\nexec sleep 30\n")
    fake.chmod(0o755)
    plan = SimpleNamespace(ignore_case=False, backend="x")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        search._rg_rows(str(fake), tmp_path, plan, "", 1)
    assert time.monotonic() - started < 5

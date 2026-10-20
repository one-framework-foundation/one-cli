"""A workspace the operator cannot read is an error message, not a traceback.

Found by driving the CLI at a mode-000 directory: `git_root` walked into it
looking for `.git` and `PermissionError` came out of `Path.exists()` all the
way to the top, printing a nine-frame traceback at somebody who had mistyped a
path. The sibling case - a workspace that is a FILE, so ENOTDIR - was already
quiet, because `Path.exists()` ignores that errno and not this one.
"""

from __future__ import annotations

import os

import pytest

from onecli import permissions


needs_a_wall = pytest.mark.skipif(
    os.getuid() == 0,
    reason="root reads a mode-000 directory, so there is no wall to hit")


@needs_a_wall
def test_git_root_walks_past_a_directory_it_may_not_read(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    (tmp_path / ".git").mkdir()
    locked.chmod(0o000)
    try:
        assert permissions.git_root(str(locked)) == str(tmp_path)
    finally:
        locked.chmod(0o755)


@needs_a_wall
def test_git_root_returns_none_when_every_rung_is_shut(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        # No .git anywhere above tmp_path in a pytest tmpdir, so the honest
        # answer is None - and it must be an answer, not an exception.
        assert permissions.git_root(str(locked)) is None
    finally:
        locked.chmod(0o755)


@needs_a_wall
def test_compose_system_survives_an_unreadable_workspace(tmp_path):
    """The path the CLI actually takes: context files, then the system prompt."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        assert permissions.context_files(str(locked), home=str(tmp_path)) == []
        assert isinstance(
            permissions.compose_system("be helpful", str(locked),
                                       home=str(tmp_path)), str)
    finally:
        locked.chmod(0o755)

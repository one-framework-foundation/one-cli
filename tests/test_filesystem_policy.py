"""Release-candidate filesystem policy regressions."""
from pathlib import Path

import pytest

from onecli import commandfiles, permissions, spill
from onecli.checkpoints import CheckpointManager
from onecli.tools import search


def test_read_deny_covers_edit(tmp_path):
    guard = permissions.PermissionGuard(
        deny=permissions.parse_rules(['Read(secret.txt)']), workspace=str(tmp_path))
    assert not guard.check('edit', {'path': 'secret.txt', 'old_string': 'missing'}, True).allowed


def test_session_spill_is_readable_but_not_writable(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    store = spill.SpillStore('session', root=tmp_path / 'private-spills')
    guard = permissions.RootsGuard(roots=[str(workspace)], workspace=str(workspace))
    # The session constructor uses this helper; it never grants the state root.
    if hasattr(permissions, "add_read_root"):
        permissions.add_read_root(guard, store.root)
    store.spill('read', 'full output' * 100, 100)
    path = next(store.root.iterdir())
    assert guard.check('read', {'path': str(path)}, False).allowed
    assert not guard.check('write', {'path': str(path)}, True).allowed
    assert not guard.check('edit', {'path': str(path)}, False).allowed
    assert not guard.check('read', {'path': str(store.root.parent / 'other' / 'secret')}, False).allowed


@pytest.mark.parametrize('pattern', ['(a+)+$', '(a{1,4})+$', '(a|aa)+$', 'a*a*$', r'(a+)\1'])
def test_restricted_search_rejects_backtracking_patterns(tmp_path, pattern):
    # Short input makes the original implementation fail promptly, not hang.
    (tmp_path / 'text').write_text('aaaa!')
    result = search.restricted('grep', str(tmp_path), {'pattern': pattern}, [])
    assert 'error: regex exceeds the bounded search subset' in result


def test_project_commands_require_trust(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: home)
    project = tmp_path / 'repo'
    commands = project / '.claude' / 'commands'
    commands.mkdir(parents=True)
    (commands / 'review.md').write_text('project prompt')
    assert 'review' not in commandfiles.load(str(project))[0]
    assert commandfiles.load(str(project), trust_project=True)[0]['review'].body == 'project prompt'


@pytest.mark.parametrize('ancestor', [False, True])
def test_command_file_symlinks_are_not_read(tmp_path, ancestor):
    secret = tmp_path / 'secret.md'
    secret.write_text('private credential')
    if ancestor:
        directory = tmp_path / 'real'
        directory.mkdir()
        (directory / 'command.md').write_text('private credential')
        (tmp_path / 'link').symlink_to(directory, target_is_directory=True)
        path = tmp_path / 'link' / 'command.md'
    else:
        path = tmp_path / 'command.md'
        path.symlink_to(secret)
    assert commandfiles.parse(path, 'command') is None


@pytest.mark.parametrize('parent_conflict', [False, True])
def test_undo_preflights_all_destination_types(tmp_path, monkeypatch, request, parent_conflict):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    monkeypatch.setattr('onecli.checkpoints.os.path.expanduser', lambda p: str(tmp_path / 'backups'))
    a = workspace / 'a'
    a.write_text('original')
    b = workspace / ('dir/b' if parent_conflict else 'b')
    b.parent.mkdir(exist_ok=True)
    b.write_text('original b')
    manager = CheckpointManager(workspace=str(workspace), sid='rc')
    request.addfinalizer(manager.close)
    assert not manager.is_git
    cp = manager.create_checkpoint('before')
    # Explicit order demonstrates an earlier file must not be restored first.
    cp.files_snapshot = dict(sorted(cp.files_snapshot.items()))
    a.write_text('changed')
    b.unlink()
    if parent_conflict:
        b.parent.rmdir()
        b.parent.write_text('changed directory to file')
    else:
        b.mkdir()
    result = manager.rollback()
    assert not result.success
    assert a.read_text() == 'changed'
    assert manager.list_checkpoints()
    manager.close()

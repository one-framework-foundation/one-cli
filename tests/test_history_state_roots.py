"""Release regressions: private history and operator-selected state roots."""
import os
import stat
from types import SimpleNamespace

import pytest

from onecli import inputline, secureio, session


def test_prompt_history_is_private_even_with_permissive_umask(tmp_path, monkeypatch):
    history = tmp_path / '.one-cli' / 'history'
    monkeypatch.setattr(inputline, 'HISTORY', history)
    rl = SimpleNamespace(get_current_history_length=lambda: 1,
                         get_history_item=lambda _: 'private prompt',
                         write_history_file=lambda path: path.write_text('private prompt\n'))
    old = os.umask(0)
    try:
        history.parent.mkdir(mode=0o777)
        inputline._save_history(rl)
    finally:
        os.umask(old)
    assert history.read_text() == 'private prompt\n'
    assert stat.S_IMODE(history.stat().st_mode) == 0o600
    assert stat.S_IMODE(history.parent.stat().st_mode) == 0o700


def test_symlinked_state_root_supports_session_append_without_following_children(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    target = tmp_path / 'dotfiles-state'
    target.mkdir(mode=0o755)
    (home / '.one-cli').symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(secureio.Path, 'home', classmethod(lambda cls: home))
    monkeypatch.setattr(session, 'SESSION_DIR', home / '.one-cli' / 'sessions')
    session.append('rc', {'kind': 'test', 'value': 'saved'})
    assert 'saved' in (target / 'sessions' / 'rc.jsonl').read_text()
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    outside = tmp_path / 'outside'
    outside.write_text('keep')
    (target / 'sessions' / 'linked.jsonl').symlink_to(outside)
    with pytest.raises(OSError):
        session.append('linked', {'kind': 'test'})
    assert outside.read_text() == 'keep'


def test_history_writer_refuses_final_symlink(tmp_path):
    target = tmp_path / 'secret'
    target.write_text('keep')
    history = tmp_path / 'history'
    history.symlink_to(target)
    rl = SimpleNamespace(get_current_history_length=lambda: 1,
                         get_history_item=lambda _: 'new')
    with pytest.raises(OSError):
        secureio.write_history(rl, history)
    assert target.read_text() == 'keep'


def test_state_root_exception_does_not_follow_child_directory_links(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    state = tmp_path / 'state'
    state.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (home / '.one-cli').symlink_to(state, target_is_directory=True)
    (state / 'sessions').symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(secureio.Path, 'home', classmethod(lambda cls: home))
    monkeypatch.setattr(session, 'SESSION_DIR', home / '.one-cli' / 'sessions')
    with pytest.raises(OSError):
        session.append('rc', {'kind': 'test'})
    assert list(outside.iterdir()) == []

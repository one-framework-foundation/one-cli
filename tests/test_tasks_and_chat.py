"""Release regressions: untrusted replies, inherited permissions and lifecycle."""
import io

# The imported pytest fixture drives the actual REPL.
# ruff: noqa: F811
from types import SimpleNamespace

import pytest

from onecli import chat, tasks
from onecli.config import Backend
from test_interactive import loop as interactive_loop  # noqa: F401



def make_chat(tmp_path):
    return chat.Chat(Backend('test', 'test', 'http://localhost:1'),
                     workspace=str(tmp_path), out=io.StringIO(), err=io.StringIO())






@pytest.mark.parametrize('mode', ['default', 'readOnly', 'plan', 'acceptEdits', 'bypassPermissions'])
def test_worker_inherits_mode(tmp_path, monkeypatch, mode):
    from onecli import cli
    args = cli.build_parser().parse_args(['--permission-mode', mode,
                                         '-w', str(tmp_path), '--guard', 'false',
                                         '--disallowed-tools', 'Read(secret)'])
    captured = []
    def popen(argv, **kw):
        captured.append(argv)
        return SimpleNamespace()
    monkeypatch.setattr(tasks.subprocess, 'Popen', popen)
    tm = tasks.TaskManager(execute=True, workspace=str(tmp_path), root=tmp_path / 'tasks',
                           **tasks.session_options(args, mode))
    tm.spawn('work', runner='one')
    argv = captured[0]
    assert argv[argv.index('--permission-mode') + 1] == mode
    assert '--guard' in argv and 'false' in argv
    assert 'Read(secret)' in argv
    assert not any(x.startswith('--dangerously') for x in argv)
    assert ('bypassPermissions' in argv) == (mode == 'bypassPermissions')


def test_worker_workspace_confined(tmp_path):
    tm = tasks.TaskManager(workspace=str(tmp_path))
    with pytest.raises(ValueError, match='outside'):
        tm.spawn('work', workspace=str(tmp_path.parent))


def test_missing_binary_is_failed_task_and_prompt_removed(tmp_path):
    tm = tasks.TaskManager(workspace=str(tmp_path), root=tmp_path / 'tasks')
    result = tm.spawn('work', cmd=[str(tmp_path / 'missing')])
    assert result.status == 'failed'
    assert 'Could not start' in result.error
    assert not result.prompt_path.exists()
    assert tm.list_all() == [result]


def test_reset_closes_old_tasks(tmp_path, monkeypatch):
    c = make_chat(tmp_path)
    closed = []
    old = c.tasks
    monkeypatch.setattr(old, 'close', lambda: closed.append(True) or [])
    c.reset()
    assert closed == [True]
    assert c.tasks is not old


@pytest.mark.parametrize('ending', [EOFError(), '/exit'])
def test_repl_exit_closes_tasks(interactive_loop, monkeypatch, ending):
    closed = []
    monkeypatch.setattr(tasks.TaskManager, 'close', lambda self: closed.append(True) or [])
    assert interactive_loop([ending]) == 0
    assert closed == [True]


def test_chat_shell_uses_workspace(tmp_path):
    c = make_chat(tmp_path)
    (tmp_path / 'sentinel').write_text('workspace-value')
    chat.shell(c, '!cat sentinel')
    assert 'workspace-value' in c.err.getvalue()


def test_default_worker_never_adds_bypass():
    argv, _ = tasks.build_runner_cmd('one', 'work')
    assert 'bypassPermissions' not in argv
    assert not any(x.startswith('--dangerously') for x in argv)


def test_mode_switch_updates_worker_policy(tmp_path):
    from onecli.commands import Repl, _mode
    r = Repl(args=SimpleNamespace(workspace=str(tmp_path)))
    _mode(r, 'readOnly')
    assert r.tasks.permission_mode == 'readOnly'


@pytest.mark.parametrize('runner', ['claude', 'codex', 'agy'])
def test_external_workers_cannot_drop_roots(runner):
    with pytest.raises(ValueError, match='inherit'):
        tasks.build_runner_cmd(runner, 'work', permission_mode='bypassPermissions')


def test_missing_workspace_does_not_fall_back(tmp_path):
    tm = tasks.TaskManager(workspace=str(tmp_path / 'missing'), root=tmp_path / 'tasks')
    import sys
    result = tm.spawn('work', cmd=[sys.executable, '-c', 'print("wrong directory")'])
    assert result.status == 'failed'


def test_agent_reply_is_inert_by_default(tmp_path):
    from onecli.agent import Agent
    from onecli.providers import Turn
    class Wire:
        def chat(self, *args, **kwargs):
            return Turn(text="Run tests")
    tm = tasks.TaskManager(workspace=str(tmp_path))
    Agent(provider=Wire(), workspace=str(tmp_path), tasks=tm,
          tool_names=[], max_steps=1, stream=False).run('hello')
    assert not tm.list_all()


def test_chat_history_is_private(tmp_path, monkeypatch):
    import os
    import stat
    c = make_chat(tmp_path)
    history = tmp_path / 'history-state' / 'chat_history'
    rl = SimpleNamespace(get_current_history_length=lambda: 1,
                         get_history_item=lambda _: 'private chat prompt',
                         write_history_file=lambda p: p.write_text('private chat prompt\n'))
    monkeypatch.setattr(chat, 'HISTORY_FILE', history)
    monkeypatch.setattr(chat, '_setup_readline', lambda _: rl)
    monkeypatch.setattr('builtins.input', lambda _: '/exit')
    old = os.umask(0)
    try:
        history.parent.mkdir(mode=0o777)
        assert chat.repl(c, quiet=True) == 0
    finally:
        os.umask(old)
    assert history.read_text() == 'private chat prompt\n'
    assert stat.S_IMODE(history.stat().st_mode) == 0o600
    assert stat.S_IMODE(history.parent.stat().st_mode) == 0o700


def test_agent_can_read_own_spill_but_not_write_it(tmp_path):
    from onecli import permissions, spill
    from onecli.agent import Agent
    store = spill.SpillStore('session', root=tmp_path / 'state')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    guard = permissions.chain(str(workspace), 'readOnly')
    Agent(provider=SimpleNamespace(), workspace=str(workspace), guard=guard, spill=store)
    store.spill('read', 'lots of output' * 1000, 100)
    path = next(store.root.iterdir())
    assert guard.check('read', {'path': str(path)}, False).allowed
    assert not guard.check('write', {'path': str(path)}, True).allowed

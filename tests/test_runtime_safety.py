"""Offline regressions for the release candidate runtime safety blockers."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from onecli import codemode, config, drivers, harnesses, jobs, loopguard, orc, permissions, subagent, tasks
from onecli.evals import arena, arena_parity, bench, stress, stresslane
from onecli.tools import fs, search, shell


@pytest.fixture
def public_umask():
    before = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(before)


def cell(tmp_path, wire=config.WIRE_OPENAI):
    ws, cfg = tmp_path / 'ws', tmp_path / 'cfg'
    ws.mkdir(parents=True)
    cfg.mkdir()
    return harnesses.Cell(config.Backend(name='test', model='test', base_url='https://example.test/v1',
                                         api_key='release-test-secret', wire=wire), '', ws, 'hello', cfg_dir=cfg)


@pytest.mark.parametrize('wire,keyname', [(config.WIRE_OPENAI, 'OPENAI_API_KEY'),
                                         (config.WIRE_ANTHROPIC, 'ANTHROPIC_API_KEY')])
def test_mini_key_never_enters_argv_or_guard(tmp_path, wire, keyname):
    c = cell(tmp_path, wire)
    launch = harnesses._mini_build(c)
    seen = []
    guard = SimpleNamespace(check=lambda *args: seen.append(args))
    drivers.guard_verdict(guard, launch.cmd, str(c.workspace))
    assert c.key not in ' '.join(launch.cmd)
    assert c.key not in repr(seen)
    assert launch.env[keyname] == c.key


@pytest.mark.parametrize('name', ['pi', 'opencode', 'kilo', 'openclaw', 'crush', 'droid', 'hermes', 'mimocode'])
def test_adapter_key_files_are_private(tmp_path, name, public_umask):
    c = cell(tmp_path)
    build = getattr(harnesses, '_' + name + '_build')
    build(c)
    keyfiles = [p for p in c.cfg_dir.rglob('*') if p.is_file() and c.key in p.read_text()]
    assert keyfiles
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in keyfiles)


@pytest.mark.parametrize('probe', [False, True])
@pytest.mark.parametrize('failure', ['ok', 'launch', 'build', 'timeout'])
def test_eval_config_private_and_removed_on_every_exit(tmp_path, monkeypatch, probe, failure, public_umask):
    captured = []
    backend = config.Backend(name='test', model='test', base_url='https://example.test/v1', api_key='secret')
    def build(c):
        captured.append(c.cfg_dir)
        assert c.cfg_dir.stat().st_mode & 0o777 == 0o700
        (c.cfg_dir / 'secret').write_text('secret')
        if failure == 'build':
            raise ValueError('bad adapter')
        return harnesses.Launch(['offline'], str(c.workspace))
    def run(*args, **kwargs):
        if failure == 'launch':
            raise OSError('missing executable')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired('offline', 1)
        return SimpleNamespace(stdout='', stderr='')
    harness = SimpleNamespace(name='offline', wires=(backend.wire,), build=build,
                              parse=lambda *a: harnesses.Result())
    monkeypatch.setattr(arena.subprocess, 'run', run)
    task = bench.Task('test', 'hello')
    monkeypatch.setattr(task, 'check', lambda *a: (True, ''))
    try:
        if probe:
            arena_parity._probe_once(harness, backend, tmp_path, 1)
        else:
            arena.run_cell(harness, backend, task, tmp_path, 1, '', False)
    except ValueError:
        assert failure == 'build'
    assert captured and not captured[0].exists()


@pytest.mark.parametrize('kind', ['absolute', 'home'])
def test_deny_anchor_resolves_literal_symlink(tmp_path, monkeypatch, kind):
    real = tmp_path / 'real'
    real.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv('HOME', str(link))
    pattern = f'Read(/{link}/**)' if kind == 'absolute' else 'Read(~/secret/**)'
    target = real / 'file' if kind == 'absolute' else real / 'secret' / 'file'
    guard = permissions.PermissionGuard(deny=permissions.parse_rules([pattern]), workspace=str(tmp_path))
    assert not guard.check('read', {'path': str(target)}, False).allowed


def stopped(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return state == 'Z'
    except FileNotFoundError:
        return True


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, 'process state did not converge'
        time.sleep(.01)


@pytest.mark.parametrize('reap', [False, True])
def test_close_kills_term_ignoring_child_after_leader_exit(tmp_path, monkeypatch, reap):
    monkeypatch.setattr(jobs, 'GRACE', .05)
    store = jobs.JobStore('test', tmp_path / 'jobs')
    ready = tmp_path / 'ready'
    source = f'import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open({str(ready)!r},"w").write(str(os.getpid())); time.sleep(30)'
    job = store.start(f'{shlex.quote(sys.executable)} -c {shlex.quote(source)} &', str(tmp_path))
    try:
        wait_until(ready.exists)
        pid = int(ready.read_text())
        job.proc.wait(timeout=3)
        if reap:
            store.poll()
        store.close()
        wait_until(lambda: stopped(pid))
    finally:
        try:
            os.killpg(job.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        job.proc.wait(timeout=3)


def test_task_cancel_escalates_and_reaps(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'GRACE', .05)
    proc = subprocess.Popen([sys.executable, '-u', '-c',
                             'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready"); time.sleep(30)'],
                            stdout=subprocess.PIPE, start_new_session=True, text=True)
    try:
        assert proc.stdout.readline().strip() == 'ready'
        task = tasks.DetachedTask('1', 'test', proc=proc)
        task.cancel()
        assert proc.returncode == -signal.SIGKILL
        assert task.exit_code == proc.returncode
        assert task.status == 'failed'
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)
        proc.stdout.close()


@pytest.mark.parametrize('value', ['bad', 'nan', 'inf', 0, -1])
@pytest.mark.parametrize('tool', ['code', 'bash'])
def test_invalid_deadline_never_starts_process(tmp_path, monkeypatch, value, tool):
    launched = []
    def popen(*a, **kw):
        launched.append(a)
        raise OSError('unexpected start')
    monkeypatch.setattr(subprocess, 'Popen', popen)
    if tool == 'code':
        out = codemode.execute('pass', str(tmp_path), lambda *a: ('', False), timeout=value).output
    else:
        out = shell.bash(str(tmp_path), 'echo hello', timeout=value)
    assert not launched
    assert 'timeout' in out


def test_fractional_string_deadline_is_normalized_and_runs(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    children = []
    def popen(*a, **kw):
        proc = real_popen(*a, **kw)
        children.append(proc)
        return proc
    monkeypatch.setattr(subprocess, 'Popen', popen)
    try:
        _fractional_deadline(tmp_path)
    finally:
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)


def _fractional_deadline(tmp_path):
    out = codemode.execute('print("ok")', str(tmp_path), lambda *a: ('', False), timeout='5.5')
    assert out.exit_code == 0 and 'ok' in out.output
    assert shell.bash(str(tmp_path), 'echo ok', timeout='5.5').startswith('exit 0')
    guard = loopguard.LoopGuard(deadline=10)
    assert guard.clamp('bash', {'timeout': '5.5'})['timeout'] == 5


def test_code_binding_error_still_kills_and_reaps(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    children = []
    def popen(*a, **kw):
        proc = real_popen(*a, **kw)
        children.append(proc)
        return proc
    monkeypatch.setattr(subprocess, 'Popen', popen)
    def fail(*args):
        raise RuntimeError('binding failed')
    try:
        with pytest.raises(RuntimeError, match='binding failed'):
            codemode.execute('one.read(path="x")', str(tmp_path), fail)
        assert children[0].returncode is not None
    finally:
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)


@pytest.mark.parametrize('loader,shape', [('orc', 'single'), ('orc', 'map'), ('orc', 'list'),
                                           ('preset', 'map'), ('preset', 'list')])
@pytest.mark.parametrize('value', ['invalid', {'invalid': 1}])
def test_bad_config_item_is_named_and_skipped(tmp_path, loader, shape, value):
    bad = {'name': 'bad', 'max_depth': value} if loader == 'orc' else {'name': 'bad', 'timeout': value}
    body = bad if shape == 'single' else {'bad': bad, 'good': {}} if shape == 'map' else [bad, {'name': 'good'}]
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(body))
    load = orc.load_orcs if loader == 'orc' else subagent.load_presets
    configs, errors = load([str(path)])
    assert 'bad' not in configs
    assert errors and str(path) in errors[0] and 'bad' in errors[0]
    if shape != 'single':
        assert 'good' in configs


@pytest.mark.parametrize('symlink', [False, True])
def test_restricted_glob_does_not_hint_denied_names(tmp_path, symlink):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'confidential.txt').write_text('secret')
    ws = tmp_path / 'ws'
    ws.mkdir()
    if symlink:
        (ws / 'private').symlink_to(outside, target_is_directory=True)
    else:
        (ws / 'private').mkdir()
        (ws / 'private' / 'confidential.txt').write_text('secret')
    guards = permissions.build(str(ws), disallowed=['Read(private/**)'])
    out = search.restricted('glob', str(ws), {'pattern': 'private/confidentia/*.txt'}, guards)
    assert 'confidential.txt' not in out


def test_missing_hint_never_follows_symlink(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'confidential.txt').touch()
    (tmp_path / 'link').symlink_to(outside, target_is_directory=True)
    assert 'confidential.txt' not in fs.missing(tmp_path / 'link' / 'confidentia')


def test_injection_checker_fails_when_stub_requested_marker_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(stress, 'SCRATCH', tmp_path)
    # Remap the old shared /tmp paths into this test's private directory so
    # the pre-fix proof cannot touch another user's sentinel.
    def safe_path(value):
        p = Path(value)
        return tmp_path / p.name if p.parent == Path('/tmp') else p
    monkeypatch.setattr(stress, 'Path', safe_path)
    markers = []
    def run(args, ws, ceiling):
        prompt = args[-1]
        replies = []
        handler = object.__new__(stresslane.Handler)
        handler.reply = lambda text: replies.append(text)
        handler.m_stub_escape({'messages': [{'role': 'user', 'content': prompt}]})
        # Simulate an escaped agent following exactly the stub's requested path.
        marker = safe_path(replies[0].split('write it to ', 1)[1].split(', then', 1)[0])
        assert marker.parent == tmp_path and marker.parent != ws
        markers.append(marker)
        marker.write_text('escaped')
        return 0, '', '', .1
    monkeypatch.setattr(stress, 'run_one', run)
    assert not stress.c_injection().ok
    assert not stress.c_injection().ok
    assert len(set(markers)) == 2
    assert all(not p.exists() for p in markers)



@pytest.mark.parametrize('url', ['HTTPS://internal.example/x', 'https://INTERNAL.example/x',
                                 'https://internal.example:443/x', 'https://user@internal.example/x',
                                 'https://internal.example./x'])
def test_url_deny_normalizes_origin(tmp_path, url):
    guard = permissions.PermissionGuard(deny=permissions.parse_rules(['WebFetch(https://internal.example/*)']),
                                        workspace=str(tmp_path))
    assert not guard.check('web_fetch', {'url': url}, False).allowed
    assert guard.check('web_fetch', {'url': 'https://internal.example.evil/x'}, False).allowed
    assert guard.check('web_fetch', {'url': 'https://internal.example:444/x'}, False).allowed
    assert guard.check('web_fetch', {'url': 'https://internal.example:0/x'}, False).allowed


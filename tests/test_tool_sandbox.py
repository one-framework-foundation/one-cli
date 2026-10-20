import os
import shlex
import socket
import time

import pytest

from onecli.sandbox.shell import SandboxError, run_shell


@pytest.fixture
def roots(tmp_path):
    source, stage = tmp_path / 'source', tmp_path / 'stage'
    source.mkdir()
    stage.mkdir()
    return source, stage


def test_actual_namespace_and_snapshot(roots, tmp_path, monkeypatch):
    source, stage = roots
    secret = tmp_path / 'host-secret'
    secret.write_text('sentinel-do-not-copy')
    (source / 'input').write_text('original')
    monkeypatch.setenv('SECRET_SENTINEL', 'host-value')
    command = f'test ! -e {shlex.quote(str(secret))} && test -z "$SECRET_SENTINEL" && test ! -e /home && test ! -e /etc && echo changed > input && pwd'
    result = run_shell(source, stage, command)
    assert result.returncode == 0, result.output
    assert result.output.strip() == '/workspace'
    assert (source / 'input').read_text() == 'original'
    assert (result.workspace / 'input').read_text() == 'changed\n'
    assert secret.read_text() == 'sentinel-do-not-copy'


def test_localhost_unreachable(roots):
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        port = server.getsockname()[1]
        code = f'import socket; s=socket.socket(); s.settimeout(.2); assert s.connect_ex(("127.0.0.1",{port})) != 0'
        result = run_shell(*roots, 'python3 -c ' + shlex.quote(code))
    assert result.returncode == 0, result.output


@pytest.mark.parametrize('kind', ['socket', 'symlink', 'hardlink', 'fifo'])
def test_unsafe_input_refused(roots, tmp_path, kind):
    source, stage = roots
    target = source / 'bad'
    sock = None
    if kind == 'socket':
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(target))
    elif kind == 'symlink':
        target.symlink_to(tmp_path)
    elif kind == 'hardlink':
        original = tmp_path / 'outside'
        original.write_text('outside')
        os.link(original, target)
    else:
        os.mkfifo(target)
    try:
        with pytest.raises(SandboxError):
            run_shell(source, stage, 'true')
        assert not list(stage.iterdir())
    finally:
        if sock:
            sock.close()


def test_host_unix_socket_not_visible(roots, tmp_path):
    endpoint = tmp_path / 'host.sock'
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(endpoint))
        server.listen()
        code = f'import socket; s=socket.socket(socket.AF_UNIX); assert s.connect_ex({str(endpoint)!r}) != 0'
        result = run_shell(*roots, 'python3 -c ' + shlex.quote(code))
    assert result.returncode == 0, result.output


def test_timeout_kills_descendants(roots):
    result = run_shell(*roots, '(sleep 1; echo escaped > late) & wait', timeout_seconds=.15)
    assert result.timed_out
    time.sleep(1.1)
    assert not (result.workspace / 'late').exists()


def test_output_cap_and_closed_stdout_timeout(roots):
    result = run_shell(*roots, 'yes x', max_output_bytes=128)
    assert result.output_limited
    assert len(result.output.encode()) <= 128
    result = run_shell(*roots, 'exec 1>&- 2>&-; sleep 3', timeout_seconds=.15)
    assert result.timed_out


def test_readonly_system_and_file_size_limit(roots):
    result = run_shell(*roots, 'touch /usr/sandbox-escape 2>/dev/null; test ! -e /usr/sandbox-escape && dd if=/dev/zero of=large bs=1048576 count=32 2>/dev/null')
    assert result.returncode != 0
    assert (result.workspace / 'large').stat().st_size <= 16777216


def test_snapshot_caps(roots):
    (roots[0] / 'a').write_text('123')
    with pytest.raises(SandboxError, match='byte limit'):
        run_shell(*roots, 'true', snapshot_bytes=2)
    (roots[0] / 'b').touch()
    with pytest.raises(SandboxError, match='entry limit'):
        run_shell(*roots, 'true', snapshot_entries=1)


@pytest.mark.parametrize('bad', [0, -1, float('inf'), True])
def test_invalid_timeout(roots, bad):
    with pytest.raises(SandboxError):
        run_shell(*roots, 'true', timeout_seconds=bad)


def test_no_fallback(roots, monkeypatch):
    monkeypatch.setattr(os, 'access', lambda *args: False)
    with pytest.raises(SandboxError, match='unavailable'):
        run_shell(*roots, 'touch escaped')
    assert not (roots[0] / 'escaped').exists()


def test_read_artifact_valid_and_oversize(roots):
    from onecli.sandbox.shell import read_artifact
    result = run_shell(*roots, 'mkdir sub; printf abc > sub/file')
    assert result.returncode == 0
    assert read_artifact(result, 'sub/file', max_bytes=3) == b'abc'
    with pytest.raises(SandboxError, match='byte limit'):
        read_artifact(result, 'sub/file', max_bytes=2)


@pytest.mark.parametrize('path', ['', '/', '/file', '..', '../file', 'sub/../file', './file', 'sub//file', 'sub/', 'bad\0file'])
def test_read_artifact_invalid_paths(roots, path):
    from onecli.sandbox.shell import read_artifact
    result = run_shell(*roots, 'true')
    with pytest.raises(SandboxError):
        read_artifact(result, path)


@pytest.mark.parametrize('command,path', [
    ('ln -s /etc/passwd file', 'file'),
    ('ln -s /usr parent', 'parent/bin/sh'),
    ('mkfifo file', 'file'),
    ('echo x > file; ln file second', 'file'),
    ('mkdir file', 'file'),
])
def test_read_artifact_unsafe_outputs(roots, command, path):
    from onecli.sandbox.shell import read_artifact
    result = run_shell(*roots, command)
    assert result.returncode == 0, result.output
    with pytest.raises(SandboxError):
        read_artifact(result, path)


def test_read_artifact_socket_and_root_symlink(roots):
    from onecli.sandbox.shell import SandboxResult, read_artifact
    result = run_shell(*roots, "python3 -c 'import socket; s=socket.socket(socket.AF_UNIX); s.bind(\"socket\")'")
    assert result.returncode == 0, result.output
    with pytest.raises(SandboxError):
        read_artifact(result, 'socket')
    alias = roots[1] / 'alias'
    alias.symlink_to(result.workspace, target_is_directory=True)
    forged = SandboxResult(0, '', alias)
    with pytest.raises(SandboxError):
        read_artifact(forged, 'file')

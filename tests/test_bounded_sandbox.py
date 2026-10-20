"""Actual quota + tmpfs + artifact integration; no provider traffic."""
import shlex
import pytest
from onecli.sandbox.bounded import run_bounded_shell
from onecli.sandbox.shell import SandboxError

@pytest.fixture
def roots(tmp_path):
    source, stage = tmp_path / 'source', tmp_path / 'stage'
    source.mkdir()
    stage.mkdir()
    return source, stage


def test_offline_coding_flow(roots):
    source, stage = roots
    original='def add(a,b):\n    return a-b\n'
    (source/'app.py').write_text(original)
    code="from pathlib import Path; p=Path('app.py'); p.write_text(p.read_text().replace('a-b','a+b')); from app import add; assert add(2,3)==5; print('tested')"
    r=run_bounded_shell(source,stage,'python3 -B -c '+shlex.quote(code),artifacts=['app.py'])
    assert r.returncode == 0, r.output
    assert 'tested' in r.output
    assert r.artifacts == {'app.py':b'def add(a,b):\n    return a+b\n'}
    assert (source/'app.py').read_text() == original


def test_aggregate_workspace_limit(roots):
    # Each individual file is small; total exceeds the 1 MiB mount bound.
    code="from pathlib import Path; [(Path(str(i)).write_bytes(b'x'*262144)) for i in range(12)]"
    r=run_bounded_shell(*roots,'python3 -c '+shlex.quote(code),workspace_bytes=1048576)
    assert r.returncode != 0
    assert 'No space left on device' in r.output


def test_timeout_and_capture(roots):
    r=run_bounded_shell(*roots,'sleep 10',timeout_seconds=1)
    assert r.timed_out or r.returncode != 0
    r=run_bounded_shell(*roots,'yes x',max_output_bytes=128)
    assert r.output_limited
    assert len(r.output.encode()) <= 128


def test_host_paths_not_visible(roots, tmp_path):
    hidden=tmp_path/'sentinel'
    hidden.write_text('private-sentinel')
    r=run_bounded_shell(*roots,'test ! -e '+shlex.quote(str(hidden))+' && test ! -e /etc && test ! -e /home && echo isolated')
    assert r.returncode == 0, r.output
    assert 'isolated' in r.output


@pytest.mark.parametrize('path',['/etc/passwd','../secret','a/../b',''])
def test_invalid_artifact_request_before_launch(roots,path):
    with pytest.raises((SandboxError,ValueError)):
        run_bounded_shell(*roots,'true',artifacts=[path])


def test_unsafe_artifact_refused(roots):
    with pytest.raises(SandboxError):
        run_bounded_shell(*roots,'ln -s /usr/bin/python3 out',artifacts=['out'])


def test_missing_quota_backend_never_falls_back(roots,monkeypatch):
    import onecli.sandbox.quota as quota
    def reject(*args,**kwargs):
        raise RuntimeError('quota provisioning refused')
    monkeypatch.setattr(quota,'prepare',reject)
    # Depending on direct import the failure can surface from quota setup itself;
    # integration exposes the module prepare so this test observes the actual gate.
    import onecli.sandbox.bounded as bounded
    if hasattr(bounded,'prepare'):
        monkeypatch.setattr(bounded,'prepare',reject)
    with pytest.raises((SandboxError,RuntimeError)):
        run_bounded_shell(*roots,'echo escaped > out',artifacts=['out'])
    assert not (roots[0]/'out').exists()

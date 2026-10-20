"""Release review regressions: confinement, network boundaries and disclosures."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from unittest.mock import mock_open, patch

import pytest

from onecli import checkpoints, cli, config, permissions, session
from onecli.providers.base import ProviderError


@pytest.mark.parametrize('name', ['AGENTS.md', 'CLAUDE.md', 'SOUL.md'])
@pytest.mark.parametrize('ancestor', [False, True])
def test_context_symlink_refused(tmp_path, capsys, name, ancestor):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / '.git').mkdir()
    workspace = root / 'child'
    workspace.mkdir()
    outside = tmp_path / 'private'
    outside.write_text('PRIVATE_CONTEXT_SENTINEL')
    link = (root if ancestor else workspace) / name
    link.symlink_to(outside)
    assert 'PRIVATE_CONTEXT_SENTINEL' not in permissions.project_context(str(workspace), str(tmp_path / 'home'))
    if not ancestor or name != 'SOUL.md':
        assert str(link) in capsys.readouterr().err


def test_flag_endpoint_error_never_discovers(tmp_path, monkeypatch):
    class Broken:
        def chat(self, *args, **kwargs):
            raise ProviderError('provider unavailable')

    calls = []
    def deny(*args, **kwargs):
        calls.append(args)
        raise AssertionError('unexpected socket')

    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path / 'sessions')
    monkeypatch.setattr(cli.providers, 'build', lambda *a, **kw: Broken())
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setattr(config, 'PROVIDER_DIR', tmp_path / 'providers')
    monkeypatch.setenv('ONE_LLAMACPP_HOSTS', 'http://127.0.0.1:19999')
    args = cli.build_parser().parse_args([
        '--response-gate', 'off', '--no-stream', '--no-fallback',
        '-w', str(tmp_path), '-m', 'explicit', 'hello'])
    backend = config.Backend(name='explicit', model='explicit', source='flag',
                             base_url='https://explicit.invalid/v1')
    assert cli.run_once(args, backend, None, 'run8-error') == 2
    assert calls == []


def test_relay_upload_inventory_matches_transport():
    from onecli.senses import ear
    with patch.object(ear, '_multipart', return_value={'text': 'hello'}) as post, \
         patch('builtins.open', mock_open(read_data=b'audio')):
        assert ear._parakeet('clip.wav', None)[0] == 'hello'
    assert post.call_args.args[0].endswith('/transcribe')
    assert post.call_args.args[3] == b'audio'
    for name in ('SECURITY.md', 'docs/reproducible-releases.md'):
        row = next((line for line in Path(name).read_text().splitlines()
                    if line.startswith('| Relay transcription upload |')), '')
        for text in ('multipart POST', '/transcribe', 'extracted video audio',
                     'ONE_PARAKEET_URL', 'ONE_PARAKEET_REMOTE_URL', 'no credentials',
                     'No transcript cache', 'omit audio/video attachments', '--media-route off'):
            assert text in row


def test_session_checkpoint_documentation(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path / 'sessions')
    for count in range(1, 41):
        session.append('run8-format', {'kind': 'messages', 'messages': list(range(count))})
    records = [json.loads(line) for line in session.path_for('run8-format').read_text().splitlines()]
    assert [r['kind'] for r in records] == ['messages'] + ['messages_delta'] * 39
    doc = Path('CHANGELOG.md').read_text()
    assert 'with periodic checkpoints' not in doc
    for text in ('first write in a process', 'cache eviction', 'external write',
                 'history rewrite', 'Older readers see only the last full record'):
        assert text in doc




@pytest.mark.parametrize('name', ['.env', '.env.production', 'cert.pem', 'private.key', 'id_ed25519'])
def test_untracked_secrets_never_hashed_and_undo_preserves_them(tmp_path, name):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, stderr=subprocess.DEVNULL)
    git('init')
    (tmp_path / 'tracked').write_text('before')
    (tmp_path / 'tracked.key').write_text('tracked before')
    git('add', '.')
    secret = b'untracked secret sentinel ' + name.encode()
    (tmp_path / name).write_bytes(secret)
    mgr = checkpoints.CheckpointManager(str(tmp_path))
    cp = mgr.create_checkpoint()
    assert cp is not None, mgr.last_error
    digest = hashlib.sha1(b'blob ' + str(len(secret)).encode() + b'\0' + secret).hexdigest()
    assert not (tmp_path / '.git' / 'objects' / digest[:2] / digest[2:]).exists()
    (tmp_path / 'tracked').write_text('after')
    (tmp_path / 'tracked.key').write_text('tracked after')
    (tmp_path / 'agent-created').write_text('new')
    assert mgr.rollback().success
    assert (tmp_path / name).read_bytes() == secret
    assert (tmp_path / 'tracked').read_text() == 'before'
    assert (tmp_path / 'tracked.key').read_text() == 'tracked before'
    assert not (tmp_path / 'agent-created').exists()

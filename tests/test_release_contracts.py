"""Release-blocker contracts: inventories, privacy and publication wording."""
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace

import pytest

from onecli import media
from onecli._vendor import dropins as presence
from onecli.senses import capabilities, eye, logbook, router
from onecli.senses.base import Outcome

ROOT = Path(__file__).resolve().parents[1]


def inventory(path, trigger):
    return next(line for line in (ROOT / path).read_text().splitlines()
                if line.startswith('| ' + trigger + ' |'))


@pytest.mark.parametrize('path', ['SECURITY.md', 'docs/reproducible-releases.md'])
@pytest.mark.parametrize('trigger,fragments', [
    ('Attachment capability metadata', ['GET', '/v1/models', 'selected backend',
     'public HTTPS', 'no credentials', '--media-route off', '60 seconds']),
    ('Local vision fallback discovery', ['GET', '/api/tags', 'POST', '/api/show',
     'model name', '127.0.0.1:11434', 'ONE_SENSES_OLLAMA_ENDPOINTS',
     'no credentials', '60 seconds', '--media-route off', 'ONE_CLI_VISION_BACKENDS']),
    ('Automatic image description', ['image', 'video', 'POST', '/v1/chat/completions',
     'image bytes', 'backend credentials', '--media-route off']),
])
def test_attachment_inventory(path, trigger, fragments):
    row = inventory(path, trigger)
    for fragment in fragments:
        assert fragment in row


@pytest.mark.parametrize('path', ['SECURITY.md', 'docs/reproducible-releases.md'])
def test_documented_cache_durations_match_constants(path):
    row = inventory(path, 'Local/private backend presence on `--list`, explicit `-m NAME` selection, and chat `/model`')
    assert f'{presence.PRESENCE_CACHE_TTL:g} seconds up' in row
    assert f'{presence.PRESENCE_DOWN_TTL:g} seconds for connection/health failures' in row
    assert f'{presence.PRESENCE_CACHE_TTL:g} seconds for unsuccessful OpenAI listings' in row
    for trigger in ('Attachment capability metadata', 'Local vision fallback discovery'):
        assert f'{capabilities.METADATA_CACHE_TTL:g} seconds' in inventory(path, trigger)
    assert f'{eye.FAILURE_COOLDOWN:g} seconds' in inventory(path, 'Automatic image description')
    role = inventory(path, 'Role drop-in backend resolution/discovery, including `--list`, when `ONE_PROVIDER_ROLE` is set')
    assert f'{presence.RELAY_TTL:g} seconds on success' in role
    assert f'{presence.RELAY_DOWN_TTL:g} seconds on failure' in role


@pytest.mark.parametrize('existing', [False, True])
def test_failed_first_attachment_log_is_private_under_umask_022(monkeypatch, tmp_path, existing):
    path = tmp_path / 'state' / 'senses_log.jsonl'
    if existing:
        path.parent.mkdir(mode=0o755)
        path.write_text('')
        path.chmod(0o644)
    monkeypatch.setattr(logbook, 'LOG_PATH', path)
    monkeypatch.setattr(eye, 'perceive', lambda *a, **kw: Outcome(modality='image'))
    attachment = media.Attachment('image', '/sensitive/attachment.png', 'image/png')
    previous = os.umask(0o022)
    try:
        with pytest.raises(media.MediaError, match='no rung worked'):
            router.route([attachment], SimpleNamespace(name='blind'),
                         caps=capabilities.Capabilities(), fallback=False)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    row = json.loads(path.read_text())
    assert row['path'] == attachment.path
    assert row['backend'] == 'blind' and row['ok'] is False


def test_release_artifact_visibility_and_retention():
    text = ' '.join((ROOT / 'docs/reproducible-releases.md').read_text().split())
    assert 'private CI artifacts' not in text
    assert 'wheel and source distribution' in text
    assert 'any signed-in GitHub user' in text
    assert 'read access' in text
    assert 'repository is public' in text
    workflow = (ROOT / '.github/workflows/release.yml').read_text()
    days = int(re.search(r'retention-days: (\d+)', workflow)[1])
    assert 1 <= days <= 7
    assert f'for {days} days' in text
    assert 'path: dist/*' in workflow


def test_contributing_adopted_license_and_pending_gates():
    text = ' '.join((ROOT / 'CONTRIBUTING.md').read_text().split())
    assert 'unresolved license' not in text
    assert 'one license 0.6.3' in text
    assert 'Licensed under the one license 0.6.3' in text
    assert 'pending' not in text


def test_metadata_cache_expires_at_documented_duration(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(capabilities.time, 'monotonic', lambda: clock[0])
    calls = []
    monkeypatch.setattr(capabilities, '_get_json',
                        lambda url, **kw: calls.append(url) or {})
    caps = capabilities.Capabilities()
    backend = SimpleNamespace(base_url='https://example.invalid/v1', model='test')
    for discover in (lambda: caps._llamacpp_caps(backend.base_url, backend.model),
                     lambda: capabilities.vision_backends(caps)):
        caps._documents.clear()
        caps.model_caps.clear()
        caps._probed_at = clock[0]
        monkeypatch.delenv('ONE_CLI_VISION_BACKENDS', raising=False)
        monkeypatch.setenv('ONE_SENSES_OLLAMA_ENDPOINTS', 'http://127.0.0.1:11434')
        calls.clear()
        discover()
        clock[0] += capabilities.METADATA_CACHE_TTL - 0.01
        discover()
        assert len(calls) == 1
        clock[0] += 0.02
        discover()
        assert len(calls) == 2

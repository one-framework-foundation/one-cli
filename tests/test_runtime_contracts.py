"""Runtime contract regressions, with no provider traffic."""
import json

import pytest

from onecli import cli, config, hooks, permissions, unavailable
from onecli.tools import shell


@pytest.mark.parametrize(('clause', 'target'), [
    ('./.env', '.env'), ('src/../.env', '.env'),
    ('./secrets/**', 'secrets/key'), ('src/../secrets/**', 'secrets/key'),
])
def test_relative_rule_normalization(tmp_path, clause, target):
    rule = permissions.Rule.parse(f'Read({clause})')
    assert rule.matches('read', {'path': target}, str(tmp_path))


@pytest.mark.parametrize('event', ['Notification', 'SubagentStart', 'PostCompact'])
def test_lifecycle_hooks_are_advisory(tmp_path, event):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'hooks': {event: [
        {'hooks': [{'type': 'command', 'command': 'exit 2'}]}]}}))
    built = hooks.build(str(tmp_path), paths=[str(path)])
    assert not built.errors
    assert not built.pre_tool_use('read', {}).blocked
    outcome = built._fire(event, {})
    assert outcome.ran == 1
    assert outcome.advisories
    assert not outcome.blocked


def test_unknown_hook_warns_without_blocking(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'hooks': {'FutureLifecycleEvent': []}}))
    built = hooks.build(str(tmp_path), paths=[str(path)])
    assert 'unknown hook event' in built.errors[0]
    assert not built.pre_tool_use('read', {}).blocked


def test_bash_without_proc_preserves_output_and_side_effect(tmp_path, monkeypatch):
    original = shell.os.listdir

    def listdir(path):
        if path == '/proc':
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(shell.os, 'listdir', listdir)
    assert shell.bash(str(tmp_path), 'printf written > result; printf hello') == 'exit 0\nhello'
    assert (tmp_path / 'result').read_text() == 'written'


@pytest.mark.parametrize('mode', ['ask', 'error', 'fallback'])
def test_explicit_backend_failure_does_not_discover(monkeypatch, capsys, mode):
    args = cli.build_parser().parse_args(['--on-unavailable-model', mode])
    primary = config.Backend('explicit', 'm', 'http://private.invalid/v1', source='flag')
    monkeypatch.setattr(config, 'discover', lambda: pytest.fail('discovery contacted other hosts'))
    assert unavailable.resolve(args, primary, 'explicit backend is offline') is None
    assert 'explicit backend is offline' in capsys.readouterr().err


@pytest.mark.parametrize('flag,value', [
    ('--max-budget-usd', '0.01'), ('--max-steps', '1'), ('--max-input-tokens', '1'),
])
def test_chat_refuses_unsupported_limits(monkeypatch, capsys, flag, value):
    from onecli import chat
    monkeypatch.setattr(chat, 'from_cli', lambda *a: pytest.fail('uncapped chat started'))
    assert cli.main(['--chat', flag, value, '-p', 'fixture']) == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize('document', ['SECURITY.md', 'docs/reproducible-releases.md'])
def test_hosted_transcription_inventory_matches_upload_routes(document):
    from pathlib import Path
    from onecli.senses import ear
    text = (Path(__file__).resolve().parents[1] / document).read_text()
    rows = [row for row in text.splitlines() if row.startswith('|')]
    for endpoint, credential in [(ear.DEEPGRAM_URL, 'DEEPGRAM_API_KEY'),
                                 (ear.GROQ_URL, 'GROQ_API_KEY')]:
        assert any(endpoint in row and credential in row and
                   '--media-route off' in row and 'POST' in row and
                   'audio' in row for row in rows)

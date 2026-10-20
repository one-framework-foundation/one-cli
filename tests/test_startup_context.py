"""Real CLI entry paths and hook subprocesses, with an in-process provider."""
import copy
import json
import shlex
import sys
import pytest
from onecli import cli, config, hooks, session, fallback
from onecli.providers.base import Turn

@pytest.fixture
def harness(tmp_path, monkeypatch):
    seen = []
    class Wire:
        def chat(self, messages, tools, **kwargs):
            seen.append(copy.deepcopy(messages))
            return Turn(text='fixture answer')
    backend = config.Backend(name='fixture', model='fixture', base_url='http://fixture.invalid/v1', source='flag')
    monkeypatch.setattr(config, 'resolve', lambda **kw: backend)
    monkeypatch.setattr(cli, '_context_settings', lambda *a: {'chain': fallback.Chain([fallback.Rung('fixture', Wire(), 'fixture')]), 'autocompact': False})
    for name in ('_guards', '_spawner', '_mcp'):
        monkeypatch.setattr(cli, name, lambda *a: None)
    monkeypatch.setattr(cli, '_presets', lambda *a: {})
    monkeypatch.setattr(cli, '_response_gate', lambda *a, **kw: 'off')
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(cli, '_banner', lambda *a: '')
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path / 'sessions')
    monkeypatch.setattr(cli.joblib, 'JOB_DIR', tmp_path / 'jobs')
    monkeypatch.setattr(cli.spilllib, 'SPILL_DIR', tmp_path / 'spill')
    script = tmp_path / 'hook.py'
    script.write_text('import json,sys\np=json.load(sys.stdin)\nprint("startup-sentinel-" + p["source"])\n')
    settings = tmp_path / 'hooks.json'
    settings.write_text(json.dumps({'hooks': {'SessionStart': [{'hooks': [{'type': 'command', 'command': shlex.join([sys.executable, str(script)])}]}]}}))
    monkeypatch.setattr(hooks, 'settings_paths', lambda *a, **kw: [str(settings)])
    args = ['--workspace', str(tmp_path), '--system', 'fixture policy', '--no-project-context', '--no-stream', '--no-fallback', '--quiet']
    return args, seen, script

@pytest.mark.parametrize('interactive', [False, True])
@pytest.mark.parametrize('resume', [False, True])
def test_entry_paths_send_startup_context_once(harness, monkeypatch, capsys, interactive, resume):
    args, seen, _ = harness
    if resume:
        session.append('saved', {'kind': 'messages', 'messages': [{'role': 'system', 'content': 'saved policy'}, {'role': 'user', 'content': 'earlier question'}, {'role': 'assistant', 'content': 'earlier answer'}]})
        args += ['--resume', 'saved']
    if interactive:
        prompts = iter(['first question', 'second question', '/exit'])
        monkeypatch.setattr('builtins.input', lambda *a: next(prompts))
    else:
        args += ['--json', 'first question']
    assert cli.main(args) == 0
    sentinel = 'startup-sentinel-' + ('resume' if resume else 'startup')
    assert len(seen) == (2 if interactive else 1)
    for request in seen:
        assert str(request).count(sentinel) == 1
        assert sentinel in request[0]['content']
        assert not any(sentinel in str(m) for m in request if m['role'] == 'user')
        if resume:
            assert request[0]['content'].startswith('saved policy')
    if not interactive:
        assert json.loads(capsys.readouterr().out)['text'] == 'fixture answer'

def test_resume_refresh_and_no_hooks_remove_stale_context(harness, capsys):
    args, seen, _ = harness
    assert cli.main(args + ['--json', 'first question']) == 0
    sid = json.loads(capsys.readouterr().out)['session']
    assert cli.main(args + ['--resume', sid, '--json', 'next question']) == 0
    capsys.readouterr()
    assert 'startup-sentinel-startup' not in str(seen[-1])
    assert str(seen[-1]).count('startup-sentinel-resume') == 1
    assert cli.main(args + ['--resume', sid, '--no-hooks', '--json', 'last question']) == 0
    assert 'startup-sentinel-' not in str(seen[-1])

def test_blocked_startup_never_reaches_provider(harness, capsys):
    args, seen, script = harness
    script.write_text('print(\'{"continue":false,"stopReason":"fixture refusal"}\')\n')
    assert cli.main(args + ['--json', 'first question']) == 2
    assert seen == []
    assert 'fixture refusal' in capsys.readouterr().err


def test_refresh_preserves_policy_suffix_and_request_bytes():
    from types import SimpleNamespace
    args = SimpleNamespace(_session_start_context='current')
    old = SimpleNamespace(_session_start_context='old')
    saved = [{'role': 'system', 'content': 'policy' + cli._startup_block(old) + '\nextra policy'},
             {'role': 'user', 'content': 'current'}]
    refreshed = cli._startup_history(args, saved)
    assert refreshed[0]['content'] == 'policy\nextra policy' + cli._startup_block(args)
    assert cli._startup_history(args, refreshed) == refreshed
    assert saved[0]['content'] == 'policy' + cli._startup_block(old) + '\nextra policy'
    assert refreshed[1] == saved[1]


def test_history_without_system_does_not_duplicate_context():
    from types import SimpleNamespace
    args = SimpleNamespace(_session_start_context='current')
    refreshed = cli._startup_history(args, [{'role': 'user', 'content': 'hi'}])
    assert cli._startup_history(args, refreshed) == refreshed

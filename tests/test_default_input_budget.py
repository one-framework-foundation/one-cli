"""Context capacity must not silently become a lifetime processing budget."""
from types import SimpleNamespace

import pytest

from onecli import cli, config, fallback
from onecli.agent import Agent
from test_input_metering import Replay


@pytest.mark.parametrize('window', [4096, 200000, 1000000])
@pytest.mark.parametrize('limit', [None, 0, 400000])
def test_input_budget_is_explicit(monkeypatch, window, limit):
    monkeypatch.setattr(config, 'price_for', lambda *a: None)
    monkeypatch.setattr(config, 'resolve_context_window', lambda *a: window)
    monkeypatch.setattr(fallback, 'build', lambda *a, **k: fallback.Chain([], source='off'))
    args = cli.build_parser().parse_args(['--json', '--no-fallback'] +
        ([] if limit is None else ['--max-input-tokens', str(limit)]))
    backend = SimpleNamespace(model='fixture', name='fixture', base_url='fixture', source='flag')
    settings = cli._context_settings(args, backend)
    assert settings['max_input_tokens'] == limit
    assert settings['context_window'] == window


def test_long_healthy_tool_loop_finishes_without_implicit_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'price_for', lambda *a: None)
    monkeypatch.setattr(config, 'resolve_context_window', lambda *a: 200000)
    monkeypatch.setattr(fallback, 'build', lambda *a, **k: fallback.Chain([], source='off'))
    args = cli.build_parser().parse_args(['--json', '--no-fallback'])
    settings = cli._context_settings(args, SimpleNamespace(
        model='fixture', name='fixture', base_url='fixture', source='flag'))
    settings.pop('chain')
    provider = Replay([(86000, 81000)] * 40)
    (tmp_path / 'small.txt').write_text('evidence\n')
    result = Agent(provider, str(tmp_path), max_steps=45, **settings).run('inspect')
    assert result.stopped == 'answered', result.error
    assert result.tool_calls == 40
    assert result.cost['metered_input_tokens'] > 400000
    assert result.cost['context_tokens'] < 200000

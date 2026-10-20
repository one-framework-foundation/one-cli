from types import SimpleNamespace

import pytest

from onecli import cli, config, fallback


def args(*extra):
    return cli.build_parser().parse_args([
        '--require-priced-budget', '--no-fallback', '--max-budget-usd', '1', *extra])


@pytest.mark.parametrize('budget', [0, -1, float('nan'), float('inf'), -float('inf'), True])
def test_invalid_budget_before_discovery(monkeypatch, budget):
    a = args()
    a.max_budget_usd = budget
    monkeypatch.setattr(config, 'price_for', lambda *a: (1, 2))
    monkeypatch.setattr(config, 'resolve_context_window', lambda *a: pytest.fail('discovery'))
    with pytest.raises(ValueError, match='priced budget'):
        cli._context_settings(a, SimpleNamespace(model='fixture', name='fixture'))


@pytest.mark.parametrize('price', [None, (), (1,), (1, 2, 3), '12', (True, 1),
                                  (-1, 2), (1, float('nan')), (float('inf'), 1)])
def test_invalid_price(monkeypatch, price):
    monkeypatch.setattr(config, 'price_for', lambda *a: price)
    with pytest.raises(ValueError, match='priced budget'):
        cli._priced_budget(args(), SimpleNamespace(model='fixture', name='fixture'))


@pytest.mark.parametrize('price', [(0, 0), (0.5, 2)])
def test_valid_price_and_no_fallback(monkeypatch, price):
    monkeypatch.setattr(config, 'price_for', lambda *a: price)
    monkeypatch.setenv('ONE_CLI_FALLBACK', 'expensive')
    monkeypatch.setattr(config, 'resolve_context_window', lambda *a: 4096)
    def build(backend, spec, **kwargs):
        assert spec == []
        return fallback.Chain([], source='off')
    monkeypatch.setattr(fallback, 'build', build)
    result = cli._context_settings(args('--json'), SimpleNamespace(
        model='fixture', name='fixture', base_url='fixture', source='flag'))
    assert result['price'] == price


@pytest.mark.parametrize('spec', [None, ['auto'], ['expensive']])
def test_requires_explicit_no_fallback(monkeypatch, spec):
    a = args()
    a.fallback_model = spec
    monkeypatch.setattr(config, 'price_for', lambda *a: (1, 2))
    with pytest.raises(ValueError, match='no-fallback'):
        cli._priced_budget(a, SimpleNamespace(model='fixture', name='fixture'))


def test_main_rejects_before_attachments(monkeypatch, capsys):
    monkeypatch.setattr(config, 'resolve', lambda **kw: SimpleNamespace(model='fixture', name='fixture'))
    monkeypatch.setattr(config, 'price_for', lambda *a: None)
    monkeypatch.setattr(cli, 'collect_attachments', lambda *a: pytest.fail('attachments'))
    assert cli.main(['--require-priced-budget', '--no-fallback', '--max-budget-usd', '1', '-p', 'fixture']) == 2
    assert 'priced budget' in capsys.readouterr().err


def test_chat_rejects(monkeypatch, capsys):
    from onecli import chat
    monkeypatch.setattr(chat, 'from_cli', lambda *a: pytest.fail('chat started'))
    assert cli.main(['--chat', '--require-priced-budget', '-p', 'fixture']) == 2
    assert 'unsupported' in capsys.readouterr().err


def test_legacy_unpriced_unchanged(monkeypatch):
    monkeypatch.setattr(config, 'price_for', lambda *a: None)
    a = cli.build_parser().parse_args([])
    assert cli._priced_budget(a, SimpleNamespace(model='fixture', name='fixture')) is None

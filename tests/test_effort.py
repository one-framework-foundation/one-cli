import pytest
from onecli import effort, providers, config


def test_policy_override_and_unknown_backend():
    assert effort.select('auto', 'triage') == ('low', 'triage')
    assert effort.select('high', 'triage') == ('high', 'explicit')
    body, decision = effort.configure({'existing': True}, 'auto', 'triage')
    assert body == {'existing': True}
    assert not decision['applied']
    with pytest.raises(ValueError, match='no declared'):
        effort.configure({}, 'high')


@pytest.mark.parametrize('wire', ['openai', 'anthropic'])
def test_provider_profiles_preserve_options_and_never_leak(wire):
    extra = {'options': {'existing': 1}, 'effort_profiles': {'low': {'options': {'budget': 32}}}}
    b = config.Backend('test', 'fixture', 'http://unused.invalid', wire=wire, extra=extra)
    p = providers.build(b, effort='auto', task_class='triage')
    assert p.extra_body == {'options': {'existing': 1, 'budget': 32}}
    assert p.effort_decision['applied']
    assert extra['options'] == {'existing': 1}




def test_cli_effort_applies_to_each_fallback(monkeypatch):
    from onecli import cli, fallback
    args = cli.build_parser().parse_args(['--effort', 'high', '--task-class', 'triage', '--fallback-model', 'second', '--json'])
    monkeypatch.setattr(config, 'resolve_context_window', lambda *a: 8192)
    first = config.Backend('first', 'fixture', 'http://unused.invalid', extra={'effort_profiles': {'high': {'budget': 100}}})
    second = config.Backend('second', 'fixture', 'http://unused.invalid', extra={'effort_profiles': {'high': {'budget': 200}}})
    monkeypatch.setattr(fallback, 'resolve_backends', lambda *a: ([second], []))
    monkeypatch.setattr(fallback, 'drop_unhealthy', lambda b: (b, []))
    chain = cli._context_settings(args, first)['chain']
    assert [r.provider.extra_body['budget'] for r in chain.rungs] == [100, 200]
    assert chain.as_dict()['effort']['second']['source'] == 'explicit'




MISTRAL_PROFILES = {lvl: {'reasoning_effort': lvl}
                    for lvl in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')}


@pytest.mark.parametrize('level', sorted(MISTRAL_PROFILES))
def test_mistral_reasoning_effort_profile_reaches_request_body(level):
    extra = {'temperature': 0.7, 'effort_profiles': MISTRAL_PROFILES}
    b = config.Backend('mistral-large-4', 'mistral-large-4', 'http://unused.invalid', extra=extra)
    p = providers.build(b, effort=level)
    assert p.extra_body == {'temperature': 0.7, 'reasoning_effort': level}
    assert 'effort_profiles' not in p.extra_body


def test_cli_accepts_every_mistral_effort_level():
    from onecli import cli
    for level in MISTRAL_PROFILES:
        assert cli.build_parser().parse_args(['--effort', level]).effort == level


def test_mistral_large_4_window_is_not_the_family_default():
    assert config.context_window_for('mistral-large-4') == 524288
    assert config.context_window_for('mistral-medium-latest') == 131072


def test_typed_content_blocks_split_thinking_from_answer():
    from onecli.providers.openai_wire import OpenAIWire, split_content
    blocks = [{'type': 'thinking', 'thinking': [{'type': 'text', 'text': 'work it out'}], 'closed': True},
              {'type': 'text', 'text': '391'}]
    assert split_content(blocks) == ('391', 'work it out')
    assert split_content('plain') == ('plain', '')
    assert split_content(None) == ('', '')
    turn = OpenAIWire._to_turn({'choices': [{'message': {'content': blocks}, 'finish_reason': 'stop'}]})
    assert turn.text == '391' and turn.thinking == 'work it out'

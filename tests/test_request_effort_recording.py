import json
import pytest
from onecli import cli, config, providers, session
from onecli.providers.openai_wire import OpenAIWire
from onecli.providers.anthropic_wire import AnthropicWire

@pytest.fixture(autouse=True)
def _declared_output_maximum(monkeypatch):
    """These fakes serve /messages only; the Models API lookup is tested in
    test_max_tokens.py."""
    monkeypatch.setattr(AnthropicWire, "declared_max_output", lambda self: 4096)



@pytest.mark.parametrize('wire', ['openai', 'anthropic'])
@pytest.mark.parametrize('stream', [False, True])
def test_effort_observed_from_sent_body(monkeypatch, wire, stream):
    backend = config.Backend('fixture', 'fixture', 'http://unused.invalid', wire=wire,
        extra={'reasoning_effort': 'none', 'effort_profiles': {'high': {'reasoning_effort': 'high'}}})
    provider = providers.build(backend, effort='high')
    sent = []
    response = ({'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
                if wire == 'openai' else {'content': [{'type': 'text', 'text': 'ok'}], 'stop_reason': 'end_turn'})
    def capture(body, *args):
        sent.append(body)
        return response
    if stream:
        monkeypatch.setattr(provider, '_stream', capture)
    elif wire == 'openai':
        monkeypatch.setattr(provider, '_post', capture)
    else:
        from contextlib import nullcontext
        from io import BytesIO
        def open_request(request, **kwargs):
            sent.append(json.loads(request.data))
            return nullcontext(BytesIO(json.dumps(response).encode()))
        monkeypatch.setattr('urllib.request.urlopen', open_request)
    turn = provider.chat([{'role': 'user', 'content': 'private prompt'}],
                         on_delta=(lambda *a: None) if stream else None)
    assert len(sent) == 1
    assert turn.request_effort['model'] == sent[0]['model']
    assert turn.request_effort['wire'] == wire
    assert turn.request_effort['scope'] == 'initial_request'
    assert turn.request_effort['controls'] == {'reasoning_effort': 'high'}
    assert 'private prompt' not in json.dumps(turn.request_effort)


def test_omission_is_observed_without_inventing_default():
    from onecli.providers.base import effort_observation
    assert effort_observation({'model': 'fixture', 'messages': ['secret']}, 'openai')['controls'] == {}
    assert effort_observation({'output_config': {'effort': 'high', 'private': 'secret'}},
                              'anthropic')['controls'] == {'output_config': {'effort': 'high'}}
    assert effort_observation({'thinking': {'type': 'enabled', 'budget_tokens': 1024,
                              'private': 'secret'}}, 'anthropic')['controls'] == {
                              'thinking': {'type': 'enabled', 'budget_tokens': 1024}}


def test_cli_persists_effort_in_json_and_run_record(monkeypatch, tmp_path, capsys):
    backend = config.Backend('fixture', 'fixture', 'http://unused.invalid',
        extra={'effort_profiles': {'high': {'reasoning_effort': 'high'}}})
    monkeypatch.setattr(config, 'resolve_context_window', lambda *a: 8192)
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path/'sessions')
    monkeypatch.setattr(cli, '_hooks', lambda *a: None)
    monkeypatch.setattr(cli, '_mcp', lambda *a: None)
    monkeypatch.setattr(OpenAIWire, '_post', lambda self, body: {
        'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]})
    args = cli.build_parser().parse_args(['--json', '--no-stream', '--no-fallback',
            '--no-project-context', '--response-gate', 'off', '--max-depth', '0', '--effort', 'high',
            '--workspace', str(tmp_path), 'hello'])
    assert cli.run_once(args, backend, None, 'effort-fixture') == 0
    output = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in session.path_for('effort-fixture').read_text().splitlines()]
    run = next(row for row in rows if row.get('kind') == 'run')
    assert run['request_effort'] == output['request_effort']
    assert run['request_effort'][0]['controls'] == {'reasoning_effort': 'high'}

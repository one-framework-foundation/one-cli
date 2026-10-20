import pytest
from onecli.agent import Agent
from onecli.providers.openai_wire import OpenAIWire
from onecli.providers.base import ProviderError


def reply(text):
    return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}


def test_json_transport_is_assessment_only(tmp_path, monkeypatch):
    wire = OpenAIWire('http://unused', 'test')
    bodies = []
    def post(body):
        # Swift rejects system messages anywhere after the first position.
        assert body['messages'][0]['role'] == 'system'
        assert all(m['role'] != 'system' for m in body['messages'][1:])
        bodies.append(body)
        return reply('{"decision":"proceed","message":"I will inspect."}' if len(bodies)==1 else 'Done')
    monkeypatch.setattr(wire, '_post', post)
    result = Agent(wire, str(tmp_path), response_gate='enforce').run('inspect')
    assert result.stopped == 'answered'
    assert bodies[0]['response_format'] == {'type':'json_object'}
    # The assessment call offers no executable tools: the wire drops the
    # key entirely and pins tool_choice to none. The tool turn that follows
    # restores them. See test_response_gate.py for the same contract.
    assert 'tools' not in bodies[0]
    assert bodies[0]['tool_choice'] == 'none'
    assert bodies[1]['tools']
    assert 'response_format' not in bodies[1]
    assert wire.extra_body == {}


@pytest.mark.parametrize('with_system', [False, True])
def test_assessment_preserves_history_and_uses_only_leading_system(with_system):
    import copy
    from onecli.response_gate import assessment_history
    history = [{'role': 'user', 'content': 'Hello'}]
    if with_system:
        history.insert(0, {'role': 'system', 'content': 'Keep all safety rules.'})
    original = copy.deepcopy(history)
    request = assessment_history(history)
    assert history == original
    if with_system:
        assert request[:-1] == history
        assert request[-1]['role'] == 'user'
        assert 'JSON transport envelope' in request[-1]['content']
    else:
        assert [m['role'] for m in request] == ['system', 'user']
        assert 'JSON transport envelope' in request[0]['content']


def test_explicit_json_rejection_falls_back_once(monkeypatch):
    wire = OpenAIWire('http://unused', 'test')
    bodies=[]
    def post(body):
        bodies.append(body)
        if len(bodies)==1:
            raise ProviderError('HTTP 400: response_format json_object is not supported')
        return reply('{"decision":"answer","message":"Hello"}')
    monkeypatch.setattr(wire, '_post', post)
    wire.chat_json([{'role':'user','content':'hi'}], [])
    assert len(bodies)==2
    assert 'response_format' not in bodies[1]


def test_other_errors_do_not_trigger_format_fallback(monkeypatch):
    wire = OpenAIWire('http://unused', 'test')
    def post(body):
        raise ProviderError('HTTP 429: rate limited')
    monkeypatch.setattr(wire, '_post', post)
    with pytest.raises(ProviderError, match='429'):
        wire.chat_json([], [])

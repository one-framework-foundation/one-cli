import copy
import pytest
from onecli.providers.openai_wire import OpenAIWire


@pytest.mark.parametrize('stream', [False, True])
def test_empty_tools_override_provider_defaults_without_mutation(monkeypatch, stream):
    defaults = {'tools': [{'type': 'function', 'function': {'name': 'old'}}],
                'tool_choice': 'required'}
    original = copy.deepcopy(defaults)
    wire = OpenAIWire('http://unused', 'fixture', extra_body=defaults)
    bodies = []
    def capture(body, *args):
        bodies.append(body)
        return {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}
    monkeypatch.setattr(wire, '_post', capture)
    monkeypatch.setattr(wire, '_stream', capture)
    wire.chat([], [], on_delta=(lambda *args: None) if stream else None)
    assert bodies[0]['tool_choice'] == 'none'
    assert 'tools' not in bodies[0]
    assert defaults == original
    wire.chat([], original['tools'])
    assert bodies[1]['tools'] == original['tools']
    assert bodies[1]['tool_choice'] == 'required'


def test_unspecified_tools_preserve_provider_default(monkeypatch):
    wire = OpenAIWire('http://unused', 'fixture')
    bodies = []
    monkeypatch.setattr(wire, '_post', lambda b: bodies.append(b) or {'choices': [{'message': {'content': 'OK'}}]})
    wire.chat([])
    assert 'tool_choice' not in bodies[0]


def test_json_fallback_keeps_explicit_tool_ban(monkeypatch):
    from onecli.providers.base import ProviderError
    wire = OpenAIWire('http://unused', 'fixture')
    bodies = []
    def post(body):
        bodies.append(body)
        if len(bodies) == 1:
            raise ProviderError('HTTP 400 response_format unsupported')
        return {'choices': [{'message': {'content': '{"decision":"answer","message":"OK"}'}}]}
    monkeypatch.setattr(wire, '_post', post)
    wire.chat_json([], [])
    assert len(bodies) == 2
    assert all(b['tool_choice'] == 'none' and 'tools' not in b for b in bodies)

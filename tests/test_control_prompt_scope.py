"""Transport controls must not become durable user instructions."""
import pytest
from onecli.agent import Agent
from onecli import progress_policy, response_gate
from onecli.providers.base import ProviderError, Turn
from test_response_gate import Provider, assessment, write_call


def test_assessment_control_does_not_reach_tool_turn_or_resume(tmp_path):
    provider = Provider(assessment(), write_call(), Turn(text='Done'))
    result = Agent(provider, str(tmp_path), response_gate='enforce',
                   progress_policy={'enabled': False}).run('write proof')
    assert result.stopped == 'answered'
    assert response_gate.ASSESS in provider.requests[0][0][-1]['content']
    for messages in [provider.requests[1][0], result.messages]:
        assert not any(response_gate.ASSESS in str(m.get('content', '')) for m in messages)
    assert (tmp_path / 'proof.txt').read_text() == 'verified'


@pytest.mark.parametrize('fails', [False, True])
def test_progress_control_is_request_only_even_on_transport_failure(tmp_path, fails):
    class Wire(Provider):
        def chat(self, messages, tools, **kwargs):
            if fails and any(str(m.get('content', '')).startswith(progress_policy.UPDATE) for m in messages):
                raise ProviderError('HTTP 403: denied')
            return super().chat(messages, tools, **kwargs)
    wire = Wire(write_call(), Turn(text='Proof written.'), Turn(text='Done'))
    snapshots = []
    result = Agent(wire, str(tmp_path), max_steps=4,
                   progress_policy={'enabled': True, 'tool_calls': 1, 'min_seconds': 0},
                   on_event=lambda k,d: snapshots.append(list(d['messages'])) if k == 'checkpoint' else None
                   ).run('write proof')
    assert result.stopped == ('error' if fails else 'answered')
    if not fails:
        assert any(str(m.get('content', '')).startswith(progress_policy.UPDATE) for m in wire.requests[1][0])
    for messages in snapshots + [result.messages]:
        assert not any(str(m.get('content', '')).startswith(progress_policy.UPDATE) for m in messages)

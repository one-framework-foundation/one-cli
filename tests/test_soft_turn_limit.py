import pytest

from onecli import cli
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall
from test_response_gate import Provider, write_call


def test_soft_limit_finishes_with_evidence_without_more_tools(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('onecli.agent.time.monotonic', lambda: clock[0])

    class SlowProvider(Provider):
        def chat(self, *args, **kwargs):
            result = super().chat(*args, **kwargs)
            clock[0] = 11.0
            return result

    provider = SlowProvider(write_call(), Turn(text='Written; verification remains incomplete.'))
    result = Agent(provider, str(tmp_path), soft_turn_seconds=10, max_steps=8).run('write and verify')
    assert (tmp_path / 'proof.txt').read_text() == 'verified'
    assert result.stopped == 'budget_time'
    assert 'incomplete' in result.error
    assert len(provider.requests) == 2
    assert any('soft wall-clock limit' in str(m.get('content', ''))
               for m in provider.requests[1][0])


@pytest.mark.parametrize('seconds', [True, 0, -1, float('inf'), float('nan'), '10'])
def test_soft_limit_rejects_invalid_values(tmp_path, seconds):
    with pytest.raises(ValueError, match='finite and positive'):
        Agent(Provider(), str(tmp_path), soft_turn_seconds=seconds)


def test_cli_parses_soft_limit():
    assert cli.build_parser().parse_args(['--soft-turn-seconds', '10']).soft_turn_seconds == 10


@pytest.mark.parametrize('final', [Turn(), Turn(text='cut off', finish_reason='length'),
    Turn(tool_calls=[ToolCall('rogue', 'write', {'path': 'forbidden.txt', 'content': 'bad'})])])
def test_soft_limit_cannot_retry_or_execute_more_tools(tmp_path, monkeypatch, final):
    clock = [0.0]
    monkeypatch.setattr('onecli.agent.time.monotonic', lambda: clock[0])

    class SlowProvider(Provider):
        def chat(self, *args, **kwargs):
            result = super().chat(*args, **kwargs)
            clock[0] = 11.0
            return result

    provider = SlowProvider(write_call(), final)
    result = Agent(provider, str(tmp_path), soft_turn_seconds=10).run('write')
    assert result.stopped == 'budget_time'
    assert len(provider.requests) == 2
    assert not (tmp_path / 'forbidden.txt').exists()

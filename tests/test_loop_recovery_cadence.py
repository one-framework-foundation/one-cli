"""Multi-step recovery and measured cadence without live provider costs."""
import pytest
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall, ProviderError
from onecli import progress_policy as progress
from test_response_gate import Provider, write_call


def read_call(i):
    return Turn(tool_calls=[ToolCall(str(i), 'read', {'path': 'proof.txt'})])


def test_separate_empty_spells_recover_without_exponential_token_boost(tmp_path):
    wire = Provider(write_call(), Turn(finish_reason='length'), read_call(2),
                    Turn(finish_reason='length'), read_call(3), Turn(text='Verified'))
    result = Agent(wire, str(tmp_path), max_steps=8, max_tokens=1000).run('write and verify')
    assert result.stopped == 'answered'
    assert result.tool_calls == 3
    assert [kw['max_tokens'] for _, _, kw in wire.requests] == [1000, 1000, 4000, 4000, 4000, 4000]


def test_consecutive_empty_turns_stop_after_a_configured_retry(tmp_path):
    wire = Provider(write_call(), Turn(finish_reason='length'), Turn(finish_reason='length'))
    result = Agent(wire, str(tmp_path), max_steps=8,
                   loop_policy={'empty_retries': 1}).run('write and verify')
    assert result.stopped == 'truncated'
    assert len(wire.requests) == 3


@pytest.mark.parametrize('reason', ['stop', 'length'])
def test_empty_mid_task_keeps_tools_and_finishes_real_work(tmp_path, reason):
    provider = Provider(write_call(), Turn(finish_reason=reason), read_call(2), Turn(text='Verified'))
    result = Agent(provider, str(tmp_path), max_steps=7).run('write and verify proof')
    assert result.text == 'Verified' and result.tool_calls == 2
    recovery, schemas, _ = provider.requests[2]
    assert schemas
    assert 'work' in recovery[-1]['content'] and 'tools' in recovery[-1]['content']
    assert result.steps[2].tool_results[0][1].strip() == '1\tverified'
    assert not any('previous turn' in str(m.get('content')) for m in result.messages)
    assert not any('previous turn' in str(m.get('content')) for m in provider.requests[3][0])


@pytest.mark.parametrize('fails', [False, True])
def test_final_budget_control_never_poison_resume(tmp_path, fails):
    class Wire(Provider):
        def chat(self, messages, tools, **kwargs):
            if len(self.requests) == 1 and fails:
                self.requests.append((list(messages), tools, kwargs))
                raise ProviderError('fixture transport failure')
            return super().chat(messages, tools, **kwargs)
    wire = Wire(read_call(1), Turn(text='Incomplete'))
    snapshots = []
    result = Agent(wire, str(tmp_path), max_steps=2,
        on_event=lambda k,d: snapshots.append(list(d['messages'])) if k == 'checkpoint' else None).run('inspect')
    assert wire.requests[1][1] == wire.requests[0][1]  # stable cached tool prefix
    marker = 'This is the final response step'
    assert any(marker in str(m.get('content')) for m in wire.requests[1][0])
    for messages in snapshots + [result.messages]:
        assert not any(marker in str(m.get('content')) for m in messages)
    resumed = Provider(write_call(), Turn(text='Recovered'))
    answer = Agent(resumed, str(tmp_path), max_steps=3).run('continue and write proof', result.messages)
    assert answer.text == 'Recovered'
    assert (tmp_path / 'proof.txt').read_text() == 'verified'


def test_long_silent_sequence_gets_measured_reminders_without_extra_requests(tmp_path, monkeypatch):
    (tmp_path / 'proof.txt').write_text('verified')
    clock = [0.0]
    monkeypatch.setattr('onecli.agent.time.monotonic', lambda: clock[0])
    class SlowWire(Provider):
        def chat(self, messages, tools, **kwargs):
            clock[0] += 8
            return super().chat(messages, tools, **kwargs)
    wire = SlowWire(*[read_call(i) for i in range(8)], Turn(text='Verified'))
    seen = []
    result = Agent(wire, str(tmp_path), max_steps=12,
        progress_policy={'enabled': True, 'tool_calls': 0, 'seconds': 30},
        on_event=lambda k,d: seen.append((k,d))).run('inspect repeatedly for a cadence fixture')
    reminders = [m['content'] for req,_,_ in wire.requests for m in req
                 if str(m.get('content', '')).startswith(progress.UPDATE)]
    assert len(wire.requests) == 9 and result.tool_calls == 8
    assert len(reminders) == 2
    assert 'seconds_since_public=32' in reminders[0]
    assert 'seconds_since_public=64' in reminders[1]
    assert all('trigger=elapsed_silence' in r and 'completed_calls_since_check=4' in r for r in reminders)
    assert [d['text'] for k,d in seen if k == 'assistant_text'] == ['Verified']
    assert all(not str(m.get('content', '')).startswith(progress.UPDATE) for m in result.messages)


def test_failure_and_recovery_change_reminder_state():
    c = progress.Cadence(progress.resolve(), 0)
    c.completed('exit 7\nupstream unavailable')
    assert 'trigger=tool_failure' in c.reminder(1)
    assert 'failure_observed=true' in c.reminder(1)
    c.defer(1)
    c.completed('exit 0\nlocal fallback loaded')
    assert not c.due(2)
    assert 'trigger=elapsed_silence' in c.reminder(31)
    assert 'failure_observed=false' in c.reminder(31)

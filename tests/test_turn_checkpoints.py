import json

import pytest

from onecli.agent import Agent
from onecli.providers.base import ToolCall, Turn


def test_checkpoint_preserves_completed_tool_group_before_next_request(tmp_path):
    checkpoints = []

    class Interrupted:
        calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Turn(tool_calls=[ToolCall(id='a', name='ls', arguments={})])
            raise KeyboardInterrupt

    def record(kind, data):
        if kind == 'checkpoint':
            checkpoints.append(json.loads(json.dumps(data['messages'])))

    with pytest.raises(KeyboardInterrupt):
        Agent(Interrupted(), str(tmp_path), on_event=record).run('list files')
    assert len(checkpoints) >= 2
    assert checkpoints[-1][-1]['role'] == 'tool'
    assert checkpoints[-1][-1]['tool_call_id'] == 'a'


def test_cli_saves_checkpoint_and_announces_session_before_interruption(tmp_path, monkeypatch, capsys):
    from onecli import cli, session
    from onecli.config import Backend

    class Interrupted:
        calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Turn(tool_calls=[ToolCall(id='a', name='ls', arguments={})])
            assert session.load_messages('recovery')[-1]['role'] == 'tool'
            raise KeyboardInterrupt

    provider = Interrupted()
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path / 'sessions')
    monkeypatch.setattr(cli.providers, 'build', lambda *a, **kw: provider)
    args = cli.build_parser().parse_args(['--response-gate', 'off', '--json', '--stream', '--no-fallback',
                                        '-w', str(tmp_path), 'list'])
    assert cli.run_once(args, Backend(name='test', model='test', base_url='http://unused'),
                        None, 'recovery') == 130
    assert session.load_messages('recovery')[-1]['tool_call_id'] == 'a'
    captured = capsys.readouterr()
    assert captured.out == ''
    assert {'e':'session', 'id':'recovery'} in [json.loads(x) for x in captured.err.splitlines()
                                             if x.startswith('{')]


def test_the_progress_prompt_enters_the_history_once_per_conversation(tmp_path):
    """Each "Continue with the task." re-ran the agent over the same history
    and appended the style prompt again."""
    from onecli import progress_policy

    class Model:
        def chat(self, messages, tools=None, **kwargs):
            return Turn(text='ok', finish_reason='stop')

    agent = Agent(Model(), str(tmp_path), progress_policy={'enabled': True})
    history = agent.run('start').messages
    for _ in range(3):
        history = agent.run('Continue with the task.', messages=history).messages
    assert sum(m.get('content') == progress_policy.PROMPT for m in history) == 1
    assert sum(m.get('content') == 'Continue with the task.' for m in history) == 3


def test_each_assistant_turn_persists_finish_reason_usage_and_reasoning(tmp_path, monkeypatch):
    """A degenerate turn must be attributable afterwards: was it cut by the
    output cap, how much was reasoning, what did the model think. None of
    that survives in the message history."""
    from onecli import cli, session
    from onecli.config import Backend

    class Model:
        calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Turn(thinking='list first', finish_reason='tool_calls',
                            usage={'prompt_tokens': 50, 'completion_tokens': 9,
                                   'completion_tokens_details': {'reasoning_tokens': 4}},
                            tool_calls=[ToolCall(id='a', name='ls', arguments={})])
            return Turn(text='W. W. W.', thinking='lost thread', finish_reason='stop',
                        usage={'prompt_tokens': 70, 'completion_tokens': 4096})

    provider = Model()
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path / 'sessions')
    monkeypatch.setattr(cli.providers, 'build', lambda *a, **kw: provider)
    args = cli.build_parser().parse_args(['--response-gate', 'off', '--json', '--no-stream',
                                        '--no-fallback', '-w', str(tmp_path), 'list'])
    cli.run_once(args, Backend(name='test', model='test', base_url='http://unused'), None, 'turns')
    turns = session.load_records('turns', 'assistant_turn')
    messages = session.load_messages('turns')
    assert [t['finish_reason'] for t in turns] == ['tool_calls', 'stop']
    assert [t['reasoning'] for t in turns] == ['list first', 'lost thread']
    assert turns[0]['usage']['completion_tokens_details'] == {'reasoning_tokens': 4}
    assert turns[1]['usage']['completion_tokens'] == 4096
    assert messages[turns[0]['index']]['tool_calls'][0]['id'] == 'a'
    assert messages[turns[1]['index']]['content'] == 'W. W. W.'

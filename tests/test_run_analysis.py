import json
import os
from pathlib import Path

import pytest
from onecli.agent import Agent
from onecli.providers.base import ToolCall, Turn


def records():
    return [[json.loads(line) for line in p.read_text().splitlines()]
            for p in Path(os.environ['ONE_CLI_ANALYSIS_DIR']).glob('*.jsonl')]


class Scripted:
    def __init__(self, turns):
        self.turns = iter(turns)
    def chat(self, *args, **kwargs):
        turn = next(self.turns)
        if isinstance(turn, BaseException):
            raise turn
        return turn


def test_zero_tool_and_reused_agent_get_distinct_receipts(tmp_path):
    agent = Agent(Scripted([Turn(text='first'), Turn(text='second')]), str(tmp_path))
    callback = agent.on_event
    agent.run('secret first prompt')
    agent.run('secret second prompt')
    logs = records()
    assert len(logs) == 2
    assert agent.on_event is callback
    for log in logs:
        assert log[0]['kind'] == 'start'
        assert log[-1]['metrics']['stopped'] == 'answered'
        assert log[-1]['metrics']['tool_calls'] == 0
        assert log[-1]['metrics']['task_success'] == 'not_verified'
        assert 'secret' not in json.dumps(log)


def test_denial_return_and_secret_redaction(tmp_path):
    def callback(kind, data):
        return {'deny': 'test permission'} if kind == 'tool_start' else None
    turns = [Turn(tool_calls=[ToolCall(id=str(i), name='write',
                  arguments={'path':'secret-path','content':'secret-value'})]) for i in range(3)]
    result = Agent(Scripted(turns + [Turn(text='done')]), str(tmp_path),
                   on_event=callback).run('secret-prompt')
    assert len(result.denied) == 3
    assert not (tmp_path/'secret-path').exists()
    log = records()[0]
    assert 'secret' not in json.dumps(log)
    assert log[-1]['metrics']['repeated_calls'] == 2
    assert log[-1]['metrics']['denied_calls'] == 3


@pytest.mark.parametrize('exc,status', [(KeyboardInterrupt(), 'interrupt'), (RuntimeError('secret'), 'crash')])
def test_exception_recorded_and_original_exception_preserved(tmp_path, exc, status):
    agent = Agent(Scripted([exc]), str(tmp_path))
    callback = agent.on_event
    with pytest.raises(type(exc)):
        agent.run('private')
    assert agent.on_event is callback
    assert records()[0][-1]['metrics']['stopped'] == status
    assert 'secret' not in json.dumps(records())


def test_nested_run_parent_link(tmp_path):
    child = Agent(Scripted([Turn(text='child')]), str(tmp_path))
    class Parent:
        def chat(self, *args, **kwargs):
            child.run('child')
            return Turn(text='parent')
    Agent(Parent(), str(tmp_path)).run('parent')
    logs = records()
    parent = next(x for x in logs if x[0]['parent_run_id'] is None)
    nested = next(x for x in logs if x[0]['parent_run_id'] is not None)
    assert nested[0]['parent_run_id'] == parent[0]['run_id']


def test_unwritable_analysis_does_not_fail_answer(tmp_path, monkeypatch, capsys):
    target = tmp_path / 'file'
    target.write_text('not a directory')
    monkeypatch.setenv('ONE_CLI_ANALYSIS_DIR', str(target))
    assert Agent(Scripted([Turn(text='ok')]), str(tmp_path)).run('hello').text == 'ok'
    assert capsys.readouterr().err.count('could not be persisted') == 1


def test_summary_never_calls_unfinished_receipt_success_or_running(tmp_path):
    from onecli.evals.run_analysis import summary
    (tmp_path/'unfinished.jsonl').write_text(json.dumps({'kind':'start'})+'\npartial')
    value = summary(tmp_path)
    assert value['without_final_analysis'] == 1
    assert value['malformed_records'] == 1
    assert value['stops'] == {}
    assert value['task_success'] == 'not_verified'

import io
import json
import pytest
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall
from onecli import progress_policy as progress, cli, subagent, events
from test_response_gate import Provider, assessment, write_call


def test_checkpoint_keeps_tools_and_continues_in_same_request(tmp_path):
    provider = Provider(assessment(), write_call(),
        Turn(text='The proof is written; checking it.',
             tool_calls=[ToolCall('r', 'read', {'path': 'proof.txt'})]),
        Turn(text='Verified'))
    seen = []
    result = Agent(provider, str(tmp_path), response_gate='enforce', stream=True,
        progress_policy={'enabled': True, 'tool_calls': 1, 'min_seconds': 0},
        max_steps=6, on_event=lambda k,d: seen.append((k,d))).run('write and verify')
    assert result.text == 'Verified' and result.tool_calls == 2
    assert len(provider.requests) == 4
    assert provider.requests[2][1]  # normal gated tools remain available
    assert 'on_delta' in provider.requests[2][2]
    assert any(str(m.get('content', '')).startswith(progress.UPDATE) for m in provider.requests[2][0])
    assert all(not str(m.get('content', '')).startswith(progress.UPDATE) for m in result.messages)
    assert [d['text'] for k,d in seen if k == 'assistant_text'] == [
        'The proof is written; checking it.', 'Verified']


def test_completed_task_at_checkpoint_has_exactly_one_final(tmp_path):
    provider = Provider(write_call(), Turn(text='Verified'), Turn(text='Redundant recap'))
    seen = []
    result = Agent(provider, str(tmp_path), max_steps=5,
        progress_policy={'enabled': True, 'tool_calls': 1, 'min_seconds': 0},
        on_event=lambda k,d: seen.append((k,d))).run('write proof')
    assert result.text == 'Verified' and result.stopped == 'answered'
    assert len(provider.requests) == 2
    assert [d['text'] for k,d in seen if k == 'assistant_text'] == ['Verified']
    assert not any(k == 'progress_update' for k,d in seen)


def test_error_can_finish_without_mandatory_progress_round(tmp_path):
    for enabled in (True, False):
        provider = Provider(Turn(tool_calls=[ToolCall('r', 'read', {'path': 'absent'})]),
                            Turn(text='File not found'))
        result = Agent(provider, str(tmp_path),
            progress_policy={'enabled': enabled}).run('read missing')
        assert result.text == 'File not found' and len(provider.requests) == 2
        assert any(str(m.get('content', '')).startswith(progress.UPDATE) for m in provider.requests[1][0]) == enabled


def test_elapsed_silence_counts_only_after_work_and_resets():
    c=progress.Cadence(progress.resolve(),0)
    assert not c.due(100)
    c.completed('ok')
    assert not c.due(29)
    assert c.due(30)
    c.sent(31)
    assert not c.due(60)
    for _ in range(5):
        c.completed("ok")
    assert not c.due(32)
    assert c.due(41)
    c.sent(33)
    c.completed("blocked: no", True)
    assert c.due(33)


def test_silent_checkpoint_continues_real_tools(tmp_path):
    provider = Provider(write_call(),
        Turn(tool_calls=[ToolCall('r', 'read', {'path': 'proof.txt'})]), Turn(text='Verified'))
    seen = []
    result = Agent(provider, str(tmp_path), max_steps=5,
        progress_policy={'enabled': True, 'tool_calls': 1, 'min_seconds': 0},
        on_event=lambda k,d: seen.append((k,d))).run('write and verify')
    assert result.text == 'Verified' and result.tool_calls == 2
    assert len(provider.requests) == 3
    assert [d['text'] for k,d in seen if k == 'assistant_text'] == ['Verified']


def test_considered_checkpoint_does_not_forge_delivery_or_repeat_on_every_tool():
    c = progress.Cadence(progress.resolve(), 0)
    c.completed('ok')
    assert c.due(30)
    c.defer(30)
    c.completed('ok')
    assert c.last_public == 0 and not c.due(31)
    assert c.due(60)


def test_skipped_checkpoint_preserves_last_public_time():
    c=progress.Cadence(progress.resolve(),0)
    c.completed('error: missing')
    assert c.due(30)
    c.defer()
    assert c.last_public==0 and not c.due(31)
    c.completed('ok')
    assert c.due(32)


def test_resume_does_not_carry_old_silence(tmp_path):
    provider=Provider(write_call(),Turn(text='done'),Turn(text='hi'))
    a=Agent(provider,str(tmp_path),max_steps=2,progress_policy={'enabled':True,'tool_calls':1,'min_seconds':0})
    result = a.run("write")
    second = a.run("hi", result.messages)
    assert second.text=='hi' and second.tool_calls==0


def test_config_inheritance_flags_and_resolved_graph(tmp_path):
    path = tmp_path / "orc.json"
    path.write_text(json.dumps({"progress_policy": {"seconds": 45}}))
    args=cli.build_parser().parse_args(['--orc',str(path),'--progress-tool-calls','3'])
    p=cli._progress_policy(args)
    assert p==dict(enabled=True,seconds=45,tool_calls=3,on_error=True,min_seconds=10.0)
    child=subagent._preset_from('quiet',{'progress_policy':{'enabled':False}})
    spawner=subagent.Spawner(parent=None)
    Agent(Provider(),str(tmp_path),spawner=spawner,progress_policy=p)
    assert spawner.build(subagent.GENERAL).progress_policy==p
    assert spawner.build(child).progress_policy=={**p,'enabled':False}
    graph=cli._resolved_orc(args,{'quiet':child}).as_dict()
    assert graph['orchestrator']['progress_policy']==p
    assert graph['workers'][0]['progress_policy']=={**p,'enabled':False}


@pytest.mark.parametrize('value',[{'seconds':float('nan')},{'seconds':-1},{'tool_calls':True},{'tool_calls':1.2},{'enabled':'true'},{'other':2}])
def test_invalid_policy_rejected(value):
    with pytest.raises(ValueError):
        progress.resolve(value)


def test_completed_progress_event_is_public_only_when_requested():
    out = io.StringIO()
    events.emitter(out, include_text=True)(
        "progress_update", {"text": "Found the config."}
    )
    assert json.loads(out.getvalue())['e']=='progress_update'
    out = io.StringIO()
    events.emitter(out)("progress_update", {"text": "Found the config."})
    assert not out.getvalue()


def test_checkpoint_tools_still_pass_permission_gate(tmp_path):
    from onecli.guard import build
    provider = Provider(
        Turn(tool_calls=[ToolCall('r', 'read', {'path': 'absent'})]), write_call(),
        Turn(text='Write denied'))
    result = Agent(provider, str(tmp_path), guard=build(read_only=True), max_steps=5,
        progress_policy={'enabled': True, 'tool_calls': 1, 'min_seconds': 0}).run('inspect and write')
    assert not (tmp_path / 'proof.txt').exists()
    assert result.text == 'Write denied'

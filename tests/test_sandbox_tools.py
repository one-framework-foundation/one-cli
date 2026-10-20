import json
import threading
import pytest
from onecli import tools
from onecli.sandbox import tools as st
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall
from test_request_budget_isolated import budget_fixture


def adapter(tmp_path, **kw):
    source = tmp_path / "source"
    source.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    (source/'app.py').write_text('value = 1\n')
    return st.SandboxTools(st.CodingGrant('coding',str(source),str(stage),**kw))

@pytest.mark.parametrize('mask',['planner','code-reviewer','visual-worker','unknown','orchestrator'])
def test_unsupported_masks_fail_closed(tmp_path,mask):
    with pytest.raises(ValueError,match='mask'):
        st.CodingGrant(mask,str(tmp_path),str(tmp_path))

@pytest.mark.parametrize('kwargs',[{'max_calls':True},{'max_calls':9},{'timeout_seconds':31},
    {'workspace_bytes':0},{'artifacts':['app.py']},{'artifacts':('../x',)},
    {'artifacts':('/x',)},{'artifacts':('x','x')}])
def test_invalid_host_grants(tmp_path,kwargs):
    with pytest.raises(ValueError):
        adapter(tmp_path, **kwargs)


@pytest.mark.parametrize('name,args',[('bash',{'command':'true'}),('sandbox_exec',{}),
    ('sandbox_exec',{'command':'true','source':'/'}),('sandbox_exec',{'command':'true','artifacts':['x']}),
    ('sandbox_exec',{'command':'true','timeout_seconds':300}),('sandbox_exec',{'command':'\0'}),
    ('sandbox_exec',{'command':True}),('task',{})])
def test_model_cannot_expand_authority(tmp_path,monkeypatch,name,args):
    a=adapter(tmp_path)
    monkeypatch.setattr(st,'run_bounded_shell',lambda *a,**k:pytest.fail('executed'))
    assert a.execute(name,args).startswith('blocked:')


def test_failure_consumes_admission_and_does_not_leak_paths(tmp_path,monkeypatch):
    a=adapter(tmp_path,max_calls=1)
    def fail(*a,**kw): raise st.SandboxError('/secret-host-path')
    monkeypatch.setattr(st,'run_bounded_shell',fail)
    result=a.execute(a.name,{'command':'true'})
    assert result.startswith('error:') and '/secret' not in result
    assert a.execute(a.name,{'command':'true'}).startswith('blocked:')


def test_serial_admission(tmp_path,monkeypatch):
    a = adapter(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def wait(*args,**kw):
        entered.set()
        assert release.wait(5)
        raise st.SandboxError('fixture')
    monkeypatch.setattr(st,'run_bounded_shell',wait)
    thread = threading.Thread(target=a.execute, args=(a.name, {"command": "true"}))
    thread.start()
    try:
        assert entered.wait(5)
        assert 'busy' in a.execute(a.name,{'command':'true'})
    finally:
        release.set()
        thread.join(5)
    a.close()
    assert 'closed' in a.execute(a.name,{'command':'true'})


def test_actual_agent_repair_and_host_inspection(tmp_path,monkeypatch):
    a=adapter(tmp_path,artifacts=('app.py',),max_calls=1)
    admission,wire,_=budget_fixture(tmp_path)
    seen=[]
    turns=iter([Turn(text='',tool_calls=[ToolCall(id='1',name='sandbox_exec',arguments={
        'command':"printf 'value = 2\\n' > app.py; python3 -B -c 'import app; assert app.value == 2'"})]),Turn(text='candidate ready')])
    def chat(*args,**kwargs):
        seen.append(args[1])
        return next(turns)
    monkeypatch.setattr(wire,'chat',chat)
    monkeypatch.setattr(tools,'run',lambda *a,**kw:pytest.fail('legacy toolkit invoked'))
    agent=Agent(wire,str(tmp_path),request_admission=admission,request_budget_isolated=True,
                sandbox_tools=a,max_tokens=100,max_steps=4)
    result=agent.run('repair')
    assert result.stopped == 'answered',result.error
    payload=json.loads(result.steps[0].tool_results[0][1])
    assert payload['returncode'] == 0 and payload['untrusted']
    assert a.artifacts() == {'app.py':b'value = 2\n'}
    assert (tmp_path/'source/app.py').read_text() == 'value = 1\n'
    assert seen[0][0]['function']['name'] == 'sandbox_exec'
    assert agent.autocompact is False

@pytest.mark.parametrize('name',['bash','code','task','web_fetch','mcp__x'])
def test_agent_denies_legacy_tools_with_adapter(tmp_path,monkeypatch,name):
    a = adapter(tmp_path)
    admission, wire, _ = budget_fixture(tmp_path)
    turns=iter([Turn(text='',tool_calls=[ToolCall(id='1',name=name,arguments={})]),Turn(text='done')])
    monkeypatch.setattr(wire,'chat',lambda *a,**kw:next(turns))
    monkeypatch.setattr(tools,'run',lambda *a,**kw:pytest.fail('legacy dispatch'))
    agent=Agent(wire,str(tmp_path),request_admission=admission,request_budget_isolated=True,sandbox_tools=a,max_tokens=100)
    result=agent.run('test')
    assert result.steps[0].denials
    output,denied=agent._code_caller(result.steps[0])('sandbox_exec',{'command':'true'})
    assert denied and 'blocked' in output


def test_adapter_requires_isolated_mode(tmp_path):
    with pytest.raises(ValueError,match='isolated'):
        Agent(None,str(tmp_path),sandbox_tools=adapter(tmp_path))


def test_real_wire_reserves_each_tool_turn(tmp_path,monkeypatch):
    import io
    from onecli.request_admission import RequestAdmission
    from onecli.providers.openai_wire import OpenAIWire
    from test_request_admission import policy
    a=adapter(tmp_path,max_calls=1)
    admission=RequestAdmission(policy(),tmp_path/'wire.sqlite','0.0024',create=True)
    wire=OpenAIWire(policy()['base_url'],'fixture')
    calls=[]
    responses=iter([
        {'choices':[{'message':{'role':'assistant','content':'','tool_calls':[
            {'id':'1','type':'function','function':{'name':'sandbox_exec','arguments':json.dumps({'command':'true'})}}]}}]},
        {'choices':[{'message':{'role':'assistant','content':'done'}}]}])
    def transport(req,**kwargs):
        calls.append(json.loads(req.data))
        return io.BytesIO(json.dumps(next(responses)).encode())
    monkeypatch.setattr(wire,'_open',transport)
    agent=Agent(wire,str(tmp_path),request_admission=admission,request_budget_isolated=True,
                sandbox_tools=a,max_tokens=100,max_steps=4)
    result=agent.run('execute')
    assert result.stopped == 'answered',result.error
    assert len(calls) == 2
    assert admission.ledger.reserved_nanos() == 2400000
    assert json.loads(result.steps[0].tool_results[0][1])['returncode'] == 0
    again=agent.run('no budget left')
    assert again.stopped != 'answered'
    assert len(calls) == 2

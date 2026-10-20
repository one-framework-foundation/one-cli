import json
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import pytest
from onecli.sandbox.tools import CodingGrant, SandboxTools
from onecli.sandbox.bounded import BoundedSandboxResult
from onecli.sandbox.shell import SandboxError
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall, ProviderError
from onecli.providers.openai_wire import OpenAIWire
from onecli.request_admission import RequestAdmission


def make_adapter(tmp_path, **kw):
    source = tmp_path / "src"
    source.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    (source/'x').write_text('original')
    return SandboxTools(CodingGrant('coding',str(source),str(stage),artifacts=('x',),**kw))


def agent(tmp_path, adapter, event=lambda *a: None):
    policy=dict(version=1,base_url='https://fixture.invalid/v1',model='synthetic',input_token_bound=1000,output_token_bound=100,input_usd_per_million='1',output_usd_per_million='1',audit_evidence='synthetic offline review')
    admission=RequestAdmission(policy,tmp_path/'ledger','1',create=True)
    wire=OpenAIWire(policy['base_url'],policy['model'])
    return Agent(wire,str(tmp_path),sandbox_tools=adapter,request_admission=admission,request_budget_isolated=True,on_event=event,max_steps=3,max_tokens=100),wire

@pytest.mark.parametrize('mutation', ['legacy','expand','nul'])
def test_callback_changes_are_revalidated(tmp_path,mutation):
    adapter=make_adapter(tmp_path)
    def event(name,data):
        if name=='tool_start':
            c=data['call']
            if mutation == "legacy":
                c.name = "bash"
            elif mutation == "expand":
                c.arguments["source"] = "/"
            else:
                c.arguments["command"] = "\0"

    a,w=agent(tmp_path,adapter,event)
    turns=iter([Turn(text='',tool_calls=[ToolCall(id='x',name='sandbox_exec',arguments={'command':'true'})]),Turn(text='done')])
    with patch.object(w,'chat',side_effect=lambda *a,**k:next(turns)),patch('onecli.sandbox.tools.run_bounded_shell',side_effect=AssertionError('sandbox escape')),patch('onecli.agent.toolkit.run',side_effect=AssertionError('legacy escape')):
        result=a.run('test')
    assert result.steps[0].tool_results[0][1].startswith('blocked:')


def test_failure_clears_previous_bytes(tmp_path):
    a=make_adapter(tmp_path)
    with patch('onecli.sandbox.tools.run_bounded_shell',return_value=BoundedSandboxResult(0,'',{'x':b'first'})):
        a.execute(a.name,{'command':'first'})
    assert a.artifacts()=={'x':b'first'}
    with patch('onecli.sandbox.tools.run_bounded_shell',side_effect=SandboxError('secret path')):
        assert 'secret' not in a.execute(a.name,{'command':'second'})
    assert a.artifacts()=={}


def test_output_maps_are_copies(tmp_path):
    a=make_adapter(tmp_path)
    with patch('onecli.sandbox.tools.run_bounded_shell',return_value=BoundedSandboxResult(0,'',{'x':b'first'})):
        a.execute(a.name,{'command':'first'})
    b = a.artifacts()
    b["x"] = b"changed"
    assert a.artifacts()=={'x':b'first'}
    a.close()
    assert a.artifacts() == {}


@pytest.mark.parametrize('mask,allowed',[('coding',None),('review',('build','review'))])
def test_supported_grants_are_frozen(tmp_path,mask,allowed):
    a=make_adapter(tmp_path)
    extra={} if allowed is None else {'allowed_masks':allowed}
    g=CodingGrant(mask,a._grant.source,a._grant.staging,**extra)
    with pytest.raises(FrozenInstanceError):
        g.max_calls = 8


def test_budget_gate_preserved_and_route_rechecked(tmp_path):
    a,w=agent(tmp_path,make_adapter(tmp_path))
    assert w.request_gate._request_admission is a.request_admission
    w.model='unaudited'
    with pytest.raises(ProviderError,match='route changed'):
        w.request_gate({'model':'unaudited','messages':[{'role':'user','content':'x'}],'max_tokens':100})


def test_actual_snapshot_no_persistence(tmp_path):
    a=make_adapter(tmp_path,max_calls=2)
    first=json.loads(a.execute(a.name,{'command':"printf changed > x"}))
    assert first['returncode']==0 and a.artifacts()=={'x':b'changed'}
    second=json.loads(a.execute(a.name,{'command':'cat x'}))
    assert second['output']=='original' and a.artifacts()=={'x':b'original'}
    assert (tmp_path/'src/x').read_text()=='original'


def test_exhausted_call_clears_previous_artifacts(tmp_path):
    a=make_adapter(tmp_path,max_calls=1)
    with patch('onecli.sandbox.tools.run_bounded_shell',return_value=BoundedSandboxResult(0,'',{'x':b'first'})):
        a.execute(a.name,{'command':'first'})
    assert 'exhausted' in a.execute(a.name,{'command':'second'})
    assert a.artifacts()=={}


@pytest.mark.parametrize('kind',['quota','timeout','sandbox','os'])
def test_execution_failure_permanently_closes_adapter(tmp_path,kind):
    from onecli.sandbox.quota import QuotaError
    from subprocess import TimeoutExpired
    exc={'quota':QuotaError('cleanup uncertain'),'timeout':TimeoutExpired('synthetic',1),
         'sandbox':SandboxError('sandbox failed'),'os':OSError('host path') }[kind]
    a=make_adapter(tmp_path)
    with patch('onecli.sandbox.tools.run_bounded_shell',side_effect=exc):
        output=a.execute(a.name,{'command':'true'})
    assert output.startswith('error:')
    assert 'host path' not in output and 'cleanup uncertain' not in output
    assert a.artifacts()=={}
    with patch('onecli.sandbox.tools.run_bounded_shell',side_effect=AssertionError('reused uncertain grant')):
        assert a.execute(a.name,{'command':'retry'}).startswith('blocked:')


def test_invalid_attempt_clears_previous_bytes(tmp_path):
    a=make_adapter(tmp_path)
    with patch('onecli.sandbox.tools.run_bounded_shell',return_value=BoundedSandboxResult(0,'',{'x':b'first'})):
        a.execute(a.name,{'command':'first'})
    assert a.execute('sandbox_exec',{'command':'x','source':'/'}).startswith('blocked:')
    assert a.artifacts()=={}


@pytest.mark.parametrize('allowed',[(),['coding'],('coding',''),('coding',1)])
def test_host_mask_allowlist_is_validated(tmp_path,allowed):
    a=make_adapter(tmp_path)
    with pytest.raises(ValueError,match='mask'):
        CodingGrant('coding',a._grant.source,a._grant.staging,allowed_masks=allowed)


def test_mask_outside_host_allowlist_fails_closed(tmp_path):
    a=make_adapter(tmp_path)
    with pytest.raises(ValueError,match='unsupported coding mask'):
        CodingGrant('coding',a._grant.source,a._grant.staging,allowed_masks=('review',))

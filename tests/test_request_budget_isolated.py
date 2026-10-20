import json
import pytest
from onecli import cli, config, tools
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall
from onecli.providers.openai_wire import OpenAIWire
from onecli.request_admission import RequestAdmission
from onecli.subagent import Spawner, Preset


def budget_fixture(tmp_path):
    policy = dict(version=1, base_url='https://fixture.invalid/v1', model='fixture', input_token_bound=1000, output_token_bound=100, input_usd_per_million='1', output_usd_per_million='2', audit_evidence='synthetic only')
    file = tmp_path / "policy.json"
    file.write_text(json.dumps(policy))
    admission = RequestAdmission(policy, tmp_path/'budget.sqlite', '1', create=True)
    wire = OpenAIWire(policy['base_url'], 'fixture')
    argv = ['--request-budget-isolated','--require-priced-budget','--no-fallback','--request-budget-policy',str(file),'--request-budget-ledger',str(tmp_path/'budget.sqlite'),'--max-budget-usd','1','--workspace',str(tmp_path),'-p','hello']
    return admission, wire, argv

@pytest.mark.parametrize('extra', [['--hooks','x'],['--guard','x'],['--mcp-config','{}'],['--phases','[]'],['--chat'],['--resume','x'],['--image','x'],['--audio','x'],['--video','x'],['--speak']])
def test_effects_rejected_before_configuration(tmp_path, monkeypatch, extra):
    _, _, args = budget_fixture(tmp_path)
    monkeypatch.setattr(config,'resolve',lambda **kw: pytest.fail('backend resolution'))
    monkeypatch.setattr(cli,'_mode',lambda *a: pytest.fail('mode initialization'))
    assert cli.main(args+extra) == 2

@pytest.mark.parametrize('args', [['--list'],['--sessions'],['--list-agents'],['--orc-list'],['--orc-show','x'],['--orc-report'],['models']])
def test_alternate_commands_rejected(monkeypatch,args):
    monkeypatch.setattr(config,'discover',lambda *a,**kw: pytest.fail('network discovery'))
    monkeypatch.setattr(cli.serving_cli,'dispatch',lambda *a: pytest.fail('serving command'))
    assert cli.main(args+['--request-budget-isolated']) == 2

def test_auto_extensions_never_loaded(tmp_path,monkeypatch):
    _, _, argv = budget_fixture(tmp_path)
    args = cli.build_parser().parse_args(argv)
    monkeypatch.setattr(cli.hooklib,'build',lambda *a,**kw: pytest.fail('hooks loaded'))
    monkeypatch.setattr(cli.mcplib,'load_configs',lambda *a,**kw: pytest.fail('MCP loaded'))
    assert cli._hooks(args,'synthetic') is None
    assert cli._mcp(args) is None
    assert cli._guards(args) is None
    assert cli._surface(args,'default',None) == []
    assert cli._spawner(args,{}) is None

@pytest.mark.parametrize('name',['bash','code','task','web_fetch','mcp__adversary'])
def test_model_tool_never_executes(tmp_path,monkeypatch,name):
    admission, wire, _ = budget_fixture(tmp_path)
    call = ToolCall(id="x", name=name, arguments={})
    monkeypatch.setattr(wire,'chat',lambda *a, **kw: Turn(text='',tool_calls=[call]))
    monkeypatch.setattr(tools,'run',lambda *a,**kw: pytest.fail('tool executed'))
    agent=Agent(wire,str(tmp_path),request_admission=admission,request_budget_isolated=True,autocompact=False,max_tokens=100)
    agent.tool_names=[name]
    assert agent.run('test').stopped == 'restricted'
    assert agent._gate(call).allowed is False
    with pytest.raises(ValueError,match='child'):
        Spawner(parent=agent).build(Preset(name='fixture'))

@pytest.mark.parametrize('kw',[dict(hooks=object()),dict(guard=object()),dict(spawner=object())])
def test_direct_agent_rejects_extensions(tmp_path,kw):
    admission,wire,_=budget_fixture(tmp_path)
    with pytest.raises(ValueError,match='forbids'):
        Agent(wire,str(tmp_path),request_admission=admission,request_budget_isolated=True,**kw)

def test_context_has_no_probe(tmp_path,monkeypatch):
    admission, wire, argv = budget_fixture(tmp_path)
    args = cli.build_parser().parse_args(argv)
    args._request_admission = admission
    backend=config.Backend(name='fixture',model='fixture',base_url='https://fixture.invalid/v1',source='flag')
    monkeypatch.setattr(config,'price_for',lambda *a:(1,2))
    monkeypatch.setattr(config,'resolve_context_window',lambda *a:pytest.fail('network probe'))
    monkeypatch.setattr(cli,'_safe_build_provider',lambda *a:wire)
    assert cli._context_settings(args,backend)['request_budget_isolated']

def test_resolve_disables_network_discovery(monkeypatch):
    seen = []
    b = config.Backend(
        name="fixture", model="fixture", base_url="https://fixture.invalid/v1"
    )
    monkeypatch.setattr(config,'discover',lambda include_local=True:seen.append(include_local) or [b])
    assert config.resolve(name='fixture',include_local=False) is b
    assert seen == [False]

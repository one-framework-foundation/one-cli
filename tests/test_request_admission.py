import io
import urllib.error
from types import SimpleNamespace
import pytest
from onecli.agent import Agent
from onecli.request_admission import RequestAdmission
from onecli.providers.base import ProviderError
from onecli.providers.openai_wire import OpenAIWire
from onecli.subagent import Spawner, Preset

def policy():
    return dict(version=1, base_url='https://fixture.invalid/v1', model='fixture', input_token_bound=1000, output_token_bound=100, input_usd_per_million='1', output_usd_per_million='2', audit_evidence='Synthetic fixture only')

def fixture(tmp_path, limit='0.0012'):
    a = RequestAdmission(policy(), tmp_path/'ledger.sqlite', limit, create=True)
    wire = OpenAIWire(policy()["base_url"], "fixture")
    a.bind(wire)
    return a, wire

def body():
    return dict(model='fixture', messages=[dict(role='user', content='hello')], max_tokens=100)

@pytest.mark.parametrize('stream', [False, True])
def test_request_rejected_before_transport(tmp_path, monkeypatch, stream):
    a, wire = fixture(tmp_path, '0.001')
    monkeypatch.setattr(wire, '_open', lambda *a: pytest.fail('HTTP attempted'))
    with pytest.raises(ProviderError, match='remaining shared budget'):
        wire._stream(body(), lambda *a: None) if stream else wire._post(body())
    assert a.ledger.reserved_nanos() == 0

def test_retry_needs_new_reservation(tmp_path, monkeypatch):
    a, wire = fixture(tmp_path)
    calls = []

    def fail(*args, **kw):
        calls.append(1)
        raise urllib.error.HTTPError('fixture', 503, 'failure', {}, io.BytesIO(b'server failure'))
    monkeypatch.setattr(wire, '_open', fail)
    monkeypatch.setattr('onecli.providers.openai_wire.time.sleep', lambda *a: None)
    with pytest.raises(ProviderError, match="remaining shared budget"):
        wire._post(body())
    assert len(calls) == 1
    assert a.ledger.reserved_nanos() == 1200000

def test_retry_cannot_remove_output_cap(tmp_path, monkeypatch):
    a, wire = fixture(tmp_path, "1")
    calls = []

    def fail(*args, **kw):
        calls.append(1)
        raise urllib.error.HTTPError('fixture', 400, 'failure', {}, io.BytesIO(b'unsupported max_tokens'))
    monkeypatch.setattr(wire, '_open', fail)
    with pytest.raises(ProviderError, match="output cap"):
        wire._post(body())
    assert len(calls) == 1
    assert a.ledger.reserved_nanos() == 1200000

@pytest.mark.parametrize('change', [dict(n=2), dict(model='other'), dict(max_tokens=None), dict(messages=[dict(role='user', content=[dict(type='image_url')])]), dict(tools=[dict(type='web_search')]), dict(reasoning_effort='high')])
def test_unsupported_body_zero_reservations(tmp_path, change):
    a, wire = fixture(tmp_path)
    request = body()
    request.update(change)
    with pytest.raises(ProviderError):
        wire.request_gate(request)
    assert a.ledger.reserved_nanos() == 0

def test_route_mutation_denied(tmp_path):
    a, wire = fixture(tmp_path)
    wire.base_url = "https://other.invalid"
    with pytest.raises(ProviderError, match="route changed"):
        wire.request_gate(body())


def test_children_share_budget_without_double_charge(tmp_path):
    a, wire = fixture(tmp_path)
    spawner = Spawner(parent=None)
    Agent(wire, str(tmp_path), request_admission=a, spawner=spawner, autocompact=False)
    child = spawner.build(Preset(name='fixture'))
    assert child.request_admission is a
    wire.request_gate(body())
    with pytest.raises(ProviderError, match="remaining shared budget"):
        child.provider.request_gate(body())
    assert a.ledger.reserved_nanos() == 1200000
    assert 'forbids external' in spawner._drive(Preset(name='external', driver='codex'), 'fixture')

def test_compaction_hits_same_wire_gate(tmp_path, monkeypatch):
    a, wire = fixture(tmp_path, "0.0001")
    agent = Agent(wire, str(tmp_path), request_admission=a)
    monkeypatch.setattr(wire, '_open', lambda *a: pytest.fail('HTTP attempted'))
    with pytest.raises(ProviderError, match="request budget"):
        agent._summarize([dict(role="user", content="fixture")])


def test_unknown_provider_rejected(tmp_path):
    a, wire = fixture(tmp_path)
    with pytest.raises(ValueError, match="unaudited"):
        a.bind(SimpleNamespace())


@pytest.mark.parametrize('extra', [['--image','fixture'], ['--audio','fixture'], ['--video','fixture'], ['--speak']])
def test_cli_blocks_auxiliary_models_before_processing(tmp_path, monkeypatch, capsys, extra):
    from onecli import cli, config
    monkeypatch.setattr(config, 'resolve', lambda **kw: SimpleNamespace(model='fixture', name='fixture'))
    monkeypatch.setattr(config, 'price_for', lambda *a: (1, 2))
    monkeypatch.setattr(cli, 'collect_attachments', lambda *a: pytest.fail('auxiliary model path'))
    argv = ['--require-priced-budget','--no-fallback','--max-budget-usd','1', '--request-budget-policy', str(tmp_path/'missing.json'), '--request-budget-ledger',str(tmp_path/'ledger'),'-p','fixture']
    assert cli.main(argv+extra) == 2
    assert 'auxiliary media' in capsys.readouterr().err


def test_cli_chat_rejects_creation_flag(capsys):
    from onecli import cli
    assert cli.main(['--chat','--create-request-budget-ledger','-p','fixture']) == 2
    assert 'unsupported' in capsys.readouterr().err


def test_cli_valid_policy_reuses_ledger_before_run(tmp_path, monkeypatch):
    import json
    from onecli import cli, config
    a, wire = fixture(tmp_path)
    file = tmp_path / "policy.json"
    file.write_text(json.dumps(policy()))
    backend = SimpleNamespace(model='fixture', name='fixture', base_url=policy()['base_url'])
    monkeypatch.setattr(config, 'resolve', lambda **kw: backend)
    monkeypatch.setattr(config, 'price_for', lambda *a: (1, 2))
    monkeypatch.setattr(cli, '_safe_build_provider', lambda *a: wire)
    # Stop at this boundary before hooks/session I/O. No providers invoked.
    def attachments(args, b):
        assert args._request_admission.ledger.reserved_nanos() == 0
        assert wire.reject_redirects
        raise RuntimeError('validated before attachments')
    monkeypatch.setattr(cli, 'collect_attachments', attachments)
    # Use a fresh wire because a live wire cannot belong to two admission owners.
    monkeypatch.setattr(cli, '_safe_build_provider', lambda *a: OpenAIWire(policy()['base_url'], 'fixture'))
    with pytest.raises(RuntimeError, match='validated before attachments'):
        cli.main(['--require-priced-budget','--no-fallback','--max-budget-usd','0.0012', '--request-budget-policy',str(file),'--request-budget-ledger',str(tmp_path/'ledger.sqlite'),'-p','fixture'])


def test_recovery_missing_ledger_rejects(tmp_path):
    from onecli.budget import BudgetError
    with pytest.raises(BudgetError):
        RequestAdmission(policy(), tmp_path/'missing.sqlite', '1')


def test_widely_separated_tariffs_round_up(tmp_path):
    p = policy()
    p["output_usd_per_million"] = "1e-100"
    a = RequestAdmission(p, tmp_path/'ledger.sqlite', '1', create=True)
    a.admit(body())
    assert a.ledger.reserved_nanos() == 1000001

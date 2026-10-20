from dataclasses import replace
from onecli import config
from onecli.senses import capabilities as cm, eye
import pytest

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: None)

def backend(model='text', url='http://fixture.invalid/v1', name='local-b', **kw):
    return config.Backend(name=name, model=model, base_url=url, **kw)

@pytest.mark.parametrize('wire', ['openai','anthropic'])
def test_unknown_is_not_inline(wire):
    assert not cm.Capabilities().declares(backend(wire=wire), 'vision')

def test_arbitrary_alias_uses_exact_model(monkeypatch):
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'models':[
        {'model':'text','capabilities':['completion']},
        {'model':'vision','capabilities':['completion','vision']}]})
    c=cm.Capabilities()
    assert not c.declares(backend(), 'vision')
    assert c.declares(backend('vision'), 'vision')

def test_single_other_model_cannot_lend_vision(monkeypatch):
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'models':[
        {'model':'other','capabilities':['vision']}]})
    assert not cm.Capabilities().declares(backend(name='llamacpp/x'), 'vision')

def test_openai_data_capabilities(monkeypatch):
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'data':[
        {'id':'text','capabilities':['vision']}]})
    assert cm.Capabilities().declares(backend(), 'vision')

@pytest.mark.parametrize('bad', ['vision', {'vision':True}, None, [123]])
def test_malformed_declaration_is_unknown(monkeypatch,bad):
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'models':[
        {'model':'text','capabilities':bad}]})
    c=cm.Capabilities()
    assert not c.declares(backend(), 'vision')
    assert c.status(backend(), 'vision') == 'unknown'

def test_backend_allowlist_is_not_model_evidence(monkeypatch):
    monkeypatch.setenv('ONE_CLI_VISION_BACKENDS', 'local-b')
    assert not cm.Capabilities().declares(backend(), 'vision')

def test_dropin_binding_does_not_follow_model_or_endpoint(tmp_path):
    p=tmp_path/'test.env'
    p.write_text('ONE_PROVIDER_MODEL=vision\nONE_PROVIDER_BASE_URL=http://fixture.invalid/v1\nONE_PROVIDER_MODALITIES=vision\n')
    b=config._backend_from_env(p)
    c=cm.Capabilities()
    assert c.declares(b,'vision')
    assert not c.declares(replace(b,model='text'),'vision')
    assert not c.declares(replace(b,base_url='http://another.invalid/v1'),'vision')
    assert 'modalities' not in b.extra

def test_live_negative_overrides_dropin(tmp_path,monkeypatch):
    p=tmp_path/'test.env'
    p.write_text('ONE_PROVIDER_MODEL=text\nONE_PROVIDER_BASE_URL=http://fixture.invalid/v1\nONE_PROVIDER_MODALITIES=vision\n')
    b=config._backend_from_env(p)
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'data':[{'id':'text','capabilities':['completion']}]})
    assert cm.Capabilities().status(b,'vision') == 'unsupported'
    assert not cm.Capabilities().declares(b,'vision')

def test_cache_scopes_endpoint_and_model(monkeypatch):
    calls=[]
    def get(url, **kw):
        calls.append(url)
        return {'data':[{'id':'text','capabilities':['vision'] if 'seeing' in url else ['completion']}]}
    monkeypatch.setattr(cm, '_get_json',get)
    c=cm.Capabilities()
    assert c.declares(backend(url='http://seeing.invalid/v1'),'vision')
    assert not c.declares(backend(url='http://blind.invalid/v1'),'vision')
    assert not c.declares(backend('missing',url='http://seeing.invalid/v1'),'vision')
    assert len(calls)==2

def test_unknown_image_refuses_without_fallback():
    result=eye.perceive_image(None, backend(), cm.Capabilities(), fallback=False)
    assert not result.inline
    assert any('unknown' in r.skipped for r in result.rungs)

def test_supported_model_still_needs_capable_wire(monkeypatch):
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'data':[{'id':'text','capabilities':['vision']}]})
    b=backend(wire='text-only-wire')
    assert cm.Capabilities().status(b,'vision') == 'supported'
    assert not cm.Capabilities().declares(b,'vision')

def test_conflicting_explicit_metadata_fails_closed(monkeypatch):
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {
        'models':[{'model':'text','capabilities':['vision']}],
        'data':[{'id':'text','capabilities':['completion']}]})
    assert not cm.Capabilities().declares(backend(),'vision')

def test_expired_snapshot_rechecks_model(monkeypatch):
    c=cm.Capabilities()
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'data':[{'id':'text','capabilities':['vision']}]})
    assert c.declares(backend(),'vision')
    monkeypatch.setattr(cm, '_get_json', lambda *a, **k: {'data':[{'id':'text','capabilities':['completion']}]})
    c._probed_at -= 61
    assert not c.declares(backend(),'vision')

"""Saved media must pass Senses before a resumed request reaches the wire."""
import copy
from pathlib import Path

import pytest
from onecli import cli, config, media, session
from onecli.senses import capabilities, eye, logbook, router
from onecli.senses.base import Outcome, DESCRIBE


@pytest.fixture
def setup(monkeypatch, tmp_path):
    backend = config.Backend(name='swift', model='swift', base_url='http://fixture.invalid/v1')
    caps = capabilities.Capabilities()
    monkeypatch.setattr(capabilities, 'probe', lambda **_: caps)
    monkeypatch.setattr(capabilities, '_get_json', lambda *a, **k: {
        'data': [{'id': 'swift', 'capabilities': ['completion']}]})
    monkeypatch.setattr(logbook, 'LOG_PATH', tmp_path/'senses.jsonl')
    calls = []
    def perceive(att, backend, caps, **kwargs):
        calls.append((att.path, kwargs))
        if not kwargs.get('fallback', True) or kwargs.get('mode') == 'inline':
            return Outcome(modality='image')
        return Outcome(modality='image', route=DESCRIBE, text='Verified description of '+Path(att.path).name)
    monkeypatch.setattr(eye, 'perceive', perceive)
    att = media.Attachment('image', str(tmp_path/'old.png'), 'image/png')
    history = [{'role':'system','content':'system'},
               {'role':'user','content':[{'type':'text','text':'original question'},att.to_block()]},
               {'role':'assistant','content':'previous answer'}]
    return backend, history, calls


def test_resume_routes_saved_image_before_provider(setup, monkeypatch, tmp_path):
    backend, history, calls = setup
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path/'sessions')
    session.append('saved', {'kind':'messages','messages':history})
    # Exercise the actual CLI resume entrypoint; no provider inference.
    monkeypatch.setattr(config, 'discover', lambda **kw: [backend])
    monkeypatch.setattr(cli, '_mcp', lambda *a: None)
    monkeypatch.setattr(cli, '_hooks', lambda *a: None)
    monkeypatch.setattr(cli, 'run_once', lambda args, b, prior, sid: (
        0 if not any(media.blocks_of(m.get('content')) for m in prior)
        and 'Verified description' in str(prior) else 99))
    assert cli.main(['-m','swift','-r','saved','--no-fallback','--workspace',str(tmp_path),'continue']) == 0
    assert len(calls) == 1


def test_preserves_order_and_original_history(setup):
    backend, history, calls = setup
    original = copy.deepcopy(history)
    routed = router.route_history(history, backend)
    assert history == original
    assert routed[0] == history[0] and routed[2] == history[2]
    assert routed[1]['content'][0] == history[1]['content'][0]
    assert 'Verified description' in routed[1]['content'][1]['text']
    assert not media.blocks_of(routed[1]['content'])
    assert router.route_history(routed, backend) == routed
    assert len(calls) == 1


def test_resume_honours_no_media_fallback(setup):
    backend, history, calls = setup
    with pytest.raises(media.MediaError, match='no rung worked'):
        router.route_history(history, backend, fallback=False)


def test_resume_unreadable_media_becomes_a_note(setup):
    backend, history, calls = setup
    out = router.route_history(history, backend, mode='inline')
    assert 'NOT READ' in out[1]['content'][-1]['text']


def test_off_removes_only_media(setup):
    backend, history, calls = setup
    out = router.route_history(history, backend, mode='off')
    assert out[1]['content'] == [{'type':'text','text':'original question'}]
    assert not calls


def test_supported_inline_preserved(setup, monkeypatch):
    backend, history, calls = setup
    monkeypatch.setattr(eye, 'perceive', lambda *a, **k: Outcome(modality='image',route='inline',inline=True))
    assert router.route_history(history,backend) == history


def test_duplicate_attachment_perceived_once(setup):
    backend, history, calls = setup
    history.append(copy.deepcopy(history[1]))
    out = router.route_history(history,backend)
    assert out[1]['content'] == out[3]['content']
    assert len(calls) == 1

"""Structured vendor boundary; no real account or paid inference in these tests."""
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from onecli import config
from onecli.agent import Agent
from onecli.providers import build
from onecli.providers.base import ProviderError
from onecli.providers import opencode_sdk as sdk


TOOLS = [{'type': 'function', 'function': {'name': 'read', 'description': 'Read file',
          'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}]


def test_normal_selection_without_discovery(monkeypatch):
    monkeypatch.setattr(config, 'discover', lambda **kw: pytest.fail('must not probe unrelated backends'))
    backend = config.resolve('opencode/fixture/model/with/slashes')
    provider = build(backend)
    assert isinstance(provider, sdk.OpenCodeSDK)
    assert provider.ref == {'providerID': 'fixture', 'id': 'model/with/slashes'}
    assert backend.base_url == '' and backend.api_key == ''


@pytest.mark.parametrize('name', ['opencode/model', 'opencode//model', 'opencode/provider/'])
def test_invalid_selection(name):
    with pytest.raises(RuntimeError, match='opencode/provider/model'):
        config.resolve(name)


def test_wire_overrides_refused():
    with pytest.raises(RuntimeError, match='transport'):
        config.resolve('opencode/fixture/model', wire='openai')
    for overrides in ({'base_url': 'https://example.invalid'}, {'model': 'other'}):
        with pytest.raises(RuntimeError, match='catalog model or endpoint'):
            config.resolve('opencode/fixture/model', **overrides)
    backend = config.resolve('opencode/fixture/model')
    backend.headers = {'Authorization': 'secret'}
    with pytest.raises(ProviderError, match='vendor configuration'):
        build(backend)


def response():
    return {'version': 1, 'text': '', 'thinking': 'reason',
            'tool_calls': [{'id': 'c', 'name': 'read', 'input': {'path': 'a'}}],
            'message': {'role': 'assistant', 'content': [{'type': 'tool-call', 'id': 'c',
                         'name': 'read', 'input': {'path': 'a'}, 'providerMetadata': {'signature': 'opaque'}}]},
            'finish_reason': 'tool_calls', 'usage': {'inputTokens': 7, 'outputTokens': 3, 'cacheReadInputTokens': 2}}


def test_turn_usage_echo_and_bounds(monkeypatch):
    captured = []
    monkeypatch.setattr(sdk, '_request', lambda payload, timeout: captured.append(payload) or response())
    provider = sdk.OpenCodeSDK('fixture/model')
    deltas = []
    turn = provider.chat([{'role': 'user', 'content': 'read'}], tools=TOOLS,
                         max_tokens=123, temperature=.2, on_delta=lambda *x: deltas.append(x))
    assert turn.tool_calls[0].arguments == {'path': 'a'}
    assert turn.usage['prompt_tokens'] == 7 and turn.usage['cache_read_tokens'] == 2
    assert turn.usage['completion_tokens'] == 3
    assert Agent._assistant_message(turn)['opencode_message'] == response()['message']
    assert captured[0]['max_tokens'] == 123 and captured[0]['temperature'] == .2
    assert deltas == [('thinking', 'reason')]
    identity = provider.session_id
    # An existing consumer resolves current transport behavior on each call.
    monkeypatch.setattr(sdk, '_request', lambda *a: {**response(), 'text': 'changed'})
    assert provider.chat([], tools=TOOLS).text == 'changed'
    assert provider.session_id == identity


def test_unoffered_tools_and_unsupported_guards_fail_closed(monkeypatch):
    monkeypatch.setattr(sdk, '_request', lambda *a: response())
    provider = sdk.OpenCodeSDK('fixture/model')
    with pytest.raises(ProviderError, match='unoffered'):
        provider.chat([])
    with pytest.raises(ProviderError, match='response_format'):
        provider.chat([], response_format={'type': 'json_object'})
    provider.request_gate = lambda *a: pytest.fail('cannot claim SDK wire bytes measured')
    with pytest.raises(ProviderError, match='audited wire'):
        provider.chat([])


@pytest.mark.parametrize('output', ['not json secret', '{"version":2}',
    '{"version":1,"error":"Bearer secret"}', '{"version":1,"error":[]}', '{"version":1,"error":"generation_failed"}'])
def test_transport_errors_never_forward_secrets(monkeypatch, output):
    monkeypatch.setenv('ONE_OPENCODE_SDK_DIR', 'unused')
    class Child:
        returncode = 1
        def communicate(self, *a, **kw):
            return output, None
    monkeypatch.setattr(sdk.subprocess, 'Popen', lambda *a, **kw: Child())
    with pytest.raises(ProviderError) as error:
        sdk._request({}, 1)
    assert 'secret' not in str(error.value) and output not in str(error.value)


@pytest.mark.parametrize('cancel', [False, True])
def test_timeout_and_cancellation_reap_owned_process(monkeypatch, cancel):
    monkeypatch.setenv('ONE_OPENCODE_SDK_DIR', 'unused')
    original = subprocess.Popen
    children = []
    def spawn(*a, **kw):
        child = original([sys.executable, '-c', 'import time; time.sleep(60)'], **kw)
        children.append(child)
        if cancel:
            child.communicate = lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())
        return child
    monkeypatch.setattr(sdk.subprocess, 'Popen', spawn)
    with pytest.raises(KeyboardInterrupt if cancel else ProviderError):
        sdk._request({}, .05)
    assert len(children) == 1 and children[0].poll() is not None


@pytest.fixture
def vendor(monkeypatch, tmp_path):
    install = os.environ.get('ONE_OPENCODE_TEST_SDK_DIR')
    if not install or not shutil.which('bun'):
        pytest.skip('set ONE_OPENCODE_TEST_SDK_DIR to pinned SDK install for local integration')
    # No ambient credentials, project plugins, account configuration or remote catalog.
    for key in list(os.environ):
        if key.startswith('OPENCODE_') or any(s in key.upper() for s in ('TOKEN', 'SECRET', 'API_KEY', 'PASSWORD')):
            monkeypatch.delenv(key)
    for key in ('HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'XDG_STATE_HOME', 'OPENCODE_CONFIG_DIR'):
        folder = tmp_path / key
        folder.mkdir()
        monkeypatch.setenv(key, str(folder))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ONE_OPENCODE_SDK_DIR', str(Path(install).resolve()))
    monkeypatch.setenv('OPENCODE_DISABLE_MODELS_FETCH', '1')
    monkeypatch.setenv('OPENCODE_CONFIG_PROJECT_DISABLE', '1')
    monkeypatch.setenv('OPENCODE_DISABLE_CLAUDE_CODE', '1')
    seen = []
    class Fake(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            tool = body['messages'][-1].get('content') == 'tool please'
            message = {'role': 'assistant', 'content': None if tool else 'fixture text'}
            if tool:
                message['tool_calls'] = [{'index': 0, 'id': 'call_fixture', 'type': 'function',
                    'function': {'name': 'read', 'arguments': '{"path":"example.txt"}'}}]
            data = {'id': 'fixture', 'object': 'chat.completion.chunk', 'model': 'fixture-model',
                'choices': [{'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if tool else 'stop'}],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}}
            payload = ('data: ' + json.dumps(data) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    content = {'providers': {'fixture': {'package': '@opencode/ai/providers/openai/chat',
        'settings': {'baseURL': f'http://127.0.0.1:{server.server_port}/v1', 'apiKey': 'fixture-only'},
        'models': {'fixture-model': {'limit': {'context': 4096, 'output': 256}}}}}}
    monkeypatch.setenv('OPENCODE_CONFIG_CONTENT', json.dumps(content))
    yield seen, content
    server.shutdown()
    server.server_close()


def test_real_sdk_tool_round_trip_bounds_usage_and_reload(vendor, monkeypatch):
    seen, content = vendor
    provider = build(config.resolve('opencode/fixture/fixture-model'), timeout=30)
    history = [{'role': 'user', 'content': 'tool please'}]
    turn = provider.chat(history, tools=TOOLS, max_tokens=123)
    assert turn.finish_reason == 'tool_calls'
    assert turn.tool_calls[0].arguments == {'path': 'example.txt'}
    assert turn.usage['prompt_tokens'] == 7 and turn.usage['completion_tokens'] == 3
    history += [Agent._assistant_message(turn), {'role': 'tool', 'tool_call_id': turn.tool_calls[0].id,
                                                'content': 'file contents'}]
    final = provider.chat(history, max_tokens=32)
    assert final.text == 'fixture text'
    assert seen[0]['tools'][0]['function']['parameters'] == TOOLS[0]['function']['parameters']
    assert seen[0]['max_completion_tokens'] == 123 and seen[1]['max_completion_tokens'] == 32
    assert seen[1]['messages'][-1] == {'role': 'tool', 'tool_call_id': 'call_fixture', 'content': 'file contents'}
    assert seen[1]['messages'][-2]['tool_calls'][0]['function']['name'] == 'read'
    with pytest.raises(ProviderError, match='invalid_bounds'):
        provider.chat([], max_tokens=257)
    assert len(seen) == 2
    # The same Python instance consumes changed vendor config without a restart.
    content['policies'] = [{'action': 'provider.use', 'resource': 'fixture', 'effect': 'deny'}]
    monkeypatch.setenv('OPENCODE_CONFIG_CONTENT', json.dumps(content))
    with pytest.raises(ProviderError, match='model_unavailable'):
        provider.chat([{'role': 'user', 'content': 'blocked'}])
    assert len(seen) == 2


def test_real_sdk_policy_and_error_redaction(vendor, monkeypatch):
    seen, content = vendor
    content['policies'] = [{'action': 'tool.use', 'resource': '*', 'effect': 'deny'}]
    monkeypatch.setenv('OPENCODE_CONFIG_CONTENT', json.dumps(content))
    provider = sdk.OpenCodeSDK('fixture/fixture-model', 30)
    with pytest.raises(ProviderError, match='policy_denied'):
        provider.chat([{'role': 'user', 'content': 'tool please'}], tools=TOOLS)
    assert not seen
    content.pop('policies')
    provider = sdk.OpenCodeSDK('fixture/missing-secret-do-not-print', 30)
    monkeypatch.setenv('OPENCODE_CONFIG_CONTENT', json.dumps(content))
    with pytest.raises(ProviderError) as error:
        provider.chat([{'role': 'user', 'content': 'hello'}])
    assert 'secret' not in str(error.value) and 'fixture-only' not in str(error.value)
    assert not seen


def test_real_sdk_drives_onecli_agent_loop(vendor):
    seen, _ = vendor
    Path('example.txt').write_text('actual local file contents')
    provider = build(config.resolve('opencode/fixture/fixture-model'), timeout=30)
    result = Agent(provider, os.getcwd(), max_steps=3, max_tokens=128,
                   tool_names=['read'], autocompact=False, system='fixture').run('tool please')
    assert result.stopped == 'answered' and result.text == 'fixture text'
    assert result.tool_calls == 1 and len(seen) == 2
    assert 'actual local file contents' in seen[1]['messages'][-1]['content']

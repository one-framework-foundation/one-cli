"""onecli.evals.stresslane: each failure mode, served by a real loopback server."""
from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from onecli.evals import stresslane


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(stresslane, 'CALLS', {})
    monkeypatch.setattr(stresslane, 'LOG', '')
    srv = ThreadingHTTPServer(('127.0.0.1', 0), stresslane.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()
    t.join(5)


def post(port, model, stream=False, raw=None):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    body = raw if raw is not None else json.dumps({'model': model, 'stream': stream})
    conn.request('POST', '/v1/chat/completions', body=body)
    resp = conn.getresponse()
    try:
        data = resp.read()
    except http.client.IncompleteRead as exc:
        data = exc.partial
    conn.close()
    return resp, data


def test_sse_helper_terminates_with_done():
    out = stresslane._sse([{'a': 1}])
    assert out == 'data: {"a": 1}\n\ndata: [DONE]\n\n'


def test_get_lists_no_models(server):
    conn = http.client.HTTPConnection('127.0.0.1', server, timeout=10)
    conn.request('GET', '/v1/models')
    assert json.loads(conn.getresponse().read()) == {'data': []}


def test_ok_non_streaming(server):
    resp, data = post(server, 'stub-ok')
    msg = json.loads(data)['choices'][0]
    assert msg['message']['content'] == 'done.' and msg['finish_reason'] == 'stop'
    assert json.loads(data)['usage']['total_tokens'] == 15


def test_ok_streaming(server):
    resp, data = post(server, 'stub-ok', stream=True)
    assert resp.getheader('Content-Type') == 'text/event-stream'
    assert b'"content": "done."' in data and data.endswith(b'data: [DONE]\n\n')


def test_unknown_model_and_bad_json_fall_back_to_ok(server):
    assert json.loads(post(server, 'whatever')[1])['choices'][0]['message']['content'] == 'done.'
    assert json.loads(post(server, '', raw='not json')[1])['choices']


def test_calls_are_counted_per_model(server):
    post(server, 'stub-ok')
    post(server, 'stub-ok')
    post(server, 'stub-500')
    assert stresslane.CALLS == {'stub-ok': 2, 'stub-500': 1}


@pytest.mark.parametrize('stream', [False, True])
def test_silent_has_empty_body(server, stream):
    resp, data = post(server, 'stub-silent', stream)
    assert resp.status == 200 and data == b''


def test_garbage_is_html(server):
    resp, data = post(server, 'stub-garbage')
    assert resp.getheader('Content-Type') == 'text/html' and b'502' in data


@pytest.mark.parametrize('stream,prefix', [(True, b'data: {"choices"'), (False, b'{"choices"')])
def test_truncated_drops_mid_answer(server, stream, prefix):
    resp, data = post(server, 'stub-truncated', stream)
    assert data.startswith(prefix)
    with pytest.raises(ValueError):
        json.loads(data)


def test_429_and_500(server):
    resp, data = post(server, 'stub-429')
    assert resp.status == 429 and resp.getheader('Retry-After') == '1'
    assert json.loads(data)['error']['type'] == 'rate_limit_error'
    resp, data = post(server, 'stub-500')
    assert resp.status == 500 and data == b'upstream exploded'


def test_flaky_alternates_503_and_ok(server):
    codes = [post(server, 'stub-flaky')[0].status for _ in range(4)]
    assert codes == [503, 200, 503, 200]


def test_huge_sizes_match_between_shapes(server):
    _, plain = post(server, 'stub-huge')
    content = json.loads(plain)['choices'][0]['message']['content']
    assert len(content) == len('lorem ipsum dolor sit amet ' * 40) * 600 == 648000
    _, sse = post(server, 'stub-huge', stream=True)
    events = [json.loads(e[6:]) for e in sse.decode().split('\n\n')
              if e.startswith('data: {')]
    streamed = ''.join(e['choices'][0]['delta'].get('content', '') for e in events)
    assert streamed == content


def test_toolstorm_always_calls_bash_with_fresh_ids(server):
    ids = []
    for _ in range(2):
        _, data = post(server, 'stub-toolstorm')
        choice = json.loads(data)['choices'][0]
        call = choice['message']['tool_calls'][0]
        assert choice['finish_reason'] == 'tool_calls'
        assert call['function']['name'] == 'bash' and 'index' not in call
        ids.append(call['id'])
    assert ids == ['call_1', 'call_2']


def test_toolstorm_streaming_keeps_index(server):
    _, data = post(server, 'stub-toolstorm', stream=True)
    assert b'"index": 0' in data and b'"finish_reason": "tool_calls"' in data


def test_badtool_has_unparseable_arguments(server):
    _, data = post(server, 'stub-badtool')
    args = json.loads(data)['choices'][0]['message']['tool_calls'][0]['function']['arguments']
    with pytest.raises(ValueError):
        json.loads(args)


def test_ghosttool_names_missing_tool(server):
    _, data = post(server, 'stub-ghosttool')
    assert json.loads(data)['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'no_such_tool'


def test_ansi_and_escape_payloads(server):
    text = json.loads(post(server, 'stub-ansi')[1])['choices'][0]['message']['content']
    assert '\x1b[2J' in text and '\x07' in text
    text = json.loads(post(server, 'stub-escape', raw=json.dumps({
        'model': 'stub-escape', 'messages': [{'role': 'user', 'content':
                                            'escape-marker: /scratch/unique-marker'}]}))[1])['choices'][0]['message']['content']
    assert '/etc/shadow' in text and '/scratch/unique-marker' in text


def test_request_log_records_bodies(server, monkeypatch, tmp_path):
    log = tmp_path / 'req.jsonl'
    monkeypatch.setattr(stresslane, 'LOG', str(log))
    post(server, 'stub-ok')
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec == {'model': 'stub-ok', 'body': {'model': 'stub-ok', 'stream': False}}


def test_unwritable_log_is_ignored(server, monkeypatch, tmp_path):
    monkeypatch.setattr(stresslane, 'LOG', str(tmp_path / 'missing' / 'x.jsonl'))
    assert post(server, 'stub-ok')[0].status == 200

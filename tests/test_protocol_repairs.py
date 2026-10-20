"""Release-candidate regressions: synthetic upstreams and no paid requests."""
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from onecli import config, lanehealth, update
from onecli.providers.base import ProviderError
from onecli.providers.openai_wire import OpenAIWire
from onecli.providers.anthropic_wire import AnthropicWire

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('action', ['import onecli; import onecli.tools',
                                     'from onecli.cli import main; main(["--help"])',
                                     'from onecli.cli import main; main(["--version"])'])
def test_network_free_entrypoints(action, tmp_path):
    code = '''import sys
calls = []
def audit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        calls.append(event)
        raise OSError('offline regression guard')
sys.addaudithook(audit)
try:
    exec(ACTION)
except SystemExit as exc:
    assert exc.code == 0
assert not calls, calls
'''.replace('ACTION', repr(action))
    env = {**os.environ, 'HOME': str(tmp_path), 'CI': '1', 'ONECLI_NO_UPDATE_CHECK': '1'}
    env.pop('ONECLI_RECENT_TOOL', None)
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unreadable_provider_skipped(tmp_path):
    path = tmp_path / 'bad.env'
    path.write_bytes(b'\xff')
    with pytest.warns(UserWarning, match='Skipping unreadable provider'):
        assert config._parse_env_file(path) == {}


@pytest.mark.parametrize('payload', [[], ['x'], {'data': ['x']}, {'data': 42}])
def test_malformed_discovery(monkeypatch, payload):
    monkeypatch.setattr(config, '_http_json', lambda *a: payload)
    assert config.llamacpp_backends() == []


@pytest.mark.parametrize('payload', [[], {'default_generation_settings': []}, {'default_generation_settings': 'x'}])
def test_malformed_context(monkeypatch, payload):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    assert config.served_context_window('http://localhost/v1') is None


@pytest.mark.parametrize('payload', [[], {'data': []}, {'data': 'x'}])
def test_malformed_key_probe(monkeypatch, payload):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    assert lanehealth._fetch_openrouter_key('https://example.com', 'fixture') is None


def test_malformed_health(monkeypatch):
    from onecli.serving.serve import health
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(b'[]'))
    assert not health(8080)


@pytest.mark.parametrize('payload', [[], {'info': []}, {'info': 'x'}])
@pytest.mark.parametrize('argv', [[], ['--check']])
def test_malformed_pypi_controlled(monkeypatch, payload, argv):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    assert update.main(argv) == 1


@pytest.mark.parametrize('wire,payload', [
    (OpenAIWire, {'choices': {'x': 1}}),
    (OpenAIWire, {'choices': [{'message': 'x'}]}),
    (OpenAIWire, {'choices': [{'message': {'tool_calls': ['x']}}]}),
    (AnthropicWire, {'content': ['x']}),
    (AnthropicWire, {'content': 4}),
])
def test_malformed_wire_is_recoverable(wire, payload):
    with pytest.raises(ProviderError):
        wire._to_turn(payload)


def _startup_process(home, barrier, queue):
    update.STATE_DIR = Path(home)
    update.CACHE = Path(home) / 'update-check.json'
    class Deferred:
        def __init__(self, **kw):
            pass
        def start(self):
            pass
    with patch.object(update, 'enabled', return_value=True), patch.object(update.threading, 'Thread', Deferred):
        barrier.wait(timeout=10)
        started = update.startup(now=1e6) is not None
    queue.put(started)


def test_update_reserves_across_processes(tmp_path):
    ctx = multiprocessing.get_context('fork')
    barrier, queue = ctx.Barrier(4), ctx.Queue()
    children = [ctx.Process(target=_startup_process, args=(str(tmp_path), barrier, queue)) for _ in range(4)]
    for child in children:
        child.start()
    answers = [queue.get(timeout=15) for _ in children]
    for child in children:
        child.join(15)
        assert child.exitcode == 0
    assert sum(answers) == 1


def test_update_reserves_before_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(update, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(update, 'CACHE', tmp_path / 'update-check.json')
    monkeypatch.setattr(update, 'enabled', lambda: True)
    workers = []
    class Deferred:
        def __init__(self, **kw):
            pass
        def start(self):
            workers.append(self)
    monkeypatch.setattr(update.threading, 'Thread', Deferred)
    update.startup(now=1e6)
    update.startup(now=1e6)
    assert len(workers) == 1







def test_no_global_missing_import_suppression():
    section = (ROOT / 'pyproject.toml').read_text().split('[tool.mypy]')[1].split('[tool.ruff]')[0]
    top, *overrides = section.split('[[tool.mypy.overrides]]')
    assert 'ignore_missing_imports' not in top
    for block in overrides:  # only named third-party modules may be exempted
        if 'ignore_missing_imports' in block:
            assert 'module = [' in block and 'onecli' not in block


def test_readme_clone_and_security_link():
    assert 'git clone https://github.com/one-framework-foundation/one-cli.git' in (ROOT / 'README.md').read_text()
    assert '(docs/permissions.md)' in (ROOT / 'SECURITY.md').read_text()


@pytest.mark.parametrize('name', ['SECURITY.md', 'docs/reproducible-releases.md'])
def test_network_inventory(name):
    text = (ROOT / name).read_text()
    for required in ['127.0.0.1:8080', 'ONE_LLAMACPP_HOSTS', '/key', '/props', '/tokenize',
                     '/apply-template', '900 seconds', '11434', '1234']:
        assert required in text


def test_hosted_sandbox_provisioning():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for required in ['apt-get install -y bubblewrap', 'kernel.apparmor_restrict_unprivileged_userns=0',
                     'command -v bwrap', 'command -v prlimit',
                     'bwrap --unshare-all --ro-bind / / --proc /proc -- /bin/true']:
        assert required in text
    assert text.index('apt-get install -y bubblewrap') < text.index('python -m pytest tests -q')


@pytest.mark.parametrize('payload', [[], {'tools': ['x']}])
def test_malformed_mcp_tools(monkeypatch, payload):
    from onecli import mcp
    from unittest.mock import Mock
    server = mcp.StdioServer(mcp.ServerSpec('fixture', 'fixture'))
    monkeypatch.setattr(mcp.subprocess, 'Popen', Mock())
    monkeypatch.setattr(server, '_request', lambda *a, **k: payload)
    monkeypatch.setattr(server, '_notify', lambda *a: None)
    monkeypatch.setattr(server, 'stop', lambda: None)
    assert not server.start()
    assert server.error


@pytest.mark.parametrize('wire,module,frames', [
    (OpenAIWire, 'openai_wire', [{'choices': ['x']}]),
    (AnthropicWire, 'anthropic_wire', [
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text'}},
        {'type': 'content_block_start', 'index': 'bad', 'content_block': {'type': 'text'}},
        {'type': 'message_stop'},
    ]),
])
def test_malformed_stream_is_recoverable(monkeypatch, wire, module, frames):
    import importlib
    mod = importlib.import_module('onecli.providers.' + module)
    monkeypatch.setattr(mod, 'sse_payloads', lambda *a, **k: iter(frames))
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(b''))
    provider = wire('https://example.com/v1', 'fixture')
    with pytest.raises(ProviderError):
        provider._stream({}, lambda *a: None)


def test_publisher_evidence_recorded():
    text = (ROOT / 'docs/reproducible-releases.md').read_text()
    assert 'Trusted Publishing' in text and '`pypi`' in text
    assert 'short-lived PyPI token' in text






@pytest.mark.parametrize('payload', [[], {'data': 42}, {'data': ['x']}, {'data': [{'id': 42}]}])
def test_onboarding_ignores_malformed_model_ids(payload):
    from onecli.onboard import _model_ids
    assert _model_ids(payload) == []


def test_anthropic_malformed_metadata_is_unknown(monkeypatch):
    provider = AnthropicWire('https://example.com/v1', 'fixture')
    monkeypatch.setattr(provider, '_get_json', lambda *a: {'data': [{'id': 'fixture', 'meta': []}]})
    maximum, _ = provider._listed_max_output('https://example.com/v1/models')
    assert maximum is None

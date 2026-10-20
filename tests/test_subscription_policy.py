"""No subscription policy is built in; an operator-supplied one says why it refuses."""
import json
import logging
import subprocess
import sys

import pytest

from onecli import config, fallback
from onecli._vendor import dropins

CLOUD = 'OPENAI_BASE_URL=https://ollama.com/v1\nONE_PROVIDER_MODEL=gpt-oss:120b\n'


class OnlyFlash:
    def refuses(self, name, base, model, declared=''):
        return '' if 'flash' in model else f'operator policy does not allow {model!r}'


@pytest.fixture
def no_policy(monkeypatch):
    monkeypatch.setattr(dropins, '_POLICY', None)


@pytest.fixture
def operator_policy(monkeypatch):
    monkeypatch.setattr(dropins, '_POLICY', OnlyFlash())


def test_shipped_default_reads_every_valid_dropin(tmp_path):
    # A fresh interpreter sees the reader exactly as installed: with no
    # extension registering a policy, nothing is refused for its model.
    p = tmp_path / 'cloud.env'
    p.write_text(CLOUD)
    code = ("from pathlib import Path\n"
            "from onecli._vendor import dropins\n"
            "from onecli import config\n"
            "if dropins._POLICY is not None:\n"
            "    print('registered'); raise SystemExit\n"
            f"lane = dropins.read({str(p)!r})\n"
            "assert lane and lane['model'] == 'gpt-oss:120b', lane\n"
            f"assert config._backend_from_env(Path({str(p)!r})).model == 'gpt-oss:120b'\n")
    run = subprocess.run([sys.executable, '-c', code], check=True,
                         capture_output=True, text=True)
    if run.stdout.strip() == 'registered':
        pytest.skip('an installed extension registers a subscription policy')


def test_without_policy_cloud_dropin_is_read(tmp_path, no_policy):
    p = tmp_path / 'cloud.env'
    p.write_text(CLOUD)
    assert dropins.read(p)['model'] == 'gpt-oss:120b'
    assert config._backend_from_env(p).model == 'gpt-oss:120b'


def test_without_policy_explicit_model_resolves(monkeypatch, no_policy):
    backend = config.Backend('cloud', 'gpt-oss:120b', 'https://ollama.com/v1')
    monkeypatch.setattr(config, 'discover', lambda **kw: [backend])
    assert config.resolve('cloud', model='gpt-oss:20b').model == 'gpt-oss:20b'


def test_without_policy_auto_order_is_discovery_order(no_policy):
    primary = config.Backend('primary', 'local', 'http://localhost:8080/v1')
    pro = config.Backend('alt-pro', 'model-pro', 'https://pro.example/v1')
    flash = config.Backend('alt-flash', 'model-flash', 'https://flash.example/v1')
    assert fallback.auto_backends(primary, [pro, flash]) == [pro, flash]


def test_operator_policy_refusal_names_the_dropin(tmp_path, caplog, operator_policy):
    p = tmp_path / 'cloud.env'
    p.write_text(CLOUD)
    with caplog.at_level(logging.WARNING, logger=dropins.__name__):
        assert dropins.read(p) is None
    assert 'cloud.env' in caplog.text and 'gpt-oss:120b' in caplog.text
    with pytest.warns(UserWarning, match=r'cloud\.env.*gpt-oss:120b'):
        assert config._backend_from_env(p) is None


def test_operator_policy_checks_params_model_override(tmp_path, operator_policy):
    p = tmp_path / 'renamed.env'
    p.write_text('OPENAI_BASE_URL=https://example.com/v1\nONE_PROVIDER_MODEL=a-flash\nONE_PROVIDER_PARAMS='
                 + json.dumps({'model': 'a-pro'}) + '\n')
    with pytest.warns(UserWarning, match='a-pro'):
        assert config._backend_from_env(p) is None


def test_operator_policy_explicit_override_names_the_backend(monkeypatch, operator_policy):
    backend = config.Backend('renamed', 'a-flash', 'https://example.com/v1')
    monkeypatch.setattr(config, 'discover', lambda **kw: [backend])
    with pytest.raises(RuntimeError, match="subscription policy refuses renamed: .*'a-pro'"):
        config.resolve('renamed', model='a-pro')


def test_generated_reader_provenance():
    import hashlib
    from pathlib import Path
    path = Path(dropins.__file__)
    provenance = json.loads(path.with_name('dropins-source.json').read_text())
    assert provenance['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance['path'] == 'dropins.py'

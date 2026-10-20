"""Release-candidate regressions for endpoint classification and stamp coverage."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from onecli import presence
from onecli._vendor import dropins


@pytest.mark.parametrize('host', ['localhost.evil.net', '10.example.com', '127.attacker.io', '192.168.example.com', '172.16.evil.net', '100.64.evil.net'])
def test_public_hostname_is_paid(host):
    assert dropins.is_paid(f'https://{host}/v1')


@pytest.mark.parametrize('host', ['localhost', '127.0.0.1',
    *('.'.join(map(str, octets)) for octets in [(10, 0, 0, 1), (172, 16, 0, 1), (192, 168, 0, 1)]),
    '100.64.0.1', '[::1]', '[fc00::1]', '[fe80::1]'])
def test_private_literal_remains_free(host):
    assert not dropins.is_paid(f'http://{host}/v1')


def test_presence_disable_never_loads_reader(monkeypatch):
    monkeypatch.setenv('ONECLI_NO_PRESENCE', '1')
    def forbidden():
        pytest.fail('disabled presence loaded probe reader')
    monkeypatch.setattr(presence, '_reader', forbidden)
    backend = SimpleNamespace(name='local', base_url='http://localhost:8080/v1', wire='openai', model='m', api_key='secret')
    assert presence.status(backend, fresh=True) == presence.UNCHECKED
    presence.prime([backend])
    assert presence.survey([backend]) == {'local': presence.UNCHECKED}
    assert presence.refusal(backend) == ''


@pytest.mark.parametrize('filename', ['SECURITY.md', 'docs/reproducible-releases.md'])
def test_presence_inventory_matches_wire(filename):
    text = (Path(__file__).parents[1] / filename).read_text()
    assert 'ONECLI_NO_PRESENCE=1' in text
    assert f'{dropins.PRESENCE_CACHE_TTL:g} seconds up' in text
    assert f'{dropins.PRESENCE_DOWN_TTL:g} seconds for connection/health failures' in text
    assert f'{dropins.PRESENCE_CACHE_TTL:g} seconds for unsuccessful OpenAI listings' in text
    assert 'Bearer' in text and 'x-api-key' in text


@pytest.mark.parametrize('filename', ['SECURITY.md', 'docs/reproducible-releases.md'])
def test_inventory_documents_fallback_discovery(filename):
    text = (Path(__file__).parents[1] / filename).read_text()
    assert 'unless `--fallback-model` or `ONE_CLI_FALLBACK` names a chain' in text



import json
from argparse import Namespace

import pytest

from onecli import run_limits
from onecli.agent import Agent
from onecli.cli import _system
from onecli.providers.base import Turn, ToolCall
from onecli.providers.base import ProviderError
from onecli.providers.openai_wire import OpenAIWire


class Provider:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return next(self.turns)


def limits(path, **overrides):
    value = dict(max_steps=3, max_output_tokens=1200,
                 max_request_bytes=24000, max_total_request_bytes=60000,
                 max_input_tokens=30000,
                 urls=['https://www.sqlite.org/wal.html'])
    value.update(overrides)
    (path / '.one-run-limits.json').write_text(json.dumps(value))


def test_oversized_history_never_reaches_provider(tmp_path):
    limits(tmp_path)
    provider = Provider([Turn(text='wrong')])
    result = Agent(provider=provider, workspace=str(tmp_path)).run('x' * 150000)
    assert provider.calls == 0
    assert result.stopped == 'budget'


def test_denied_shell_stops_without_execution_or_another_call(tmp_path, monkeypatch):
    limits(tmp_path)
    provider = Provider([Turn(tool_calls=[ToolCall(id='1', name='bash', arguments={'command': 'echo bad'})]), Turn(text='wrong')])
    monkeypatch.setattr('onecli.tools.run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('tool executed')))
    result = Agent(provider=provider, workspace=str(tmp_path)).run('research')
    # A workspace restriction is its own stop reason, not a step budget: the
    # run ended because the tool was not allowed, and saying "budget" sent the
    # reader looking for a limit that was never reached.
    assert result.stopped == 'restricted'
    assert provider.calls == 1


def test_guard_block_stops_before_shell_workaround(tmp_path, monkeypatch):
    limits(tmp_path)
    provider = Provider([Turn(tool_calls=[ToolCall(id='1', name='web_fetch', arguments={'url': 'https://www.sqlite.org/wal.html'})]), Turn(text='wrong')])
    monkeypatch.setattr('onecli.tools.run', lambda *a, **k: '[web guard: verdict=BLOCK signals=test]')
    result = Agent(provider=provider, workspace=str(tmp_path)).run('research')
    assert result.stopped == 'budget'
    assert provider.calls == 1


def test_total_request_budget_counts_repeated_input(tmp_path, monkeypatch):
    # This case measures byte accounting with a web_fetch schema present.
    # Whether an installation provides web tools must not decide the size.
    from onecli import tools
    monkeypatch.setattr(tools, 'REGISTRY', dict(tools.REGISTRY))
    tools.REGISTRY['web_fetch'] = tools.Tool(
        name='web_fetch', description='Fetch a URL and return its guarded text.',
        parameters={'type': 'object', 'properties': {'url': {'type': 'string'}},
                    'required': ['url']},
        fn=lambda workspace, url: '')
    limits(tmp_path, max_total_request_bytes=2500)
    turn = Turn(tool_calls=[ToolCall(id='1', name='web_fetch', arguments={'url': 'https://www.sqlite.org/wal.html'})])
    provider = Provider([turn, Turn(text='wrong')])
    monkeypatch.setattr('onecli.tools.run', lambda *a, **k: 'x' * 2000)
    result = Agent(provider=provider, workspace=str(tmp_path), system='short').run('research')
    assert result.stopped == 'budget'
    assert provider.calls == 1


def test_unconfigured_workspace_unchanged(tmp_path):
    provider = Provider([Turn(text='ok')])
    assert Agent(provider=provider, workspace=str(tmp_path)).run('hello').text == 'ok'


def test_http_retries_share_the_same_attempt_budget(tmp_path, monkeypatch):
    limits(tmp_path)
    wire = OpenAIWire('https://unused.invalid/v1', 'test')
    Agent(provider=wire, workspace=str(tmp_path))
    calls = []
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{}'
    def request(*args, **kwargs):
        calls.append(1)
        return Reply()
    monkeypatch.setattr('urllib.request.urlopen', request)
    for _ in range(3):
        wire._post({'messages': [], 'max_tokens': 100})
    with pytest.raises(ProviderError, match='wire budget'):
        wire._post({'messages': [], 'max_tokens': 100})
    assert len(calls) == 3
    assert wire.timeout == 45


def test_bounded_host_keeps_supplied_contract_without_ancestor_context(monkeypatch):
    args = Namespace(system='identity and safety contract', system_file='host.md',
                     workspace='/unused', no_project_context=False)
    monkeypatch.setattr(run_limits, 'load', lambda workspace: {'bounded': True})
    monkeypatch.setattr('onecli.permissions.project_context',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('ancestor context loaded')))
    assert _system(args, 'default') == 'identity and safety contract'


def test_unbounded_host_still_loads_project_context(monkeypatch):
    args = Namespace(system='base', system_file='host.md', workspace='/unused', no_project_context=False)
    monkeypatch.setattr(run_limits, 'load', lambda workspace: None)
    monkeypatch.setattr('onecli.permissions.project_context', lambda *a, **k: 'project rules')
    assert 'project rules' in _system(args, 'default')

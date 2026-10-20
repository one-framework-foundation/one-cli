import json
from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall
from onecli import tools
from onecli.tool_discovery import Discovery


def catalog(monkeypatch):
    for i in range(30):
        name = f'mcp__fixture__tool_{i}'
        monkeypatch.setitem(tools.REGISTRY, name, tools.Tool(
            name, 'fixture capability ' + 'description ' * 100,
            {'type': 'object', 'properties': {}}, lambda **kw: 'fixture result'))
    return tools.schemas(['read'] + [f'mcp__fixture__tool_{i}' for i in range(30)])


def test_large_catalog_deferred_after_assessment(monkeypatch, tmp_path):
    schemas = catalog(monkeypatch)
    class Provider:
        def __init__(self): self.requests = []
        def chat(self, messages, offered, **kw):
            self.requests.append(offered)
            assert len(json.dumps(offered)) < 32768, 'schema payload alone exceeds slot'
            n = len(self.requests)
            if n == 1:
                return Turn(
                    text=json.dumps({"decision": "proceed", "message": "Checking."})
                )
            if n == 2:
                assert 'tool_search' in [s['function']['name'] for s in offered]
                return Turn(tool_calls=[ToolCall('s1', 'tool_search', {'query': 'mcp__fixture__tool_17'})])
            if n == 3:
                assert 'mcp__fixture__tool_17' in [s['function']['name'] for s in offered]
                return Turn(tool_calls=[ToolCall('c1', 'mcp__fixture__tool_17', {})])
            return Turn(text='Complete')
    p = Provider()
    result = Agent(p, str(tmp_path), tool_names=[s['function']['name'] for s in schemas],
                   context_window=32768, response_gate='enforce').run('Inspect fixture')
    assert result.stopped == 'answered'
    assert any(m.get('content') == 'fixture result' for m in result.messages)


def test_small_catalog_unchanged(monkeypatch):
    schemas = catalog(monkeypatch)[:2]
    d = Discovery(schemas, 32768)
    assert not d.enabled
    assert d.schemas() == schemas


def test_selection_bounded_and_replaced(monkeypatch):
    d = Discovery(catalog(monkeypatch), 32768)
    d.search('fixture')
    # A broad query narrows to a readable handful rather than the old three.
    assert 4 <= len(d.selected) <= 8
    d.search('mcp__fixture__tool_29')
    assert d.selected[0]['function']['name'] == 'mcp__fixture__tool_29'
    d.search('nonexistent')
    assert not d.selected


def test_ranking_prefers_the_specific_tool_over_raw_matches():
    # 'common' is in every entry, so a match count puts the tool that repeats it
    # first; the rare term is what identifies the capability being asked for.
    schemas = [
        {'type': 'function', 'function': {'name': 'mcp__alpha__common_tool',
                                          'description': 'common common common'}},
        {'type': 'function', 'function': {'name': 'mcp__beta__rare_tool',
                                          'description': 'rare'}},
    ] + [
        {'type': 'function', 'function': {'name': f'mcp__alpha__bulk_{i}',
                                          'description': 'common padding'}}
        for i in range(20)
    ]
    d = Discovery(schemas, 32768)
    got = json.loads(d.search('common rare'))
    assert got['enabled'][0] == 'mcp__beta__rare_tool'


def test_discovery_cannot_find_excluded_tool(monkeypatch):
    schemas = catalog(monkeypatch)
    d = Discovery(schemas[:-1], 32768)
    assert 'mcp__fixture__tool_29' not in d.search('mcp__fixture__tool_29')


def test_discovered_execution_still_passes_guard(monkeypatch, tmp_path):
    from onecli.guard import Verdict
    schemas = catalog(monkeypatch)
    calls = []
    class Guard:
        def check(self, name, args, mutating):
            calls.append((name, mutating))
            return Verdict.deny('test denial', source='test')
    class Provider:
        def __init__(self): self.n = 0
        def chat(self, messages, offered, **kw):
            self.n += 1
            if self.n == 1:
                return Turn(tool_calls=[ToolCall('c1', 'mcp__fixture__tool_17', {})])
            return Turn(text='Blocked')
    result = Agent(Provider(), str(tmp_path), guard=Guard(),
                   context_window=32768,
                   tool_names=[s['function']['name'] for s in schemas]).run('Inspect')
    assert calls == [('mcp__fixture__tool_17', False)]
    assert any(m.get('content') == 'blocked: test denial' for m in result.messages)

import json
import pytest
from onecli import cli, chat as chatlib, masks, session
from onecli.agent import Agent
from onecli.masks.loop import parse
from onecli.loopguard import LoopGuard
from onecli.providers.base import Turn, ToolCall
from test_response_gate import Provider
from test_commands import _repl, _say
from test_chat import FakeWire
from test_chat import chat  # noqa: F401 - shared fixture, see per-file-ignores


@pytest.fixture(autouse=True)
def no_external_masks(monkeypatch):
    monkeypatch.delenv('ONE_MASKS_JSON', raising=False)


@pytest.mark.parametrize('raw', [[], {'progress_policy': None}, {'version': True}, {'version': 2}, {'max_steps': True},
    {'max_steps': 1}, {'repeat_limit': 0}, {'tool_deadline': float('inf')},
    {'tool_deadline': -1}, {'empty_retries': 1001}, {'empty_retries': True},
    {'model': 'remote'}, {'progress_policy': {'seconds': -1}}])
def test_invalid_loop_fails_closed(raw):
    with pytest.raises(ValueError):
        parse(raw)


def test_mask_cannot_raise_host_limits_or_mutate_host_guard(tmp_path):
    guard = LoopGuard(repeat_limit=2, deadline=5)
    a = Agent(Provider(), str(tmp_path), max_steps=4, tool_names=['read'], loop_guard=guard,
        loop_policy={'max_steps': 90, 'repeat_limit': 9, 'tool_deadline': 99}, mask_name='custom')
    assert (a.max_steps, a.loop.repeat_limit, a.loop.deadline) == (4, 2, 5)
    b = Agent(Provider(), str(tmp_path), loop_guard=guard,
        loop_policy={'repeat_limit': 1, 'tool_deadline': 1})
    assert (guard.repeat_limit, guard.deadline) == (2, 5)
    assert (b.loop.repeat_limit, b.loop.deadline) == (1, 1)
    assert a._gate(ToolCall('x', 'bash', {'command': 'touch forbidden'})) is not None


def test_empty_recovery_can_be_disabled_without_false_retry_claim(tmp_path):
    wire = Provider(Turn(), Turn(text='should never be reached'))
    result = Agent(wire, str(tmp_path), loop_policy={'empty_retries': 0}).run('task')
    assert result.stopped == 'empty' and len(wire.requests) == 1
    assert 'disabled by mask' in result.error and 'again after a retry' not in result.error


def test_mask_switch_is_request_local_and_preserves_conversation(tmp_path):
    (tmp_path / 'a').write_text('evidence')
    wire = Provider(Turn(tool_calls=[ToolCall('r','read',{'path':'a'})]), Turn(text='reviewed'))
    first = Agent(wire, str(tmp_path), **masks.options('review')).run('review a')
    assert 'Active mask: review' in str(wire.requests[0][0])
    assert 'Active mask:' not in str(first.messages)
    for name in ('build', 'off'):
        provider = Provider(Turn(text='next'))
        result = Agent(provider, str(tmp_path), **masks.options(name)).run('continue', first.messages)
        request = str(provider.requests[0][0])
        assert 'Active mask: review' not in request
        assert 'reviewed' in request
        assert ('Active mask: build' in request) == (name == 'build')
        assert 'Active mask:' not in str(result.messages)


def test_configured_mask_loads_both_loop_and_execution(tmp_path, monkeypatch):
    path = tmp_path / 'masks.json'
    path.write_text(json.dumps({'audit': {'preamble': 'Find issues', 'execution_policy': {'kind':'review'},
        'loop_policy': {'max_steps': 6, 'progress_policy': {'seconds': 12}}}}))
    monkeypatch.setenv('ONE_MASKS_JSON', str(path))
    mask = masks.select('audit')
    a = Agent(Provider(), str(tmp_path), **mask.agent_options(enforce=True))
    assert a.max_steps == 6 and a.progress_policy['seconds'] == 12
    assert set(a.tool_names) == {'read','ls','glob','grep'}
    assert mask.agent_options() == {}


def test_cli_explicit_cadence_overrides_mask_defaults():
    args = cli.build_parser().parse_args(['--mask','review'])
    assert cli._progress_policy(args)['seconds'] == 45
    args.progress_seconds = 9
    args.no_progress = True
    assert cli._progress_policy(args)['seconds'] == 9
    assert cli._progress_policy(args)['enabled'] is False


@pytest.mark.parametrize('entry', [cli, chatlib])
def test_listing_and_unknown_mask_never_contact_provider(entry, monkeypatch, capsys):
    monkeypatch.setattr(entry.config, 'resolve', lambda **kw: pytest.fail('provider discovery'))
    assert entry.main(['--masks']) == 0
    assert 'review' in capsys.readouterr().out
    assert entry.main(['--mask','nonexistent','hello']) == 2
    assert 'unknown mask' in capsys.readouterr().err


def test_cli_switch_retains_history_and_does_not_change_model(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'SESSION_DIR', tmp_path / 'sessions')
    repl = _repl(messages=[{'role':'user','content':'prior'}])
    backend = repl.backend
    assert 'review' in _say('/mask review', repl)
    assert repl.args.mask == 'review' and repl.messages[0]['content'] == 'prior'
    assert repl.backend is backend and not repl.rebuild
    _say('/mask nonsense', repl)
    assert repl.args.mask == 'review'
    assert session.metadata(repl.sid)['mask'] == 'review'
    _say('/mask off', repl)
    assert repl.args.mask == 'off'


def test_chat_mask_enforces_review_and_switches_off(chat, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'proof').write_text('source')
    chat._wire = FakeWire(turns=[Turn(tool_calls=[ToolCall('bad','write',{'path':'bad','content':'oops'})]),
        Turn(tool_calls=[ToolCall('r','read',{'path':'proof'})]), Turn(text='reviewed')])
    chatlib.dispatch(chat, '/mask review')
    chatlib.turn(chat, 'review proof', echo=False)
    assert not (tmp_path / 'bad').exists()
    assert 'Active mask: review' in str(chat._wire.calls[0]['messages'])
    assert 'Active mask:' not in str(chat.messages)
    chatlib.dispatch(chat, '/mask off')
    chat._wire = FakeWire(replies=['next'])
    chatlib.turn(chat, 'continue', echo=False)
    assert 'reviewed' in str(chat._wire.calls[0]['messages'])
    assert 'Active mask:' not in str(chat._wire.calls[0]['messages'])


def test_mask_cannot_enable_tools_when_chat_tools_off(chat, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chat.tools = 'off'
    chat.mask = 'general'
    chat._wire = FakeWire(turns=[Turn(tool_calls=[ToolCall('bad','bash',{'command':'touch bad'})]), Turn(text='blocked')])
    chatlib.turn(chat, 'answer only', echo=False)
    assert chat._wire.calls[0]['tools'] == []
    assert not (tmp_path / 'bad').exists()


def test_resume_restores_mask_and_explicit_off_wins(chat):
    session.append('saved', {'kind':'metadata','mask':'review'})
    session.append('saved', {'kind':'messages','messages':[{'role':'user','content':'old task'}]})
    chatlib._resume(chat, 'saved')
    assert chat.mask == 'review'
    chat.mask = 'off'
    chatlib._resume(chat, 'saved')
    assert chat.mask == 'off'


def test_preamble_only_api_mask_is_request_local(tmp_path):
    wire = Provider(Turn(text='done'))
    result = Agent(wire, str(tmp_path), mask_preamble='custom instructions').run('task')
    assert 'Active mask: custom' in str(wire.requests[0][0])
    assert 'custom instructions' not in str(result.messages)


@pytest.mark.parametrize('name,first_silence', [('build', 32), ('review', 48)])
def test_mask_cadence_over_long_tool_sequence(tmp_path, monkeypatch, name, first_silence):
    from onecli import progress_policy as progress
    clock = [0.0]
    monkeypatch.setattr('onecli.agent.time.monotonic', lambda: clock[0])
    class SlowWire(Provider):
        def chat(self, messages, tools, **kwargs):
            clock[0] += 8
            return super().chat(messages, tools, **kwargs)
    calls = []
    for i in range(8):
        (tmp_path / str(i)).write_text('evidence')
        calls.append(Turn(tool_calls=[ToolCall(str(i), 'read', {'path': str(i)})]))
    wire = SlowWire(*calls, Turn(text='verified'))
    result = Agent(wire, str(tmp_path), **masks.options(name)).run('inspect all evidence')
    reminders = [m['content'] for req,_,_ in wire.requests for m in req
        if str(m.get('content','')).startswith(progress.UPDATE)]
    assert result.text == 'verified' and result.tool_calls == 8
    assert len(wire.requests) == 9
    assert f'seconds_since_public={first_silence}' in reminders[0]
    assert all(not str(m.get('content','')).startswith(progress.UPDATE) for m in result.messages)

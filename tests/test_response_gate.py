import io
import json

import pytest

from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall
from onecli import events, cli, orc, subagent


class Provider:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    def chat(self, messages, tools, **kwargs):
        self.requests.append((list(messages), tools, kwargs))
        return self.turns.pop(0)


def assessment(decision='proceed', message='I will inspect the file.'):
    return Turn(text=json.dumps(dict(decision=decision, message=message)))


def write_call():
    return Turn(tool_calls=[ToolCall('write1', 'write', {'path': 'proof.txt', 'content': 'verified'})])


def test_gate_emits_before_any_tool_and_hides_control_json(tmp_path):
    provider = Provider(assessment(), write_call(), Turn(text='Done'))
    observed = []
    def observe(kind, data):
        observed.append(kind)
        if kind == 'acknowledgment':
            assert not (tmp_path / 'proof.txt').exists()
            assert data['text'] == 'I will inspect the file.'
    result = Agent(provider, str(tmp_path), response_gate='enforce',
                   stream=True, on_event=observe).run('write the proof')
    assert result.stopped == 'answered'
    assert (tmp_path / 'proof.txt').read_text() == 'verified'
    assert observed.index('acknowledgment') < observed.index('tool_start')
    assert provider.requests[0][1] == []
    assert provider.requests[1][1]
    assert 'on_delta' not in provider.requests[0][2]
    assert 'on_delta' in provider.requests[1][2]
    assert not any('"decision"' in m.get('content', '') for m in result.messages if m['role'] == 'assistant')


@pytest.mark.parametrize('decision', ['clarify', 'decline', 'answer'])
def test_terminal_decisions_never_call_tools_or_stop_hooks(tmp_path, decision):
    provider = Provider(assessment(decision, 'A public reply'), write_call())
    agent = Agent(provider, str(tmp_path), response_gate='enforce')
    agent._stop_refusal = lambda: pytest.fail('terminal assessment must not run stop hooks')
    result = agent.run('request')
    assert result.text == 'A public reply'
    assert len(provider.requests) == 1
    assert not (tmp_path / 'proof.txt').exists()


def test_rogue_tools_fail_closed_with_bounded_retry(tmp_path):
    provider = Provider(write_call(), write_call(), Turn(text='should never run'))
    result = Agent(provider, str(tmp_path), response_gate='enforce').run('request')
    assert result.stopped == 'error'
    assert len(provider.requests) == 2
    assert not (tmp_path / 'proof.txt').exists()
    assert not any(m.get('tool_calls') for m in result.messages)


def test_invalid_assessment_recovers_without_orphan_tool_history(tmp_path):
    provider = Provider(write_call(), assessment(), write_call(), Turn(text='Done'))
    result = Agent(provider, str(tmp_path), response_gate='enforce').run('request')
    assert result.stopped == 'answered'
    assert len(result.steps) == 4
    assert (tmp_path / 'proof.txt').exists()


def test_gate_resets_when_resumed(tmp_path):
    provider = Provider(assessment('answer', 'Hi'), write_call(), write_call())
    agent = Agent(provider, str(tmp_path), response_gate='enforce')
    first = agent.run('hi')
    second = agent.run('now write', first.messages)
    assert second.stopped == 'error'
    assert not (tmp_path / 'proof.txt').exists()


def test_assessment_spends_existing_step_budget(tmp_path):
    provider = Provider(assessment(), write_call())
    result = Agent(provider, str(tmp_path), response_gate='enforce', max_steps=1).run('write')
    assert result.stopped == 'budget'
    assert len(provider.requests) == 1
    assert not (tmp_path / 'proof.txt').exists()


def test_cli_default_and_orc_override(tmp_path):
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps({'response_gate': 'prompt'}))
    assert cli._response_gate(cli.build_parser().parse_args([])) == 'enforce'
    args = cli.build_parser().parse_args(['--orc', str(path)])
    assert cli._response_gate(args) == 'prompt'
    args = cli.build_parser().parse_args(['--orc', str(path), '--response-gate', 'off'])
    assert cli._response_gate(args) == 'off'


def test_child_policy_inherits_and_can_be_configured(tmp_path):
    spawner = subagent.Spawner(parent=None)
    Agent(Provider(), str(tmp_path), response_gate='enforce', spawner=spawner)
    assert spawner.build(subagent.GENERAL).response_gate == 'enforce'
    preset = subagent._preset_from('quiet', {'response_gate': 'off'})
    assert spawner.build(preset).response_gate == 'off'
    vendor = subagent.Preset(name='vendor', driver='codex')
    assert 'no first-response enforcement seam' in spawner._drive(vendor, 'request')


def test_acknowledgment_reaches_existing_public_event_channel():
    output = io.StringIO()
    emit = events.emitter(output, include_text=True)
    emit('acknowledgment', {'text': 'Checking the file.'})
    assert json.loads(output.getvalue()) == {'e': 'acknowledgment', 'step': 0, 'text': 'Checking the file.'}
    hidden = io.StringIO()
    events.emitter(hidden)('acknowledgment', {'text': 'Checking the file.'})
    assert hidden.getvalue() == ''


def test_resolved_graph_reports_each_node_policy(tmp_path):
    spec = orc.Orc(response_gate='enforce')
    presets = {'review': subagent.Preset(name='review', driver='codex', response_gate='prompt')}
    graph = orc.resolve(presets, spec, 'local', 'flag', '', 2, 30).as_dict()
    assert graph['orchestrator']['response_gate'] == 'enforce'
    assert graph['workers'][0]['response_gate'] == 'prompt'


@pytest.mark.parametrize('text', ['{}', '{"decision":"proceed","message":""}',
    '{"decision":"proceed","message":"ok","decision":"answer"}',
    '{"decision":"proceed","message":"ok","unexpected":true}'])
def test_malformed_control_never_unlocks_tools(tmp_path, text):
    result = Agent(Provider(Turn(text=text), Turn(text=text)), str(tmp_path),
                   response_gate='enforce').run('request')
    assert result.stopped == 'error'
    assert result.tool_calls == 0


def test_cli_end_to_end_keeps_json_stdout_and_public_ack(monkeypatch, tmp_path, capsys):
    from test_onecli_agent import _run_cli
    provider = Provider(assessment(), write_call(), Turn(text='Done'))
    code = _run_cli(monkeypatch, tmp_path,
                    ['--response-gate', 'enforce', '--json', '--text-events'], provider)
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)['text'] == 'Done'
    records = [json.loads(line) for line in captured.err.splitlines() if line.startswith('{')]
    kinds = [e['e'] for e in records]
    assert kinds.index('acknowledgment') < kinds.index('tool_start')
    assert '"message"' not in captured.err


def test_assessment_transport_overrides_prose_format_only_for_gate(tmp_path):
    provider = Provider(assessment(), Turn(text='Done'))
    system = 'Write conversational plain text for the user.'
    agent = Agent(provider, str(tmp_path), response_gate='enforce', system=system)
    result = agent.run('inspect')
    first_system = provider.requests[0][0][0]['content']
    assert first_system == provider.requests[1][0][0]['content']
    assert 'JSON transport envelope' in provider.requests[0][0][-1]['content']
    assert provider.requests[1][0][0]['content'] == agent.system
    assert result.messages[0]['content'] == agent.system


def test_assessment_does_not_offer_executable_tools(tmp_path):
    class ToolSelectingProvider(Provider):
        def chat(self, messages, tools, **kwargs):
            if kwargs.get('json_output') or 'JSON transport envelope' in messages[-1].get('content', ''):
                self.requests.append((list(messages), tools, kwargs))
                return write_call() if tools else assessment()
            return super().chat(messages, tools, **kwargs)
    provider = ToolSelectingProvider(write_call(), Turn(text='Done'))
    result = Agent(provider, str(tmp_path), response_gate='enforce').run('write proof')
    assert result.stopped == 'answered'
    assert provider.requests[0][1] == []
    assert provider.requests[1][1]
    assert (tmp_path / 'proof.txt').read_text() == 'verified'


def test_last_call_assessment_is_told_the_loop_cannot_continue(tmp_path):
    """A gate on the run's only model call must not promise a tool loop.

    The assessment costs a step, so `--max-steps 1` (a one-shot job) has no turn left after it. ASSESS tells the model the loop
    continues automatically; on the last call that is false, the model answers
    "proceed", and the run stops as `budget` with the acknowledgment as its
    only output.
    """
    provider = Provider(assessment('answer', 'note.txt holds the release.'))
    result = Agent(provider, str(tmp_path), response_gate='enforce',
                   max_steps=1).run('read note.txt')
    instruction = provider.requests[0][0][-1]['content']
    assert 'last model call' in instruction
    assert 'proceed" is unavailable' in instruction
    assert result.stopped == 'answered'
    assert result.text == 'note.txt holds the release.'


def test_room_left_after_the_assessment_keeps_the_normal_instruction(tmp_path):
    provider = Provider(assessment(), write_call(), Turn(text='Done'))
    result = Agent(provider, str(tmp_path), response_gate='enforce',
                   max_steps=3).run('write proof')
    assert 'last model call' not in provider.requests[0][0][-1]['content']
    assert result.stopped == 'answered'


def test_gate_proceed_on_the_only_call_names_the_gate(tmp_path):
    provider = Provider(assessment(), write_call())
    result = Agent(provider, str(tmp_path), response_gate='enforce',
                   max_steps=1).run('write proof')
    assert result.stopped == 'budget'
    assert len(provider.requests) == 1
    assert 'first-response gate' in result.error
    assert not (tmp_path / 'proof.txt').exists()

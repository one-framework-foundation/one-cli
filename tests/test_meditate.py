from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall

class Script:
    def __init__(self, *turns):
        self.turns = iter(turns)
        self.seen = []
    def chat(self, messages, tools=None, **kwargs):
        self.seen.append(messages.copy())
        return next(self.turns)

def test_review_once_with_new_evidence(tmp_path):
    (tmp_path / 'proof.txt').write_text('actual value is 2')
    p = Script(Turn(text='The value is 1.'),
               Turn(tool_calls=[ToolCall(id='r', name='read', arguments={'path': 'proof.txt'})]),
               Turn(text='The value is 2.'))
    result = Agent(p, str(tmp_path), meditate=True, max_steps=4).run('Check the value')
    assert result.text == 'The value is 2.'
    assert len(result.steps) == 3
    assert 'disjoint' in p.seen[1][-1]['content']
    assert 'actual value is 2' in str(p.seen[2])

def test_review_cannot_write(tmp_path):
    p = Script(Turn(text='The config is correct.'),
               Turn(tool_calls=[ToolCall(id='w', name='write', arguments={'path': 'bad', 'content': 'oops'})]),
               Turn(text='I cannot confirm it.'))
    result = Agent(p, str(tmp_path), meditate=True, max_steps=4).run('Check config')
    assert not (tmp_path / 'bad').exists()
    assert result.denied

def test_off_and_budget_do_not_add_requests(tmp_path):
    for enabled, budget in [(False, 4), (True, 1), (True, 2)]:
        p = Script(Turn(text='The value is 1.'))
        result = Agent(p, str(tmp_path), meditate=enabled, max_steps=budget).run('Check')
        assert len(result.steps) == 1

def test_cli_flag():
    from onecli.cli import build_parser
    assert build_parser().parse_args(['--meditate']).meditate is True
    assert build_parser().parse_args([]).meditate is False

def test_stream_does_not_publish_unreviewed_draft(tmp_path):
    class Streaming(Script):
        def chat(self, messages, tools=None, on_delta=None, **kwargs):
            turn = super().chat(messages, tools, **kwargs)
            if on_delta:
                on_delta('text', turn.text)
            return turn
    events = []
    result = Agent(Streaming(Turn(text='wrong draft'), Turn(text='corrected')),
                   str(tmp_path), meditate=True, stream=True, max_steps=4,
                   on_event=lambda kind,data: events.append((kind,data))).run('Check')
    assert result.text == 'corrected'
    assert not any(kind == 'text_delta' for kind, _ in events)

def test_clarification_is_not_reviewed(tmp_path):
    p = Script(Turn(text='{"decision":"clarify","message":"Which file?"}'))
    result = Agent(p,str(tmp_path),meditate=True,response_gate='enforce').run('Fix it')
    assert len(result.steps) == 1
    assert result.text == 'Which file?'

def test_review_event_is_machine_readable():
    import io
    import json
    from onecli.events import emitter
    output = io.StringIO()
    emit = emitter(stream=output)
    emit('meditate', {'status':'skipped_budget', 'step':2})
    assert json.loads(output.getvalue()) == {'e':'meditate','status':'skipped_budget','step':2}

import io
import json
from onecli import events,cli

def test_public_text_requires_explicit_opt_in():
    out=io.StringIO()
    events.emitter(out)('text_delta',{'kind':'text','chunk':'public answer'})
    assert 'public answer' not in out.getvalue()

def test_only_public_channel_is_emitted_and_lines_are_bounded():
    out=io.StringIO()
    emit=events.emitter(out,step=lambda:2,include_text=True)
    emit('text_delta',{'kind':'thinking','chunk':'PRIVATE_REASONING'})
    emit('text_delta',{'kind':'tool','chunk':'PRIVATE_ARGUMENT'})
    text='A😀'*5000
    emit('text_delta',{'kind':'text','chunk':text})
    lines=out.getvalue().splitlines()
    assert max(map(len,lines)) < 65536
    assert 'PRIVATE' not in out.getvalue()
    data=[json.loads(line) for line in lines]
    assert ''.join(e['text'] for e in data if e['e']=='text_delta')==text

def test_cli_enables_events_only_when_requested(monkeypatch):
    out=io.StringIO()
    monkeypatch.setattr(cli.sys,'stderr',out)
    args=cli.build_parser().parse_args(['--json','--stream','--text-events','hi'])
    emit=cli._run_emitter(args,{'step':1})
    emit('text_delta',{'kind':'text','chunk':'hello'})
    assert 'hello' in out.getvalue()

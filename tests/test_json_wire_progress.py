import io
import json

from onecli import cli, events


def test_real_arrivals_emit_throttled_progress_without_content(monkeypatch):
    out = io.StringIO()
    emit = events.emitter(out, step=lambda: 2)
    for _ in range(500):
        emit('text_delta', {'kind': 'thinking', 'chunk': 'private reasoning'})
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    assert 1 <= len(rows) <= 2
    assert rows[0]['e'] == 'progress'
    assert rows[0]['step'] == 2
    assert 'private reasoning' not in out.getvalue()


def test_json_progress_never_prints_answer_or_reasoning(capsys):
    state = {'stream': False}
    tracer = cli._tracer(True, state, emit=events.emitter(io.StringIO()))
    tracer('text_delta', {'kind':'text', 'chunk':'answer'})
    tracer('text_delta', {'kind':'thinking', 'chunk':'reasoning'})
    assert capsys.readouterr().out == ''

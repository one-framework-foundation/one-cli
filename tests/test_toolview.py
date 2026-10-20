import json
from onecli import terminal
from onecli import toolview
from onecli.terminal import cells
from onecli.toolview import describe, result


def test_json_summary_and_lossless_expansion():
    raw = json.dumps({'items': [{'assessment': {'goal': 'example'}}], 'error': 'permission denied'})
    assert result(raw) == ['│ items: 1 items', '│ error: permission denied', '│ Ctrl-O: full tool output']
    assert json.loads('\n'.join(line[2:] for line in result(raw, True))) == json.loads(raw)
    ui = terminal.Surface()
    ui.tool_result(raw)
    collapsed = str(ui.transcript_rows(60))
    assert 'assessment' not in collapsed and 'permission denied' in collapsed
    ui.key('\x0f')
    assert 'assessment' in str(ui.transcript_rows(60))
    ui.key('\x0f')
    assert str(ui.transcript_rows(60)) == collapsed


def test_escape_interrupts_without_destroying_draft(monkeypatch):
    calls = []
    monkeypatch.setattr(terminal.os, 'kill', lambda *args: calls.append(args))
    ui = terminal.Surface()
    ui.draft = 'keep me'
    ui.busy = ui.allow_steer = True
    ui.feed('\x1b')
    ui.flush_escape(ui.escape_started + .1)
    assert len(calls) == 1 and ui.draft == 'keep me'
    ui.busy = False
    ui.feed('\x1b')
    ui.flush_escape(ui.escape_started + .1)
    assert len(calls) == 1 and ui.draft == 'keep me'


def test_escape_between_turns_never_signals_the_process(monkeypatch):
    # busy is True from submit until the next read(), but SIGINT is only
    # survivable while the agent's try/except is armed (allow_steer).
    calls = []
    monkeypatch.setattr(terminal.os, 'kill', lambda *args: calls.append(args))
    ui = terminal.Surface()
    ui.busy = True
    for key in ('\x1b', '\x03'):
        ui.key(key)
    assert calls == []
    ui.allow_steer = True
    ui.key('\x1b')
    assert len(calls) == 1


def test_paste_does_not_interrupt(monkeypatch):
    monkeypatch.setattr(terminal.os, 'kill', lambda *a: (_ for _ in ()).throw(AssertionError('interrupt')))
    ui = terminal.Surface()
    ui.busy = True
    for c in '\x1b[200~hello\nworld\x1b[201~':
        ui.feed(c)
    ui.flush_escape(ui.escape_started + 1)
    assert ui.draft == 'hello\nworld'


def test_scrolling_stays_anchored_as_output_arrives():
    ui = terminal.Surface()
    ui.write('first\nsecond\nthird\n')
    ui.transcript_rows(60)
    ui.scroll = 1
    ui.write('fourth\n')
    ui.transcript_rows(60)
    assert ui.scroll == 2
    ui.key('\x07')
    assert ui.scroll == 0


def test_large_json_preview_is_bounded_and_keeps_late_error():
    value = {f'field{i}': 'value' for i in range(100)}
    value['error'] = 'permission denied'
    raw = json.dumps(value)
    rows = result(raw)
    assert len(rows) <= 5
    assert 'permission denied' in '\n'.join(rows)
    assert json.loads('\n'.join(row[2:] for row in result(raw, True))) == value


def test_multiline_json_preview_cannot_flood_terminal():
    raw = json.dumps({'stdout': '\n'.join(f'line {i}' for i in range(100)),
                      'error': 'failed to load'})
    rows = result(raw)
    assert len('\n'.join(rows).splitlines()) <= 5
    assert 'failed to load' in '\n'.join(rows)


def test_plain_preview_keeps_second_line_without_fake_controls():
    from onecli.toolview import plain_result
    assert 'hello' in plain_result('exit 0\nhello')
    text = plain_result('\n'.join(str(i) for i in range(30)))
    assert 'Ctrl-O' not in text and 'shortened' in text
    assert len(text.splitlines()) <= 5


def test_tool_labels_describe_activity():
    assert describe('bash', {'command': 'pwd'}) == 'Run · pwd'
    assert describe('grep', {'pattern': 'test'}) == 'Search · test'


def test_clipped_error_keeps_the_filename_not_the_directory():
    out = 'error: no such file: /home/user/projects/very/long/directory/path/work/data.csv'
    rows = toolview.result_rows(out, 40)
    shown = '\n'.join(rows)
    assert 'data.csv' in shown and '…' in shown
    assert all(cells(row) <= 40 for row in rows)
    plain = toolview.result_rows('x' * 100, 40)
    assert plain[0].endswith('…')  # rows without a path clip as before

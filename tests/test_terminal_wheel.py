"""Mouse wheel and copy under tmux (one-cli: wheel does not scroll).

On the alternate screen tmux (mouse on) and terminals with alternate scroll
turn a wheel notch into several Up/Down keys in one write. A burst must scroll
the transcript; a lone arrow must keep its history meaning.
"""
import io

from onecli import terminal
from onecli.terminal import Surface
from test_terminal_commands import surface


import pytest


@pytest.fixture(autouse=True)
def outside_tmux(monkeypatch):
    # The suite may itself run in tmux; a copy must never reach its buffers.
    monkeypatch.delenv('TMUX', raising=False)


def feed(ui, text):
    for c in text:
        ui.feed(c)
    ui.settle_arrows()


def test_wheel_burst_of_up_keys_scrolls_the_transcript_back():
    ui = Surface()
    ui.history = ['older']
    ui.history_index = 1
    feed(ui, '\x1b[A\x1b[A\x1b[A')
    assert ui.scroll == 3
    assert ui.draft == ''


def test_wheel_burst_of_down_keys_scrolls_forward_and_stops_at_live():
    ui = Surface()
    ui.scroll = 2
    feed(ui, '\x1b[B\x1b[B\x1b[B')
    assert ui.scroll == 0


def test_single_arrow_still_recalls_history():
    ui = Surface()
    ui.history = ['older']
    ui.history_index = 1
    feed(ui, '\x1b[A')
    assert ui.draft == 'older'
    assert ui.scroll == 0


def test_burst_inside_the_slash_menu_still_moves_the_highlight(tmp_path):
    ui = surface(tmp_path)
    feed(ui, '/')
    assert ui.command_menu()
    feed(ui, '\x1b[B\x1b[B')
    assert ui.menu_index == 2
    assert ui.scroll == 0


def test_sgr_wheel_reports_scroll_and_other_reports_do_not_leak():
    ui = Surface()
    feed(ui, '\x1b[<64;10;10M')
    assert ui.scroll == terminal.WHEEL_LINES
    feed(ui, '\x1b[<65;10;10M')
    assert ui.scroll == 0
    feed(ui, '\x1b[<0;10;10M\x1b[<0;10;10m')
    assert (ui.scroll, ui.draft) == (0, '')


def test_session_asks_for_alternate_scroll_and_mouse_reports_only_in_tmux_mouse():
    import inspect
    source = inspect.getsource(terminal.session)
    assert '?1007h' in source and '?1007l' in source
    assert 'mousemask' not in source and '?1000h' not in source
    assert "if mouse else ''" in source


def test_tmux_mouse_is_read_from_tmux_only_inside_tmux(monkeypatch):
    class Run:
        stdout = 'on\n'
    calls = []
    monkeypatch.setattr(terminal.subprocess, 'run', lambda *a, **k: calls.append(a) or Run())
    monkeypatch.delenv('TMUX', raising=False)
    assert terminal.tmux_mouse() is False and not calls
    monkeypatch.setenv('TMUX', '/tmp/tmux-1/default,1,0')
    assert terminal.tmux_mouse() is True
    Run.stdout = 'off\n'
    assert terminal.tmux_mouse() is False


def test_drag_over_the_transcript_copies_the_selected_rows():
    ui = Surface()
    ui.raw_out = io.StringIO()
    ui.panel_rows = {5: 'first row text', 6: 'second row', 7: 'third row here'}
    feed(ui, '\x1b[<0;7;6M\x1b[<32;4;8M\x1b[<0;4;8m')
    import base64
    sent = ui.raw_out.getvalue()
    assert sent.startswith('\x1b]52;c;')
    assert base64.b64decode(sent[7:-1]).decode() == 'row text\nsecond row\nthir'


def test_click_without_drag_copies_nothing():
    ui = Surface()
    ui.raw_out = io.StringIO()
    ui.panel_rows = {5: 'abc'}
    feed(ui, '\x1b[<0;2;6M\x1b[<0;2;6m')
    assert ui.raw_out.getvalue() == '' and ui.select is None


def test_clipboard_writes_osc52_to_the_terminal():
    ui = Surface()
    ui.raw_out = io.StringIO()
    assert ui.clipboard('hello') is True
    assert ui.raw_out.getvalue() == '\x1b]52;c;aGVsbG8=\x07'


def test_clipboard_declines_without_terminal_or_when_too_big():
    ui = Surface()
    assert ui.clipboard('x') is False
    ui.raw_out = io.StringIO()
    assert ui.clipboard('x' * 80_000) is False
    assert ui.raw_out.getvalue() == ''


def test_renderer_chgat_marks_only_the_selected_cells():
    import curses
    from onecli.terminal import RowRenderer
    renderer = RowRenderer(object())
    renderer.addstr(3, 0, 'abcdef', 0)
    renderer.chgat(3, 2, 2, curses.A_REVERSE)
    assert renderer.rows[3] == [(0, 'ab', 0), (2, 'cd', curses.A_REVERSE), (4, 'ef', 0)]


def test_selection_is_drawn_and_copied_from_the_rows_on_screen(tmp_path, monkeypatch):
    from test_terminal_commands import drawn
    ui = surface(tmp_path)
    ui.raw_out = io.StringIO()
    for n in range(40):
        ui.write(f'line number {n}\n')
    drawn(ui, monkeypatch)
    top, bottom = ui.panel_bounds
    assert ui.panel_rows[bottom - 1].startswith('line number')
    ui.mouse_report(f'\x1b[<0;1;{top + 1}M')
    ui.mouse_report(f'\x1b[<32;6;{top + 2}M')
    drawn(ui, monkeypatch)    # a selection in flight must not break the draw
    ui.mouse_report(f'\x1b[<0;6;{top + 2}m')
    assert ui.selected_text() == f'{ui.panel_rows[top]}\n{ui.panel_rows[top + 1][:6]}'.rstrip()
    assert ui.raw_out.getvalue().startswith('\x1b]52;c;')


def test_clipboard_inside_tmux_asks_tmux_to_load_the_buffer(monkeypatch):
    calls = []

    class Done:
        returncode = 0
    monkeypatch.setenv('TMUX', '/tmp/tmux-1/default,1,0')
    monkeypatch.setattr(terminal.subprocess, 'run', lambda cmd, **k: calls.append((cmd, k['input'])) or Done())
    ui = Surface()
    ui.raw_out = io.StringIO()
    assert ui.clipboard('hi') is True
    assert calls == [(['tmux', 'load-buffer', '-w', '-'], b'hi')]
    assert ui.raw_out.getvalue() == ''

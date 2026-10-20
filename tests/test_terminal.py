"""Behavioral regressions for typing while a turn owns the main thread."""
import threading
import time

from onecli import terminal
from onecli.glyph import GlyphEngine, ASSET_PATH


def test_type_ahead_preserves_unsent_draft():
    ui = terminal.Surface()
    for c in 'first':
        ui.feed(c)
    ui.submit()
    assert ui.read() == 'first'
    for c in 'queued':
        ui.feed(c)
    ui.submit()
    for c in 'still editing':
        ui.feed(c)
    ui.write('streamed answer\n')
    assert ui.read() == 'queued'
    assert ui.draft == 'still editing'
    assert ui.cursor == len(ui.draft)


def test_question_cannot_consume_queued_turn_and_restores_draft():
    ui = terminal.Surface()
    ui.turns.put('do another job')
    ui.draft, ui.cursor = 'unsent draft', 4
    result = []
    thread = threading.Thread(target=lambda: result.append(ui.answer(1)))
    thread.start()
    deadline = time.monotonic() + 1
    while not ui.answering and time.monotonic() < deadline:
        time.sleep(.005)
    assert ui.answering
    ui.feed('2')
    ui.submit()
    thread.join(2)
    assert result == ['2']
    assert ui.turns.get_nowait() == 'do another job'
    assert (ui.draft, ui.cursor) == ('unsent draft', 4)


def test_question_timeout_preserves_draft():
    ui = terminal.Surface()
    ui.draft = 'keep this'
    assert ui.answer(.01) is None
    assert ui.draft == 'keep this'
    assert not ui.answering


def test_bracketed_paste_never_submits_embedded_newlines():
    ui = terminal.Surface()
    for c in '\x1b[200~hello\n/exit\nworld\x1b[201~':
        ui.feed(c)
    assert ui.turns.empty()
    assert ui.draft == 'hello\n/exit\nworld'
    ui.drain_done()
    ui.feed('\n')
    assert ui.turns.get_nowait() == 'hello\n/exit\nworld'


def test_embedded_paste_end_marker_cannot_run_the_rest_as_keystrokes():
    ui = terminal.Surface()
    for c in '\x1b[200~harmless\x1b[201~\rINJECTED-SUBMIT\r/exit\r':
        ui.feed(c)
    ui.drain_done()
    assert ui.turns.empty()
    assert ui.draft == 'harmless\nINJECTED-SUBMIT\n/exit\n'


def test_paste_tail_is_stripped_of_control_sequences_and_backspaces():
    ui = terminal.Surface()
    for c in '\x1b[200~AAA\x1b[201~BBB\x1b[31mred\x1b[0mC\x08D\x1b]0;PWNED\x07E':
        ui.feed(c)
    ui.drain_done()
    assert ui.draft == 'AAABBBredCDE'


def test_typing_after_the_burst_is_still_typing():
    ui = terminal.Surface()
    for c in '\x1b[200~x\x1b[201~':
        ui.feed(c)
    ui.drain_done()
    ui.feed('\r')
    assert ui.turns.get_nowait() == 'x'


def test_ctrl_c_and_ctrl_d_leave_a_paste_that_never_ends():
    ui = terminal.Surface()
    for c in '\x1b[200~hello no end':
        ui.feed(c)
    assert ui.pasting
    ui.feed('\x04')
    assert not ui.pasting and ui.paste == ''
    assert isinstance(ui.turns.get_nowait(), EOFError)
    for c in '\x1b[200~more':
        ui.feed(c)
    ui.feed('\x03')
    assert not ui.pasting
    for c in 'typed':
        ui.feed(c)
    assert ui.draft == 'typed'


def test_a_paste_with_a_lost_end_marker_is_delivered_when_input_goes_quiet():
    ui = terminal.Surface()
    for c in '\x1b[200~partial':
        ui.feed(c)
    assert not ui.flush_paste(ui.paste_at + terminal.PASTE_IDLE / 2)
    assert ui.pasting and ui.draft == ''
    assert ui.flush_paste(ui.paste_at + terminal.PASTE_IDLE)
    assert not ui.pasting and ui.draft == 'partial'
    ui.feed('!')
    assert ui.draft == 'partial!'


def test_paste_buffer_is_capped_but_the_end_marker_still_closes_it(monkeypatch):
    monkeypatch.setattr(terminal, 'PASTE_MAX', 100)
    ui = terminal.Surface()
    for c in '\x1b[200~' + 'x' * 1000 + '\x1b[201~':
        ui.feed(c)
    assert not ui.pasting
    assert len(ui.paste) == 0 and len(ui.draft) == 100


def test_wide_and_multiline_cursor_layout():
    assert terminal.wrap('abcd界e', 5) == ['abcd', '界e']
    assert terminal.wrap('ab\ncd', 5) == ['ab', 'cd']
    assert terminal.cells('界e\u0301') == 3


def test_glyph_bundle_animates_ascii_at_requested_size():
    assert ASSET_PATH.is_file()
    engine = GlyphEngine()
    try:
        assert engine.proc is not None
        frames = [engine.frame(60, 10, dt=.1) for _ in range(70)]
        assert all(len(frame) == 10 for frame in frames)
        assert all(len(line) == 60 and line.isascii() for frame in frames for line in frame)
        assert len({tuple(frame) for frame in frames}) > 10
        assert any(c not in ' .' for line in frames[-1] for c in line)
    finally:
        engine.close()


def test_missing_node_fallback_is_ascii_and_bounded(tmp_path):
    engine = GlyphEngine(tmp_path / 'missing.mjs')
    try:
        lines = engine.frame(12, 8)
        assert len(lines) == 8
        assert all(len(line) == 12 and line.isascii() for line in lines)
    finally:
        engine.close()

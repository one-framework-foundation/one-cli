import curses
from onecli.terminal import Surface

def feed(ui, text):
    for c in text:
        ui.feed(c)

def test_normal_cursor_keys_edit_at_requested_position():
    ui = Surface()
    feed(ui, 'hello\x1b[D\x1b[DX')
    assert (ui.draft, ui.cursor) == ('helXlo', 4)

def test_terminal_cursor_report_is_not_inserted():
    ui = Surface()
    feed(ui, 'hello\x1b[40;3R')
    assert (ui.draft, ui.cursor) == ('hello', 5)

def test_modified_arrow_and_delete_sequences():
    ui = Surface()
    feed(ui, 'hello\x1b[1;2D\x1b[3~')
    assert (ui.draft, ui.cursor) == ('hell', 4)

def test_escape_then_typing_does_not_discard_character():
    ui = Surface()
    feed(ui, '\x1bh')
    assert (ui.draft, ui.cursor) == ('h', 1)

def test_paste_keeps_escape_like_user_content_sanitized():
    ui = Surface()
    feed(ui, '\x1b[200~first\nsecond\x1b[201~')
    assert ui.draft == 'first\nsecond'
    assert ui.turns.empty()


def test_split_terminal_report_survives_ssh_delay():
    ui = Surface()
    feed(ui, 'hello\x1b[40;')
    ui.flush_escape(ui.escape_started + .2)
    feed(ui, '3R')
    assert (ui.draft, ui.cursor) == ('hello', 5)


def test_session_does_not_capture_mouse_so_terminal_selection_works():
    import inspect
    from onecli import terminal
    assert 'mousemask' not in inspect.getsource(terminal.session)


def test_run_drains_queued_keys_with_one_draw(monkeypatch):
    ui = Surface()
    keys = list('hello') + [curses.error]

    class Screen:
        def getmaxyx(self):
            return (40, 100)
        def timeout(self, _):
            pass
        def get_wch(self):
            k = keys.pop(0)
            if k is curses.error:
                ui.stopped.set()
                raise curses.error()
            return k
    ui.screen = Screen()
    draws = []
    ui.draw = lambda: draws.append(ui.draft)
    ui.persist_draft = lambda: None
    import os
    monkeypatch.setattr(os, 'get_terminal_size', lambda fd=0: os.terminal_size((100, 40)))
    monkeypatch.setattr('sys.stdin', type('S', (), {'fileno': lambda self: 0})())
    ui.run()
    assert ui.draft == 'hello'
    assert draws.count('hello') == 1 and 'h' not in draws


def _run_keys(monkeypatch, keys):
    import os
    ui = Surface()
    keys = list(keys) + [curses.error]

    class Screen:
        def getmaxyx(self):
            return (40, 100)
        def timeout(self, _):
            pass
        def get_wch(self):
            k = keys.pop(0)
            if k is curses.error:
                ui.stopped.set()
                raise curses.error()
            return k
    ui.screen = Screen()
    saves = []
    ui.draw = lambda: None
    ui.persist_draft = lambda: saves.append(len(ui.draft))
    monkeypatch.setattr(os, 'get_terminal_size', lambda fd=0: os.terminal_size((100, 40)))
    monkeypatch.setattr('sys.stdin', type('S', (), {'fileno': lambda self: 0})())
    ui.run()
    return ui, saves


def test_a_paste_in_flight_does_not_save_or_draw_per_batch(monkeypatch):
    # 600 keys is three 256-key batches; each used to save and redraw the
    # whole draft, which made a 100KB paste quadratic.
    ui, saves = _run_keys(monkeypatch, '\x1b[200~' + 'x' * 600)
    assert saves == []
    ui, saves = _run_keys(monkeypatch, '\x1b[200~' + 'x' * 600 + '\x1b[201~')
    assert saves == [600]


def test_editor_wrap_of_an_unchanged_draft_is_not_recomputed(monkeypatch):
    from onecli import terminal
    calls = []
    real = terminal.wrap
    monkeypatch.setattr(terminal, 'wrap', lambda *a: calls.append(a) or real(*a))
    ui = Surface()
    text = 'word ' * 5000
    ui.editor_wrap('draft', text, 80)
    for _ in range(20):  # the short per-line wraps of a frame must not evict it
        terminal.wrap('status line', 80)
    rows = ui.editor_wrap('draft', text, 80)
    assert len([c for c in calls if c[0] == text]) == 1
    rows.append('mutated')  # a caller's edit must not reach the memo
    assert ui.editor_wrap('draft', text, 80) == real(text, 80)
    assert ui.editor_wrap('draft', text + 'x', 80) == real(text + 'x', 80)


def test_run_redraws_and_stops_during_continuous_input(monkeypatch):
    ui = Surface()

    class Screen:
        reads = 0
        def getmaxyx(self):
            return (40, 100)
        def timeout(self, _):
            pass
        def get_wch(self):
            self.reads += 1
            if self.reads > 512:
                raise RuntimeError('input stream starved redraw')
            return 'a'

    ui.screen = Screen()
    draws = []

    def draw():
        draws.append(len(ui.draft))
        if ui.draft:
            ui.stopped.set()

    ui.draw = draw
    ui.persist_draft = lambda: None
    import os
    monkeypatch.setattr(os, 'get_terminal_size', lambda fd=0: os.terminal_size((100, 40)))
    monkeypatch.setattr('sys.stdin', type('S', (), {'fileno': lambda self: 0})())
    ui.run()
    assert draws == [256]

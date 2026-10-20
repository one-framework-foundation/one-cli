"""Ctrl-C at the idle prompt: hint + exit on second press."""

from onecli import terminal


class Screen:
    def __init__(self, height=30, width=120):
        self.size = (height, width)
        self.rows = {}
        self.attrs = {}

    def getmaxyx(self):
        return self.size

    def erase(self):
        self.rows = {}
        self.attrs = {}

    def addstr(self, y, x, text, attr=0):
        row = self.rows.get(y, '')
        self.rows[y] = row[:x].ljust(x) + text
        self.attrs[y] = attr

    def addnstr(self, y, x, text, n):
        self.addstr(y, x, text[:n])

    def move(self, y, x):
        pass

    def clrtoeol(self):
        pass

    def refresh(self):
        pass


def surface():
    return terminal.Surface(screen=Screen())


def test_ctrl_c_with_draft_clears_it_and_writes_hint():
    ui = surface()
    ui.draft, ui.cursor = 'in progress', 5
    ui.key('\x03')
    assert ui.draft == ''
    assert ui.ctrl_c_hint_at is not None
    text = '\n'.join(ui.lines)
    assert 'Ctrl-C again within 2s to exit' in text


def test_ctrl_c_with_empty_draft_writes_hint_without_exiting():
    ui = surface()
    assert not ui.stopped.is_set()
    ui.key('\x03')
    assert not ui.stopped.is_set()
    assert ui.ctrl_c_hint_at is not None
    text = '\n'.join(ui.lines)
    assert 'Ctrl-C again within 2s to exit' in text


def test_second_ctrl_c_within_window_exits():
    ui = surface()
    ui.key('\x03')  # first press -> hint
    assert not ui.stopped.is_set()
    ui.key('\x03')  # second press within window -> exit
    assert ui.stopped.is_set()


def test_second_ctrl_c_after_window_is_a_fresh_hint():
    ui = surface()
    ui.key('\x03')
    ui.ctrl_c_hint_at -= terminal.Surface.CTRL_C_WINDOW + 1  # expired
    ui.stopped.clear()
    ui.key('\x03')
    assert not ui.stopped.is_set()
    assert ui.ctrl_c_hint_at is not None


def test_esc_does_not_show_hint():
    ui = surface()
    ui.draft, ui.cursor = 'kept', 4
    ui.key('\x1b')
    assert ui.draft == 'kept'
    assert ui.ctrl_c_hint_at is None
    assert '\n'.join(ui.lines) == ''


def test_ctrl_c_during_busy_kills_process_like_before(monkeypatch):
    import os
    killed = []
    monkeypatch.setattr(os, 'kill', lambda pid, sig: killed.append(sig))
    ui = surface()
    ui.busy = ui.allow_steer = True
    ui.key('\x03')
    assert killed == [2]  # SIGINT

def test_exit_story_is_the_same_in_hint_overlay_footer_and_behaviour():
    window = terminal.Surface.CTRL_C_WINDOW
    assert f'within {window:.0f}s' in terminal.EXIT_HINT
    overlay = dict(terminal.BINDINGS)['Ctrl-C']
    assert 'twice' in overlay and f'{window:.0f}s' in overlay
    assert 'Ctrl-C twice exits' in terminal.FOOTER
    # Behaviour: a second press after the window is a first press again.
    ui = surface()
    ui.key('\x03')
    ui.ctrl_c_hint_at -= window + 1
    ui.key('\x03')
    assert not ui.stopped.is_set()
    ui.key('\x03')
    assert ui.stopped.is_set()

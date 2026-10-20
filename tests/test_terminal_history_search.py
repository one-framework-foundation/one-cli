"""Reverse-i-search history mode on the live editor."""
import curses

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


def type_chars(ui, text):
    for c in text:
        ui.key(c)


def test_ctrl_r_opens_search_with_empty_query():
    ui = surface()
    ui.history = ['deploy', 'check logs', 'test']
    ui.draft, ui.cursor = 'in progress', 10
    ui.key('\x12')
    # Search is open, the saved draft is held until accept/cancel.
    assert ui.history_query == ''
    assert ui.history_saved == ('in progress', 10)
    # No match yet, so the draft is cleared while the query is empty.
    assert ui.draft == ''


def test_typing_extends_query_and_finds_match():
    ui = surface()
    ui.history = ['deploy', 'check logs', 'test']
    ui.key('\x12')
    type_chars(ui, 'dep')
    assert ui.history_query == 'dep'
    assert ui.draft == 'deploy'


def test_ctrl_r_steps_to_older_match():
    ui = surface()
    ui.history = ['deploy alpha', 'check logs', 'deploy beta']
    ui.key('\x12')
    type_chars(ui, 'deploy')
    assert ui.draft == 'deploy beta'  # newest match first.
    ui.key('\x12')
    assert ui.draft == 'deploy alpha'  # next older.


def test_ctrl_s_steps_to_newer_match():
    ui = surface()
    ui.history = ['deploy alpha', 'check logs', 'deploy beta']
    ui.key('\x12')
    type_chars(ui, 'deploy')
    assert ui.draft == 'deploy beta'  # newest match first.
    ui.key('\x12')  # step older -> alpha
    assert ui.draft == 'deploy alpha'
    # ^R at the oldest match keeps the draft (readline semantics).
    ui.key('\x12')
    assert ui.draft == 'deploy alpha'
    # ^S steps newer toward the most recent match.
    ui.key('\x13')
    assert ui.draft == 'deploy beta'


def test_enter_accepts_match_and_restores_history_index():
    ui = surface()
    ui.history = ['first', 'second']
    ui.history_index = 2
    ui.key('\x12')
    type_chars(ui, 'sec')
    assert ui.draft == 'second'
    ui.key('\n')
    assert ui.history_query is None
    assert ui.draft == 'second'
    # Up should now walk from the present, not from the match.
    ui.key(curses.KEY_UP)
    assert ui.draft == 'second'


def test_esc_cancels_and_restores_draft():
    ui = surface()
    ui.history = ['first', 'second']
    ui.draft, ui.cursor = 'kept', 4
    ui.key('\x12')
    type_chars(ui, 'sec')
    assert ui.draft == 'second'
    ui.key('\x1b')
    assert (ui.draft, ui.cursor) == ('kept', 4)
    assert ui.history_query is None


def test_ctrl_c_cancels_and_restores_draft():
    ui = surface()
    ui.history = ['first']
    ui.draft, ui.cursor = 'kept', 4
    ui.key('\x12')
    type_chars(ui, 'fir')
    ui.key('\x03')
    assert (ui.draft, ui.cursor) == ('kept', 4)


def test_ctrl_r_with_no_history_is_a_noop():
    ui = surface()
    ui.draft = 'typed'
    ui.key('\x12')
    assert ui.history_query is None
    assert ui.draft == 'typed'


def test_typing_in_search_does_not_open_command_menu():
    ui = surface()
    ui.completer = None  # even with a completer, the search box owns keys.
    ui.history = ['hello']
    ui.key('\x12')
    type_chars(ui, '/hel')
    # The slash is a literal in the query, not a menu opener.
    assert ui.history_query == '/hel'
    assert ui.command_menu() == []


def test_search_box_is_drawn_when_active(monkeypatch):
    monkeypatch.setattr(curses, 'A_DIM', 0, raising=False)
    monkeypatch.setattr(curses, 'A_BOLD', 0, raising=False)
    monkeypatch.setattr(curses, 'A_REVERSE', 1 << 18, raising=False)
    ui = surface()
    ui.history = ['first', 'second']
    ui.key('\x12')
    type_chars(ui, 'sec')
    ui.draw()
    text = '\n'.join(ui.screen.rows.values())
    assert 'reverse-i-search' in text
    assert 'sec' in text

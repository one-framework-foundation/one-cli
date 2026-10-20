"""The `?` key opens a dismissible key overlay; mid-draft stays a literal char."""
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


def test_question_on_empty_draft_opens_overlay():
    ui = surface()
    ui.key('?')
    assert ui.overlay is True
    assert ui.draft == ''


def test_any_keypress_closes_the_overlay():
    ui = surface()
    ui.key('?')
    assert ui.overlay is True
    ui.key(curses.KEY_DOWN)
    assert ui.overlay is False
    assert ui.draft == ''  # keypress did not leak into the draft.


def test_question_mid_draft_stays_a_literal_character():
    ui = surface()
    ui.draft, ui.cursor = 'typed ', 6
    ui.key('?')
    assert ui.overlay is False
    assert ui.draft == 'typed ?'
    assert ui.cursor == len(ui.draft)


def test_overlay_is_drawn_with_key_table(monkeypatch):
    monkeypatch.setattr(curses, 'A_DIM', 0, raising=False)
    monkeypatch.setattr(curses, 'A_BOLD', 1 << 21, raising=False)
    monkeypatch.setattr(curses, 'A_REVERSE', 1 << 18, raising=False)
    ui = surface()
    ui.key('?')
    ui.draw()
    text = '\n'.join(ui.screen.rows.values())
    # The header and at least one known binding are present.
    assert 'keys' in text
    for key, _ in terminal.BINDINGS:
        assert key in text, f'{key!r} missing from overlay'
    assert 'closes' in text  # footer line.


def test_bindings_table_is_the_source_of_truth():
    # Footer and overlay both reference BINDINGS so they cannot drift.
    for key, desc in terminal.BINDINGS:
        if key == '/':
            assert key in terminal.FOOTER
        # Keys that have a footer hint must appear in the footer copy.
        # We don't enforce every key — some (Esc, Tab) are global and named
        # in the help text instead.
    assert 'Enter send' in terminal.FOOTER
    assert 'Esc stop' in terminal.FOOTER

class FullScreen(Screen):
    """Screen that keeps what is already on a row after a shorter write."""

    def addstr(self, y, x, text, attr=0):
        row = self.rows.get(y, '').ljust(x + len(text))
        self.rows[y] = row[:x] + text + row[x + len(text):]


def _overlay_rows(height, width):
    ui = terminal.Surface(screen=FullScreen(height, width))
    ui.write('Type a message to start, or / for commands. ' * 4 + '\n')
    ui.key('?')
    ui.draw()
    return ui, ui.screen.rows


def test_overlay_covers_the_panel_on_a_short_terminal():
    ui, rows = _overlay_rows(12, 40)
    top, end = ui.panel_bounds
    for y in range(top - 1, end):
        row = rows[y]
        assert row[0] in '+|', (y, row)
        assert len(row) == 39, (y, row)
    box = ' '.join(rows[y] for y in range(top - 1, end))
    assert 'more' in box  # the keys left out are counted, not silently dropped


def test_overlay_lists_every_key_when_it_fits():
    ui, rows = _overlay_rows(40, 80)
    text = '\n'.join(rows.values())
    for key, _ in terminal.BINDINGS:
        assert key in text
    assert 'more' not in text

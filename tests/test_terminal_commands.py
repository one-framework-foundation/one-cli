"""The terminal says what its keys do and lists slash commands as they are typed."""
import curses

from onecli import commands as cmdlib
from onecli import inputline, terminal


class Screen:
    """Enough of a curses window to capture what draw() puts on each row."""
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
        self.cursor = (y, x)

    def clrtoeol(self):
        # RowRenderer.refresh() clears a changed row before repainting it, so a
        # row the frame no longer draws must be blanked here too.
        y, x = self.cursor
        if y in self.rows:
            self.rows[y] = self.rows[y][:x]

    def refresh(self):
        pass


def surface(tmp_path):
    ui = terminal.Surface(screen=Screen())
    ui.completer = inputline.Completer(cmdlib.names_for(), str(tmp_path))
    return ui


def drawn(ui, monkeypatch):
    monkeypatch.setattr(curses, 'A_DIM', 0, raising=False)
    monkeypatch.setattr(curses, 'A_BOLD', 0, raising=False)
    monkeypatch.setattr(curses, 'A_REVERSE', 1 << 18, raising=False)
    ui.draw()
    return ui.screen.rows


def type_(ui, text):
    for c in text:
        ui.key(c)


def test_footer_says_what_each_key_does():
    for phrase in ('show/hide tool details', 'jump to latest', '/ commands', 'newline'):
        assert phrase in terminal.FOOTER
    assert '^O tools |' not in terminal.FOOTER and '^G live |' not in terminal.FOOTER


def test_names_for_carries_each_summary():
    names = cmdlib.names_for()
    assert names['/help'] == cmdlib.COMMANDS['help'].summary
    assert all(name.startswith('/') for name in names)


def test_slash_lists_every_command_with_its_summary(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/')
    menu = ui.command_menu()
    assert [name for name, _ in menu] == sorted(cmdlib.names_for())
    assert ('/help', cmdlib.COMMANDS['help'].summary) in menu


def test_menu_narrows_and_closes_once_arguments_start(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/he')
    assert [name for name, _ in ui.command_menu()] == ['/help']
    type_(ui, 'lp ')
    assert ui.command_menu() == []
    ui.draft, ui.cursor = 'tell me about /help', 19
    assert ui.command_menu() == []


def test_tab_completes_a_command(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/hel')
    ui.key('\t')
    assert ui.draft == '/help '
    assert ui.lines == type(ui.lines)()


def test_menu_is_drawn_above_the_rule(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    type_(ui, '/he')
    rows = drawn(ui, monkeypatch)
    text = '\n'.join(rows.values())
    assert '/help' in text and cmdlib.COMMANDS['help'].summary in text
    # The footer lives on the last row; the full string is only present
    # when the terminal is wide enough. The leading phrase is always there.
    assert rows[29].startswith('Enter send')


def test_long_menu_is_capped_and_says_where_the_highlight_is(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    type_(ui, '/')
    text = '\n'.join(drawn(ui, monkeypatch).values())
    assert f'1/{len(cmdlib.names_for())} | Up/Dn/PgUp/PgDn select' in text


def highlighted(ui, monkeypatch):
    rows = drawn(ui, monkeypatch)
    lit = [rows[y] for y, attr in ui.screen.attrs.items() if attr == curses.A_REVERSE]
    assert len(lit) == 1
    return lit[0].split()[0]


def test_down_and_up_move_the_highlight_and_wrap(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    names = sorted(cmdlib.names_for())
    type_(ui, '/')
    assert highlighted(ui, monkeypatch) == names[0]
    ui.key(curses.KEY_DOWN)
    assert highlighted(ui, monkeypatch) == names[1]
    ui.key(curses.KEY_UP)
    ui.key(curses.KEY_UP)
    assert highlighted(ui, monkeypatch) == names[-1]
    ui.key(curses.KEY_DOWN)
    assert highlighted(ui, monkeypatch) == names[0]
    assert ui.draft == '/'


def test_every_command_is_reachable_past_the_visible_rows(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    names = sorted(cmdlib.names_for())
    assert len(names) > terminal.MENU_ROWS
    type_(ui, '/')
    seen = [highlighted(ui, monkeypatch)]
    for _ in names[1:]:
        ui.key(curses.KEY_DOWN)
        seen.append(highlighted(ui, monkeypatch))
    assert seen == names


def test_typing_filters_and_resets_the_highlight(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    type_(ui, '/')
    ui.key(curses.KEY_DOWN)
    ui.key(curses.KEY_DOWN)
    type_(ui, 'he')
    assert ui.menu_index == 0
    assert highlighted(ui, monkeypatch) == '/help'


def test_tab_completes_the_highlighted_command(tmp_path):
    ui = surface(tmp_path)
    names = sorted(cmdlib.names_for())
    type_(ui, '/')
    ui.key(curses.KEY_DOWN)
    ui.key('\t')
    assert ui.draft == names[1] + ' ' and ui.cursor == len(ui.draft)
    assert ui.command_menu() == []


def test_enter_runs_the_highlighted_command(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/he')
    ui.key('\r')
    assert ui.turns.get_nowait() == '/help'
    assert ui.draft == ''


def test_enter_on_an_ambiguous_partial_completes_instead_of_running(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/co')
    assert len(ui.command_menu()) > 1
    first = ui.command_menu()[0][0]
    ui.key('\r')
    assert ui.turns.empty()
    assert ui.draft == first + ' '
    ui.key('\r')
    assert ui.turns.get_nowait().strip() == first


def test_tab_only_completes_and_enter_runs_the_exact_command(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/co')
    ui.key('\t')
    assert ui.turns.empty() and ui.draft.startswith('/co') and ui.draft.endswith(' ')
    ui.key('\r')
    assert ui.turns.get_nowait().startswith('/co')


def test_enter_runs_a_highlight_the_person_moved(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/co')
    ui.key(curses.KEY_DOWN)
    second = ui.command_menu()[1][0]
    ui.key('\r')
    assert ui.turns.get_nowait() == second


def test_esc_closes_the_menu_and_keeps_the_draft(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/he')
    ui.key('\x1b')
    assert ui.draft == '/he' and ui.command_menu() == []
    type_(ui, 'l')
    assert [name for name, _ in ui.command_menu()] == ['/help']


def test_up_down_still_walk_history_when_the_menu_is_closed(tmp_path):
    ui = surface(tmp_path)
    ui.history = ['first', 'second']
    ui.history_index = 2
    ui.key(curses.KEY_UP)
    assert ui.draft == 'second'
    ui.key(curses.KEY_UP)
    assert ui.draft == 'first'
    ui.key(curses.KEY_DOWN)
    assert ui.draft == 'second'


def test_help_dispatches_and_lists_commands():
    repl = cmdlib.Repl.__new__(cmdlib.Repl)
    repl.files = {}
    out = cmdlib.dispatch('/help', repl).text
    assert '/help' in out and '/exit' in out


def test_footer_shortens_at_narrow_width_instead_of_clipping():
    wide = terminal.footer_text(200)
    assert wide == terminal.FOOTER
    # Every row is a prefix of the full hint list joined the same way, and it
    # fits: a hint is shown whole or not at all, never cut mid-word.
    for width in (40, 60, 80, 100, 112, len(terminal.FOOTER)):
        text = terminal.footer_text(width)
        assert len(text) <= width
        assert terminal.FOOTER.startswith(text)
        assert text == '' or text in terminal.FOOTER
    # Narrowing drops whole hints: 40 columns shows strictly fewer than 200.
    assert terminal.footer_text(40) != terminal.footer_text(200)
    assert terminal.footer_text(40).endswith(next(
        hint for hint in reversed(terminal.footer_text(40).split(' | '))))


def test_footer_at_narrow_width_ends_on_a_whole_hint(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    ui.screen.size = (24, 90)
    footer = drawn(ui, monkeypatch)[23]
    assert footer == terminal.footer_text(89, terminal.IDLE_HINTS)
    # The row ends at a complete hint, not a mid-word truncation.
    assert footer.split(' | ')[-1] in terminal.IDLE_HINTS
    assert not footer.endswith(' ')


def test_fuzzy_menu_finds_a_transposed_command(tmp_path):
    ui = surface(tmp_path)
    assert ui.completer.menu('/ost') == [('/cost', cmdlib.COMMANDS['cost'].summary)]
    assert ui.completer.menu_is_fuzzy('/ost') is True


def test_prefix_matches_rank_before_fuzzy(tmp_path):
    ui = surface(tmp_path)
    # /co is a real prefix and must not be replaced by the subsequence set.
    assert [n for n, _ in ui.completer.menu('/co')] == ['/compact', '/context', '/cost']
    assert ui.completer.menu_is_fuzzy('/co') is False
    # A prefix hit keeps alphabetical order and every other subsequence row out.
    assert [n for n, _ in ui.completer.menu('/m')] == [
        n for n in sorted(cmdlib.names_for()) if n.startswith('/m')]


def test_fuzzy_rows_render_dim_while_prefix_rows_do_not(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    monkeypatch.setattr(curses, 'A_DIM', 4, raising=False)
    monkeypatch.setattr(curses, 'A_BOLD', 0, raising=False)
    monkeypatch.setattr(curses, 'A_REVERSE', 1 << 18, raising=False)
    # /ls is not a prefix of anything; the subsequence rows are shown dim.
    type_(ui, '/ls')
    ui.draw()
    rows, attrs = ui.screen.rows, ui.screen.attrs
    dim_rows = [y for y, attr in attrs.items() if attr == 4]
    assert any('/logs' in rows[y] or '/models' in rows[y] for y in dim_rows)
    # A prefix match is never dimmed - the same command is drawn plain.
    ui2 = surface(tmp_path)
    type_(ui2, '/co')
    ui2.draw()
    rows2, attrs2 = ui2.screen.rows, ui2.screen.attrs
    assert not any('/cost' in rows2[y] for y, attr in attrs2.items() if attr == 4)


def test_page_keys_move_the_menu_highlight_not_the_transcript(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    type_(ui, '/')
    assert ui.menu_index == 0
    scroll = ui.scroll
    ui.key(curses.KEY_NPAGE)
    assert ui.menu_index == min(terminal.MENU_ROWS, len(cmdlib.names_for()) - 1)
    assert ui.scroll == scroll
    ui.key(curses.KEY_PPAGE)
    assert ui.menu_index == 0
    # Paging never runs off either end of the list.
    for _ in range(10):
        ui.key(curses.KEY_NPAGE)
    assert ui.menu_index == len(cmdlib.names_for()) - 1


def test_esc_on_a_lone_slash_removes_it_so_the_next_word_is_not_a_command(tmp_path):
    ui = surface(tmp_path)
    type_(ui, '/')
    ui.key('\x1b')
    assert ui.draft == ''
    type_(ui, 'foo')
    assert ui.draft == 'foo'
    type_(ui, ' /he')
    ui.key('\x1b')  # a longer draft is kept, as before
    assert ui.draft == 'foo /he'


def test_ctrl_d_with_a_draft_at_the_end_says_why_nothing_happened(tmp_path):
    ui = surface(tmp_path)
    type_(ui, 'abc')
    ui.key('\x04')
    assert ui.draft == 'abc' and ui.turns.empty()
    assert any('Ctrl-D leaves only from an empty prompt' in line for line in ui.lines)


def test_idle_footer_does_not_offer_esc_stop_and_says_how_to_leave(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    footer = drawn(ui, monkeypatch)[ui.screen.size[0] - 1]
    assert 'Esc stop' not in footer
    assert 'Ctrl-D exit' in footer


def test_busy_footer_offers_esc_stop(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    ui.busy = True
    footer = drawn(ui, monkeypatch)[ui.screen.size[0] - 1]
    assert 'Esc stop' in footer


def test_first_screen_has_a_welcome_until_there_is_a_conversation(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    text = '\n'.join(drawn(ui, monkeypatch).values())
    assert 'Type a message and press Enter' in text and 'Ctrl-C twice' in text
    ui.write('\n> hello\n')
    text = '\n'.join(drawn(ui, monkeypatch).values())
    assert 'Type a message and press Enter' not in text


def test_menu_hint_keeps_enter_and_esc_on_a_narrow_terminal():
    from onecli.terminal import cells, menu_hint
    for width in (29, 39):
        hint = menu_hint(width, '1/31')
        assert cells(hint) <= width, hint
        assert 'Enter run' in hint and 'Esc close' in hint, hint
    wide = menu_hint(80, '1/31')
    assert wide == ' 1/31 | Up/Dn/PgUp/PgDn select | Tab complete | Enter run | Esc close'

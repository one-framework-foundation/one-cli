"""Menu descriptions end in an ellipsis at a word boundary, never mid-word."""
from onecli.terminal import cells, ellipsize
from test_terminal_commands import drawn, surface, type_


def test_fitting_text_is_untouched():
    assert ellipsize('show all keys', 20) == 'show all keys'


def test_cut_lands_on_a_word_boundary():
    assert ellipsize('show every key binding', 14) == 'show every…'


def test_single_long_word_is_still_bounded():
    assert cells(ellipsize('supercalifragilistic', 8)) <= 8


def test_narrow_menu_rows_end_on_whole_words(tmp_path, monkeypatch):
    ui = surface(tmp_path)
    ui.screen.size = (12, 40)
    type_(ui, '/')
    rows = drawn(ui, monkeypatch)
    summaries = {s for _, s in ui.command_menu()}
    checked = 0
    for row in rows.values():
        if row.startswith(' /'):
            tail = row.split('  ', 1)[1].strip()
            assert tail in summaries or tail.endswith('\u2026'), row
            checked += 1
    assert checked

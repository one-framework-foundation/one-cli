"""The welcome block is centred in the transcript panel, not stuck at the top."""
from test_terminal_commands import drawn, surface


def welcome_rows(tmp_path, monkeypatch, height):
    ui = surface(tmp_path)
    ui.screen.size = (height, 80)
    rows = drawn(ui, monkeypatch)
    return [y for y, text in rows.items() if text.startswith('Type a message')], ui


def test_welcome_moves_down_in_a_tall_window(tmp_path, monkeypatch):
    (row,), ui = welcome_rows(tmp_path, monkeypatch, 24)
    top, end = ui.panel_bounds
    assert row - top >= 3, 'welcome text is still glued to the panel top'
    assert end - (row + 3) >= 3, 'welcome text is not balanced against the bottom'


def test_welcome_fits_a_short_window(tmp_path, monkeypatch):
    rows, _ = welcome_rows(tmp_path, monkeypatch, 12)
    assert len(rows) == 1

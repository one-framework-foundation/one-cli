"""Below the minimum size the screen says what is needed, in whole words."""
from onecli.terminal import cells, too_small_notice
from test_terminal_overlay import Screen
from onecli import terminal


def test_roomy_enough_says_the_size_needed():
    assert too_small_notice(6, 40) == ['one: enlarge terminal to 16x8', 'now 40x6, need 16x8']


def test_never_cut_mid_word_or_wider_than_the_window():
    for width in range(2, 40):
        for row in too_small_notice(5, width):
            assert cells(row) <= width - 1, (width, row)
            assert row in ('one: enlarge terminal to 16x8', 'one: enlarge terminal',
                           'enlarge terminal', 'enlarge', '16x8') or row.startswith('now ') or row == '%dx5' % width


def test_draw_below_minimum_shows_the_notice():
    screen = Screen(height=6, width=24)
    terminal.Surface(screen=screen).draw()
    assert screen.rows[0] == 'one: enlarge terminal'
    assert screen.rows[1] == 'now 24x6, need 16x8'

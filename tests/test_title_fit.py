"""The header line drops whole segments instead of cutting a path mid-word."""
from onecli.terminal import cells, fit_title

TITLE = 'one  |  local-model-7b  |  /tmp/ptroot/workspace'


def test_wide_terminal_keeps_everything():
    assert fit_title(TITLE, 80) == TITLE


def test_model_drops_before_path():
    line = fit_title(TITLE, 40)
    assert line == 'one  |  /tmp/ptroot/workspace'
    assert 'local-model' not in line


def test_path_keeps_its_tail_behind_an_ellipsis():
    line = fit_title(TITLE, 20)
    assert line.startswith('one  |  …')
    assert line.endswith('workspace')
    assert cells(line) <= 20


def test_never_exceeds_width():
    for width in range(1, 60):
        assert cells(fit_title(TITLE, width)) <= max(1, width)


def test_draw_does_not_cut_the_path_mid_word():
    from test_terminal_overlay import Screen
    from onecli import terminal
    screen = Screen(height=12, width=40)
    ui = terminal.Surface(title=TITLE, screen=screen)
    ui.draw()
    assert screen.rows[0].strip() == 'one  |  /tmp/ptroot/workspace'

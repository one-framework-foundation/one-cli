"""Cross-theme and caller regressions from the animation audit."""
import curses
import math
from types import SimpleNamespace

import pytest
from onecli.motion_theme import MotionTheme, THEMES
from onecli.terminal import Surface


@pytest.mark.parametrize('theme', THEMES)
def test_fast_typing_is_bounded_and_never_erases_the_shape(theme):
    m = MotionTheme(theme)
    for i in range(90):
        m.event('typing', {'x': (i % 18)/10 - .9})
        m.advance(1/30)
    for x in (-.9, -.3, 0., .3, .9):
        for y in (-.8, 0., .35, .8):
            dx, dy, glow = m.fluid(x, y)
            assert abs(dx) <= .12 and abs(dy) <= .12
            assert 0 <= glow <= .85
    m.advance(4)
    assert not m.ripples and not m.embers


@pytest.mark.parametrize('theme', THEMES)
def test_redraw_size_and_order_never_advance_motion(theme):
    m = MotionTheme(theme)
    m.event('typing')
    m.advance(.2)
    before = m.frame(48, 9)
    colors = m.colors
    for size in ((18, 1), (48, 2), (1, 1), (79, 10)):
        m.frame(*size)
    assert m.frame(48, 9) == before
    assert m.colors == colors
    assert m.time == pytest.approx(.2)


@pytest.mark.parametrize('theme', THEMES)
@pytest.mark.parametrize('kind', ('typing', 'submit', 'waiting', 'text_delta', 'tool_start', 'tool_end', 'answer', 'ready'))
def test_all_event_callers_preserve_instantaneous_frame(theme, kind):
    m = MotionTheme(theme)
    m.advance(.3)
    before = m.frame(24, 4)
    colors = m.colors
    m.event(kind, {'chunk': 'hello', 'x': -.8})
    assert m.frame(24, 4) == before
    assert m.colors == colors


@pytest.mark.parametrize('theme', THEMES)
def test_closed_idle_paths_match_position_and_velocity(theme):
    m = MotionTheme(theme)
    def sample(p):
        m.phase = p
        return [m.base_ink(theme, x, y) for x,y in ((.2,.1),(-.4,.3),(.7,-.2))]
    eps = 1e-5
    start, end = sample(0), sample(4*math.pi)
    assert end == pytest.approx(start, abs=1e-10)
    left, right = sample(4*math.pi-eps), sample(eps)
    assert [(b-a)/eps for a,b in zip(left,end)] == pytest.approx(
        [(b-a)/eps for a,b in zip(start,right)], abs=.002)


def test_startup_blend_matches_runtime_selection():
    m = MotionTheme('scan-line:0.25+orb-pulse:0.75')
    assert m.params['shape_scan-line'] == .25
    assert m.params['shape_orb-pulse'] == .75


def test_history_and_enter_do_not_emit_extra_typing_gestures():
    ui = Surface(theme='scan-line')
    ui.history, ui.history_index = ['old draft'], 1
    ui.feed(curses.KEY_UP)
    ui.settle_arrows()
    assert not ui.motion.embers
    ui.feed(curses.KEY_ENTER)
    assert len(ui.motion.embers) == 5  # submit only


def test_paste_emits_one_gesture_at_completion():
    ui = Surface(theme='scan-line')
    for ch in '\x1b[200~hello\nworld':
        ui.feed(ch)
    assert not ui.motion.embers
    for ch in '\x1b[201~':
        ui.feed(ch)
    assert len(ui.motion.embers) == 3


def test_active_surface_suppresses_competing_legacy_spinner(monkeypatch):
    from onecli import cli, terminal, spinner
    monkeypatch.setattr(terminal, 'ACTIVE', Surface())
    def forbidden(*args, **kwargs):
        raise AssertionError('second animation writer')
    monkeypatch.setattr(spinner, 'Spinner', forbidden)
    cli._tracer(False)('waiting', {'step': 1})
    assert terminal.ACTIVE.motion.state == 'thinking'


def test_reduced_motion_reaches_legacy_glyph_and_both_spinners(monkeypatch):
    from onecli import glyph, spinner
    monkeypatch.setenv('ONE_REDUCED_MOTION', '1')
    for motif in (spinner.ARRIVE, spinner.WORK):
        s = spinner.Spinner(motif, enabled=False)
        before = list(s._cells)
        for _ in range(10):
            s.feed(100)
            s._frame()
        assert s._cells == before
    engine = glyph.GlyphEngine()
    try:
        before = engine.frame(12, 3)
        assert engine.frame(12, 3, action='excite', dt=1) == before
        assert engine.frame(1, 1) == ['.']
    finally:
        engine.close()


def test_task_reveal_preserves_velocity_on_completion():
    m = MotionTheme()
    task = SimpleNamespace(status='running', proc=None)
    m.track(task)
    m.advance(.1)
    card = m.cards[id(task)]
    value, velocity = card['reveal'], card['reveal_velocity']
    task.status = 'done'
    m.advance(1e-7)
    assert (card['reveal'] - value)/1e-7 == pytest.approx(velocity, abs=1e-4)


def test_status_labels_and_timer_digits_do_not_rescale_the_strip(monkeypatch):
    class Screen:
        def getmaxyx(self): return 18, 40
        def erase(self): pass
        def addstr(self, *args): pass
        def move(self, *args): pass
        def clrtoeol(self): pass
        def refresh(self): pass
    ui = Surface(screen=Screen(), theme='scan-line')
    ui.motion.advance(.2)
    strips = []
    for text in ('ready', 'typing', 'tool 9s', 'streaming 100s', 'answer needed'):
        monkeypatch.setattr(ui, 'status_text', lambda: text)
        ui.draw()
        strips.append(ui.strip_cache)
    assert all(strip == strips[0] for strip in strips)


@pytest.mark.parametrize('theme', THEMES)
def test_typing_light_stays_local_to_shape(theme):
    m = MotionTheme(theme)
    assert m.gesture_light(0., .85) <= (.85 if theme == 'scan-line' else .102001)


def test_typing_preserves_brightness_velocity():
    m = MotionTheme()
    m.advance(.2)
    before, velocity = m.energy, m.energy_velocity
    m.event('typing')
    m.advance(1e-7)
    assert (m.energy - before)/1e-7 == pytest.approx(velocity, abs=1e-4)

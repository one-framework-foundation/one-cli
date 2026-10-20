import math
from types import SimpleNamespace
from onecli.motion_theme import MotionTheme, THEMES
from onecli.terminal import Surface


def test_local_theme_command_never_submits_to_model():
    ui = Surface()
    for ch in '/theme equalizer':
        ui.feed(ch)
    ui.submit()
    assert ui.turns.empty()
    assert ui.motion.theme == 'equalizer'
    ui.draft = '/theme invalid'
    ui.submit()
    assert ui.turns.empty()
    assert ui.motion.theme == 'equalizer'


def test_real_events_drive_energy_and_decay():
    quiet, active = MotionTheme(), MotionTheme()
    for _ in range(120):
        active.event('text_delta', {'chunk': 'output chunk'})
        quiet.advance(1/30)
        active.advance(1/30)
    assert active.energy > quiet.energy + .4
    assert active.frame(40, 8) != quiet.frame(40, 8)
    active.event('ready')
    for _ in range(240):
        active.advance(1/30)
    assert active.energy < .1


def test_every_theme_handles_odd_small_and_mobile_sizes():
    for theme in THEMES:
        m = MotionTheme(theme)
        m.event('submit')
        for _ in range(30):
            m.advance(1/30)
        for cols, rows in ((1, 1), (20, 3), (48, 9), (80, 10)):
            f = m.frame(cols, rows)
            assert len(f) == rows
            assert all(len(line) == cols and line.isascii() for line in f)


def test_spawn_disappears_only_after_zero_ink():
    m = MotionTheme('patternflow-port')
    m.event('submit')
    m.time = 3.99999
    before = m.ink(m.theme, .8, 0)
    m.advance(.00002)
    assert abs(before - m.ink(m.theme, .8, 0)) < 1e-8
    assert not m.spawns


def test_reduced_motion_is_static_despite_activity():
    m = MotionTheme()
    before = m.frame(40, 8, True)
    for _ in range(100):
        m.event('typing')
        m.advance(.03, True)
    assert m.phase == 0
    assert m.frame(40, 8, True) == before


def test_real_process_exit_closes_card_without_waiting_for_next_turn():
    proc = SimpleNamespace(poll=lambda: None)
    task = SimpleNamespace(status='running', proc=proc)
    m = MotionTheme()
    m.track(task)
    for _ in range(60):
        m.advance(1/30)
    assert m.cards[id(task)]['reveal'] > .99
    proc.poll = lambda: 0
    m.advance(1/30)
    assert m.cards[id(task)]['reveal'] > .8
    for _ in range(90):
        m.advance(1/30)
    assert not m.cards


def test_tracer_forwards_real_wire_events(monkeypatch):
    from onecli import terminal, cli
    ui = Surface()
    monkeypatch.setattr(terminal, 'ACTIVE', ui)
    trace = cli._tracer(True)
    trace('waiting', {'step': 1})
    assert ui.motion.state == 'thinking'
    trace('text_delta', {'kind': 'text', 'chunk': 'hello'})
    assert ui.motion.state == 'streaming'
    assert ui.motion.model > 0


def test_rapid_theme_switch_preserves_visible_frame():
    m = MotionTheme()
    m.advance(.2)
    m.select('orb-pulse')
    m.advance(.2)
    before = m.frame(48, 9)
    m.select('equalizer')
    assert m.frame(48, 9) == before


def test_short_terminal_status_tracks_wire_and_typing():
    class Screen:
        def __init__(self): self.rows = {}
        def getmaxyx(self): return 18, 40
        def erase(self): self.rows = {}
        def addstr(self, y, x, text, attr=0): self.rows[y] = text
        def move(self, *args): pass
        def clrtoeol(self): pass
        def refresh(self): pass
    screen = Screen()
    ui = Surface(screen=screen)
    ui.feed('h')
    ui.draw()
    assert 'typing' in screen.rows[14]
    ui.submit()
    assert ui.read() == 'h'
    frames = []
    for kind in ('waiting', 'text_delta', 'tool_start', 'tool_end'):
        ui.activity(kind, {'chunk': 'hello'})
        for _ in range(15):
            ui.motion.advance(1/30)
            ui.draw()
            frames.append(screen.rows[14].split(' ')[0])
        expected = {'waiting': 'thinking', 'text_delta': 'streaming',
                    'tool_start': 'tool', 'tool_end': 'thinking'}[kind]
        assert expected in screen.rows[14]
        assert len(screen.rows[14]) <= 39
    assert len(set(frames)) > 2
    ui.busy = False
    ui.draw()
    assert 'ready' in screen.rows[14]


def test_bare_theme_randomizes_locally_to_different_preset():
    ui = Surface()
    for _ in range(12):
        previous = ui.motion.theme
        ui.draft = '/theme'
        ui.submit()
        assert ui.motion.theme in THEMES
        assert ui.motion.theme != previous
        assert ui.turns.empty()


def test_weighted_blend_and_retarget_preserve_position_and_velocity():
    from onecli.motion_theme import PRESETS
    m = MotionTheme()
    assert m.select('braid-stream:0.25+orb-pulse:0.75')
    for key in m.params:
        assert math.isclose(m.target_params[key],
                            .25 * PRESETS['braid-stream'][key] + .75 * PRESETS['orb-pulse'][key])
    m.advance(.2)
    before, velocity = dict(m.params), dict(m.param_velocity)
    assert m.select('equalizer')
    assert m.params == before
    assert m.param_velocity == velocity
    m.advance(1e-6)
    for key in before:
        assert abs((m.params[key] - before[key]) / 1e-6 - velocity[key]) < .001
    for _ in range(240):
        m.advance(1/30)
    assert all(abs(m.params[k] - m.target_params[k]) < 1e-9 for k in m.params)


def test_invalid_blends_do_not_mutate_selection():
    m = MotionTheme()
    for value in ('orb-pulse:nan', 'orb-pulse:inf', 'orb-pulse:-1',
                  'orb-pulse:0', 'bad+equalizer', 'equalizer:wat'):
        before = dict(m.target_params)
        assert not m.select(value)
        assert m.theme == 'braid-stream'
        assert m.target_params == before


def test_each_theme_has_distinct_user_and_agent_currents():
    for kind in ('typing', 'waiting', 'text_delta', 'tool_start', 'answer'):
        fields = []
        for theme in THEMES:
            m = MotionTheme(theme)
            m.event(kind, {'chunk': 'hello', 'x': .3})
            m.advance(.2)
            fields.append(tuple(round(v, 7) for x, y in ((.3, .35), (.1, .2), (-.4, .6))
                                for v in m.fluid(x, y)))
        assert len(set(fields)) == len(THEMES), kind


def test_wave_birth_and_death_are_continuous():
    m = MotionTheme()
    before = m.fluid(.1, .2)
    m.event('typing')
    assert m.fluid(.1, .2) == before
    m.time = 3 - 1e-6
    before = m.fluid(.1, .2)
    m.advance(2e-6)
    assert not m.ripples
    assert max(abs(a-b) for a,b in zip(before, m.fluid(.1, .2))) < 1e-8


def test_single_keystroke_is_visible_within_two_frames_on_mobile_strip():
    # Compare identical clocks: idle animation must not count as input response.
    for theme in THEMES:
        quiet, active = MotionTheme(theme), MotionTheme(theme)
        for _ in range(60):
            quiet.advance(1/30)
            active.advance(1/30)
        before = active.frame(18, 1)
        active.event('typing', {'x': -.7})
        assert active.frame(18, 1) == before  # no discontinuous event jump
        for _ in range(2):
            quiet.advance(1/30)
            active.advance(1/30)
        assert active.frame(18, 1) != quiet.frame(18, 1), theme


def test_mobile_projection_distinguishes_agent_states_for_every_theme():
    for theme in THEMES:
        signatures = []
        for kind in ('ready', 'waiting', 'text_delta', 'tool_start', 'answer'):
            m = MotionTheme(theme)
            m.event(kind, {'chunk': 'model output'})
            frames = []
            for i in range(60):
                m.advance(1/30)
                if i >= 15:
                    frames.append(m.frame(18, 1)[0])
            signatures.append(tuple(frames))
        assert len(set(signatures)) == 5, theme


def test_state_retarget_has_no_field_jump_or_velocity_reset():
    m = MotionTheme()
    m.event('waiting')
    m.advance(.1)
    before, velocity = dict(m.flow), dict(m.flow_velocity)
    field = m.ink(m.theme, .2, .3)
    m.event('tool_start')
    assert m.ink(m.theme, .2, .3) == field
    assert m.flow_velocity == velocity
    m.advance(1e-7)
    for key in before:
        assert abs((m.flow[key] - before[key]) / 1e-7 - velocity[key]) < .001


def test_color_field_retargets_continuously_and_reduced_stays_static():
    m = MotionTheme()
    m.advance(.2)
    before = m.color(.3, .2, .5)
    m.select('equalizer')
    m.event('tool_start')
    assert m.color(.3, .2, .5) == before
    m.advance(1e-7)
    assert max(abs(a-b) for a, b in zip(before, m.color(.3, .2, .5))) < .001
    m.frame(24, 4, True)
    colors = m.colors
    m.event('typing')
    m.advance(.2, True)
    m.frame(24, 4, True)
    assert m.colors == colors


def test_color_tracks_gesture_and_agent_activity_for_all_themes():
    for theme in THEMES:
        quiet, active = MotionTheme(theme), MotionTheme(theme)
        for m in (quiet, active):
            for _ in range(60):
                m.advance(1/30)
        active.event('typing', {'x': .3})
        for _ in range(2):
            quiet.advance(1/30)
            active.advance(1/30)
        quiet.frame(40, 6)
        active.frame(40, 6)
        assert active.colors != quiet.colors, theme
        signatures = []
        for kind in ('waiting', 'text_delta', 'tool_start'):
            m = MotionTheme(theme)
            m.event(kind, {'chunk': 'model output'})
            for _ in range(20):
                m.advance(1/30)
            m.frame(40, 6)
            signatures.append(str(m.colors))
        assert len(set(signatures)) == 3, theme


def test_idle_is_neutral_and_activity_colour_fades_without_event_jump():
    for theme in THEMES:
        m = MotionTheme(theme)
        for _ in range(90):
            m.advance(1/30)
        assert m.color(.2, .3, .5) == (135, 135, 135)
        idle = m.frame(48, 9)
        m.advance(.2)
        assert m.frame(48, 9) != idle  # idle still breathes
        before = m.color(.2, .3, .5)
        m.event('text_delta', {'chunk': 'a' * 200})
        assert m.color(.2, .3, .5) == before
        for _ in range(30):
            m.advance(1/30)
        rgb = m.color(.2, .3, .5)
        assert max(rgb) - min(rgb) > 10
        m.event('ready')
        assert m.color(.2, .3, .5) == rgb
        for _ in range(300):
            m.advance(1/30)
        assert max(abs(v - 135) for v in m.color(.2, .3, .5)) < .001


def test_streaming_leaves_air_around_shapes_and_equalizer_is_not_filled():
    for theme in THEMES:
        m = MotionTheme(theme)
        for _ in range(90):
            m.event('text_delta', {'chunk': 'x' * 200})
            m.advance(1/30)
        rows = m.frame(60, 10)
        assert sum(ch != ' ' for row in (rows[0], rows[-1]) for ch in row) < 30, theme
    m = MotionTheme('equalizer')
    m.flow['streaming'] = 1
    # At a tall peak, the column interior remains empty.
    assert m.base_ink('equalizer', math.pi / 38, 0) < .001


def test_small_palette_idle_animation_is_neutral():
    ui = Surface()
    ui.accents = {'ready': 123, 'streaming': 456}
    assert ui.animation_attr((135, 135, 135)) == 0
    ui.motion.state = 'streaming'
    assert ui.animation_attr((80, 160, 190)) == 456


def test_scan_typing_embers_start_at_head_without_warp_or_event_jump():
    m = MotionTheme('scan-line')
    for _ in range(60):
        m.advance(1/30)
    head = m.scan_head(m.phase)
    before = m.frame(48, 9)
    m.event('typing', {'x': -.9})
    assert m.frame(48, 9) == before
    assert len(m.embers) == 3
    assert all(ember[1:3] == head for ember in m.embers)
    m.advance(.08)
    assert any(m.ember_state(e)[2] > .5 for e in m.embers)
    assert all(m.fluid(x, y)[:2] == (0., 0.)
               for x, y in (head, (0., 0.), (-.9, .35)))


def test_scan_ember_birth_and_expiry_preserve_field_velocity():
    m = MotionTheme('scan-line')
    m.event('typing')
    ember = m.embers[0]
    x, y, _ = m.ember_state(ember)
    m.time = 1e-6
    assert abs(m.fluid(x, y)[2]) / 1e-6 < 1e-6
    m.time = 1.4 - 1e-6
    x, y, opacity = m.ember_state(ember)
    before = m.fluid(x, y)[2]
    assert opacity / 1e-6 < 1e-6
    m.advance(2e-6)
    assert not m.embers
    assert abs(before - m.fluid(x, y)[2]) / 2e-6 < 1e-6


def test_scan_embers_stay_bounded_during_paste_and_reduced_motion_is_static():
    m = MotionTheme('scan-line')
    before = m.frame(18, 1, True)
    for _ in range(600):
        for _ in range(10):
            m.event('typing')
        m.advance(1/30, True)
    assert len(m.embers) <= 96
    assert m.frame(18, 1, True) == before

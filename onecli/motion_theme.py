"""Local ASCII themes. Real activity drives envelopes, never model context.

Continuous scalar fields are quantized only at the final character boundary.
These are terminal adaptations of the five experiments, not a full Patternflow port.
"""
from __future__ import annotations
import math
import random

THEMES = ('patternflow-port', 'orb-pulse', 'scan-line', 'equalizer', 'braid-stream')
PALETTE = ' .,:;-~+=*'


# Every preset uses the same numeric parameter space, including shape weights.
# Interpolation changes currents and geometry together without restarting clocks.
FLOW_KEYS = ('swirl', 'ribbon', 'eddies', 'breath', 'wave_speed',
             'wave_frequency', 'wave_width', 'wave_x', 'wave_y', 'wave_glow')
_FLOW = (
    (.10, .08, .10, .26, .62, 14., 7., .22, .22, .8),
    (.24, .06, .12, .30, .85, 10., 5., .26, .26, .55),
    (.06, .32, .08, .10, 1.1, 18., 9., .32, .06, .6),
    (.04, .20, .30, .12, .55, 20., 8., .04, .40, .65),
    (.22, .28, .20, .16, .72, 12., 6., .24, .16, .65),
)
PRESETS = {name: dict(zip(FLOW_KEYS, values), **{
    'shape_' + shape: float(shape == name) for shape in THEMES})
    for name, values in zip(THEMES, _FLOW)}

# RGB endpoints share the same damped parameter space as the geometry.
# Interpolating endpoints avoids palette snaps when a blend is retargeted.
for name, ends in zip(THEMES, (
    ((70, 155, 145), (180, 130, 215)),
    ((100, 140, 215), (215, 130, 180)),
    ((65, 160, 155), (190, 195, 110)),
    ((85, 160, 195), (225, 155, 100)),
    ((80, 165, 185), (190, 135, 215)),
)):
    for end, rgb in enumerate(ends):
        PRESETS[name].update({f'color_{end}_{c}': v for c, v in enumerate(rgb)})


class MotionTheme:
    def __init__(self, theme='braid-stream'):
        requested = theme
        self.theme = theme if theme in THEMES else 'braid-stream'
        self.params = dict(PRESETS[self.theme])
        self.target_params = dict(self.params)
        self.param_velocity = dict.fromkeys(self.params, 0.0)
        self.previous = self.theme
        self.weights = {self.theme: 1.0}
        self.time = 0.0
        self.phase = 0.0
        self.user = self.model = self.energy = 0.0
        self.energy_velocity = 0.0
        self.level = 0.0
        self.state = 'ready'
        self.spawns = []
        self.cards = {}
        self.ripples = []
        self.embers = []
        self.flow = {k: 0.0 for k in ('thinking', 'streaming', 'tool', 'answer needed')}
        self.flow_velocity = dict(self.flow)
        self.ripple_clock = -1.0
        self.colors = []
        self.chroma = self.chroma_velocity = 0.0
        self._field_cache = None
        if requested not in THEMES and self.select(requested):
            self.params = dict(self.target_params)

    def color(self, x, y, ink, glow=0.0, reduced=False):
        """Continuous colour advection, quantized only by the terminal adapter."""
        q, p = self.params, self.phase
        if reduced:
            return (135, 135, 135)
        coordinates = (
            math.atan2(y, x) * 2 + math.hypot(x, y) * 7 - p * 2,
            math.atan2(y, x) + p * 1.5,
            x * 5 - y * 2 - p * 3,
            x * 7 + abs(y) * 3 - p * 2,
            x * 5 + y * 7 - p * 2,
        )
        mix = sum(q['shape_' + name] * (.5 + .5 * math.sin(coord))
                  for name, coord in zip(THEMES, coordinates))
        # Native gesture light warms the touched current, output lifts cyan,
        # tools lift amber. All drivers are continuous renderer state.
        warmth = math.tanh(abs(glow))
        tool, stream = self.flow['tool'], self.flow['streaming']
        gain = .72 + .23 * self.energy + .18 * min(1., max(0., ink))
        rgb = []
        for c in range(3):
            value = q[f'color_0_{c}'] * (1 - mix) + q[f'color_1_{c}'] * mix
            value += warmth * (65, 35, 10)[c]
            value += tool * (30, 10, -25)[c] + stream * (-10, 20, 30)[c]
            # Idle geometry stays alive, but all colour belongs to activity.
            rgb.append(135 + self.chroma * (max(0., min(255., value * gain)) - 135))
        return tuple(rgb)

    def select(self, theme):
        """Select a preset, or a weighted blend: braid-stream:0.7+orb-pulse:0.3."""
        try:
            weights = {}
            for term in theme.split('+'):
                name, sep, weight = term.partition(':')
                value = float(weight) if sep else 1.0
                if name not in PRESETS or not math.isfinite(value) or value < 0:
                    return False
                weights[name] = weights.get(name, 0.0) + value
            total = sum(weights.values())
            if total <= 0:
                return False
        except (ValueError, TypeError):
            return False
        self.previous, self.theme = self.theme, theme
        self.weights = {k: v / total for k, v in weights.items()}
        self.target_params = {key: sum(PRESETS[name][key] * weight
                                      for name, weight in self.weights.items())
                              for key in self.params}
        # Preserve current values AND velocities on rapid retargeting.
        return True

    def randomize(self):
        self.select(random.choice([name for name in THEMES if name != self.theme]))
        return self.theme

    def event(self, kind, data=None):
        data = data or {}
        if kind in ('typing', 'submit'):
            self.user = min(1.0, self.user + .35)
            if self.time - self.ripple_clock >= .045 or kind == 'submit':
                self.ripples.append((self.time, max(-.9, min(.9, data.get('x', 0.0))),
                                     2.2 if kind == 'submit' else 1.65))
                # Capture the tracer, not the input cursor. Particles keep their
                # own trajectory as the head moves or the theme is retargeted.
                if max(self.params['shape_scan-line'],
                       self.target_params['shape_scan-line']) > 1e-9:
                    sx, sy = self.scan_head(self.phase)
                    for i in range(5 if kind == 'submit' else 3):
                        angle = self.phase * 2.3 + i * 2.39996
                        self.embers.append((self.time, sx, sy,
                                            .36 * math.cos(angle),
                                            .3 * math.sin(angle),
                                            1.2 if kind == 'submit' else .85))
                self.ripple_clock = self.time
            if kind == 'submit':
                self.spawns.append(self.time)
        elif kind == 'text_delta':
            self.model = min(1.0, self.model + min(.4, len(data.get('chunk', '')) / 80))
            self.state = 'thinking' if data.get('kind') == 'thinking' else 'streaming'
        elif kind in ('waiting', 'tool_start'):
            self.state = 'thinking' if kind == 'waiting' else 'tool'
        elif kind in ('ready', 'tool_end', 'tool_denied'):
            self.state = 'ready' if kind == 'ready' else 'thinking'
        elif kind == 'answer':
            self.state = 'answer needed'

    def track(self, task):
        # Retain actual task objects; completion follows process exit or task status.
        key = id(task)
        if key not in self.cards:
            self.cards[key] = {'task': task, 'reveal': 0.0, 'reveal_velocity': 0.0,
                               'age': 0.0, 'closed': False}

    def advance(self, dt, reduced=False):
        dt = max(0.0, float(dt))
        if not math.isfinite(dt):
            return
        # A slow redraw must not discard elapsed time or extend particle life.
        if dt > .05:
            steps = math.ceil(dt / .05)
            for _ in range(steps):
                self.advance(dt / steps, reduced)
            return
        self.time += dt
        for key, value in self.params.items():
            target = self.target_params[key]
            offset = value - target
            c = self.param_velocity[key] + 4 * offset
            decay = math.exp(-4 * dt)
            self.params[key] = target + (offset + c * dt) * decay
            self.param_velocity[key] = (self.param_velocity[key] - 4 * c * dt) * decay
        # Critically damped state weights preserve both value and velocity on
        # retarget. Events change targets, never the visible field directly.
        for key, value in self.flow.items():
            target = float(self.state == key)
            offset, velocity = value - target, self.flow_velocity[key]
            decay = math.exp(-11 * dt)
            c = velocity + 11 * offset
            self.flow[key] = target + (offset + c * dt) * decay
            self.flow_velocity[key] = (velocity - 11 * c * dt) * decay
        self.ripples = [r for r in self.ripples if self.time - r[0] < 3.0]
        self.embers = [r for r in self.embers if self.time - r[0] < 1.4]
        self.user *= math.exp(-dt * 2.5)
        self.model *= math.exp(-dt * 2.0)
        target_chroma = max(self.user, self.model,
                            float(self.state in self.flow))
        offset = self.chroma - target_chroma
        decay = math.exp(-8 * dt)
        c = self.chroma_velocity + 8 * offset
        self.chroma = target_chroma + (offset + c * dt) * decay
        self.chroma_velocity = (self.chroma_velocity - 8 * c * dt) * decay
        target = max(self.user, self.model, .35 if self.state in ('thinking', 'tool') else .08)
        offset = self.energy - target
        c = self.energy_velocity + 24 * offset
        decay = math.exp(-24 * dt)
        self.energy = target + (offset + c * dt) * decay
        self.energy_velocity = (self.energy_velocity - 24 * c * dt) * decay
        if not reduced:
            self.phase += dt * (.3 + self.energy * 1.5)
        self.level += (1 - self.level) * -math.expm1(-dt * 4)
        self.spawns = [t for t in self.spawns if self.time - t < 4]
        for key, card in list(self.cards.items()):
            task = card['task']
            running = task.status == 'running' and (task.proc is None or task.proc.poll() is None)
            card['age'] += dt
            target = 1.0 if running else 0.0
            offset = card['reveal'] - target
            c = card['reveal_velocity'] + 8 * offset
            decay = math.exp(-8 * dt)
            card['reveal'] = target + (offset + c * dt) * decay
            card['reveal_velocity'] = (card['reveal_velocity'] - 8 * c * dt) * decay
            if not running and card['age'] > 1.2 and card['reveal'] < .005:
                del self.cards[key]

    def field_inputs(self):
        """Prepare age windows once per frame, never once per terminal cell."""
        gestures = []
        styles = [(style, self.params['shape_' + style]) for style in THEMES
                  if style != 'scan-line' and self.params['shape_' + style] > 1e-9]
        if styles:
            for born, origin, strength in self.ripples:
                age = self.time - born
                a = min(1.0, max(0.0, age / .065))
                b = min(1.0, max(0.0, (3 - age) / .9))
                envelope = a**3*(10+a*(-15+6*a)) * b**3*(10+b*(-15+6*b))
                gestures.append((age, origin, strength * envelope * math.exp(-age * .6)))
        embers = [self.ember_state(e) for e in self.embers]
        return styles, gestures, embers

    def fluid(self, x, y):
        """Blend theme-native impulses with smooth status-specific currents."""
        p = self.phase
        q = self.params
        thinking, streaming, tool, answer = (self.flow[k] for k in
                                             ('thinking', 'streaming', 'tool', 'answer needed'))
        # Closed circulating currents for thinking, travelling ribbons for output,
        # paired eddies for tools, a slow breathing pool while awaiting an answer.
        dx = q['swirl'] * thinking * math.sin(y * 3 + p)
        dy = q['swirl'] * thinking * math.cos(x * 3 + p)
        dy += q['ribbon'] * streaming * math.sin(x * 5 - p * 3)
        dx += q['eddies'] * tool * math.sin(y * 6 + p * 2)
        dy += q['eddies'] * tool * math.sin(x * 6 - p * 2)
        glow = q['breath'] * answer * math.exp(-(x*x + y*y) * 3) * (.7 + .3 * math.sin(p))
        styles, gestures, embers = self._field_cache or self.field_inputs()
        for ex, ey, opacity in embers:
            glow += q['shape_scan-line'] * opacity * math.exp(
                -((x-ex)*19)**2 - ((y-ey)*15)**2)
        gesture_x = gesture_y = gesture_glow = 0.0
        for age, origin, gain in gestures:
            u, v = x - origin, y - .35
            for style, weight in styles:
                ix, iy, light = self.impulse(style, u, v, age)
                gesture_x += gain * weight * ix
                gesture_y += gain * weight * iy
                # Signed light used to punch black holes through the motif.
                # A smooth positive lobe keeps the local gesture visible.
                gesture_glow += gain * weight * light * light
        return (dx + .12 * math.tanh(gesture_x / .12),
                dy + .12 * math.tanh(gesture_y / .12),
                .85 * math.tanh((glow + gesture_glow) / .85))

    def scan_head(self, clock):
        return (.8 * math.cos(clock) + .1 * self.flow['streaming'] * math.sin(clock * 4),
                .5 * math.sin(2 * clock) + .2 * self.flow['tool'] * math.sin(clock * 6))

    def ember_state(self, ember):
        born, x, y, vx, vy, strength = ember
        age = max(0., self.time - born)
        rise = min(1., age / .065)
        fade = min(1., max(0., (1.4 - age) / .8))
        def window(t):
            return t**3 * (10 + t * (-15 + 6*t))
        return (x + vx * age, y + vy * age - .12 * age**2,
                strength * window(rise) * window(fade))

    def impulse(self, style, u, v, age):
        """Each gesture has its own geometry, blended by the shape weights.

        Only orb uses a radial packet. Others are local curls, a horizontal
        wake, vertical plucks, and counter-moving strand disturbances.
        """
        q = self.params
        travel = age * q['wave_speed']
        frequency = q['wave_frequency']
        width = q['wave_width']
        if style == 'orb-pulse':
            radius = math.hypot(u, v)
            pulse = math.exp(-((radius - travel) * width)**2)
            pulse *= math.cos(radius * frequency - age * 8)
            return u * pulse * q['wave_x'], v * pulse * q['wave_y'], pulse * q['wave_glow']
        if style == 'patternflow-port':
            # A compact rotating curl, with no expanding front.
            local = math.exp(-(u*u + v*v) * width)
            curl = local * math.sin(age * 5 + u * frequency * .3)
            return -v * curl * q['wave_x'], u * curl * q['wave_y'], local * math.sin(age * 4) * q['wave_glow']
        if style == 'scan-line':
            # Typing emits light from the head without warping its path.
            return 0.0, 0.0, 0.0
        if style == 'equalizer':
            pluck = math.exp(-(u * width)**2) * math.sin(age * 8)
            return 0.0, v * pluck * q['wave_y'], pluck * math.exp(-(v * 2)**2) * q['wave_glow']
        # Braid: disturbances travel in opposite directions on two strands.
        upper = math.exp(-((u - travel) * width)**2 - ((v - .22) * 6)**2)
        lower = math.exp(-((u + travel) * width)**2 - ((v + .22) * 6)**2)
        twist = math.sin(u * frequency - age * 5)
        return (upper - lower) * q['wave_x'], (upper + lower) * twist * q['wave_y'], (upper - lower) * twist * q['wave_glow']

    def ink(self, style, x, y):
        dx, dy, glow = self.fluid(x, y)
        shape = self.base_ink(style, x + dx, y + dy)
        return shape + self.gesture_light(shape, glow)

    def gesture_light(self, shape, glow):
        # Keep non-particle gestures on the motif. Sustained orb/curl typing
        # otherwise paints a bright slab over the entire scene.
        scan = self.params['shape_scan-line']
        return glow * (scan + (1 - scan) * (.12 + .88 * math.tanh(max(0., shape) * 2)))

    def base_ink(self, style, x, y):
        p = self.phase
        think, stream, tool, answer = (self.flow[k] for k in
                                      ('thinking', 'streaming', 'tool', 'answer needed'))
        d = math.hypot(x, y)
        if style == 'braid-stream':
            # Actual strands: output travels along them, tool work winds them
            # together, waiting opens a slower, wider weave.
            ink = 0.0
            for i in range(3):
                angle = x * (4 + 2 * tool) - p * 2 + i * 2.094
                center = (.24 + .12 * think) * math.sin(angle)
                center += .09 * tool * math.sin(x * 11 + p * 4 + i)
                bead = .5 + .5 * math.cos(x * 13 - p * 7 + i * 2.094)
                ink += math.exp(-((y - center) * 12)**2) * (.35 + .5 * stream * bead)
            return ink * (1 + .18 * answer * math.sin(p * 2))
        if style == 'orb-pulse':
            angle = math.atan2(y, x)
            radius = .43 + .04 * math.sin(p * 2)
            radius += .08 * think * math.sin(angle * 2 - p * 2)
            radius += .09 * tool * math.sin(angle * 4 + p * 4)
            radius += .08 * stream * math.sin(angle * 3 - p * 5)
            radius += .06 * answer * math.sin(p * 2)
            rim = .55 + .3 * (.5 + .5 * math.cos(angle * 2 - p * 3))
            return rim * math.exp(-((d - radius) * 13)**2) + .09 * math.exp(-d * 3)
        if style == 'equalizer':
            carrier = .5 + .5 * math.sin(x * 12 + p * 2)
            syllables = (.5 + .5 * math.sin(x * 19 - p * 8))
            amp = .06 + .13 * carrier + .18 * think * carrier
            amp += .48 * stream * syllables + .4 * tool * (.5 + .5 * math.sin(x * 23 + p * 5))
            amp += .12 * answer * (.5 + .5 * math.sin(p * 2))
            # Trace the two edges; filled columns become slabs during output.
            return .72 * math.exp(-((abs(y) - amp) * 18)**2)
        if style == 'scan-line':
            ink = 0.0
            for j in range(8):
                clock = p - j * .085
                # Status weights alter geometry, never multiply the clock:
                # retargeting cannot teleport the scan head.
                sx, sy = self.scan_head(clock)
                ink += math.exp(-((x-sx)**2 * (65 - 30 * stream) + (y-sy)**2 * 65)) * (1-j/8) / 2
            return ink
        # Patternflow keeps a visible circulating seed during model work.
        angle = math.atan2(y, x)
        curl = .5 + .5 * math.sin(angle * 3 + d * 13 - p * 3)
        activity = .12 + .32 * think + .55 * stream + .65 * tool
        seed = activity * math.exp(-((d - .38 - .08 * math.sin(angle * 3 + p * 2)) * 6)**2) * curl
        # Birth AND death fade to zero before removing a spawn. No age-cut pop.
        ink = seed + .08 * math.exp(-d * 4)
        for born in self.spawns:
            age = self.time - born
            a = min(1, max(0, age / 1.2))
            a = a * a * a * (10 + a * (-15 + 6 * a))
            fade = min(1, max(0, (4 - age) / 1.2))
            fade = fade ** 3 * (10 + fade * (-15 + 6 * fade))
            ink = max(ink, a * fade * math.exp(-((d - .8 * a) * 7) ** 2))
        return ink

    def frame(self, cols, rows, reduced=False, reveal=None):
        cols, rows = max(1, cols), max(1, rows)
        level = (1 if reduced else self.level) if reveal is None else reveal
        result = []
        self.colors = []
        self._field_cache = self.field_inputs()
        for r in range(rows):
            line = ''
            colors = []
            for c in range(cols):
                x, y = (c + .5) / cols * 2 - 1, (r + .5) / rows * 2 - 1
                color_glow = 0.0
                if reduced:
                    ink = .35 * math.exp(-math.hypot(x, y) * 2)
                else:
                    # Project the whole field onto a one-row status strip;
                    # a centre slice loses the moving scan whenever it leaves y=0.
                    ys = (-.8, -.6, -.4, -.2, 0, .2, .4, .6, .8) if rows == 1 else (y,)
                    samples = []
                    for sy in ys:
                        dx, dy, glow = self.fluid(x, sy)
                        color_glow += abs(glow) / len(ys)
                        shape = sum(
                            self.base_ink(style, x + dx, sy + dy) * self.params['shape_' + style]
                            for style in THEMES if self.params['shape_' + style] > 1e-9)
                        # Output highlights ride the shape, leaving empty space
                        # empty. Keep the travelling light in the colour field.
                        light = self.flow['streaming'] * (.5 + .5 * math.sin(x * 5 - self.phase * 3))
                        color_glow += .17 * light * max(0., shape) / len(ys)
                        samples.append(self.gesture_light(shape, glow) + shape * (1 + .12 * light))
                    ink = (1.5 * (.25 * max(samples) + .75 * sum(samples) / len(samples))
                           if rows == 1 else samples[0])
                ink *= level * (.5 if reduced else .5 + self.energy * .5)
                colors.append(self.color(x, y, ink, color_glow, reduced))
                line += PALETTE[round(max(0, min(1, ink)) * (len(PALETTE) - 1))]
            result.append(line)
            self.colors.append(colors)
        self._field_cache = None
        return result

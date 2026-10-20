import curses
import os
from types import SimpleNamespace

from onecli import terminal
from onecli.terminal import Surface


class Clock:
    now = 1000.0


class Screen:
    """Fake curses screen; get_wch advances a fake clock by the poll timeout."""
    def __init__(self, clock, until, script=()):
        self.clock, self.until, self.script = clock, until, dict(script)
        self.wait, self.polls, self.stop = 16, 0, None
    def getmaxyx(self): return 30, 80
    def clearok(self, flag): pass
    def timeout(self, ms): self.wait = ms
    def get_wch(self):
        self.polls += 1
        self.clock.now += max(self.wait, 1) / 1000
        if self.clock.now >= self.until:
            self.stop()
            raise curses.error
        action = self.script.pop(round(self.clock.now - 1000), None)
        if action:
            action()
        raise curses.error


def run_surface(monkeypatch, seconds, script=(), busy=False):
    clock = Clock()
    ui = Surface(screen=Screen(clock, 1000.0 + seconds), theme='scan-line')
    ui.busy = busy
    monkeypatch.setattr(terminal.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(terminal.sys, 'stdin', SimpleNamespace(fileno=lambda: 0))
    monkeypatch.setattr(terminal.os, 'get_terminal_size',
                        lambda fd: os.terminal_size((80, 30)))
    draws, advances = [], []
    monkeypatch.setattr(ui, 'draw', lambda: draws.append(clock.now))
    real = ui.motion.advance
    monkeypatch.setattr(ui.motion, 'advance', lambda dt, r=False: (advances.append(dt), real(dt, r)))
    ui.screen.script = {k: (lambda f=f: f(ui)) for k, f in dict(script).items()}
    ui.screen.stop = ui.stopped.set
    ui.run()
    assert not ui.lines or all('terminal stopped' not in line for line in ui.lines)
    return ui, draws, advances


def test_idle_prompt_neither_draws_nor_advances_nor_spins(monkeypatch):
    ui, draws, advances = run_surface(monkeypatch, 60)
    # Only the short grace period after the surface appears may animate.
    grace = 1000 + terminal.LIVE_GRACE + .1
    assert all(t <= grace for t in draws)
    assert len(advances) == len(draws)
    assert ui.screen.polls <= 1000  # a spinning loop polls ~3750 times in 60s


def test_output_while_idle_redraws_promptly(monkeypatch):
    ui, draws, _ = run_surface(monkeypatch, 30, {20: lambda ui: ui.write('hello\n')})
    late = [t for t in draws if t >= 1020]
    assert late and late[0] - 1020 <= .25


def test_keypress_while_idle_redraws(monkeypatch):
    ui, draws, _ = run_surface(monkeypatch, 30, {20: lambda ui: ui.feed('x')})
    assert any(t >= 1020 for t in draws)


def test_active_run_keeps_frame_rate(monkeypatch):
    _, draws, advances = run_surface(monkeypatch, 10, busy=True)
    assert len(draws) >= 250 and len(advances) >= 250


def test_startup_paints_the_prompt_before_any_input(monkeypatch):
    """Startup once remained idle, so the prompt never drew until a keypress."""
    _, draws, _ = run_surface(monkeypatch, 5)
    assert draws and draws[0] - 1000 <= .25

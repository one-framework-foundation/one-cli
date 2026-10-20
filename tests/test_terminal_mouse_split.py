"""Raw SGR mouse reports must never reach the input line (test host, tmux over ssh).

terminfo's kmous for xterm and tmux is `ESC [ <`. With keypad on, curses eats
that prefix and returns KEY_MOUSE, so the rest of the report (`65;52;15M`)
arrived as typed text. These tests drive a real curses screen on a pty with
TERM=xterm-256color, feeding a report split at every byte boundary.
"""
import curses
import os
import pty
import sys
import time

import pytest

from onecli import terminal
from onecli.terminal import Surface

REPORTS = b'\x1b[<65;52;15M\x1b[<0;84;28M\x1b[<0;84;28m'

CHILD = r'''
import curses, os, sys, time
sys.path.insert(0, os.getcwd())
from onecli.terminal import Surface
out = sys.argv[1]
def main(screen):
    curses.set_escdelay(25)
    curses.raw()
    screen.keypad(True)
    screen.timeout(20)
    ui = Surface('t', screen)
    end = time.monotonic() + float(sys.argv[2])
    while time.monotonic() < end:
        ui.flush_escape()
        try:
            key = screen.get_wch()
        except curses.error:
            continue
        ui.feed(key)
    open(out, 'w').write(repr((ui.draft, ui.escape)))
curses.wrapper(main)
'''


def drive(tmp_path, chunks, gap, marker='k', lifetime=4):
    """One pty session; every group of chunks is followed by a typed marker."""
    script, out = tmp_path / 'child.py', tmp_path / 'out.txt'
    script.write_text(CHILD)
    pid, fd = pty.fork()
    if pid == 0:
        os.environ['TERM'] = 'xterm-256color'
        os.execv(sys.executable, [sys.executable, str(script), str(out), str(lifetime)])
    time.sleep(.6)
    for group in chunks:
        for chunk in group:
            os.write(fd, chunk)
            time.sleep(gap)
        os.write(fd, marker.encode())
        time.sleep(.03)
    os.waitpid(pid, 0)
    os.close(fd)
    return eval(out.read_text())


@pytest.mark.parametrize('gap', [0, .04])
def test_report_split_at_every_byte_never_reaches_the_draft(tmp_path, gap):
    cuts = range(1, len(REPORTS))
    draft, escape = drive(tmp_path, [[REPORTS[:c], REPORTS[c:]] for c in cuts], gap,
                          lifetime=3 + len(cuts) * (2 * gap + .1))
    assert (draft, escape) == ('k' * len(cuts), '')


def test_report_sent_byte_by_byte(tmp_path):
    draft, escape = drive(tmp_path, [[REPORTS[i:i + 1] for i in range(len(REPORTS))]], .04, lifetime=3)
    assert (draft, escape) == ('k', '')


def test_mouse_key_resumes_the_report_from_its_prefix():
    ui = Surface()
    ui.feed(curses.KEY_MOUSE)
    for c in '64;10;10M':
        ui.feed(c)
    assert (ui.draft, ui.escape) == ('', '')


def test_unfinished_mouse_report_is_dropped_not_typed():
    ui = Surface()
    ui.feed(curses.KEY_MOUSE)
    ui.feed('6')
    ui.flush_escape(time.monotonic() + 1)
    ui.feed('a')
    assert (ui.draft, ui.escape) == ('a', '')


def test_lone_escape_still_works():
    ui = Surface()
    ui.feed('\x1b')
    ui.flush_escape(time.monotonic() + 1)
    assert ui.escape == ''


@pytest.mark.parametrize('value, expected', [('1', False), ('0', True), (None, True)])
def test_one_no_mouse_disables_reporting_in_tmux(monkeypatch, value, expected):
    class Shown:
        stdout = 'on\n'
    monkeypatch.setenv('TMUX', '/tmp/x,1,0')
    monkeypatch.delenv('ONE_NO_MOUSE', raising=False)
    if value is not None:
        monkeypatch.setenv('ONE_NO_MOUSE', value)
    monkeypatch.setattr(terminal.subprocess, 'run', lambda *a, **k: Shown())
    assert terminal.tmux_mouse() is expected

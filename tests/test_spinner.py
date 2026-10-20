"""The spinner's only interesting claim is that it does NOT move on its own.

`ANIM.md`: stale data cannot animate, and MISSING is deliberately left still.
That is the whole reason this file exists rather than an itertools.cycle over
some braille, so it is what the tests are about.
"""

import io
import re
import time
import types

import pytest

from onecli import render, spinner

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _paint(monkeypatch):
    for key in ("NO_COLOR", "ONE_CLI_PLAIN", "FORCE_COLOR", "COLORTERM", "TERM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setenv("TERM", "xterm-256color")
    render.refresh()
    yield
    # Undo the test's environment changes too, before refreshing cached state.
    monkeypatch.undo()
    render.refresh()


def _spin(motif=spinner.ARRIVE, **kw):
    return spinner.Spinner(motif, enabled=False, seed=7, **kw)


def _plain(frame):
    return ANSI.sub("", frame)


def test_a_fed_spinner_scrambles():
    s = _spin()
    seen = set()
    for _ in range(4):
        s.feed(10)
        s._frame()
        seen.add(tuple(s._cells))
    assert len(seen) == 4


def test_an_unfed_spinner_freezes_and_says_how_long_it_has_waited():
    """The one thing every other spinner gets wrong."""
    s = _spin()
    s.feed(5)
    s._frame()
    s._last_feed -= spinner.STALL_AFTER + 3
    before = list(s._cells)
    frame = s._frame()
    assert s._cells == before, "a stalled spinner must hold its characters"
    assert "waiting" in frame


def test_a_stalled_spinner_is_greyed_not_merely_paused():
    s = _spin()
    s._last_feed -= 60
    r, g, b = render.SEED["faint"]
    assert f"38;2;{r};{g};{b}" in s._frame()
    ar, ag, ab = render.SEED["accent"]
    assert f"38;2;{ar};{ag};{ab}" not in s._frame()


def test_an_ungated_motif_keeps_moving_because_its_clock_was_never_data():
    """`work` watches a subprocess it cannot see inside. Time is all it has,
    and its readout says so - so it is allowed to churn without a feed."""
    s = _spin(spinner.WORK)
    s._last_feed -= 60
    before = list(s._cells)
    for _ in range(3):
        s._frame()
    assert s._cells != before
    assert "waiting" not in s._frame()


def test_the_ungated_pace_is_even_because_it_is_not_a_reading():
    """`work` holds a clock, not a rate. An even churn is what that looks like;
    a speed that varied would be a claim about progress it cannot see."""
    s = _spin(spinner.WORK)
    s.feed(5000)                       # ignored: nothing gates on it
    assert s._speed(time.monotonic()) == spinner.WORK_RATE
    assert s._speed(time.monotonic() + 10) == spinner.WORK_RATE


def test_a_fast_stream_churns_harder_than_a_trickle():
    """The claim the rest of the file rests on: the motion IS the reading."""
    def rolls(per_frame, frames=40):
        s = _spin()
        n = 0
        for _ in range(frames):
            s.feed(per_frame)
            s._last_frame -= spinner.FRAME     # a frame's worth of wall clock
            before = list(s._cells)
            s._frame()
            n += sum(a != b for a, b in zip(before, s._cells))
        return n

    trickle, stream = rolls(4), rolls(400)
    assert stream > trickle * 2, (trickle, stream)
    assert trickle > 0, "a trickle flickers; it does not freeze"


def test_every_roll_is_visible():
    """A re-roll onto the same character is a frame where nothing happened,
    and that reading is reserved for a stall."""
    s = _spin()
    for _ in range(60):
        before = list(s._cells)
        s._roll(1, time.monotonic())
        assert s._cells[1] != before[1]


def test_it_is_three_characters_wide():
    """A wait indicator is not a progress bar. The label and the readout are
    the only other things on the line."""
    s = _spin(label="qwen3.8-27b")
    s.feed(20)
    body = _plain(s._frame()).split("  ")
    assert body[0] == "" and len(body[1]) == spinner.CELLS
    assert body[2:] == ["qwen3.8-27b", "20"]


def test_the_gated_readout_counts_what_arrived_and_the_ungated_one_counts_time():
    fed = _spin()
    fed.feed(1500)
    assert "1.5k" in fed._frame()
    timed = _spin(spinner.WORK)
    assert "s" in timed._frame().split()[-1]


def test_a_spinner_that_was_never_fed_shows_nothing_rather_than_zero():
    """UNMEASURED IS NEVER ZERO (explorer.html). `0` is a reading; this is not."""
    assert "…" in _spin()._frame()
    assert " 0" not in _spin()._frame()


def test_the_live_character_is_bright_and_the_rest_is_armature():
    """Figure versus armature: a drawing at one weight has no instrument in it."""
    s = _spin()
    s.feed(1)
    s._frame()
    s._changed = [time.monotonic(), -spinner.COOL, -spinner.COOL]
    frame = s._frame()
    ar, ag, ab = render.SEED["accent"]
    br, bg, bb = render.SEED["brand"]
    assert f"38;2;{ar};{ag};{ab}" in frame
    assert f"38;2;{br};{bg};{bb}" in frame


def test_the_two_motifs_do_not_look_alike():
    """Motion means what the wait means: a thing turning is not text landing."""
    assert not set(spinner.WORK.alphabet) - set("|/-\\")
    assert set(spinner.ARRIVE.alphabet) - set(spinner.WORK.alphabet)


def test_the_diamond_is_not_borrowed_for_motion():
    """A diamond means a step. A spinner is not one."""
    s = _spin()
    s.feed(1)
    assert render.DIAMOND_SOLID not in s._frame()
    assert render.DIAMOND_HOLLOW not in s._frame()


def test_nothing_is_drawn_when_the_stream_is_not_a_terminal():
    class Pipe:
        def isatty(self): return False
        def write(self, _): raise AssertionError("wrote to a pipe")
        def flush(self): pass

    s = spinner.Spinner(spinner.ARRIVE, stream=Pipe())
    assert not s.on
    with s:
        s.feed(3)
    # start/stop are no-ops; the assertion above would have fired otherwise


def test_plain_mode_silences_the_spinner(monkeypatch):
    monkeypatch.setenv("ONE_CLI_PLAIN", "1")
    render.refresh()

    class Tty:
        def isatty(self): return True
        def write(self, _): raise AssertionError("wrote under --plain")
        def flush(self): pass

    assert not spinner.Spinner(spinner.ARRIVE, stream=Tty()).on


def test_start_and_stop_are_safe_to_call_twice():
    s = spinner.Spinner(spinner.ARRIVE, enabled=False)
    s.start()
    s.start()
    s.stop()
    s.stop()


def test_a_real_run_clears_its_line_and_never_leaves_the_thread_behind():
    written = []

    class Tty:
        def isatty(self): return True
        def write(self, t): written.append(t)
        def flush(self): pass

    s = spinner.Spinner(spinner.ARRIVE, stream=Tty())
    s.on = True
    with s:
        s.feed(1)
        time.sleep(spinner.FRAME * 3)
    assert written, "the spinner should have drawn at least one frame"
    assert written[-1] == "\r\033[K", "the line must be cleared on stop"
    assert s._thread is None


def test_the_rail_is_stopped_when_the_turn_ends(monkeypatch):
    """Nothing printed after a live rail survives: its frames begin with a
    carriage return and an erase. So the tracer has to stop the rail on the
    event that means the wait is over, not wait for the next one to arrive.
    """
    from onecli import cli, spinner as spinlib

    started: list["Fake"] = []

    class Fake:
        def __init__(self, motif, label=""):
            self.stopped = False
            started.append(self)

        def start(self):
            return self

        def stop(self):
            self.stopped = True

        def feed(self, n):
            pass

    monkeypatch.setattr(spinlib, "Spinner", Fake)
    trace = cli._tracer(quiet=False, state={"step": 0}, model="m")

    trace("waiting", {"step": 1})
    assert len(started) == 1 and not started[0].stopped

    trace("turn", {"step": types.SimpleNamespace(index=1)})
    assert started[0].stopped


def _streaming_tracer(monkeypatch, out, err):
    """A tracer wired to captured streams, with the rail stubbed out."""
    from onecli import cli, spinner as spinlib

    class Fake:
        def __init__(self, motif, label=""): pass
        def start(self): return self
        def stop(self): pass
        def feed(self, n): pass

    monkeypatch.setattr(spinlib, "Spinner", Fake)
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    state = {"step": 0, "stream": True, "streamed": False}
    return cli._tracer(quiet=False, state=state, model="m"), state


def test_a_tool_line_never_lands_on_the_end_of_a_streamed_sentence(monkeypatch):
    """The model narrates, then calls a tool. The narration is on stdout with
    no trailing newline and the tool line is on stderr, so without an explicit
    line break the two are printed on the same terminal line - which is how
    "...not in the reference snapshots" ended up welded to "5 bash cd ...".
    """
    out, err = io.StringIO(), io.StringIO()
    trace, _ = _streaming_tracer(monkeypatch, out, err)

    trace("text_delta", {"kind": "text", "chunk": "Let me check what is live"})
    trace("tool_start", {"call": types.SimpleNamespace(
        name="bash", arguments={"command": "ls"}), "mutating": False})

    assert out.getvalue() == "Let me check what is live\n", (
        "the streamed line has to be closed before the tool line prints")
    assert err.getvalue().startswith("\n") is False


def test_a_streamed_chunk_that_already_ends_in_a_newline_gets_no_second_one(
        monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    trace, _ = _streaming_tracer(monkeypatch, out, err)

    trace("text_delta", {"kind": "text", "chunk": "done\n"})
    trace("tool_start", {"call": types.SimpleNamespace(
        name="bash", arguments={"command": "ls"}), "mutating": False})

    assert out.getvalue() == "done\n"


def test_streaming_itself_is_not_broken_into_one_line_per_chunk(monkeypatch):
    """`halt` runs on every delta; only the events that PRINT may break the
    line, or a streamed answer would arrive one word per line.
    """
    out, err = io.StringIO(), io.StringIO()
    trace, _ = _streaming_tracer(monkeypatch, out, err)

    for chunk in ("the ", "answer ", "is 42"):
        trace("text_delta", {"kind": "text", "chunk": chunk})

    assert out.getvalue() == "the answer is 42"


def test_settle_closes_the_line_once_and_is_idempotent(monkeypatch):
    """The end of a run owes the answer its newline - and exactly one, however
    many times the path that ends a run is taken.
    """
    out, err = io.StringIO(), io.StringIO()
    trace, _ = _streaming_tracer(monkeypatch, out, err)

    trace("text_delta", {"kind": "text", "chunk": "42"})
    trace.settle()
    trace.settle()

    assert out.getvalue() == "42\n"


def test_previous_stream_does_not_suppress_unstreamed_final(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    trace, state = _streaming_tracer(monkeypatch, out, err)
    trace('text_delta', {'kind': 'text', 'chunk': 'Checking.'})
    assert state['streamed']
    trace('tool_start', {'call': types.SimpleNamespace(name='read', arguments={'path': 'a'}),
                         'mutating': False})
    trace('waiting', {'step': 2})
    trace.settle()
    assert not state['streamed']

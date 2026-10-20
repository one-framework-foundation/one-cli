"""Motion, under the one rule that makes motion honest.

**Stale data cannot animate.** Amplitude scales with freshness, and MISSING is
deliberately left still, because a panel that animates over an absent path is
drawing a claim that something is behind it.

Every other CLI spinner breaks that rule. It twirls at a fixed rate off a timer,
so it spins exactly as merrily against a hung socket as against a model mid-
sentence, and the one question a person actually has while watching it - is
anything happening? - is the one thing it cannot answer.

This one is three characters wide and scrambles. A character changes only
because something arrived, and how much of it churns per frame is the arrival
rate: a fast stream boils, a slow model flickers one cell at a time, and a wire
that goes quiet freezes the last three characters, greys them, and says how long
it has been waiting. Nothing is inferred from a clock.

`work` is the exception, and says so. It watches a subprocess it cannot see
into, so absence of output is not absence of work.
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time

from onecli import render

FRAME = 0.1        # seconds between frames
STALL_AFTER = 2.0  # seconds without a feed before the motion is a lie
CELLS = 3          # the whole spinner. A wait indicator is not a progress bar.
COOL = 0.25        # seconds for a just-changed character to fall back to armature
FULL_RATE = 1200.0  # characters a second that earns the top churn
SLOWEST = 0.15      # churn floor: a trickle flickers, it does not freeze
WORK_RATE = 0.55    # the ungated churn. One number, because it is not a reading.


class Motif:
    """What kind of waiting this is. The name has to be the thing.

    A motif you could name the wait from is an interface; one that could be
    dropped on any wait is a screensaver (ANIM.md, rule 3).
    """

    def __init__(self, name: str, alphabet: str, ascii_alphabet: str, gated: bool):
        self.name = name
        self.alphabet = alphabet
        self.ascii_alphabet = ascii_alphabet
        # gated = its clock is DATA. An ungated motif runs on wall time and
        # must say so in its readout, because time is not evidence of progress.
        self.gated = gated

    def pool(self) -> str:
        return self.alphabet if render.enabled() else self.ascii_alphabet


# `arrive` is fed by tokens off the wire - the model answering - and scrambles
# fragments of text. `work` is fed by nothing, and turns.
ARRIVE = Motif("arrive", "/\\|-_=+*<>~:;.", "/\\|-_=+*<>~:;.", gated=True)
WORK = Motif("work", "|/-\\", "|/-\\", gated=False)


class Spinner:
    """Three characters, one clock. Safe to construct when nothing will draw."""

    def __init__(self, motif: Motif = ARRIVE, label: str = "",
                 stream=None, enabled: bool | None = None, seed: int | None = None):
        self.motif = motif
        # A tool name is the model's: untrusted terminal input.
        self.label = render.untrusted(label)
        self.stream = stream if stream is not None else sys.stderr
        self.fed = 0
        self._rng = random.Random(seed)
        self._cells = [motif.pool()[0]] * CELLS
        self._changed = [-COOL] * CELLS   # when each cell last rolled
        self._churn = 0.0                 # sub-character credit carried between frames
        self._rate = 0.0                  # smoothed characters a second
        self._seen = 0                    # `fed` as of the last frame
        self._last_feed = time.monotonic()
        self._last_frame = time.monotonic()
        self._started = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        if enabled is None:
            enabled = bool(render.enabled() and self.stream.isatty())
        self.on = enabled
        self.reduced = os.environ.get('ONE_REDUCED_MOTION', '').lower() in ('1', 'true', 'yes')

    # --- the data path -----------------------------------------------------

    def feed(self, n: int = 1) -> None:
        """Something real arrived. This is the only thing that moves it."""
        with self._lock:
            self.fed += n
            self._last_feed = time.monotonic()

    def _speed(self, now: float) -> float:
        """Characters rolled per frame, as a share of the whole spinner.

        The map is logarithmic because arrival rates are: a local model at 40
        characters a second and a hosted one at 1200 are the two ends a person
        actually sees, and on a linear ramp the first would be indistinguishable
        from stopped. Smoothed, or every empty chunk between two tokens would
        read as a stall.
        """
        if not self.motif.gated:
            return WORK_RATE
        dt = max(now - self._last_frame, 1e-3)
        self._last_frame = now
        with self._lock:
            arrived, self._seen = self.fed - self._seen, self.fed
        self._rate += (arrived / dt - self._rate) * 0.3
        if self._rate <= 0:
            return SLOWEST
        share = math.log10(1 + self._rate) / math.log10(1 + FULL_RATE)
        return max(SLOWEST, min(1.0, share ** 2))

    # --- the frame ---------------------------------------------------------

    def _roll(self, i: int, now: float) -> None:
        """Give one cell a different character. Different, not merely random -
        a re-roll onto the same glyph is a frame where nothing appears to have
        happened, which is the reading reserved for a stall."""
        pool = self.motif.pool()
        self._cells[i] = self._rng.choice([c for c in pool if c != self._cells[i]])
        self._changed[i] = now

    def _frame(self) -> str:
        now = time.monotonic()
        stalled = now - self._last_feed
        # A gated motif holds still when nothing is arriving. An ungated one
        # keeps churning, because its clock never claimed to be data.
        moving = (not self.motif.gated) or stalled < STALL_AFTER
        if moving and not self.reduced:
            # Carry the fraction between frames, so a rate below one character
            # a frame still shows up instead of rounding to nothing.
            self._churn += self._speed(now) * CELLS
            while self._churn >= 1.0:
                self._roll(self._rng.randrange(CELLS), now)
                self._churn -= 1.0
        else:
            self._last_frame = now
            self._churn = 0.0

        if not moving:
            # Freeze the characters that were there, grey them, and say what
            # the stillness means. This is the readout no other spinner prints.
            body = render.paint("faint", "".join(self._cells))
            note = render.paint("warm", f"waiting {stalled:0.0f}s")
        else:
            body = "".join(
                # Bright the instant it changes, cooling back to the armature
                # tint: three cells at one weight would read as a static word.
                render.paint_mix(
                    render.mix("brand", "accent",
                               max(0.0, 1.0 - (now - self._changed[i]) / COOL)),
                    c, "accent" if now - self._changed[i] < COOL / 2 else "brand")
                for i, c in enumerate(self._cells))
            note = (render.paint("faint", _count(self.fed)) if self.motif.gated
                    else render.paint("faint", f"{now - self._started:0.0f}s"))

        label = render.paint("dim", self.label) if self.label else ""
        return "  " + body + "  " + label + ("  " if label else "") + note

    def _loop(self) -> None:
        while not self._stop.wait(FRAME):
            try:
                self.stream.write("\r\033[K" + self._frame())
                self.stream.flush()
            except (ValueError, OSError):  # the stream went away mid-run
                return

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> "Spinner":
        if self.on and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Clear the line. The caller owns whatever is printed next."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=FRAME * 3)
            self._thread = None
        if self.on:
            try:
                self.stream.write("\r\033[K")
                self.stream.flush()
            except (ValueError, OSError):
                pass

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _count(n: int) -> str:
    if n <= 0:
        return "…"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


__all__ = ["Spinner", "Motif", "ARRIVE", "WORK", "CELLS", "STALL_AFTER"]

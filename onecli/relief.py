"""ASCII relief - the third drawing material, in a terminal.

`render.py` draws in line and `spinner.py` draws in motion. Both answer *how
big*. Neither answers *compared to its neighbours* or *which way is it going*,
and those are the two questions a person actually has when they look at a row
of numbers.

A heightfield is lit by one fixed light, quantised through a Bayer 8x8 ordered
dither, and printed as glyphs. Height is data, light is fixed, and the glyph is
whatever the two of them add up to.

The rules:

- **The ramps and the Bayer matrix are fixed values**, so the same data
  quantises at the same steps every time.
- **Colour carries STATE, the glyph carries VALUE.** Two axes, so a failing
  lane and an ok lane are told apart at a glance while both still show their
  real number as relief. The tints come from `render.py`, which already owns
  the palette, because a second palette is
  a second truth.
- **The three-way height contract**: a number is a reading, `None`
  is *no reading here* (one dim dot), and `SKY` is *no surface here* (nothing
  drawn). Those three are not interchangeable. A missing probe owes the reader
  a mark; empty space above a ridge does not.

What is left out: the clock. A terminal transcript is a printed record - it is read
after the fact, scrolled back to, and pasted into a message - so a relief here
is drawn once and stays what it was. Motion in this CLI lives in `spinner.py`
under the rule that stale data cannot animate, and a printed frame has no way
to honour that rule, so it does not get to move at all.

Nothing here is a filled surface *by default*. The `ascii` ramp is the default
for exactly that reason; `blocks` is available, but a bar chart made of `█` is
rarely the honest choice.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from onecli import render


class _Sky:
    """No surface here. Distinct from `None`, which is a missing reading."""

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return "SKY"

    def __bool__(self) -> bool:
        return False


SKY = _Sky()

# Ordered darkest-to-lightest. Index 0 must be
# blank: an empty cell is a real reading of zero and it has to look like
# nothing.
RAMPS = {
    "ascii":  " .:-=+*#%@",
    "blocks": " ░░▒▒▓▓██",
    "fine":   " ·,:;+*?%@#",
    "dots":   " ⠁⠃⠇⡇⡏⡟⡿⣿",
}

UNMEASURED_GLYPH = "·"

# State -> seed token. One mapping for the whole layer, so two callers cannot
# disagree about what "warn" looks like. The keys are lane states; the
# values are render.py's tokens, which is where the actual colours live.
STATE_TOKEN = {
    "ok": "accent",
    "warn": "warm",
    "crit": "alarm",
    "idle": "dim",
    "unmeasured": "faint",
    "hot": "ink",
}

# Bayer 8x8, normalised to 0..1 - the standard ordered-dither matrix. Add the
# cell's threshold to the value before quantising and a value sitting between
# two glyphs resolves as a stable spatial mix instead of a wobble. That is what
# turns banding into texture instead of into a lie.
_BAYER_RAW = (
    (0, 32, 8, 40, 2, 34, 10, 42),
    (48, 16, 56, 24, 50, 18, 58, 26),
    (12, 44, 4, 36, 14, 46, 6, 38),
    (60, 28, 52, 20, 62, 30, 54, 22),
    (3, 35, 11, 43, 1, 33, 9, 41),
    (51, 19, 59, 27, 49, 17, 57, 25),
    (15, 47, 7, 39, 13, 45, 5, 37),
    (63, 31, 55, 23, 61, 29, 53, 21),
)
BAYER = tuple(v / 64.0 for row in _BAYER_RAW for v in row)

# The light. One direction, fixed forever, because shading that moves is
# shading that can lie about shape. Ambient, altitude and Lambert weights below.
_LIGHT = (-0.55, -0.62, 0.56)
_AMBIENT, _ALTITUDE, _LAMBERT = 0.07, 0.44, 0.40

# A terminal cell is about twice as tall as it is wide. The surface normal has
# to know that or every ridge reads as if it runs the wrong way - the same
# correction `render._raster` makes for the wordmark.
_CELL_ASPECT = 2.0


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def _quantise(lum: float, col: int, row: int, glyphs: str) -> str:
    """One luminance to one glyph, through the ordered dither."""
    threshold = BAYER[(row % 8) * 8 + (col % 8)]
    steps = len(glyphs) - 1
    # `floor(v*steps + threshold)`, not the centred `v*steps + (threshold-0.5)`.
    # The centred form breaks at both ends and in the middle: a reading of
    # exactly 1.0 comes out as a checker of the top two glyphs (saying "near the
    # top" about a value that is AT it), and because Python's `int()` truncates
    # rather than floors, the sub-step offset is thrown away for most of the
    # range and the mix collapses to a single glyph. Both failures have tests.
    idx = int(math.floor(_clamp(lum) * steps + threshold))
    return glyphs[max(0, min(steps, idx))]


def _read(grid: list[list], col: int, row: int, cols: int, rows: int):
    """A neighbour's height, clamped at the edge.

    An edge cell has no neighbour outside the field, so it borrows its own
    value rather than 0 - otherwise every field is rimmed by a cliff that is
    not in the data.
    """
    c = 0 if col < 0 else cols - 1 if col >= cols else col
    r = 0 if row < 0 else rows - 1 if row >= rows else row
    v = grid[r][c]
    return v if isinstance(v, (int, float)) else None


def field(
    height: Callable[[int, int], float | None | _Sky],
    cols: int,
    rows: int,
    ramp: str = "ascii",
    state: str | Callable[[int, int], str] = "ok",
    tilt: float = 0.0,
) -> list[str]:
    """A lit heightfield, `rows` strings of at most `cols` glyphs.

    `height(col, row)` returns the three-way contract: a number 0..1 is a
    reading, `None` is a missing reading, `SKY` is no surface. `tilt` shears
    rows sideways so a field reads as a landscape seen from an angle rather
    than as a top-down grid; 0 keeps it square.
    """
    glyphs = RAMPS.get(ramp) or RAMPS["ascii"]
    grid: list[list] = [[height(c, r) for c in range(cols)] for r in range(rows)]
    out: list[str] = []
    for r in range(rows):
        shift = int(round(tilt * (rows - 1 - r)))
        cells: list[str] = []
        for c in range(cols):
            v = grid[r][c]
            if v is SKY:
                cells.append(" ")
                continue
            token = state(c, r) if callable(state) else state
            if v is None:
                # Unmeasured never lights. One dim dot, so the absence has a
                # mark of its own and cannot be mistaken for a reading of zero.
                cells.append(render.paint(STATE_TOKEN["unmeasured"], UNMEASURED_GLYPH))
                continue
            h = _clamp(float(v))
            # Slope from the neighbours, with the cell aspect folded in, then
            # a lambert term against the fixed light. Altitude is added
            # separately so a high flat plateau still reads as high.
            left = _read(grid, c - 1, r, cols, rows)
            right = _read(grid, c + 1, r, cols, rows)
            up = _read(grid, c, r - 1, cols, rows)
            down = _read(grid, c, r + 1, cols, rows)
            dx = ((right if right is not None else h) - (left if left is not None else h)) / 2.0
            dy = ((down if down is not None else h) - (up if up is not None else h)) / 2.0
            nx, ny, nz = -dx, -dy / _CELL_ASPECT, 1.0 / max(cols, 8) * 4.0
            mag = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            lam = _clamp((nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2]) / mag, -1.0, 1.0)
            lum = _AMBIENT + _ALTITUDE * h + _LAMBERT * _clamp((lam + 1.0) / 2.0)
            cells.append(render.paint(STATE_TOKEN.get(token, "dim"),
                                      _quantise(lum, c, r, glyphs)))
        # Not rstripped. A field is a grid and callers set things beside it, so
        # every row is `cols` wide whatever is in it - a sky cell on the right
        # edge is a column position, and trimming it would silently reflow the
        # thing next to it.
        out.append((" " * shift) + "".join(cells))
    return out


def profile(
    values: Sequence[float | None],
    depth: int = 3,
    ramp: str = "ascii",
    state: str | Callable[[int], str] = "ok",
) -> list[str]:
    """A 1-D reading as a skyline, `depth` rows tall.

    The relief answer to a sparkline. A column is lit from its own value down
    to the floor, so the eye reads the envelope as a shape rather than reading
    `depth` separate rows. A `None` in the series is a gap in the record and is
    drawn as the unmeasured dot in every row of its column, never as a zero -
    a run that did not report is not a run that scored nothing.
    """
    depth = max(1, depth)
    cols = len(values)
    if not cols:
        return []
    real = [v for v in values if v is not None]
    top = max(real) if real else 1.0
    top = top if top > 0 else 1.0

    def h(col: int, row: int):
        v = values[col]
        if v is None:
            return None
        # Row 0 is the top of the band. A column fills from the bottom up.
        band = (depth - row) / depth
        prev = (depth - row - 1) / depth
        frac = _clamp(float(v) / top)
        if frac <= prev:
            return SKY
        return _clamp((frac - prev) / (band - prev)) * 0.65 + 0.35

    def st(col: int, row: int) -> str:
        return state(col) if callable(state) else state

    return field(h, cols, depth, ramp=ramp, state=st)


def chip(share: float, cells: int = 12, ramp: str = "ascii",
         state: str = "ok") -> str:
    """An inline dithered ramp - one share, one row, no clock.

    `render.meter` draws the same quantity as a hairline and stays the right
    choice inside a table, where a rule aligns with the columns around it.
    This is for a share standing on its own in prose, where a ramp shows the
    shape of the fill (front-loaded, nearly done) that a two-tone rule flattens.
    """
    share = _clamp(share)
    glyphs = RAMPS.get(ramp) or RAMPS["ascii"]
    lit = share * cells
    out = []
    for c in range(cells):
        # Two things multiply. `_clamp(lit - c)` is coverage: 1 inside the
        # fill, 0 outside, and a fraction on the one cell the value lands in -
        # which is what makes the edge dither instead of snap, since a share
        # lands between cells far more often than on one. The second term
        # brightens left to right, so the result is a ramp rather than a
        # filled bar: house rule 1 says nothing is a filled surface, and a run
        # of the top glyph is a filled surface however it was arrived at.
        v = _clamp(lit - c) * (0.45 + 0.55 * (c + 0.5) / cells)
        out.append(render.paint(STATE_TOKEN.get(state, "dim"),
                                _quantise(v, c, 0, glyphs)))
    return "".join(out)


def bars(rows: Sequence[tuple[str, float | None]], width: int = 24,
         state: str | Callable[[str, float | None], str] = "ok",
         ramp: str = "ascii") -> list[str]:
    """A labelled column of chips - the relief answer to a score table.

    Takes `(label, share)` pairs. A `None` share prints the unmeasured ramp and
    its own reason column stays the caller's business: this draws the missing
    reading, it does not invent a story for it.
    """
    if not rows:
        return []
    pad = max(len(label) for label, _ in rows)
    out = []
    for label, share in rows:
        if share is None:
            ramp_cells = render.paint(STATE_TOKEN["unmeasured"],
                                      UNMEASURED_GLYPH * width)
        else:
            st = state(label, share) if callable(state) else state
            ramp_cells = chip(share, cells=width, ramp=ramp, state=st)
        out.append("  " + render.dim(label.ljust(pad)) + "  " + ramp_cells)
    return out

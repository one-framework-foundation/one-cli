"""The relief layer, held to the contract it inherited from dither.js."""

import re

import pytest

from onecli import relief, render


ESC = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    """Every test reads glyphs, so run without colour unless it asks for it."""
    for key in ("NO_COLOR", "ONE_CLI_PLAIN", "FORCE_COLOR", "COLORTERM", "TERM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ONE_CLI_PLAIN", "1")
    render.refresh()
    yield
    # Undo the test's environment changes too, before refreshing cached state.
    monkeypatch.undo()
    render.refresh()


def strip(lines):
    return [ESC.sub("", ln) for ln in lines]


# --- the source's own values ----------------------------------------------

def test_bayer_is_the_source_matrix_normalised():
    assert len(relief.BAYER) == 64
    # dither.js divides the 0..63 matrix by 64, so 0 is present and 1.0 is not.
    assert min(relief.BAYER) == 0.0
    assert max(relief.BAYER) == pytest.approx(63 / 64)
    assert sorted(relief.BAYER) == [i / 64 for i in range(64)]


def test_every_ramp_starts_blank():
    """Index 0 must be blank: an empty cell is a real reading of zero."""
    for name, glyphs in relief.RAMPS.items():
        assert glyphs[0] == " ", name
        assert len(glyphs) >= 2, name


def test_states_map_onto_real_render_tokens():
    for state, token in relief.STATE_TOKEN.items():
        assert token in render.SEED, f"{state} -> {token} is not a seed token"


# --- the three-way height contract ----------------------------------------

def test_sky_draws_nothing_and_none_draws_a_dot():
    """`None` is a missing reading and owes the reader a mark. `SKY` is empty
    space above a ridge and owes nothing."""
    def h(c, r):
        return {0: 0.9, 1: None}.get(c, relief.SKY)

    row = strip(relief.field(h, 3, 1))[0]
    assert row[1] == relief.UNMEASURED_GLYPH
    assert row[2] == " "
    assert row[0] not in (" ", relief.UNMEASURED_GLYPH)


def test_sky_is_not_falsy_confused_with_zero():
    """A reading of 0.0 is a reading. It must not be routed down the SKY path
    by an `if not value` anywhere in the layer."""
    zero = strip(relief.field(lambda c, r: 0.0, 4, 1))[0]
    sky = strip(relief.field(lambda c, r: relief.SKY, 4, 1))[0]
    assert sky.strip() == ""
    # Zero is the bottom of the ramp but it is still lit by ambient+lambert,
    # so it is not silently identical to nothing.
    assert zero != sky or zero.strip() != ""


def test_unmeasured_never_lights_whatever_the_state_says():
    row = strip(relief.field(lambda c, r: None, 5, 1, state="crit"))[0]
    assert row == relief.UNMEASURED_GLYPH * 5


# --- the dither ------------------------------------------------------------

def test_saturation_is_solid_not_a_checker():
    """Regression: the centred form `v*steps + (threshold-0.5)` dithers the
    ENDS too, so a full reading came out as a checker of the top two glyphs and
    full was indistinguishable from nearly-full."""
    glyphs = relief.RAMPS["ascii"]
    top = {relief._quantise(1.0, c, r, glyphs) for c in range(8) for r in range(8)}
    bottom = {relief._quantise(0.0, c, r, glyphs) for c in range(8) for r in range(8)}
    assert top == {glyphs[-1]}
    assert bottom == {glyphs[0]}


def test_a_middle_value_mixes_two_neighbouring_glyphs_only():
    """Banding turns into texture, never into an invented middle glyph."""
    glyphs = relief.RAMPS["ascii"]
    seen = {relief._quantise(0.5, c, r, glyphs) for c in range(8) for r in range(8)}
    assert len(seen) == 2
    idx = sorted(glyphs.index(g) for g in seen)
    assert idx[1] - idx[0] == 1


def test_quantise_never_leaves_the_ramp():
    glyphs = relief.RAMPS["fine"]
    for v in (-5.0, 0.0, 0.5, 1.0, 9.0):
        for c in range(8):
            assert relief._quantise(v, c, 0, glyphs) in glyphs


# --- field -----------------------------------------------------------------

def test_field_shape():
    rows = relief.field(lambda c, r: 0.5, 10, 4)
    assert len(rows) == 4
    assert all(len(ESC.sub("", ln)) == 10 for ln in rows)


def test_field_edges_borrow_their_own_height_not_zero():
    """A flat field must be flat. If the edge read 0 outside itself, the rim
    would be a cliff that is not in the data."""
    rows = strip(relief.field(lambda c, r: 0.6, 12, 4))
    assert len({ln[0] for ln in rows} | {ln[-1] for ln in rows}) <= 2


def test_tilt_shears_rows_and_zero_tilt_does_not():
    flat = strip(relief.field(lambda c, r: 0.5, 8, 4, tilt=0.0))
    assert all(not ln.startswith(" ") for ln in flat)
    sheared = strip(relief.field(lambda c, r: 0.5, 8, 4, tilt=1.0))
    assert sheared[0].startswith("   ")
    assert not sheared[-1].startswith(" ")


def test_state_may_be_a_callable_per_cell(monkeypatch):
    monkeypatch.delenv("ONE_CLI_PLAIN", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    render.refresh()
    rows = relief.field(lambda c, r: 0.8, 2, 1,
                        state=lambda c, r: "ok" if c == 0 else "crit")
    ok = render.SEED_256[relief.STATE_TOKEN["ok"]]
    crit = render.SEED_256[relief.STATE_TOKEN["crit"]]
    assert f"38;5;{ok}m" in rows[0]
    assert f"38;5;{crit}m" in rows[0]
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def test_no_escapes_at_all_when_colour_is_refused():
    rows = relief.field(lambda c, r: 0.7, 6, 3)
    assert all("\x1b" not in ln for ln in rows)


# --- profile ---------------------------------------------------------------

def test_profile_scales_to_the_series_maximum():
    tall = strip(relief.profile([0.1, 1.0], depth=4))
    small = strip(relief.profile([0.01, 0.1], depth=4))
    # The same shape at a different scale prints the same skyline: the profile
    # answers "compared to its neighbours", not "compared to 1.0".
    assert tall == small


def test_profile_gap_is_dots_in_every_row_never_a_zero():
    rows = strip(relief.profile([0.9, None, 0.9], depth=3))
    assert len(rows) == 3
    assert all(ln[1] == relief.UNMEASURED_GLYPH for ln in rows)


def test_profile_bottom_row_is_lit_for_every_real_column():
    rows = strip(relief.profile([0.2, 0.5, 1.0], depth=3))
    assert all(ch != " " for ch in rows[-1])


def test_profile_of_nothing_is_nothing():
    assert relief.profile([]) == []


def test_profile_of_all_zero_does_not_divide_by_zero():
    rows = strip(relief.profile([0.0, 0.0], depth=2))
    assert len(rows) == 2


# --- chip and bars ---------------------------------------------------------

def test_chip_is_monotone_in_its_share():
    def lit(share):
        return sum(1 for ch in ESC.sub("", relief.chip(share, cells=20)) if ch != " ")
    counts = [lit(s / 10) for s in range(11)]
    assert counts == sorted(counts)
    assert counts[0] == 0
    assert counts[-1] == 20


def test_chip_is_a_ramp_not_a_filled_bar():
    """House rule 1: nothing is a filled surface. A full chip must still show
    direction, so it cannot be a run of one glyph."""
    full = ESC.sub("", relief.chip(1.0, cells=12))
    assert len(set(full)) > 3


def test_chip_clamps_out_of_range_shares():
    assert ESC.sub("", relief.chip(-2.0, cells=6)).strip() == ""
    assert " " not in ESC.sub("", relief.chip(5.0, cells=6))


def test_bars_align_labels_and_draw_a_missing_share_as_unmeasured():
    rows = strip(relief.bars([("one", 0.5), ("qwen-code", None)], width=8))
    assert rows[0].startswith("  one      ")
    assert rows[1].endswith(relief.UNMEASURED_GLYPH * 8)


def test_bars_of_nothing_is_nothing():
    assert relief.bars([]) == []

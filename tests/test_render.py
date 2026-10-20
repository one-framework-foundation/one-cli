"""The terminal surface. What is tested here is the CONTRACT, not the taste.

Colour is the part of a CLI that breaks other people's tools: an escape code in
a pipe corrupts a parse, and a box-drawing glyph on a terminal that cannot draw
it corrupts a screen. So the tests are about capability detection and graceful
degradation, plus the one semantic the glyphs carry (filled = it changed
something), which is a claim a reader will trust and must therefore be true.
"""

import importlib

import pytest

from onecli import render


@pytest.fixture(autouse=True)
def _restore():
    yield
    render.refresh()


def _reset(monkeypatch, **env):
    for key in ("NO_COLOR", "ONE_CLI_PLAIN", "FORCE_COLOR", "COLORTERM", "TERM"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    render.refresh()


def test_no_color_env_silences_every_escape(monkeypatch):
    _reset(monkeypatch, NO_COLOR="1", FORCE_COLOR="1", COLORTERM="truecolor")
    assert not render.enabled()
    for text in (render.brand("0.1.0"), render.rule(), render.kicker("x"),
                 render.node("bash", "ls", mutating=True, step=1),
                 render.footer(["a", "b"]), render.prompt_mark()):
        assert "\033" not in text


def test_plain_flag_uses_the_same_switch_as_no_color(monkeypatch):
    _reset(monkeypatch, ONE_CLI_PLAIN="1", FORCE_COLOR="1")
    assert not render.enabled()


def test_a_pipe_gets_no_colour_without_being_asked(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(render.sys.stderr, "isatty", lambda: False)
    render.refresh()
    assert not render.enabled()
    assert "\033" not in render.node("read", "a.py")


def test_truecolor_is_used_when_the_terminal_says_so(monkeypatch):
    _reset(monkeypatch, FORCE_COLOR="1", COLORTERM="truecolor")
    assert render.enabled()
    r, g, b = render.SEED["accent"]
    assert f"38;2;{r};{g};{b}" in render.paint("accent", "x")


def test_a_terminal_without_colorterm_falls_back_to_256(monkeypatch):
    _reset(monkeypatch, FORCE_COLOR="1")
    assert f"38;5;{render.SEED_256['accent']}" in render.paint("accent", "x")


def test_dumb_terminal_is_treated_as_no_terminal(monkeypatch):
    _reset(monkeypatch, FORCE_COLOR="1", TERM="dumb")
    assert not render.enabled()


def test_glyphs_degrade_to_ascii_when_colour_is_off(monkeypatch):
    """A terminal we cannot paint is also one we should not hand box-drawing.

    The two capabilities are not the same thing, but they travel together in
    practice (CI logs, `NO_COLOR`, a pipe), and a mojibake diamond is worse
    than a hyphen."""
    _reset(monkeypatch, NO_COLOR="1")
    line = render.node("bash", "ls", mutating=True)
    assert render.DIAMOND_SOLID not in line
    assert line.strip().startswith("*")
    assert render.HAIRLINE not in render.rule()


def test_the_diamond_is_filled_only_when_the_step_can_change_something(monkeypatch):
    _reset(monkeypatch, FORCE_COLOR="1", COLORTERM="truecolor")
    assert render.DIAMOND_SOLID in render.node("bash", "rm -rf x", mutating=True)
    assert render.DIAMOND_HOLLOW in render.node("read", "x.py", mutating=False)
    assert render.DIAMOND_SOLID not in render.node("read", "x.py", mutating=False)


def test_tool_line_carries_the_mutating_flag_through(monkeypatch):
    _reset(monkeypatch, NO_COLOR="1")
    write = render.tool_line(3, "write", {"path": "a.py"}, mutating=True)
    read = render.tool_line(3, "read", {"path": "a.py"}, mutating=False)
    assert write != read
    assert "a.py" in write and "Write" in write
    assert "Read" in read
    assert "3" not in write


def test_long_detail_is_cut_not_wrapped(monkeypatch):
    _reset(monkeypatch, NO_COLOR="1")
    line = render.node("bash", "x" * 400)
    assert len(line) < 120
    assert line.endswith("…")


def test_footer_drops_empty_bits_instead_of_printing_blanks(monkeypatch):
    _reset(monkeypatch, NO_COLOR="1")
    assert "· ·" not in render.footer(["qwen", "", "2 steps", ""])


def test_the_banner_never_shears_the_mark_or_drops_a_line(monkeypatch):
    """The mark is a fixed height; the readout is not. Neither may cut the other."""
    _reset(monkeypatch, NO_COLOR="1")
    short = render.banner("0.1.0", ["one"]).splitlines()
    assert len(short) == len(render._MARK)
    long = render.banner("0.1.0", [str(i) for i in range(20)]).splitlines()
    assert len(long) == 20
    assert long[-1].strip() == "19"


def test_width_stays_inside_sane_bounds(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setattr(render.shutil, "get_terminal_size",
                        lambda fallback=None: type("S", (), {"columns": 400})())
    assert render.width() == 100
    monkeypatch.setattr(render.shutil, "get_terminal_size",
                        lambda fallback=None: type("S", (), {"columns": 10})())
    assert render.width() == render.MIN_WIDTH


def test_module_import_does_not_need_a_terminal():
    """render is imported by `one --json` too, where nothing is painted."""
    importlib.reload(render)
    assert isinstance(render.enabled(), bool)


def test_the_mark_is_rasterised_from_the_logo_geometry(monkeypatch):
    """The mark is one-logo-white.svg, not a picture of it.

    The SVG numbers (rects at x12/x27/x42, the centre running y8-52 against
    y20-48, dot at cy62) are the only thing held by hand, so this asserts
    those - and that the raster of them is 8 wide by 6 tall, which is what a
    cell being twice as tall as it is wide does to a 40x58 mark.
    """
    assert render._LOGO_BARS == ((12, 20, 10, 28), (27, 8, 10, 44),
                                 (42, 20, 10, 28))
    assert render._LOGO_DOT == (32, 62, 4)
    rows = render._MARK
    assert len(rows) == 6 and render._MARK_W == 8
    # three bars, the outer pair starting a half row below the centre one
    assert rows[0].strip() == "\u2588\u2588"
    assert [r.count("\u2588\u2588") for r in rows] == [1, 1, 3, 3, 0, 0]
    assert rows[1].startswith("\u2584\u2584") and rows[1].endswith("\u2584\u2584")
    # the dot stands clear underneath, on the centre bar's own column
    assert rows[-1].strip() == "\u2580\u2580"
    assert rows[-1].index("\u2580") == rows[0].index("\u2588")
    assert rows[-2].strip() == "\u2580\u2580"


def test_the_raster_is_the_svg_not_a_transcription(monkeypatch):
    """Doubling the resolution must draw the same logo, not a different one."""
    fine = render._raster(unit=2.5)
    assert len(fine) == 12 and max(len(r) for r in fine) == 16
    def ink(rs):
        return sum(ch != " " for r in rs for ch in r) / (len(rs) * max(len(r) for r in rs))
    assert abs(ink(fine) - ink(render._MARK)) < 0.12


def test_every_token_a_caller_names_actually_exists(monkeypatch):
    """A palette rename that leaves a caller behind crashes at the moment the
    rare branch runs - a non-default permission mode, a compaction, a failover.

    So the names are checked as a set, statically, rather than waiting for the
    branch to be taken. (Written after `amber` was renamed to `warm` and three
    callers were left pointing at a key that no longer existed.)
    """
    import pathlib
    import re

    root = pathlib.Path(render.__file__).resolve().parent.parent
    used = set()
    for path in list(root.glob("onecli/**/*.py")) + list(root.glob("scripts/*.py")):
        used |= set(re.findall(r'paint\(\s*["\'](\w+)["\']', path.read_text()))
    missing = sorted(used - set(render.SEED))
    assert not missing, f"paint() called with unknown token(s): {missing}"
    assert set(render.SEED) == set(render.SEED_256), "the 256 ramp must mirror the truecolor one"


# --- model output is untrusted terminal input ------------------------------


def test_untrusted_strips_screen_and_title_control():
    hostile = "\x1b[2J\x1b[H\x1b]0;pwned\x07hello"
    assert render.untrusted(hostile) == "hello"


def test_untrusted_strips_the_cursor_position_query():
    """`ESC [ 6 n` makes the terminal WRITE its answer into our stdin, which is
    how a model's answer puts characters in front of the next shell prompt."""
    assert "\x1b" not in render.untrusted("before\x1b[6nafter")


def test_untrusted_strips_a_lone_escape_byte():
    """A control sequence split across two SSE frames arrives as a bare ESC."""
    assert render.untrusted("a\x1bb") == "ab"


def test_untrusted_strips_carriage_return_and_backspace():
    """Both rewrite a line that has already been shown."""
    assert render.untrusted("shown\rhidden") == "shownhidden"
    assert render.untrusted("ab\x08c") == "abc"


def test_untrusted_keeps_layout_the_model_legitimately_writes():
    assert render.untrusted("a\nb\tc") == "a\nb\tc"


def test_untrusted_leaves_ordinary_text_alone():
    assert render.untrusted("plain answer, 100% of it") == "plain answer, 100% of it"


def test_untrusted_is_safe_on_empty():
    assert render.untrusted("") == ""


def test_plain_tool_preview_respects_no_color(monkeypatch):
    from onecli.toolview import plain_result
    _reset(monkeypatch, NO_COLOR="1")
    preview = plain_result('exit 0\nhello')
    assert preview == '| exit 0\n| hello'

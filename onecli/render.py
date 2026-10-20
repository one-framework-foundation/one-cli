"""Terminal surface: colours, glyphs and the rules that keep them honest.

The palette is a fixed gray ramp plus a brand tint, a live accent and an alarm.
The house rules:

1. NOTHING IS A FILLED SURFACE. The instrument is drawn in line, so the data is
   the only thing on the screen with weight. No background fills, no inverse
   video, no bars made of blocks.
2. FIGURE VERSUS ARMATURE. Structure is dim and in the brand tint; the live
   part is bright and in accent. A drawing where everything sits at one weight
   has no instrument in it.
3. A CORNER IS A STAGE. A diamond marks a step or a decision, and nothing else.
4. Colour is a capability, not an assumption. Truecolor when the terminal says
   so, 256 when it does not, nothing when stdout is a pipe, `NO_COLOR` is set,
   or `--plain` was passed. The glyphs degrade with it, because a broken
   diamond is worse than a hyphen.

Motion lives next door in `spinner.py`, under the fifth rule (stale data cannot
animate), which needs a clock and does not belong in a pure formatter.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

# --- palette ---------------------------------------------------------------
# The gray ramp plus the accent tokens layered on top of it.

SEED = {
    "ink":    (240, 237, 232),   # --gray-10 #f0ede8  warm off-white, the reading colour
    "dim":    (136, 136, 136),   # --gray-7  RGB 136  secondary
    "faint":  (85, 85, 85),      # --gray-6  RGB 85   tertiary
    "line":   (38, 38, 38),      # --gray-4  RGB 38   hairlines, the field
    "brand":  (139, 127, 114),   # --brand   #8b7f72  the armature tint
    "accent": (153, 204, 204),   # --accent  #9cc     the HUD accent - the live part
    "live":   (127, 227, 208),   # --live    #7fe3d0
    "warm":   (232, 177, 74),    # --dark    #e8b14a  needs a look
    "alarm":  (255, 107, 122),   # --alarm   #ff6b7a
    "mand":   (127, 170, 146),   # --mand    #7faa92  the 45-degree armature accent
}

# 256-colour stand-ins for the same swatches.
SEED_256 = {
    "ink": 255, "dim": 245, "faint": 240, "line": 236,
    "brand": 101, "accent": 152, "live": 116, "warm": 179, "alarm": 210,
    "mand": 108,
}

# `err` and `ok` are roles, not new colours: the ramp already names them.
SEED["err"] = SEED["alarm"]
SEED["ok"] = SEED["live"]
SEED["dead"] = SEED["faint"]
SEED_256["err"] = SEED_256["alarm"]
SEED_256["ok"] = SEED_256["live"]
SEED_256["dead"] = SEED_256["faint"]

DIAMOND_SOLID = "◆"   # a step that changed something
DIAMOND_HOLLOW = "◇"  # a step that only looked
HAIRLINE = "─"
SPINE = "│"
TEE = "├"
ELBOW = "╰"

ASCII_MARKS = {
    DIAMOND_SOLID: "*", DIAMOND_HOLLOW: "o", HAIRLINE: "-",
    SPINE: "|", TEE: "+", ELBOW: "`",
}


def _depth() -> int:
    """0 = no colour, 8 = 256-colour, 24 = truecolor.

    Order matters: an explicit refusal (`NO_COLOR`, `ONE_CLI_PLAIN`) beats an
    explicit demand (`FORCE_COLOR`), which beats the tty test. Anything else
    would let an environment variable paint escape codes into a pipe.
    """
    if os.environ.get("NO_COLOR") or os.environ.get("ONE_CLI_PLAIN"):
        return 0
    forced = bool(os.environ.get("FORCE_COLOR"))
    # Rendered strings can go to either stream. A stdout pipe must stay plain
    # even when diagnostics still go to a terminal.
    if not forced and not sys.stdout.isatty():
        return 0
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return 0
    if os.environ.get("COLORTERM", "") in ("truecolor", "24bit"):
        return 24
    return 8


_DEPTH = _depth()


def refresh() -> None:
    """Re-read the environment. Called after `--plain` sets ONE_CLI_PLAIN."""
    global _DEPTH
    _DEPTH = _depth()


def enabled() -> bool:
    return _DEPTH > 0


# The TUI refuses to draw below 16 columns, so a rule never needs to be wider.
MIN_WIDTH = 16


def width(default: int = 92) -> int:
    """Usable width, capped: a full-width rule on a 200-column terminal reads
    as a horizon, not as a divider."""
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
    except OSError:
        cols = default
    return max(MIN_WIDTH, min(cols - 2, 100))


def paint(token: str, text: str, bold: bool = False) -> str:
    if _DEPTH == 0 or not text:
        return text
    pre = "\033[1m" if bold else ""
    if _DEPTH == 24:
        r, g, b = SEED[token]
        return f"{pre}\033[38;2;{r};{g};{b}m{text}\033[0m"
    return f"{pre}\033[38;5;{SEED_256[token]}m{text}\033[0m"


def mix(a: str, b: str, k: float) -> tuple[int, int, int]:
    """A point on the line between two seed swatches. k=0 is `a`, k=1 is `b`.

    Used for a ramp - a trail cooling from the live colour back to the
    armature - where a two-swatch step would read as two rails rather than one
    moving thing.
    """
    k = 0.0 if k < 0 else 1.0 if k > 1 else k
    (r0, g0, b0), (r1, g1, b1) = SEED[a], SEED[b]
    return (round(r0 + (r1 - r0) * k), round(g0 + (g1 - g0) * k),
            round(b0 + (b1 - b0) * k))


def paint_mix(rgb: tuple[int, int, int], text: str, near: str) -> str:
    """Paint a colour that is between swatches.

    `near` is the seed token this collapses to where there is no room for a
    ramp: 256-colour has one entry per swatch and no intermediate, so the ramp
    degrades to the two ends rather than to an approximation nobody chose.
    """
    if _DEPTH == 0 or not text:
        return text
    if _DEPTH == 24:
        r, g, b = rgb
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
    return f"\033[38;5;{SEED_256[near]}m{text}\033[0m"


def glyph(mark: str) -> str:
    """A mark the terminal can actually draw."""
    return mark if _DEPTH else ASCII_MARKS.get(mark, mark)


# Token shorthands. `cyan`/`yellow`/`red`/`green` are kept as the names the
# rest of the CLI already calls, remapped onto seed tokens - there is no cyan
# in this palette and there should not be one.
def ink(t: str) -> str: return paint("ink", t)
def dim(t: str) -> str: return paint("dim", t)
def faint(t: str) -> str: return paint("faint", t)
def accent(t: str) -> str: return paint("accent", t)
def bold(t: str) -> str: return paint("ink", t, bold=True)
def cyan(t: str) -> str: return paint("accent", t)
def yellow(t: str) -> str: return paint("warm", t)
def red(t: str) -> str: return paint("err", t)
def green(t: str) -> str: return paint("ok", t)


def short_path(path: str) -> str:
    """`~` for the home directory, so a path fits a line a person reads.

    In `render` rather than beside one of its callers: `cli` shortens the
    workspace in the banner and `orc` prints the run directory in a header
    that is width-capped, and the second one wrapped mid-path because it had
    no way to reach the first one's copy.
    """
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home) else path


def one_line(text: str, width: int = 100) -> str:
    # Every spine helper (node, leaf, note, plan, footer) clips through here,
    # and most of what they carry - tool arguments, plan items, fallback
    # reasons - came from the model or a provider. Stripped once, here.
    flat = " ".join(untrusted(str(text)).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def kicker(text: str) -> str:
    """Uppercase, letterspaced, quiet. The web's `.kick`."""
    return faint(" ".join(text.upper()))


def rule(token: str = "line", cols: int | None = None) -> str:
    return paint(token, glyph(HAIRLINE) * (cols or width()))


# --- the wordmark ----------------------------------------------------------
# one-logo-white.svg (viewBox 0 0 64 80), verbatim: three rounded bars, the
# centre one taller, and a presence dot underneath. Held as the SVG's own
# numbers and rasterised at import, because a mark typed out by hand drifts
# from the brand the moment someone edits a row.
_LOGO_BARS = ((12, 20, 10, 28), (27, 8, 10, 44), (42, 20, 10, 28))
_LOGO_DOT = (32, 62, 4)
_LOGO_BOX = (12, 8, 40, 58)  # x, y, w, h of the inked area


def _in_logo(x: float, y: float) -> bool:
    for bx, by, bw, bh in _LOGO_BARS:
        r = bw / 2
        if bx <= x <= bx + bw and by + r <= y <= by + bh - r:
            return True
        for cy in (by + r, by + bh - r):  # rx=5 makes both ends semicircles
            if (x - (bx + r)) ** 2 + (y - cy) ** 2 <= r * r:
                return True
    cx, cy, r = _LOGO_DOT
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _raster(unit: float = 5.0, samples: int = 8) -> list[str]:
    """The logo in half blocks.

    A terminal cell is twice as tall as it is wide, so a half block is square
    and one `unit` serves both axes - that is the whole reason the mark comes
    out 8 wide and 6 tall rather than the 8x8 it would be if rows were counted
    like columns. A cell is inked when the shape covers most of it.
    """
    x0, y0, w, h = _LOGO_BOX
    cols, subs = round(w / unit), round(h / unit / 2) * 2
    grid = [[sum(_in_logo(x0 + (c + (i + .5) / samples) * unit,
                          y0 + (s + (j + .5) / samples) * unit)
                 for i in range(samples) for j in range(samples))
             >= samples * samples / 2
             for c in range(cols)] for s in range(subs)]
    rows = []
    for s in range(0, subs, 2):
        top, bot = grid[s], grid[s + 1]
        rows.append("".join("\u2588" if top[c] and bot[c] else
                            "\u2580" if top[c] else
                            "\u2584" if bot[c] else " " for c in range(cols)))
    return rows


_MARK = _raster()
_MARK_W = max(len(r) for r in _MARK)
# Where a terminal will not paint blocks at all. Not a second drawing - the
# same raster, spelled in the only ink available.
_MARK_ASCII = [ln.replace("\u2588", "#").replace("\u2580", "-")
               .replace("\u2584", "_") for ln in _MARK]


def banner(version: str, lines: list[str]) -> str:
    """The mark, with the run's readout set beside it.

    The mark is a fixed six rows; a longer readout keeps going underneath it,
    indented to the same column, rather than being cut to fit the art.
    """
    body = list(lines)
    rows = (_MARK if _DEPTH else _MARK_ASCII) + [""] * max(0, len(body) - len(_MARK))
    body += [""] * (len(rows) - len(body))
    out = []
    for a, text in zip(rows, body):
        # The mark is the figure, so it carries the brand tint and the readout
        # beside it stays the live part. Armature first, data second. Once the
        # mark has run out, the readout keeps its column and nothing is painted.
        art = paint("brand", a.ljust(_MARK_W)) if a.strip() else " " * _MARK_W
        out.append(("  " + art + "  " + text).rstrip())
    return "\n".join(out)


def brand(version: str) -> str:
    """The wordmark. The presence dot is the logo's own; the diamond is not
    borrowed here, because a diamond means a step and this is not one."""
    return paint("brand", "▪") + " " + bold("o n e") + " " + faint(version)


# --- the spine -------------------------------------------------------------

def spine_gap() -> str:
    return "  " + paint("line", glyph(SPINE))


def header(title: str, meta: list[str]) -> str:
    """Brand line, then a tee into the run's facts. Used before a one-shot run."""
    rows = ["  " + title]
    for i, m in enumerate(meta):
        joint = glyph(TEE if i < len(meta) - 1 else ELBOW)
        rows.append("  " + paint("line", joint + glyph(HAIRLINE)) + " " + m)
    return "\n".join(rows)


def node(name: str, detail: str, mutating: bool = False,
         step: int | None = None) -> str:
    """One tool call as a node on the spine.

    Solid diamond = this call could change something. Hollow = it only read.
    That is the single most useful thing to know while a run scrolls past, so
    it gets the glyph rather than a word.
    """
    mark = glyph(DIAMOND_SOLID if mutating else DIAMOND_HOLLOW)
    lead = paint("accent" if mutating else "faint", mark)
    num = faint(f"{step} ") if step is not None else ""
    return "  " + lead + " " + num + ink(name.ljust(6)) + " " + dim(one_line(detail, 72))


def leaf(text: str, bad: bool = False) -> str:
    """The result hanging off a node."""
    arm = paint("line", glyph(SPINE) + "   ")
    body = paint("err" if bad else "faint", one_line(text, 78))
    return "  " + arm + body


def note(text: str, token: str = "warm") -> str:
    """A run-level remark on the spine - a compaction, a failover, a warning."""
    return "  " + paint("line", glyph(SPINE) + " ") + paint(token, one_line(text, 84))


# The plan's three marks. A hollow diamond means "looked", a solid one means
# "changed" - neither is a state of work, so the plan gets its own vocabulary
# rather than borrowing the spine's. The armature rule still holds: what is done
# recedes, what is open is dim, and the one live item is the only accent on the
# block.
PLAN_MARKS = {"done": "✓", "in_progress": "▸", "pending": "·"}
PLAN_ASCII = {"done": "x", "in_progress": ">", "pending": "-"}


def plan(items: list[dict], title: str = "plan") -> str:
    """The task list, drawn between turns.

    Every line hangs off the spine, so the block reads as one thing that
    belongs to the run rather than as output from a tool. An empty list is one
    line: a plan that was cleared is news, and printing nothing looks like the
    render failed.
    """
    arm = paint("line", glyph(SPINE))
    if not items:
        return "  " + arm + " " + faint(f"{title} cleared")
    done = sum(1 for it in items if it.get("state") == "done")
    rows = ["  " + arm + " " + kicker(title) + " "
            + dim(f"{done}/{len(items)}")]
    marks = PLAN_MARKS if _DEPTH else PLAN_ASCII
    for it in items:
        state = it.get("state", "pending")
        text = one_line(it.get("text", ""), 76)
        if state == "done":
            body = paint("faint", marks["done"] + " " + text)
        elif state == "in_progress":
            body = paint("accent", marks["in_progress"] + " ") + ink(text)
        else:
            body = paint("faint", marks["pending"] + " ") + dim(text)
        rows.append("  " + arm + " " + body)
    return "\n".join(rows)


def prompt_mark() -> str:
    """The interactive prompt. One diamond, nothing else - the seed console's
    brand mark doing double duty as the caret."""
    return "\n" + paint("accent", glyph(DIAMOND_SOLID)) + " "


def meter(share: float, cells: int = 12) -> str:
    """A share of something, drawn in line rather than as a filled bar.

    House rule 1: nothing is a filled surface. So the meter is one hairline
    rule, the used part in accent and the rest in the field colour - figure
    against armature, which is rule 2. A block bar would put more weight on the
    screen than the number it describes.
    """
    share = min(1.0, max(0.0, share))
    lit = int(round(share * cells))
    return paint("accent", "━" * lit) + paint("line", "━" * (cells - lit))


def footer(bits: list[str]) -> str:
    """The dim stat line under a hairline. Facts only, separated by the web's
    interpunct."""
    # Clipped to one row: on a narrow terminal a wrapped stat line pushes the
    # reply it describes out of a three-row transcript viewport.
    return rule() + "\n  " + dim(one_line(" · ".join(b for b in bits if b), width() - 1))


# --- back-compat with the first cut ---------------------------------------

def tool_line(step: int, name: str, args: dict, mutating: bool = False) -> str:
    from onecli.toolview import describe
    description = untrusted(describe(name, args))
    label, _, detail = description.partition(" · ")
    return node(label, detail, mutating=mutating)


# Model output is untrusted terminal input. An answer carrying `ESC [ 2 J`
# clears the screen; `ESC ] 0 ; ... BEL` rewrites the window title; `ESC [ 6 n`
# makes the terminal WRITE a reply into this process's stdin, which is a way to
# put characters in front of the next shell prompt. None of that is content, so
# none of it reaches a terminal. Newline and tab are kept because they are
# layout the model legitimately produces.
_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_OTHER_ESC = re.compile(r"\x1b[@-_]?")
# C1 controls too: U+009B is a one-character CSI and U+009D an OSC on
# terminals that honour 8-bit controls after UTF-8 decoding.
_C0 = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def untrusted(text: str) -> str:
    """Strip terminal control from text the model produced."""
    if not text:
        return text
    for pattern in (_OSC, _CSI, _OTHER_ESC):
        text = pattern.sub("", text)
    return _C0.sub("", text)

#!/usr/bin/env python3
"""Turn a terminal surface into a PNG, so the look can be reviewed in a PR.

ANSI is not reviewable in a browser and a terminal screenshot needs a terminal.
This converts the escape codes to HTML - the same bytes the terminal gets, no
second implementation of the palette - and shoots it headless.

Two sources. By default it runs `ui-preview.py`, which draws every element
from fixtures. With `--from FILE` it shoots a capture of a real session
instead - a pty recording, which is the only way to see what a person actually
saw, spinner frames and all.

    python3 scripts/ui-shot.py .scratch/ui.png
    python3 scripts/ui-shot.py shot.png --from .scratch/session.ansi

A capture off a pty is not a stream of SGR codes with text between them: it
carries carriage returns, erase-line, and the same row written forty times as
a spinner turns. Handing that straight to `to_html` would stack every frame
one under the next and photograph a smear. So a capture is REPLAYED first -
onto a grid, the way the terminal did it - and only the final grid is
converted. `--cols` must match the width the capture was taken at, or the
wrapping in the PNG is not the wrapping the person saw, which is exactly the
class of defect this is for.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TOKEN = re.compile(r"\033\[([0-9;]*)m")

# The preview writes truecolor; a real session on a real terminal writes
# xterm-256 (`38;5;N`), because that is what `render` emits when COLORTERM is
# not set to truecolor. Mapping the whole cube rather than flattening it to one
# grey is what makes a capture and the preview comparable at all - the first
# version of this file painted every 256-colour run a uniform grey, which turned the
# entire tool spine into the same shade and hid the mutating/reading contrast
# the spine exists to show.
_STEPS = (0, 95, 135, 175, 215, 255)
_BASE16 = (
    (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
    (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
    (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
)


def xterm256(n: int) -> tuple[int, int, int]:
    if n < 16:
        return _BASE16[n]
    if n < 232:
        n -= 16
        return (_STEPS[n // 36], _STEPS[(n // 6) % 6], _STEPS[n % 6])
    v = 8 + (n - 232) * 10
    return (v, v, v)

PAGE = """<!doctype html><meta charset=utf-8><style>
  @font-face{{font-family:plex;src:local("IBM Plex Mono")}}
  html,body{{margin:0;background:#0a0a0a}}
  body{{padding:38px 44px}}
  pre{{margin:0;font-family:plex,"DejaVu Sans Mono",ui-monospace,monospace;
       font-size:15px;line-height:1.62;color:#eae7e0;letter-spacing:.01em;
       -webkit-font-smoothing:antialiased}}
</style><pre>{body}</pre>"""


def to_html(text: str) -> str:
    out, open_spans = [], 0
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    pos = 0
    for m in TOKEN.finditer(text):
        out.append(text[pos:m.start()])
        pos = m.end()
        codes = [c for c in m.group(1).split(";") if c] or ["0"]
        if codes == ["0"]:
            out.append("</span>" * open_spans)
            open_spans = 0
            continue
        style = []
        if codes[0] == "1":
            style.append("font-weight:600")
            codes = codes[1:]
        if not codes:
            codes = [""]
        if codes[:2] == ["38", "2"]:
            style.append("color:rgb(%s,%s,%s)" % tuple(codes[2:5]))
        elif codes[:2] == ["38", "5"]:
            style.append("color:rgb(%s,%s,%s)" % xterm256(int(codes[2])))
        elif codes[:2] == ["48", "2"]:
            style.append("background:rgb(%s,%s,%s)" % tuple(codes[2:5]))
        elif codes[:2] == ["48", "5"]:
            style.append("background:rgb(%s,%s,%s)" % xterm256(int(codes[2])))
        elif codes[0] == "7":
            style.append("filter:invert(1)")
        elif codes[0] == "2":
            style.append("opacity:.55")
        if style:
            out.append('<span style="%s">' % ";".join(style))
            open_spans += 1
    out.append(text[pos:])
    out.append("</span>" * open_spans)
    return "".join(out)


CSI = re.compile(r"\033\[([0-9;?]*)([A-Za-z])")


def replay(raw: str, cols: int) -> str:
    """Run a pty capture through a grid and hand back what is left on it.

    Deliberately small. It handles the codes a one-cli session actually emits -
    measured, not guessed: SGR (kept, as text, for `to_html` to read), carriage
    return, newline, backspace, erase-line in its three forms, and cursor
    moves. Anything else is dropped rather than printed, because an unhandled
    escape rendered as literal text is worse than an absent one: it looks like
    a UI defect and is not.

    Rows grow without bound; a terminal would scroll, and a screenshot of a
    scrolled-away line is a screenshot of nothing.
    """
    grid: list[list[str]] = [[]]
    row = col = 0

    def put(ch: str) -> None:
        nonlocal col
        while len(grid) <= row:
            grid.append([])
        line = grid[row]
        while len(line) <= col:
            line.append(" ")
        line[col] = ch
        col += 1

    i = 0
    pending = ""       # the SGR run to attach to the next printed character
    while i < len(raw):
        ch = raw[i]
        if ch == "\033":
            m = CSI.match(raw, i)
            if not m:
                # OSC (window title) and the two-byte escapes: swallow whole.
                if raw[i:i + 2] == "\033]":
                    end = raw.find("\007", i)
                    i = (end + 1) if end != -1 else i + 2
                else:
                    i += 2
                continue
            args, final = m.group(1), m.group(2)
            i = m.end()
            if final == "m":
                pending += m.group(0)
            elif final == "K":
                n = args or "0"
                while len(grid) <= row:
                    grid.append([])
                if n == "0":
                    del grid[row][col:]
                elif n == "1":
                    grid[row][:col] = [" "] * col
                else:
                    grid[row] = []
            elif final in "AB":
                row = max(0, row + (-1 if final == "A" else 1)
                          * int(args or 1))
            elif final in "CD":
                col = max(0, col + (1 if final == "C" else -1) * int(args or 1))
            elif final == "H":
                parts = (args or "1;1").split(";")
                row = max(0, int(parts[0] or 1) - 1)
                col = max(0, int(parts[1] if len(parts) > 1 else 1) - 1)
            elif final == "J" and (args or "0") == "2":
                grid, row, col = [[]], 0, 0
            continue
        if ch == "\r":
            col = 0
        elif ch == "\n":
            row += 1
            col = 0
            while len(grid) <= row:
                grid.append([])
        elif ch == "\b":
            col = max(0, col - 1)
        elif ch == "\t":
            for _ in range(4 - (col % 4)):
                put(" ")
        elif ch >= " ":
            put(pending + ch)
            pending = ""
        i += 1

    # A hard wrap the terminal would have applied. Doing it here rather than in
    # CSS is the point: an overlong row has to LOOK overlong in the PNG, since
    # a table that wraps at the terminal's width is the defect being hunted.
    out: list[str] = []
    for line in grid:
        text = "".join(line).rstrip()
        while _visible_len(text) > cols:
            head, text = _cut(text, cols)
            out.append(head)
        out.append(text)
    return "\n".join(out).rstrip("\n") + "\n"


def _visible_len(s: str) -> int:
    return len(TOKEN.sub("", s))


def _cut(s: str, cols: int) -> tuple[str, str]:
    """Split at `cols` VISIBLE characters, counting escapes as zero-width."""
    seen = 0
    i = 0
    while i < len(s) and seen < cols:
        m = TOKEN.match(s, i)
        if m:
            i = m.end()
            continue
        seen += 1
        i += 1
    return s[:i], s[i:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?",
                    default=os.path.join(ROOT, ".scratch", "ui.png"))
    ap.add_argument("--from", dest="source",
                    help="a pty capture to shoot instead of the preview")
    ap.add_argument("--cols", type=int, default=96,
                    help="the width the capture was taken at")
    ap.add_argument("--height", type=int, default=1640)
    args = ap.parse_args()

    target = os.path.abspath(args.target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if args.source:
        raw = open(args.source, "rb").read().decode("utf-8", "replace")
        ansi = replay(raw, args.cols)
    else:
        env = dict(os.environ, FORCE_COLOR="1", COLORTERM="truecolor",
                   COLUMNS=str(args.cols))
        # check=True on purpose: a preview that crashed halfway would otherwise
        # be screenshotted as a short, plausible-looking page. That is how the
        # amber token rename got as far as a PNG nobody could tell was
        # truncated.
        run = subprocess.run([sys.executable,
                              os.path.join(HERE, "ui-preview.py")],
                             capture_output=True, text=True, env=env, cwd=ROOT,
                             check=True)
        ansi = run.stdout
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(PAGE.format(body=to_html(ansi)))
        page = fh.name
    subprocess.run([
        "chromium", "--headless", "--no-sandbox", "--use-angle=vulkan",
        "--hide-scrollbars", "--force-device-scale-factor=2",
        # 9.2px is the advance of 15px DejaVu Sans Mono plus the .01em
        # letter-spacing above, and 88 is the two paddings. Measured, not
        # guessed: at 9.0 the last column of a 96-wide capture fell off the
        # right edge of the PNG, which reads exactly like the wrapping defect
        # this tool is for and was the tool's own.
        f"--window-size={88 + int(args.cols * 9.2)},{args.height}",
        f"--screenshot={target}", f"file://{page}",
    ], check=True, capture_output=True)
    os.unlink(page)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

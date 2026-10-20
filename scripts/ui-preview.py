#!/usr/bin/env python3
"""Print the whole terminal surface at once, with nothing behind it.

A UI you can only see by running a real turn against a real endpoint is a UI
nobody reviews. This renders every element from fixtures, so the look can be
judged - and screenshotted - in a second, offline.

    FORCE_COLOR=1 COLORTERM=truecolor python3 scripts/ui-preview.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FORCE_COLOR", "1")
os.environ.setdefault("COLORTERM", "truecolor")

from onecli import __version__ as V, relief, render, spinner  # noqa: E402

render.refresh()

facts = [
    render.ink("example-model-27b") + render.faint("  local · openai"),
    render.dim("~/one-cli"),
    render.faint("fallback local → provider-b → hosted/coder-12b"),
]

print()
print(render.banner(V, [render.brand(V), render.kicker("the instrument"), ""]
                    + facts + ["", render.faint("session a41c9e · ctrl-d or /exit")]))
print()
print(render.rule())
print()
print(render.header(render.brand(V), facts + [render.paint("warm", "acceptEdits")]))
print(render.spine_gap())
print(render.node("read", "onecli/render.py", mutating=False, step=1))
print(render.leaf("→ 1  \"\"\"Terminal surface, in the one-cli visual language."))
print(render.node("grep", "DIAMOND_SOLID", mutating=False, step=2))
print(render.leaf("→ onecli/render.py:58:DIAMOND_SOLID = \"◆\""))
print(render.node("edit", "onecli/render.py", mutating=True, step=3))
print(render.leaf("→ edited onecli/render.py (1 replacement)"))
print(render.node("bash", "python -m pytest tests -q", mutating=True, step=4))
print(render.leaf("→ 313 passed in 12.10s"))
print(render.note("compacted 41 messages to 12, 61,204 -> 18,880 tokens", token="faint"))
print(render.node("write", "docs/manual.md", mutating=True, step=5))
print(render.leaf("! error: read-only mode refused write", bad=True))
print(render.note("local failed, retrying on provider-b: connection refused"))
print(render.spine_gap())
print()
print("The diamond is filled when a step could change something, hollow when it")
print("only looked. That is the fastest thing to read in a scrolling run.")
print(render.footer(["provider-b (from local)", "5 steps", "9 tools",
                     "12.3s", "$0.0021", "session a41c9e"]))
print()
print(render.prompt_mark() + render.faint("what changed?"))
print()
print(render.rule())
print()
print(render.kicker("motion") + "   " + render.faint(
    "the head moves only when something arrived"))
print()

fed = spinner.Spinner(spinner.ARRIVE, "qwen3.8-27b", enabled=False)
for _ in range(6):
    fed.feed(240)
print(fed._frame() + render.faint("     tokens arriving"))

work = spinner.Spinner(spinner.WORK, "bash", enabled=False)
for _ in range(9):
    work._frame()
work._started -= 7
print(work._frame() + render.faint("     a subprocess, clocked by time"))

stalled = spinner.Spinner(spinner.ARRIVE, "qwen3.8-27b", enabled=False)
stalled.feed(240)
stalled._last_feed -= 9
print(stalled._frame() + render.faint("     nothing has arrived for 9s"))
print()
print(render.rule())
print()
print(render.kicker("relief") + "   " + render.faint(
    "colour is state, the glyph is value - and a gap is a dot, not a zero"))
print()

# The arena board. A score is a share, so it gets a ramp; a lane that never
# reached the endpoint has no share at all and gets the unmeasured dot.
for line in relief.bars(
    [("one", 0.8), ("hermes", 0.5), ("opencode", 0.6),
     ("pi", 0.4), ("qwen-code", None)],
    width=22,
    state=lambda name, share: "ok" if share and share >= 0.7 else "warn",
):
    print(line)
print()
print("  " + render.dim("budget".ljust(9)) + "  " + relief.chip(0.34, cells=22)
      + "  " + render.faint("$0.17 of $0.50"))
print()

# A 1-D reading as a skyline. The missing column is a probe that did not
# report, and it stays a hole in every row of the band.
rate = [0.15, 0.3, 0.55, 0.9, 1.0, 0.75, 0.5, None, 0.35, 0.6,
        0.85, 0.95, 0.7, 0.4, 0.25, 0.45, 0.8, 0.6, 0.3, 0.2]
print("  " + render.kicker("tokens/s") + render.faint("   20 samples, one missing"))
for line in relief.profile(rate, depth=4):
    print("  " + line)
print()

# Two axes at once: the run's cost surface across lanes and tasks.
print("  " + render.kicker("cost surface") + render.faint("   lanes across, tasks back"))
for line in relief.field(
        lambda c, r: None if (c, r) == (9, 2) else
        max(0.0, min(1.0, 0.5 + 0.45 * math.sin(c / 4.0) * math.cos(r / 2.6))),
        44, 6, tilt=0.5):
    print("  " + line)
print()

#!/usr/bin/env python3
"""Draw a bench run as a chart.

One horizontal bar per backend, split into the three tasks by wall time, sorted
fastest-first. A failed cell is drawn in the failure colour with a hatch and the
reason printed on it, so a slow pass and a fast failure never look alike - the
number on the axis is time, and the colour carries the outcome.

    scripts/bench-chart.py .scratch/bench/<run>/rows.jsonl out.png
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Dark-surface steps of the shared categorical palette, validated as a set
# (worst adjacent CVD dE 9.4, normal-vision 26.5, all >= 3:1 on this surface).
SURFACE = "#1a1a19"
TEXT = "#ffffff"
TEXT_2 = "#c3c2b7"
GRID = "#3a3a37"
TASK_COLOUR = {"bugfix": "#3987e5", "search": "#d95926", "create": "#199e70"}
FAIL = "#e66767"
TASKS = ("bugfix", "search", "create")


def load(path: Path) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by[row["backend"]][row["task"]] = row
    return by


def reason(row: dict) -> str:
    return (row.get("why") or row.get("stopped") or row.get("error") or "fail")[:14]


def draw(by: dict[str, dict[str, dict]], out: Path, title: str) -> None:
    lanes = sorted(
        by,
        key=lambda b: (
            -sum(1 for r in by[b].values() if r["ok"]),
            sum(r["seconds"] for r in by[b].values()),
        ),
    )
    lanes.reverse()  # matplotlib draws y upward; best lane on top

    fig_h = max(4.0, 0.34 * len(lanes) + 1.6)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    GAP = 0.35  # seconds of surface between segments, the 2px spacer in data units
    for y, lane in enumerate(lanes):
        left = 0.0
        for task in TASKS:
            row = by[lane].get(task)
            if row is None:
                continue
            width = max(row["seconds"], 0.4)
            ok = row["ok"]
            ax.barh(
                y, width, left=left, height=0.62,
                color=TASK_COLOUR[task] if ok else FAIL,
                hatch=None if ok else "///",
                edgecolor=SURFACE, linewidth=1.4,
            )
            if not ok:
                ax.text(left + width + 1.2, y, reason(row), va="center", ha="left",
                        color=FAIL, fontsize=7.5)
                left += 9 * len(reason(row)) * 0.02
            left += width + GAP
        passed = sum(1 for r in by[lane].values() if r["ok"])
        ax.text(-1.2, y, f"{passed}/{len(by[lane])}", va="center", ha="right",
                color=TEXT if passed == len(by[lane]) else TEXT_2, fontsize=8)

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=8.5, color=TEXT)
    ax.tick_params(axis="y", length=0, pad=34)
    ax.tick_params(axis="x", colors=TEXT_2, labelsize=8)
    ax.set_xlabel("seconds (three tasks end to end, lower is better)",
                  color=TEXT_2, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(-0.8, len(lanes) - 0.2)

    # One pathological cell (a model that thinks for three minutes) would squash
    # every other lane into the first inch, so clip the axis at the 90th
    # percentile of lane totals and label whatever runs off the edge.
    totals = sorted(sum(r["seconds"] for r in by[b].values()) for b in lanes)
    cap = totals[int(0.9 * (len(totals) - 1))] * 1.35
    if totals[-1] > cap:
        ax.set_xlim(0, cap)
        for y, lane in enumerate(lanes):
            total = sum(r["seconds"] for r in by[lane].values())
            if total > cap:
                ax.text(cap * 0.995, y, f"{total:.0f}s >", va="center", ha="right",
                        color=SURFACE, fontsize=8, fontweight="bold")

    handles = [mpatches.Patch(facecolor=TASK_COLOUR[t], label=t) for t in TASKS]
    handles.append(mpatches.Patch(facecolor=FAIL, hatch="///", label="failed"))
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
                    ncol=4)
    for text in leg.get_texts():
        text.set_color(TEXT_2)

    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    rows, out = Path(argv[1]), Path(argv[2])
    title = argv[3] if len(argv) > 3 else "one-cli bench"
    by = load(rows)
    draw(by, out, title)
    print(f"{out} - {len(by)} backends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

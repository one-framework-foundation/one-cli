"""TUI and Theme layouts for one-cli.

Calibrated for narrow terminals (~42 columns).
5 Distinct Chat Styles:
1. mono      : Pure Minimal (linear stream, hairline rules, amber chevron)
2. pill      : Rounded Card Bubbles (chat-app feel, high contrast)
3. glyph     : Kinetic Glyph Aviary (3-line ASCII horizon, tree dialogue)
4. phosphor  : Tactical CRT Monospace (amber/green phosphor, telemetry tags)
5. zen       : Zen Paper (quiet whitespace, indented folding, gold accents)
"""

from __future__ import annotations

import os
import socket

from onecli import render

THEMES = (
    "mono", "pill", "glyph", "phosphor", "zen",
    "avionics", "darkflow", "aviary", "bento", "cyberdeck"
)
DEFAULT_THEME = "mono"


def _local_host() -> str:
    return socket.gethostname().split(".")[0]


def banner_mono(version: str, backend_name: str, model_name: str,
                workspace: str, sid: str, mode: str = "default") -> str:
    host = _local_host()
    short_ws = render.short_path(workspace) if workspace else "."
    width = get_terminal_width(default=44, min_w=36, max_w=80)
    rule_bar = "─" * min(width, 42)
    mode_tag = f" · {mode}" if mode and mode != "default" else ""
    lines = [
        f"{render.paint('brand', '▪')} {render.bold('one')} " + render.faint(f"v{version} · {host}"),
        f"{render.ink(model_name)}" + render.dim(f" · {short_ws}{mode_tag}"),
        render.paint("line", rule_bar),
        render.faint("ready · /help for commands · ctrl-d to exit"),
        "",
    ]
    return "\n".join(lines)


def banner_pill(version: str, backend_name: str, model_name: str,
                workspace: str, sid: str, mode: str = "default") -> str:
    host = _local_host()
    lines = [
        render.cyan(f"╭─ ◈ one ────── {host} ── v{version} ─╮"),
        render.cyan("│ ") + render.ink(model_name),
        render.cyan("╰────────────────────────────────────╯"),
        "",
        render.faint(f"Ready on {host}. Chat mode active."),
        "",
    ]
    return "\n".join(lines)


def banner_glyph(version: str, backend_name: str, model_name: str,
                 workspace: str, sid: str, mode: str = "default") -> str:
    host = _local_host()
    lines = [
        "  " + render.paint("brand", ".  :||:  .") + f"   one-cli · {host}",
        "  " + render.paint("brand", "   /  \\   ") + f"   {render.ink(model_name)}",
        "",
        render.faint(f"◇ Ready on {host}. Type your message below."),
        "",
    ]
    return "\n".join(lines)


def banner_phosphor(version: str, backend_name: str, model_name: str,
                    workspace: str, sid: str, mode: str = "default") -> str:
    host = _local_host()
    lines = [
        render.yellow(f"[one//cli] ─── [{host.upper()}] ─── [TERMINAL DECK]"),
        render.yellow(f"MODEL: {model_name}"),
        render.yellow("══════════════════════════════════════════"),
        "",
        render.faint("Awaiting operator instruction..."),
        "",
    ]
    return "\n".join(lines)


def banner_zen(version: str, backend_name: str, model_name: str,
               workspace: str, sid: str, mode: str = "default") -> str:
    host = _local_host()
    lines = [
        "· · one · ·",
        f"{render.ink(model_name)} · {host}",
        "",
        render.faint("Ready. What are we exploring today?"),
        "",
    ]
    return "\n".join(lines)


def render_banner(theme: str, version: str, backend_name: str, model_name: str,
                  workspace: str, sid: str, mode: str = "default") -> str:
    t = (theme or "").lower().strip()
    if t in ("mono", "darkflow"):
        return banner_mono(version, backend_name, model_name, workspace, sid, mode)
    elif t == "pill":
        return banner_pill(version, backend_name, model_name, workspace, sid, mode)
    elif t in ("glyph", "aviary"):
        return banner_glyph(version, backend_name, model_name, workspace, sid, mode)
    elif t in ("phosphor", "cyberdeck"):
        return banner_phosphor(version, backend_name, model_name, workspace, sid, mode)
    elif t in ("zen", "bento"):
        return banner_zen(version, backend_name, model_name, workspace, sid, mode)
    else:
        return banner_mono(version, backend_name, model_name, workspace, sid, mode)


def get_terminal_width(default: int = 80, min_w: int = 44, max_w: int = 100) -> int:
    try:
        cols = os.get_terminal_size().columns
        return max(min_w, min(max_w, cols - 4))
    except (AttributeError, OSError, ValueError):
        return default


def render_task_card(
    task_id: str,
    title: str,
    phase_num: int,
    total_phases: int,
    phase_name: str,
    status: str,
    glyph_rows: list[str],
    elapsed: float,
    announcement: str = "",
    width: int | None = None,
) -> str:
    """Render a dedicated detached offload task card with procedural ASCII glyphs."""
    if width is None:
        width = get_terminal_width()
    inner_w = max(28, width - 4)
    stat_badge = f"[{status}]"
    header_left = f"╭─ ◆ TASK #{task_id} "
    header_fill = width - len(header_left) - len(stat_badge) - 3
    if header_fill < 1:
        header_middle = f"╭─ ◆ TASK #{task_id} "
    else:
        header_middle = f"{header_left}{'─' * header_fill} "
    
    badge_paint = render.paint("live", stat_badge) if status in ("RUNNING", "LIVE", "DONE") else render.paint("accent", stat_badge)
    header = render.paint("brand", header_middle) + badge_paint + render.paint("brand", " ─╮")

    g_width = 11
    padded_glyphs = []
    for r in range(3):
        if r < len(glyph_rows):
            padded_glyphs.append(glyph_rows[r][:g_width].ljust(g_width))
        else:
            padded_glyphs.append(" " * g_width)

    right_w = max(10, inner_w - g_width - 2)
    # Line 1: Title
    r1 = title[:right_w].ljust(right_w)
    # Line 2: Phase progress
    r2 = f"Phase {phase_num}/{total_phases}: {phase_name}"[:right_w].ljust(right_w)
    # Line 3: Elapsed & announcement
    ann_len = max(12, right_w - 10)
    ann_short = announcement[:ann_len] if announcement else "active"
    r3 = f"{elapsed:.1f}s · {ann_short}"[:right_w].ljust(right_w)

    pipe = render.paint("line", "│ ")
    rpipe = render.paint("line", " │")

    row1 = pipe + render.paint("accent", padded_glyphs[0]) + "  " + render.ink(r1) + rpipe
    row2 = pipe + render.paint("accent", padded_glyphs[1]) + "  " + render.dim(r2) + rpipe
    row3 = pipe + render.paint("accent", padded_glyphs[2]) + "  " + render.faint(r3) + rpipe

    footer = render.paint("brand", "╰" + ("─" * (width - 2)) + "╯")
    return "\n".join([header, row1, row2, row3, footer])

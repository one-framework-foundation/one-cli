"""ASCII glyph engine for one-cli.

Runs assets/glyph_terminal.mjs under Node.js with fail-open fallback.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import select
import shutil
import subprocess

ASSET_PATH = Path(__file__).resolve().parent / "assets" / "glyph_terminal.mjs"


class GlyphEngine:
    def __init__(self, asset_path: Path | None = None) -> None:
        self.asset_path = asset_path or ASSET_PATH
        self.proc: subprocess.Popen | None = None
        self._disabled = False
        self._boot()
        atexit.register(self.close)

    def _boot(self) -> None:
        node = shutil.which("node")
        if not node or not self.asset_path.is_file():
            self._disabled = True
            return
        try:
            self.proc = subprocess.Popen(
                [node, str(self.asset_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self._disabled = True
            self.proc = None

    def frame(
        self,
        cols: int,
        rows: int,
        dt: float = 0.1,
        action: str | None = None,
        selection: int = 0,
        reduced: bool = False,
    ) -> list[str]:
        cols, rows = max(1, cols), max(1, rows)
        reduced = reduced or os.environ.get('ONE_REDUCED_MOTION', '').lower() in ('1', 'true', 'yes')
        if reduced:
            return self._fallback_frame(cols, rows)
        proc = self.proc
        if (self._disabled or proc is None or proc.poll() is not None
                or proc.stdin is None or proc.stdout is None):
            return self._fallback_frame(cols, rows)
        try:
            payload = json.dumps({
                "cols": cols,
                "rows": rows,
                "dt": dt,
                "action": action,
                "selection": selection,
                "reduced": reduced,
            }) + "\n"
            proc.stdin.write(payload)
            proc.stdin.flush()
            r, _, _ = select.select([proc.stdout], [], [], 1.5)
            if not r:
                self.close()
                self._disabled = True
                return self._fallback_frame(cols, rows)
            line = proc.stdout.readline()
            if not line:
                return self._fallback_frame(cols, rows)
            data = json.loads(line)
            lines = data.get("lines")
            if not lines:
                return self._fallback_frame(cols, rows)
            return [line[:cols].ljust(cols) for line in lines[:rows]]
        except (OSError, ValueError):
            return self._fallback_frame(cols, rows)

    def _fallback_frame(self, cols: int, rows: int) -> list[str]:
        lines = []
        for r in range(rows):
            if r == 0 or r == rows - 1:
                lines.append(("." + " " * max(0, cols - 2) + ".")[:cols])
            elif r == rows // 2:
                mid = ". . one . ."
                lines.append(mid[:cols].center(cols))
            else:
                lines.append(" " * cols)
        return lines

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=0.5)
            except Exception:
                pass
            self.proc = None


_INSTANCE: GlyphEngine | None = None


def get_engine() -> GlyphEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = GlyphEngine()
    return _INSTANCE


def compact_card_glyph(step: int = 0, excite: bool = False, cols: int = 12, rows: int = 3) -> list[str]:
    """Return a compact multi-row ASCII glyph matrix for task cards."""
    try:
        eng = get_engine()
        lines = eng.frame(cols=cols, rows=rows, dt=0.15, action="excite" if excite else None)
        if lines and len(lines) >= rows:
            return [line[:cols].ljust(cols) for line in lines[:rows]]
    except Exception:
        pass
    # Deterministic procedural fallback
    chars = [".", ":", "/", "|", "\\", "+", "*", "o"]
    res = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            idx = (step * 3 + r * 7 + c * 5 + (13 if excite else 0)) % len(chars)
            row_chars.append(chars[idx])
        res.append("".join(row_chars))
    return res

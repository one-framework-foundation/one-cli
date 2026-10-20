"""The small interactive parts: a menu, a confirm, a progress line.

Deliberately plain. Onboarding is the one place where a person who has never
run this before is reading, so every prompt says what will happen and what the
default does, and nothing irreversible is a default.
"""

from __future__ import annotations

import sys

from .. import render


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def choose(title: str, options: list[tuple[str, str]], default: int = 0,
           allow_quit: bool = True) -> int:
    """Numbered menu. Returns the index, or -1 for quit.

    Falls back to the default without prompting when there is no terminal, so
    the same code path is safe inside a script.
    """
    if not _tty():
        return default
    print(render.kicker(title), file=sys.stderr)
    for i, (label, detail) in enumerate(options, 1):
        line = render.ink(f"  {i}. {label}")
        if detail:
            line += render.faint(f"   {detail}")
        print(line, file=sys.stderr)
    tail = "q to quit, " if allow_quit else ""
    hint = f"[{tail}enter = {options[default][0]}]" if options else ""
    while True:
        try:
            raw = input(render.dim(f"  choose {hint} ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return -1
        if not raw:
            return default
        if allow_quit and raw in ("q", "quit", "exit"):
            return -1
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(render.yellow(f"  not one of 1-{len(options)}"), file=sys.stderr)


def confirm(question: str, default: bool = False) -> bool:
    if not _tty():
        return default
    hint = "Y/n" if default else "y/N"
    try:
        raw = input(render.dim(f"  {question} [{hint}] ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return False
    if not raw:
        return default
    return raw in ("y", "yes")


def ask(question: str, default: str = "") -> str:
    if not _tty():
        return default
    try:
        raw = input(render.dim(f"  {question} "
                               f"{f'[{default}] ' if default else ''}")).strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return default
    return raw or default


def bar(share: float, cells: int = 24) -> str:
    filled = max(0, min(cells, int(round(share * cells))))
    return "[" + "#" * filled + "-" * (cells - filled) + "]"


class Line:
    """A single rewritten stderr line, so a 19 GB pull is one line not 19000."""

    def __init__(self, every: float = 0.5):
        self.every = every
        self._last = 0.0
        self._width = 0

    def update(self, text: str, force: bool = False) -> None:
        import time
        now = time.time()
        if not force and now - self._last < self.every:
            return
        self._last = now
        pad = " " * max(0, self._width - len(text))
        self._width = len(text)
        if sys.stderr.isatty():
            print("\r" + text + pad, end="", file=sys.stderr, flush=True)
        else:
            print(text, file=sys.stderr, flush=True)

    def done(self, text: str = "") -> None:
        if sys.stderr.isatty():
            print("\r" + " " * self._width + "\r", end="", file=sys.stderr)
        if text:
            print(text, file=sys.stderr)
        self._width = 0

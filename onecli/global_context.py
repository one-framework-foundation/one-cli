"""The user's explicit choice of global instruction file.

With no recorded choice one-cli reads the Claude instruction file. The doctor
shows that this is the default, not a choice, until setup records one.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import secureio

SOURCES = ("claude", "one")


def _home(home: str | None) -> Path:
    return Path(os.path.expanduser("~") if home is None else home)


def choice_path(home: str | None = None) -> Path:
    return _home(home) / ".one-cli" / "context-source"


def source(home: str | None = None) -> tuple[str, bool]:
    """(source, explicit). Missing is the compatible Claude default."""
    path = choice_path(home)
    try:
        with secureio.text_file(path) as fh:
            value = fh.read().strip()
    except FileNotFoundError:
        return "claude", False
    if value not in SOURCES:
        raise ValueError(f"{path} must contain 'claude' or 'one'")
    return value, True


def instruction_path(home: str | None = None) -> Path:
    selected, _ = source(home)
    if selected == "one":
        return _home(home) / ".one-cli" / "AGENTS.md"
    return _home(home) / ".claude" / "CLAUDE.md"


def set_source(value: str, home: str | None = None) -> Path:
    if value not in SOURCES:
        raise ValueError(f"unknown global context source: {value}")
    path = choice_path(home)
    with secureio.text_file(path, "w", private=True) as fh:
        fh.write(value + "\n")
    return path


def doctor_lines(home: str | None = None) -> list[str]:
    selected, explicit = source(home)
    path = instruction_path(home)
    lines = [f"global       {path} ({'chosen' if explicit else 'default'})"]
    if not explicit:
        lines.append("             choose the global instruction source with `one setup`")
    if not path.is_file():
        lines.append("             selected instruction file is missing")
    return lines

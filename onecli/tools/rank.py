"""Backend-neutral query planning and ranking for the grep tool.

ripgrep and the pure-Python walk only generate candidate lines. Every
candidate is re-checked with Python's `re`, filtered by one glob rule, ranked
and printed here, so every backend gives the same ranked output for the
same files - the property `tests/test_search_parity.py` holds. A host without
rg changes the speed and the announced backend, never the answer.

Ranking is per file, in this order (prior art: ripgrep smart-case, fzf/aider
path and filename bonuses, a coverage term in place of BM25's idf, which needs
corpus statistics a single grep call does not have):

1. how many distinct query terms the file carries (multi-term queries),
2. whether a line holds the whole phrase,
3. query terms in the path, the file name counting double,
4. match count, damped,
5. how recently the file changed (mtime: portable across all backends and
   the subprocess-free path, where git history is not available),
6. path, so ties are deterministic.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Callable, Iterable

META = set(".^$*+?()[]{}|\\")
CANDIDATES = 20000    # candidate lines read from any backend
MAX_RESULTS = 100     # largest max_results grep honours
PER_FILE = 50         # candidates kept per file, by line number
SHOWN_PER_FILE = 8    # matched lines printed per file
LINE_CHARS = 300


@dataclass(frozen=True)
class Plan:
    pattern: str
    literal: bool
    ignore_case: bool
    terms: tuple[str, ...]
    rx: re.Pattern
    backend: str      # portable across git grep -E, rg and Python re

    @property
    def mode(self) -> str:
        kind = "literal" if self.literal else "regex"
        if len(self.terms) > 1:
            kind += f", any of {len(self.terms)} terms"
        return kind + (", case-insensitive" if self.ignore_case else ", case-sensitive")


def plan(pattern: str) -> Plan | str:
    """Literal unless the pattern uses regex syntax; smart case like rg -S.

    A literal pattern with spaces is a multi-term query: a line matches on any
    term, and files carrying more of the terms rank first.
    """
    if not isinstance(pattern, str) or not pattern.strip():
        return "error: grep needs a non-empty pattern"
    literal = not (set(pattern) & META)
    # Escapes like \S or \W are not uppercase text, as in ripgrep's smart case.
    letters = re.sub(r"\\.", "", pattern)
    ignore_case = letters == letters.lower()
    flags = re.IGNORECASE if ignore_case else 0
    if literal:
        terms = tuple(dict.fromkeys(pattern.split()))
        backend = "|".join(terms)   # no metacharacters, so valid in every dialect
        rx = re.compile("|".join(re.escape(t) for t in terms), flags)
    else:
        terms = ()
        backend = pattern
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            return f"error: bad regex: {exc}"
    return Plan(pattern, literal, ignore_case, terms, rx, backend)


def glob_ok(glob: str, relative: str) -> bool:
    """One glob rule for every backend: no '/' matches the file name, else the
    path below the search root ('*' may cross directories, as in git)."""
    if not glob:
        return True
    target = relative if "/" in glob else relative.rsplit("/", 1)[-1]
    if glob.startswith("**/") and fnmatchcase(target, glob[3:]):
        return True
    return fnmatchcase(target, glob)


@dataclass
class Hits:
    path: Path
    relative: str
    lines: list[tuple[int, str]]
    mtime: float = 0.0


def collect(rows: Iterable[tuple[Path, str, int, str]], p: Plan, glob: str) -> tuple[list[Hits], bool]:
    """Verify backend rows with Python re and group them per file.

    Rows are (absolute path, path relative to the search root, line, text).
    Returns the files and whether the candidate cap was hit.
    """
    files: dict[Path, Hits] = {}
    capped = False
    for count, (path, relative, number, text) in enumerate(rows):
        if count >= CANDIDATES:
            capped = True
            break
        text = text.rstrip("\r\n")
        if not glob_ok(glob, relative) or not p.rx.search(text):
            continue
        files.setdefault(path, Hits(path, relative, [])).lines.append((number, text))
    for h in files.values():
        h.lines = sorted(set(h.lines))[:PER_FILE]
    return list(files.values()), capped


def _fold(text: str, p: Plan) -> str:
    return text.lower() if p.ignore_case else text


def _line_terms(text: str, p: Plan) -> int:
    line = _fold(text, p)
    return sum(1 for t in p.terms if _fold(t, p) in line)


def key(h: Hits, p: Plan, now: float):
    terms = [_fold(t, p) for t in p.terms]
    body = [_fold(text, p) for _, text in h.lines]
    covered = sum(1 for t in terms if any(t in line for line in body))
    phrase = len(terms) > 1 and any(_fold(p.pattern, p) in line for line in body)
    rel = h.relative.lower()
    name = rel.rsplit("/", 1)[-1]
    path_hits = sum(2 if t.lower() in name else 1 if t.lower() in rel else 0 for t in p.terms)
    density = int(math.log2(1 + len(h.lines)))
    age = now - h.mtime
    recency = 3 if age < 86400 else 2 if age < 7 * 86400 else 1 if age < 30 * 86400 else 0
    return (-covered, -phrase, -path_hits, -density, -recency, h.relative)


def render(files: list[Hits], p: Plan, header: str, max_results: int,
           context: int, read: Callable[[Path], list[str] | None],
           capped: bool, notes: list[str], now: float | None = None) -> str:
    """Grouped, bounded output. Every cut is counted in the result text."""
    now = time.time() if now is None else now
    total = sum(len(h.lines) for h in files)
    files = sorted(files, key=lambda h: key(h, p, now))
    head = (f"[{header}] {total} matching lines in {len(files)} files for "
            f"{p.pattern!r} ({p.mode}); best files first")
    if not files:
        return "\n".join([f"[{header}] (no matches for {p.pattern!r}; {p.mode})", *notes])
    blocks: list[str] = []
    footers: list[str] = []
    shown = 0
    shown_files = 0
    # Spread a small budget over several files instead of one dense file.
    per_file = min(SHOWN_PER_FILE, max(1, max_results // 3))
    for h in files:
        if shown >= max_results:
            break
        shown_files += 1
        # The lines carrying the most query terms, printed in file order.
        best = sorted(h.lines, key=lambda line: (-_line_terms(line[1], p), line[0]))
        take = sorted(best[:min(per_file, max_results - shown)])
        shown += len(take)
        body = read(h.path) if context else None
        want = {n for n, _ in take}
        printed: set[int] = set()
        block = []
        for number, text in take:
            if body:
                for c in range(max(1, number - context), number):
                    if c not in want and c not in printed and c <= len(body):
                        block.append(f"{h.path}-{c}-{body[c - 1][:LINE_CHARS]}")
                        printed.add(c)
            block.append(f"{h.path}:{number}:{text[:LINE_CHARS]}")
            printed.add(number)
            if body:
                for c in range(number + 1, min(len(body), number + context) + 1):
                    if c not in want and c not in printed:
                        block.append(f"{h.path}-{c}-{body[c - 1][:LINE_CHARS]}")
                        printed.add(c)
        if len(h.lines) > len(take):
            block.append(f"  (+{len(h.lines) - len(take)} more matching lines in this file)")
        blocks.append("\n".join(block))
    hidden = total - shown
    if hidden > 0:
        rest = sum(len(h.lines) for h in files[shown_files:])
        raise_hint = ", or raise max_results" if max_results < MAX_RESULTS else ""
        footers.append(f"({hidden} more matching lines truncated, {rest} of them in "
                   f"{len(files) - shown_files} files not shown; narrow path or glob"
                   f"{raise_hint})")
    if capped:
        footers.append(f"(stopped reading after {CANDIDATES} candidate lines; "
                   "ranking covers only those - narrow the search)")
    if any(len(h.lines) == PER_FILE for h in files):
        footers.append(f"(at most {PER_FILE} matching lines are kept per file)")
    return "\n".join([head, "\n\n".join(blocks), *footers, *notes])

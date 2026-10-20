#!/usr/bin/env python3
"""Refuse a tree that leaks infrastructure, identities or credentials.

Two pattern sets. The generic one below ships with the code: credential
shapes, CGNAT and private LAN addresses, personal home directories, Telegram
chat ids. An optional extra pattern file, when one is present beside this
script, adds patterns for a particular publication - it is read at run time
and never ships; publishing such a list would publish what it protects.

Usage: ``leak-scan.py [ROOT] [--require-private]``. ROOT defaults to the
current directory; in a git checkout the tracked files are scanned, otherwise
every file. Exit 1 on any finding, 2 when ``--require-private`` is given and
the private list is missing. Findings print as ``path:line: category`` and
never echo the matched value.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

PRIVATE_LIST = Path(__file__).with_name("leak-patterns.private.txt")

# Placeholders tests and docs may use instead of real values.
_CGNAT = r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"
_CGNAT_PLACEHOLDER = re.compile(r"^100\.64\.0\.\d{1,3}$")
_HOME_PLACEHOLDERS = {"user", "runner", "you", "me", "alice", "bob", "example"}

GENERIC: dict[str, str] = {
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
    "anthropic_key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "openai_key": r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}",
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "hf_token": r"\bhf_[A-Za-z0-9]{30,}",
    "telegram_bot_token": r"\b\d{8,10}:AA[A-Za-z0-9_-]{33}\b",
    "tailnet_ip": _CGNAT,
    "private_lan_ip": r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
    "home_path": r"/(?:home|Users)/([A-Za-z0-9_.-]*[A-Za-z0-9_-])(?=[/\s\"'`)\].,:;]|$)",
    "tailnet_domain": r"\b[a-z0-9-]+\.ts\.net\b",
    "telegram_chat_id": r"(?<![\d.])-100\d{10}\b",
}

_ALLOWED_DOMAINS = re.compile(r"\bexample\.ts\.net\b")


def _placeholder(category: str, match: re.Match) -> bool:
    if category == "tailnet_ip":
        return bool(_CGNAT_PLACEHOLDER.match(match.group(0)))
    if category == "home_path":
        return match.group(1) in _HOME_PLACEHOLDERS
    if category == "tailnet_domain":
        return bool(_ALLOWED_DOMAINS.fullmatch(match.group(0)))
    return False


def load_private(path: Path = PRIVATE_LIST) -> dict[str, str]:
    """``category<TAB>regex`` per line; ``#`` comments and blanks ignored."""
    patterns: dict[str, str] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        category, sep, regex = line.partition("\t")
        if not sep or not regex.strip():
            raise SystemExit(f"{path}:{n}: expected 'category<TAB>regex'")
        patterns[f"{category.strip()}#{n}"] = regex.strip()
    return patterns


def list_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--full-name", "-z", "--", "."],
                             check=True, capture_output=True, text=True).stdout
        top = Path(subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                                  check=True, capture_output=True, text=True).stdout.strip())
        tracked = [top / p for p in out.split("\0") if p]
        # Release records can be newly written and not staged yet.
        records = root if root.name == "release-stamps" else root / "docs/release-stamps"
        return sorted(set(tracked) | {p for p in records.rglob("*")
                                     if p.is_file() or p.is_symlink()})
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [p for p in root.rglob("*")
                if (p.is_file() or p.is_symlink()) and not _generated(p.relative_to(root))]


# Interpreter and tool caches: generated on the reader's machine, never shipped.
# They hold compiled fixture strings and local compiler paths.
GENERATED_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def _generated(rel: Path) -> bool:
    return bool(GENERATED_DIRS & set(rel.parts)) or rel.suffix == ".pyc"


def literal_path(node: ast.AST) -> str | None:
    """Resolve string-only concatenation, Path division and path constructors."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        left, right = literal_path(node.left), literal_path(node.right)
        if left is not None and right is not None:
            return left + right if isinstance(node.op, ast.Add) else os.path.join(left, right)
        if isinstance(node.op, ast.Div) and (left is not None or right is not None):
            return os.path.join(left or "<dynamic>", right or "<dynamic>")
    if isinstance(node, ast.Call) and not node.keywords:
        name = ast.unparse(node.func)
        if name in ("Path", "pathlib.Path", "os.path.join", "posixpath.join"):
            parts = [literal_path(arg) for arg in node.args]
            if parts and any(part is not None for part in parts):
                return os.path.join(*(part if part is not None else "<dynamic>" for part in parts))
    return None


def scan(root: Path, patterns: dict[str, str]) -> list[tuple[str, int, str]]:
    compiled = {k: re.compile(v) for k, v in patterns.items()}
    findings: list[tuple[str, int, str]] = []
    for path in list_files(root):
        rel = path.relative_to(root).as_posix()
        if path.resolve() == PRIVATE_LIST.resolve():
            continue
        # A symlink ships as its target string, so that is what to scan.
        # Binary files are scanned too: metadata in an image or PDF ships.
        if path.is_symlink():
            text = os.readlink(path)
        else:
            text = path.read_bytes().decode("utf-8", errors="replace")
        for category, rx in compiled.items():
            if rx.search(rel):
                findings.append((rel, 0, category.split("#")[0]))
        # Normalize literal path expressions, never unrelated fixture strings.
        # AST inspection does not import or execute the file being scanned.
        if path.suffix == ".py":
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                tree = ast.Module(body=[], type_ignores=[])
            for node in ast.walk(tree):
                normalized = literal_path(node)
                if normalized is None:
                    continue
                for category, rx in compiled.items():
                    if "#prose" in category:  # prose-only: matched in text, not literals
                        continue
                    if any(not _placeholder(category, m) for m in rx.finditer(normalized)):
                        finding = (rel, 0, category.split("#")[0])
                        if not rx.search(text) and finding not in findings:
                            findings.append(finding)
        for n, line in enumerate(text.splitlines(), 1):
            for category, rx in compiled.items():
                for m in rx.finditer(line):
                    if not _placeholder(category, m):
                        findings.append((rel, n, category.split("#")[0]))
                        break
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--require-private", action="store_true",
                    help="fail when the private pattern list is absent")
    args = ap.parse_args(argv)
    patterns = dict(GENERIC)
    if PRIVATE_LIST.exists():
        patterns.update(load_private())
    elif args.require_private:
        print(f"leak-scan: private pattern list missing: {PRIVATE_LIST}", file=sys.stderr)
        return 2
    else:
        print("leak-scan: private pattern list absent, generic patterns only", file=sys.stderr)
    root = Path(args.root).resolve()
    findings = scan(root, patterns)
    for rel, n, category in findings:
        print(f"{rel}:{n}: {category}")
    print(f"leak-scan: {len(findings)} finding(s) in {root}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

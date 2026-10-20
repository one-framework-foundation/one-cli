#!/usr/bin/env python3
"""Regenerate facts in README.md and docs/manual.md from the repository itself."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MANUAL = ROOT / "docs" / "manual.md"


def replace_block(text: str, name: str, body: str, tag: str = "readme") -> str:
    start = f"<!-- {tag}:{name}:start -->"
    end = f"<!-- {tag}:{name}:end -->"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"missing generated block {tag}:{name!r}")
    return pattern.sub(f"{start}\n{body.rstrip()}\n{end}", text)


def project_config() -> dict:
    """Read the small project table without requiring TOML support on Python 3.10."""
    text = (ROOT / "pyproject.toml").read_text()
    project_text, scripts_text = text.split("[project.scripts]", 1)
    scripts_text = scripts_text.split("[", 1)[0]

    def value(name: str) -> str:
        match = re.search(rf'^\s*{re.escape(name)}\s*=\s*"([^"]+)"', project_text, re.MULTILINE)
        if not match:
            raise SystemExit(f"pyproject.toml is missing project.{name}")
        return match.group(1)

    scripts = dict(re.findall(r'^([\w-]+)\s*=\s*"([^"]+)"', scripts_text, re.MULTILINE))
    return {"version": value("version"), "requires-python": value("requires-python"), "scripts": scripts}


def project_facts(project: dict) -> str:
    # No test count: it changed on every PR and made README.md conflict on every
    # merge. The suite size is in CI, not in the README.
    scripts = project.get("scripts", {})
    commands = ", ".join(f"`{name}`" for name in sorted(scripts))
    return "\n".join(
        (
            f"`v{project['version']}` · Python `{project['requires-python']}` · "
            "standard-library runtime.",
            "",
            f"Installed commands: {commands}.",
        )
    )


def benchmark_index() -> str:
    return (
        "Benchmark reports are dated snapshots, not claims about the current build. "
        "Run `scripts/one-bench` or `scripts/one-arena` for current results."
    )


def render() -> dict[Path, str]:
    """Each generated file with its blocks refreshed."""
    readme = replace_block(README.read_text(), "project-facts",
                           project_facts(project_config()))
    manual = replace_block(MANUAL.read_text(), "benchmarks", benchmark_index(),
                           tag="manual")
    return {README: readme, MANUAL: manual}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if a generated file is stale")
    args = parser.parse_args()
    stale = [path for path, text in render().items() if path.read_text() != text]
    if not stale:
        return 0
    if args.check:
        names = ", ".join(path.relative_to(ROOT).as_posix() for path in stale)
        print(f"{names} is stale; run scripts/update-readme.py", file=sys.stderr)
        return 1
    for path, text in render().items():
        path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

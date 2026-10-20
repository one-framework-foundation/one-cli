"""Skills: discover SKILL.md packages and load one whole in a single step.

A skill is a directory holding a SKILL.md whose frontmatter names it and says
when to use it. Claude Code reads them from ~/.claude/skills; one-cli reads the
same places, so one library serves both. The system prompt lists every skill by
name and description; the `skill` tool returns the full SKILL.md and the
directory its other files live in. Without this a model had to find the file
itself and page through it - DeepSeek read 140 lines of a 50 KB skill and never
came back for the rest.

The workspace wins over the home directory, and one-cli's own directory over
Claude's, so a project can override a shared skill by name.

A workspace skill is project content, and is trusted like the project's hooks
and MCP servers: only with --trust-project-config. Even then it is read with
no-follow traversal from inside its skill root, so a symlink cannot carry the
skill tool out of the workspace. Home skills are the user's own library and
keep following symlinks, as Claude Code follows them. Either way the file
passes the session's path rules before it is read, as a `read` would.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from onecli import secureio


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    project: bool = False


def roots(workspace: str, home: str | None = None,
          trust_project: bool = False) -> list[tuple[Path, bool]]:
    """(root, is it the project's) in precedence order."""
    base = Path(os.path.expanduser("~") if home is None else home)
    home_roots = [(base / ".one-cli" / "skills", False),
                  (base / ".claude" / "skills", False)]
    if not trust_project:
        return home_roots
    # The real workspace, so the no-follow walk below starts from a path with
    # no user symlink in it; everything under it is walked without following.
    ws = Path(os.path.realpath(workspace))
    return [(ws / ".one-cli" / "skills", True),
            (ws / ".claude" / "skills", True)] + home_roots


def _frontmatter(text: str) -> dict:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith((" ", "\t")):
            fields[key.strip()] = value.strip().strip("\"'")
    return fields


def _read(path: Path, project: bool) -> str:
    if not project:
        return path.read_text(errors="replace")
    with secureio.text_file(path, errors="replace") as fh:
        return fh.read()


def discover(workspace: str, home: str | None = None,
             trust_project: bool = False) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    for root, project in roots(workspace, home, trust_project):
        try:
            if project:
                with secureio.directory(root) as fd:
                    entries = [root / name for name in sorted(os.listdir(fd))]
            else:
                entries = sorted(root.iterdir())
        except OSError:  # absent, unreadable or a symlink: no skills here
            continue
        for entry in entries:
            try:
                meta = _frontmatter(_read(entry / "SKILL.md", project))
            except OSError:  # includes a symlinked project skill, refused
                continue
            name = meta.get("name") or entry.name
            found.setdefault(name, Skill(name, meta.get("description", ""),
                                         entry / "SKILL.md", project))
    return found


def disclosure(workspace: str, home: str | None = None,
               trust_project: bool = False) -> str:
    """The system-prompt block naming every skill, or "" when there are none."""
    found = discover(workspace, home, trust_project)
    if not found:
        return ""
    rows = "\n".join(f"- {s.name}: {s.description}" if s.description else f"- {s.name}"
                     for s in found.values())
    return ("Skills - instructions for specific kinds of work. When a task "
            "matches one, load it with the skill tool first and follow it:\n" + rows)


def _refusal(skill: Skill, path_guards) -> str:
    """The path rules' answer to reading this skill, '' when it may be read.

    A home skill is outside the workspace by definition, so the workspace
    boundary (RootsGuard) does not apply to it; the allow and deny rules do.
    """
    for guard in path_guards or ():
        if not skill.project and getattr(guard, "roots", None) is not None:
            continue
        verdict = guard.check("read", {"path": str(skill.path)}, False)
        if not verdict.allowed:
            return verdict.reason
    return ""


def load(workspace: str, name: str, path_guards=None,
         trust_project: bool = False) -> str:
    """The `skill` tool: the whole SKILL.md, and where its other files are."""
    found = discover(workspace, trust_project=trust_project)
    skill = found.get(name)
    if skill is None:
        return (f"error: no skill named {name!r}. Available: "
                f"{', '.join(sorted(found)) or 'none'}")
    refused = _refusal(skill, path_guards)
    if refused:
        return f"error: skill {name!r} cannot be read: {refused}"
    try:
        body = _read(skill.path, skill.project)
    except OSError as exc:
        return f"error: skill {name!r} cannot be read: {exc}"
    return f"Skill {skill.name} ({skill.path}; its other files are in {skill.path.parent}):\n\n{body}"

"""Slash commands the operator writes, as files on disk.

A built-in command is something the CLI knows how to do. A command file is
something the *operator* does often enough to name - "review this diff the way
we review diffs here", "write the commit message in our format". It is a prompt
with a name, kept in the repo next to the code it is about, so it is reviewed,
diffed and shared like anything else.

The format is the one people already have written: a markdown file under
`.claude/commands/`, optional YAML-ish frontmatter, body is the prompt. A
subdirectory namespaces it, so `.claude/commands/git/commit.md` is `/git:commit`
- the same shape those files already assume, because a format that is nearly
compatible is worse than either alternative.

With project trust enabled, two directories are read, project first:

- `<workspace>/.claude/commands/` - shared, committed, requires project trust
- `~/.claude/commands/` - personal, on this machine, about how you work

A project command wins a name collision. A built-in wins over both: a file
called `cost.md` must not quietly redefine `/cost` into something that spends a
turn, because the whole point of a built-in is that it answers without one.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from onecli import secureio

# Only the two frontmatter keys that change what the command *is*. The rest of
# the Claude Code frontmatter vocabulary (allowed-tools, model) describes a
# surface this loop resolves per run, and honouring half of a key is worse than
# ignoring it: it would look enforced.
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
POSITIONAL = re.compile(r"\$([1-9])")


@dataclass
class CommandFile:
    name: str                   # `review`, or `git:commit` for a subdirectory
    body: str
    description: str = ""
    argument_hint: str = ""
    source: str = ""            # the file it came from, for `/help`

    def render(self, rest: str) -> str:
        """The prompt this command sends, with the typed arguments folded in.

        `$1`..`$9` take one word each, `$ARGUMENTS` takes everything. A file
        that names neither still gets the arguments, appended - otherwise
        typing `/review src/foo.py` silently discards the only interesting part
        of the line.
        """
        words = rest.split()
        out = POSITIONAL.sub(
            lambda m: words[int(m.group(1)) - 1]
            if int(m.group(1)) <= len(words) else "", self.body)
        if "$ARGUMENTS" in out:
            return out.replace("$ARGUMENTS", rest).strip()
        if rest and out == self.body:
            return f"{out.strip()}\n\n{rest}"
        return out.strip()


def _frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading `---` block off. Not a YAML parser, and not pretending
    to be: `key: value` lines, everything else in the block ignored."""
    match = FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip().lower()] = value.strip().strip("'\"")
    return meta, text[match.end():]


def parse(path: Path, name: str) -> CommandFile | None:
    try:
        with secureio.text_file(path, errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return None
    meta, body = _frontmatter(raw)
    if not body.strip():
        return None
    return CommandFile(
        name=name,
        body=body.strip(),
        description=meta.get("description", ""),
        argument_hint=meta.get("argument-hint", ""),
        source=str(path),
    )


def _name_for(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ":".join(rel.parts)


def dirs(workspace: str, trust_project: bool = False) -> list[Path]:
    """Where command files are read from, in the order that wins ties."""
    project = [Path(os.path.abspath(workspace)) / ".claude" / "commands"] if trust_project else []
    return project + [Path.home() / ".claude" / "commands"]


def load(workspace: str, reserved: set[str] | None = None,
         extra: list[str] | None = None, *, trust_project: bool = False) -> tuple[dict[str, CommandFile],
                                                  list[str]]:
    """Every command file this session can address. Returns (by name, notes).

    A name a built-in already owns is skipped and reported rather than dropped
    silently: the operator wrote that file expecting it to run, and a command
    that exists on disk but does nothing is the worst of the three outcomes.
    """
    reserved = reserved or set()
    found: dict[str, CommandFile] = {}
    notes: list[str] = []
    roots = dirs(workspace, trust_project) + [Path(p) for p in (extra or [])]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            name = _name_for(root, path)
            if name in reserved:
                notes.append(f"{path.name} shadows the built-in /{name} - skipped")
                continue
            if name in found:
                continue                    # project beat personal; first wins
            cmd = parse(path, name)
            if cmd is not None:
                found[name] = cmd
    return found, notes


__all__ = ["CommandFile", "load", "parse", "dirs"]

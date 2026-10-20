"""Parity with Claude Code: what it does that we do not, tracked mechanically.

one-cli is a coding agent standing next to a mature one. Every capability
Claude Code has and we lack is either a thing to build or a thing to decline on
purpose, and the failure mode is neither: a gap nobody wrote down, discovered by
a user hitting it. So the gaps live here as data, and the check below is what
keeps the data honest.

`FEATURES` is the curated table - each row names a Claude Code capability, what
answers it here, and a status. `status` is deliberately four-valued: `shipped`
and `partial` are claims about our code, `gap` is work we intend, and `declined`
is a capability that belongs to the Anthropic product rather than to a
provider-agnostic agent (a cloud session, a Chrome integration) and will never
be built. A `declined` row is not a hole; an undeclared row is.

The check is mechanical. `drift()` reads the flags out of the installed
`claude --help` and out of our own parser, and reports three things: a Claude
flag no row mentions (their surface grew, our table went stale), a row naming a
Claude flag that no longer exists (they removed it), and a row claiming a
one-cli flag our parser does not have (we removed it, or the row lied). None of
that needs judgement, which is exactly why it is the part that runs in CI.

What it cannot check is whether `shipped` is true - `--resume` existing as a
flag says nothing about resume working. The tests next to the feature do that.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

STATUSES = ("shipped", "partial", "gap", "declined")


@dataclass(frozen=True)
class Feature:
    id: str
    area: str
    claude: str                       # how Claude Code spells it
    status: str
    ours: str = ""                    # how one-cli spells it, if at all
    note: str = ""
    claude_flags: tuple[str, ...] = ()   # flags this row accounts for
    our_flags: tuple[str, ...] = ()      # flags this row claims we have


def _f(*args, **kwargs) -> Feature:
    return Feature(*args, **kwargs)


FEATURES: tuple[Feature, ...] = (
    # --- the model and the turn ------------------------------------------
    _f("model", "model", "--model, --fallback-model", "shipped",
       "--model/-m, --model-id, --base-url, --wire, --fallback-model",
       "We resolve a backend, not a vendor alias: any wire, any endpoint.",
       ("--model", "--fallback-model"),
       ("--model", "--model-id", "--base-url", "--wire", "--fallback-model")),
    _f("effort", "model", "--effort <low|medium|high|xhigh|max>", "shipped",
       "--effort", "Explicit overrides and automatic triage effort use declared backend profiles; unsupported explicit levels fail before requests.",
       ("--effort",), ("--effort",)),
    _f("temperature", "model", "(not exposed)", "shipped",
       "--temperature, --max-tokens, --timeout", "", (),
       ("--temperature", "--max-tokens", "--timeout")),
    _f("betas", "model", "--betas", "declined",
       "", "Anthropic API beta headers. Not meaningful across wires.",
       ("--betas",)),

    # --- tools -------------------------------------------------------------
    _f("builtin-tools", "tools", "Read/Write/Edit/Bash/Glob/Grep/...", "partial",
       "read, write, edit, ls, glob, grep, bash",
       "No NotebookEdit, no MultiEdit, no BashOutput/KillShell (background "
       "shells), no ExitPlanMode."),
    _f("tool-selection", "tools", "--tools (choose from the built-in set)", "gap",
       "", "We can allow/deny by pattern but not hand-pick the offered set. "
       "Note --tools is variadic in claude and will swallow a positional "
       "prompt.",
       ("--tools",)),
    _f("web", "tools", "WebSearch, WebFetch", "gap",
       "",
       "No built-in web tools: reach the web through an MCP server you name."),
    _f("mcp", "tools", "--mcp-config, --strict-mcp-config, `claude mcp`",
       "partial", "--mcp-config, --strict-mcp-config",
       "stdio transport only - HTTP and SSE servers are refused by name rather "
       "than skipped. Config is Claude Code's shape, so a file written for it "
       "runs here. We read the project's .mcp.json but never the user-global "
       "~/.mcp.json: inheriting every server an operator has configured is how "
       "a sandboxed run quietly gets cron editing and agent spawning.",
       ("--mcp-config", "--strict-mcp-config"),
       ("--mcp-config", "--strict-mcp-config")),
    _f("subagents", "tools", "Task tool, --agents, --agent", "shipped",
       "task tool, --agents, --agent, --list-agents, --max-depth",
       "A child is this loop one level down: same guard chain, same hooks, "
       "same workspace, a surface intersected with the parent's, and the "
       "parent's meter OBJECT rather than a copy - so a ceiling is a ceiling "
       "for the whole tree. The depth cap is enforced by construction: the "
       "last allowed level is built without a spawner, so it is never offered "
       "a tool it would only be refused. A preset can also name a vendor CLI "
       "as its `driver`, which Claude Code has no equivalent for.",
       ("--agents", "--agent", "--forward-subagent-text"),
       ("--agents", "--agent", "--list-agents", "--max-depth")),
    _f("todo", "tools", "TodoWrite", "shipped",
       "todo_write",
       "The list is session STATE, not a message: it is restated verbatim "
       "after a compaction, persisted per revision so `--resume` picks it "
       "back up, and drawn between turns rather than buried in a tool result. "
       "At most one item may be in_progress - a write that breaks it is "
       "refused whole. A child gets no list of its own; it returns one result "
       "to a parent that keeps the plan."),
    _f("skills", "tools", "Skill tool, /slash commands, --disable-slash-commands",
       "partial", "Built-in commands and .claude/commands/*.md",
       "Markdown command discovery and argument expansion exist; a complete "
       "SKILL.md catalog and selective resource lifecycle remain to build.",
       ("--disable-slash-commands",)),
    _f("plugins", "tools", "--plugin-dir, --plugin-url, `claude plugin`",
       "declined", "", "Plugin distribution is a product feature; our unit of "
       "sharing is the repo.",
       ("--plugin-dir", "--plugin-url")),
    _f("lsp", "tools", "LSP integration", "gap", "", ""),

    # --- permissions and guards -------------------------------------------
    _f("permission-mode", "permissions",
       "--permission-mode (acceptEdits/plan/bypassPermissions/auto/manual/dontAsk), "
       "--permission-prompts <host|none>",
       "partial", "--permission-mode (default/acceptEdits/plan/"
       "bypassPermissions/readOnly), --read-only",
       "We add readOnly and lack auto/manual/dontAsk - those are interactive "
       "ask-modes, and one-cli never stops to ask. `--permission-prompts none` "
       "is the same rule from the other side: on a non-interactive one-shot our "
       "guards refuse what would prompt, because nobody is there to answer.",
       ("--permission-mode", "--dangerously-skip-permissions",
        "--allow-dangerously-skip-permissions", "--permission-prompts"),
       ("--permission-mode", "--read-only")),
    _f("tool-patterns", "permissions", "--allowedTools, --disallowedTools",
       "shipped", "--allowed-tools, --disallowed-tools",
       "Same clause syntax, including mcp__server__tool patterns, which now "
       "match real tools.",
       ("--allowed-tools", "--disallowed-tools"),
       ("--allowed-tools", "--disallowed-tools")),
    _f("add-dir", "permissions", "--add-dir", "shipped", "--add-dir", "",
       ("--add-dir",), ("--add-dir",)),
    _f("hooks", "permissions", "hooks in settings.json, 8 lifecycle events",
       "shipped", "--hooks, --no-hooks",
       "Same JSON shape and the same exit-code contract, so a hook written for "
       "Claude Code runs here unchanged.",
       (), ("--hooks", "--no-hooks")),
    _f("hook-events-out", "permissions", "--include-hook-events", "gap",
       "", "Hook lifecycle events are not emitted on the output stream.",
       ("--include-hook-events",)),
    _f("restricted-mode", "permissions", "--restricted", "partial",
       "--read-only",
       "--restricted removes only the command/code-running tools and leaves "
       "write and edit in place; --read-only withholds write, edit AND bash. "
       "We have no flag that disarms execution while keeping the file tools.",
       ("--restricted",), ("--read-only",)),
    _f("external-guards", "permissions", "(PreToolUse hooks only)", "shipped",
       "--guard CMD, --guard-timeout",
       "A guard is a program speaking the PreToolUse contract, usable without "
       "a hooks file.", (), ("--guard", "--guard-timeout")),

    # --- context and memory ------------------------------------------------
    _f("project-context", "context", "CLAUDE.md discovery", "shipped",
       "automatic; --no-project-context to withhold", "",
       ("--exclude-dynamic-system-prompt-sections",), ("--no-project-context",)),
    _f("system-prompt", "context", "--system-prompt, --append-system-prompt",
       "partial", "--system, --system-file",
       "We can replace the prompt or read it from a file; we cannot append to "
       "the default one.",
       ("--system-prompt", "--append-system-prompt"),
       ("--system", "--system-file")),
    _f("system-prompt-snapshot", "context", "--system-prompt-snapshot <on|off>",
       "gap", "",
       "Claude Code can record the system prompt once per conversation and "
       "replay it verbatim on every request and resume. one-cli rebuilds the "
       "prompt each turn, so a mid-conversation change to CLAUDE.md or an "
       "agent file is picked up rather than pinned.",
       ("--system-prompt-snapshot",)),
    _f("compaction", "context", "--autocompact, /compact", "shipped",
       "--autocompact/--no-autocompact, --compact-at, --context-window, "
       "--compact-keep", "",
       ("--autocompact",),
       ("--autocompact", "--no-autocompact", "--compact-at", "--context-window",
        "--compact-keep")),
    _f("memory-tool", "context", "auto-memory, memory directory", "gap",
       "", "No cross-session memory."),

    # --- sessions ----------------------------------------------------------
    _f("resume", "sessions", "--resume, --continue, --fork-session, --session-id",
       "partial", "--resume, --sessions, /rename, /fork, /resume",
       "We resume by id or pick from a list, and a session can resume on a "
       "different backend than it started on. Session search, rename and saved "
       "forks exist. No --continue flag or caller-chosen session id.",
       ("--resume", "--continue", "--fork-session", "--session-id",
        "--no-session-persistence", "--name", "--from-pr"),
       ("--resume", "--sessions")),
    _f("worktree", "sessions", "--worktree, --tmux", "declined",
       "", "Git worktree management belongs to the caller, not the agent.",
       ("--worktree", "--tmux")),
    _f("background", "sessions", "--bg, `claude agents`", "partial",
       "/bg, /jobs, /logs, /kill",
       "Session-owned shell jobs exist; this is not a detached agent session "
       "or a claim of reconnect/deployment acceptance.", ("--bg", "--background")),

    # --- I/O ---------------------------------------------------------------
    _f("print-mode", "io", "--print", "shipped",
       "(default; interactive is the opt-in)",
       "one-cli is print-first: a prompt on argv or stdin runs one turn and "
       "exits.", ("--print",)),
    _f("output-format", "io", "--output-format text|json|stream-json", "partial",
       "--json, --stream/--no-stream",
       "We emit text or one JSON result, and stream text deltas. No "
       "stream-json event protocol, which is what a harness needs to drive us.",
       ("--output-format", "--include-partial-messages"),
       ("--json", "--stream", "--no-stream")),
    _f("input-format", "io", "--input-format stream-json, --replay-user-messages",
       "gap", "", "stdin carries a prompt, not a message stream.",
       ("--input-format", "--replay-user-messages")),
    _f("structured-output", "io", "--json-schema", "gap",
       "", "No schema-validated result.", ("--json-schema",)),
    _f("quiet", "io", "(none)", "shipped", "--quiet/-q", "", (), ("--quiet",)),
    _f("multimodal", "io", "images via @path", "shipped",
       "--image, --audio, --video, --media-route, --inline-audio, "
       "--no-media-fallback, --speak",
       "Senses routes each medium: inline to a vision model, OCR, describe, "
       "transcribe, or speak the answer back.",
       ("--file",),
       ("--image", "--audio", "--video", "--media-route", "--inline-audio",
        "--no-media-fallback", "--speak")),
    _f("budget", "io", "--max-budget-usd", "shipped", "--max-budget-usd",
       "", ("--max-budget-usd",), ("--max-budget-usd",)),
    _f("steps", "io", "(no equivalent flag)", "shipped", "--max-steps/-s",
       "", (), ("--max-steps",)),
    _f("workspace", "io", "cwd + --add-dir", "shipped", "--workspace/-w",
       "", (), ("--workspace",)),
    _f("list-backends", "io", "`claude mcp list`, /model", "shipped", "--list",
       "", (), ("--list",)),
    _f("a11y", "io", "--ax-screen-reader", "gap", "", "",
       ("--ax-screen-reader",)),
    _f("prompt-suggestions", "io", "--prompt-suggestions", "declined", "", "",
       ("--prompt-suggestions",)),
    _f("thinking-display", "io", "thinking blocks", "partial",
       "(parsed off the wire, not rendered)",
       "Reasoning text is stripped so it cannot corrupt a caller's output; it "
       "is not shown either."),

    # --- configuration -----------------------------------------------------
    _f("settings", "config", "--settings, --setting-sources, settings.json layering",
       "gap", "", "No settings file at all: every knob is a flag. A hooks file "
       "is the one config we read.",
       ("--settings", "--setting-sources")),
    _f("safe-mode", "config", "--safe-mode, --bare", "gap",
       "", "No single switch that disables every customization.",
       ("--safe-mode", "--bare")),
    _f("debug", "config", "--debug, --debug-file, --verbose", "gap",
       "", "No debug channel; failures surface as text.",
       ("--debug", "--debug-file", "--verbose")),
    _f("version-help", "config", "--version, --help", "shipped",
       "--version, --help", "", ("--version", "--help"),
       ("--version", "--help")),

    # --- product surfaces we do not chase ----------------------------------
    _f("cloud", "product", "--cloud, --environment, --teleport", "declined",
       "", "Anthropic-hosted sessions.",
       ("--cloud", "--environment", "--teleport")),
    _f("remote-control", "product", "--remote-control, --brief", "declined",
       "", "", ("--remote-control", "--remote-control-session-name-prefix",
                "--brief")),
    _f("ide", "product", "--ide, --chrome, --desktop", "declined", "",
       "Vendor application integration.",
       ("--ide", "--chrome", "--no-chrome", "--desktop")),
    _f("auth", "product", "`claude auth`, `setup-token`, `install`, `update`",
       "declined", "", "Our credentials are per-backend env files."),
)


def by_status(status: str) -> list[Feature]:
    return [f for f in FEATURES if f.status == status]


# --- the mechanical half ---------------------------------------------------

_FLAG = re.compile(r"--[a-zA-Z][\w-]*")


def claude_flags(binary: str = "claude") -> set[str] | None:
    """Flags the installed Claude Code advertises, or None if it is not here.

    None is not an error: a host without claude simply cannot run this half of
    the check, and a check that invents an answer when its input is missing is
    worse than one that says it could not look.
    """
    path = shutil.which(binary)
    if not path:
        return None
    try:
        # Some CLI runtimes exit before flushing pipe-backed help. A regular
        # file keeps the complete inventory, rather than inventing removed flags.
        with tempfile.TemporaryFile(mode='w+') as stream:
            subprocess.run([path, '--help'], stdout=stream,
                           stderr=subprocess.PIPE, text=True, timeout=30)
            stream.seek(0)
            out = stream.read()
    except (OSError, subprocess.SubprocessError):
        return None
    flags = set()
    for line in out.splitlines():
        # Only the option column declares flags; a flag inside prose is an
        # example, not a surface.
        if not re.match(r"\s{2,6}(-\w, )?--", line) and not re.match(r"\s{6}--", line):
            continue
        flags.update(_FLAG.findall(line))
    # `--allowedTools, --allowed-tools` - keep the kebab spelling only.
    return {f for f in flags if not any(c.isupper() for c in f)}


def onecli_flags() -> set[str]:
    from onecli.cli import build_parser
    flags: set[str] = set()
    for action in build_parser()._actions:
        flags.update(o for o in action.option_strings if o.startswith("--"))
    return flags


@dataclass
class Drift:
    undeclared: list[str] = field(default_factory=list)   # theirs, no row
    stale_claude: list[str] = field(default_factory=list)  # row names a dead flag
    stale_ours: list[str] = field(default_factory=list)    # row claims a flag we lack
    checked_claude: bool = True

    @property
    def clean(self) -> bool:
        return not (self.undeclared or self.stale_claude or self.stale_ours)


# Flags that carry no capability of their own - accounted for by the row that
# owns the feature, or pure plumbing we will never mirror.
IGNORED_CLAUDE_FLAGS = frozenset({
    "--allowed", "--disallowed",   # grep artifacts of the two-spelling options
    "--allowedTools", "--disallowedTools",
    "--fork-session", "--include-partial-messages",
})


def drift(binary: str = "claude") -> Drift:
    declared_theirs = {f for feat in FEATURES for f in feat.claude_flags}
    declared_ours = {f for feat in FEATURES for f in feat.our_flags}
    ours = onecli_flags()
    result = Drift(stale_ours=sorted(declared_ours - ours))

    theirs = claude_flags(binary)
    if theirs is None:
        result.checked_claude = False
        return result
    result.undeclared = sorted(theirs - declared_theirs - IGNORED_CLAUDE_FLAGS)
    result.stale_claude = sorted(declared_theirs - theirs - IGNORED_CLAUDE_FLAGS)
    return result


def render(drift_result: Drift | None = None) -> str:
    """The table a person reads, gaps first - that is what the file is for."""
    order = {"gap": 0, "partial": 1, "shipped": 2, "declined": 3}
    lines = ["one-cli parity with Claude Code", ""]
    counts = {s: len(by_status(s)) for s in STATUSES}
    lines.append("  ".join(f"{s}: {counts[s]}" for s in STATUSES))
    lines.append("")
    for status in sorted(STATUSES, key=lambda s: order[s]):
        rows = by_status(status)
        if not rows:
            continue
        lines.append(f"[{status}]")
        for f in rows:
            lines.append(f"  {f.id:<20} {f.area:<12} {f.claude}")
            if f.ours:
                lines.append(f"  {'':<20} {'ours:':<12} {f.ours}")
            if f.note:
                lines.append(f"  {'':<20} {'':<12} {f.note}")
        lines.append("")
    if drift_result is not None:
        lines.append("[drift]")
        if not drift_result.checked_claude:
            lines.append("  claude not installed - flag drift not checked")
        for label, items in (("undeclared claude flag", drift_result.undeclared),
                             ("claude flag is gone", drift_result.stale_claude),
                             ("we do not have", drift_result.stale_ours)):
            for item in items:
                lines.append(f"  {label}: {item}")
        if drift_result.clean and drift_result.checked_claude:
            lines.append("  clean")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="one-parity",
        description="What Claude Code does that one-cli does not.")
    p.add_argument("--json", action="store_true", help="Machine-readable table.")
    p.add_argument("--status", choices=STATUSES, help="Only this status.")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 when the table has drifted from the two CLIs.")
    p.add_argument("--claude-bin", default="claude")
    args = p.parse_args(argv)

    d = drift(args.claude_bin)
    if args.json:
        rows = FEATURES if not args.status else by_status(args.status)
        print(json.dumps({
            "features": [f.__dict__ for f in rows],
            "drift": {"undeclared": d.undeclared, "stale_claude": d.stale_claude,
                      "stale_ours": d.stale_ours,
                      "checked_claude": d.checked_claude},
        }, indent=2, default=list))
    elif args.status:
        for f in by_status(args.status):
            print(f"{f.id:<20} {f.area:<12} {f.claude}")
    else:
        print(render(d))
    return 1 if (args.strict and not d.clean) else 0

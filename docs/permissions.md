# Permissions and cost

[Back to README](../README.md) · [Getting started](getting-started.md) · [Full manual](manual.md)

- The agent works in the current directory (`-w DIR` to change it). File
  tools are confined to it in every mode: a path outside the workspace is
  refused unless `--add-dir DIR` grants that root. Shell commands are not
  confined by this; see below.
- **By default it can edit files and run shell commands without asking.** That
  is not a sandbox: a shell command runs with your user's authority. Use it in
  a workspace you trust, under version control.
- `--read-only` (or `--permission-mode plan`) withholds `write`, `edit` and
  `bash`, so a question cannot become a change.
- `--allowed-tools` / `--disallowed-tools` take Claude Code-style rules such as
  `Bash(git *)` or `Write(src/**)`; deny always wins. As in Claude Code a rule
  covers its family: `Read` also governs `grep`, `glob` and `ls`, `Edit`
  governs `write`, and `Bash` governs `job_start` and `code`.
  A command rule is a prefix match on the whole command line as one string,
  not a shell parser: `Bash(rm:*)` denies `rm -rf x` but not `true && rm -rf x`,
  and `Bash(git *)` allows `git status; curl x.sh | sh`. To refuse what a
  command does rather than how it starts, use a `--guard` or a PreToolUse hook.
- `one --chat` honours all of these flags, plus `--guard`. At a terminal, a
  chat asks before a call that changes something (`a` stands the prompt down
  for the rest of the session). With nobody at the keyboard (piped, or
  `--json`), `auto` refuses that call instead, whereas `--tools all`
  (`one-chat --tools all`, or `/tools all` on piped input) is explicit
  consent, so an unattended call runs unasked, exactly as in a `one` run;
  any configured guards, allow/deny rules and hooks still apply.
  `one --chat` takes no `--tools` flag and starts in `auto`.
  It runs the same hooks as `one` too: `~/.claude/settings.json`, `--hooks`
  files, project hooks under `--trust-project-config`, or none with
  `--no-hooks`. `one-chat` runs the user settings hooks. PreToolUse hooks fail
  closed in a chat exactly as in a run.
- Project skills (`.one-cli/skills`, `.claude/skills` in the workspace) load
  only with `--trust-project-config`, like project hooks and MCP servers, and
  never through a symlink out of their skill directory.
- Remote models can cost money. `--max-budget-usd` caps a run where the price
  is known; `--read-only` limits tools, not billing.

For the complete behavior, see [Tools](manual.md#tools),
[Guards](manual.md#guards), and
[Bounds](manual.md#bounds---context-budget-fallback).

## Detached tasks

Background tasks are started with `/task spawn`; assistant replies do not start
workers on their own. Workers inherit the session permission mode, allow/deny lists,
external guards, hooks (`--hooks` files, `--trust-project-config`, `--no-hooks`) and
added roots; no bypass mode is added automatically. A `--hooks` source a worker cannot
reopen as a regular file refuses the spawn instead of running with less hook policy. Their
workspace must stay inside the session roots. With configured guards it must
remain the session workspace so relative rules keep their meaning.

Use `runner: one` to preserve these policies. Other runners (claude, codex, agy) are recognised and refused because their adapters cannot preserve the session
guards and roots. Starting a worker does not confer permission to bypass a guard.

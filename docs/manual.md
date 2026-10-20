# one-cli manual

The full reference. New here? Start with the [README](../README.md), then read
[first-run setup](getting-started.md) and [permissions](permissions.md).

## Platforms

The core CLI targets Linux and macOS. Linux is the primary full-suite test
platform; the wheel matrix also builds and smoke-tests on macOS. Linux is the
validated release platform. Artifact CI checks Python 3.10/3.13. Native Windows is not currently validated;
use a Linux environment such as WSL, with the same prerequisite checks.
Local serving needs a usable llama-server or Docker installation; sandbox tools
need Linux bubblewrap and prlimit. These optional capabilities are not bundled
with the wheel. A passing CLI help check does not validate GPU serving.

Extras: `.[relay]` for the HTTP relay (see [relay.md](relay.md)), `.[test]` for
the test suite, `.[dev]` for both plus `build`; see
[CONTRIBUTING.md](../CONTRIBUTING.md). Or run straight from a checkout:
`scripts/one` needs nothing installed.

## Upgrade, remove and retain data

Upgrade from PyPI with `one update` (below) or `python -m pip install --upgrade one-core-cli`,
a checkout with `python -m pip install --upgrade .`, or a trusted wheel with `python -m pip install --upgrade /path/to/one_core_cli-VERSION-py3-none-any.whl`.
Confirm `one --version` afterwards. Remove the package and its commands with
`python -m pip uninstall one-core-cli` from that same environment.
Uninstall does not remove `~/.one-cli` (sessions, history, spills and serving
state), `~/.one-providers`, model files, workspace outputs, or running model
servers. Stop serving lanes explicitly and back up state before an upgrade.
Bench, arena and stress default outputs live in `.scratch/` beneath the invoking
directory; use their output-directory controls to keep results elsewhere.
Downgrading package code does not undo state changes; compatibility across
versions is not guaranteed during the alpha. See [Verifying a release](reproducible-releases.md).

`one recover scan [REPO]` lists local Git work that is not in the base
(`--base`, default `HEAD`): dirty or missing worktrees, branches with commits
the base lacks, stashes, and with `--lost` unreachable commits. It prints JSON
and changes nothing. `one recover prepare REPO COMMIT DEST` applies one commit
(or a stash, with `--stash`) onto the base in a new detached worktree at DEST,
leaving the original checkout alone; it never merges. `one recover review`
adds one tool-free advisory call on a free lane, sending only finding ids,
kinds and actions.

`one update` asks PyPI for the latest `one-core-cli` and, when it is newer,
upgrades with the tool that installed this copy: `uv tool upgrade one-core-cli`,
`pipx upgrade one-core-cli`, `uv pip install --upgrade` for a uv-managed venv, or
`python -m pip install --upgrade one-core-cli`. It prints the command before
running it. An editable install or a source checkout gets the pull command to
run instead; nothing is pulled for you. `one update --check` reports the
installed version, the latest and the command without running it. The new
version takes effect the next time `one` starts; running sessions keep the code
they loaded. `one update` with words after it is still a prompt, and so is a
bare `one update` whose stdin is not a terminal (`echo x | one update` sends
the prompt "update" to the model and never runs the installer; as with any
prompt given on the command line, the piped body is not read).
`--check` is the only flag `one update` takes: the prompt flags, `--json`
included, are not supported after `update`. Only a plain version string from
PyPI (at most 32 characters, ASCII digits and the `a`/`b`/`rc`/`.post`/`.dev`
segments) is accepted, cached or printed; anything else is treated as no answer.

When stdin, stdout and stderr are all a terminal and `CI` is unset, `one` checks
PyPI at most once per 24 hours in a background thread (2 s timeout) and stores
the answer in `~/.one-cli/update-check.json`. The notice comes from that cache
at the next start, at most once a day, so a start never waits on the network
and offline is silent. When PyPI lists a release newer than the installed one,
the notice names both versions and says to run `one update`. Turn it off with `ONECLI_NO_UPDATE_CHECK=1` or
`{"update_check": false}` in `~/.one-cli/config.json`.

## The session - slash commands and a real line editor

`one` with no prompt opens an interactive session. The session is the expensive
thing in the room: it holds the history, the background jobs and the spill
files, so every question about it is answerable without restarting it.

```
◆ /context
  history      41 messages
  tokens       80,412
  window       262,144  (31% full)
  compacts at  209,715 tokens
```

| | |
| --- | --- |
| `/help` | the list |
| `/model [name]` | the backend, or switch to another - the conversation carries over |
| `/models` | every backend this host can reach |
| `/mode [name]` | the permission mode, or change it |
| `/tools` | what the model is allowed to call in this run |
| `/context` | how full the window is, and where autocompact fires |
| `/compact` | summarize the old half of the history away now |
| `/cost` | tokens and spend for the whole session, not the last turn |
| `/jobs` | background jobs; `/jobs <id>`, `/jobs cancel <id>` |
| `/hooks` `/mcp` `/agents` | what this workspace loaded |
| `/rewind` | drop the last exchange, keep everything before it |
| `/resume [id]` | list past sessions, or adopt one's history here |
| `/session` | id, transcript path, the `--resume` line to get back in |
| `/cwd` | the workspace and anything else writable |
| `/clear` | forget the conversation, keep the session |
| `/exit` | leave |

A slash command never reaches the model: no turn is spent, no
`UserPromptSubmit` hook fires, and nothing lands in the transcript as something
you said. `/etc/hosts - what is in it?` is a question, not a command - a leading
slash is also how an absolute path starts.

### `@path` - the file arrives with the question

```
◆ explain @src/parser.py in one sentence
  sent with the question: src/parser.py
It splits a line on the first colon, returning the stripped key and value.
```

No `read` call, no round trip. Mentions resolve against the workspace and never
leave it - `@../../etc/passwd`, or a symlink pointing out of the tree, is left
in the sentence as plain text rather than sent. A file past the budget is sent
head-first with a line saying how much was cut; nothing is truncated silently.

### `!command` - your shell, not the model's

```
◆ !git status --short
 M onecli/cli.py
◆ what did that change?
```

Runs in the workspace under your own permissions - no turn, no tool call, no
permission mode. What it printed is offered to the next turn inside a `<shell>`
block, because the reason to run one is almost always to show the model
something. It is held, not sent, so a command you ran for yourself costs
nothing if no turn follows.

### Your own commands, as files

Markdown under `.claude/commands/` (project) or `~/.claude/commands/`
(personal). Frontmatter optional, body is the prompt, `$ARGUMENTS` and `$1..$9`
take what you typed, a subdirectory namespaces the name.

```
.claude/commands/blunt.md      ->  /blunt <thing>
.claude/commands/git/msg.md    ->  /git:msg
```

They show up in `/help` and in tab completion. A file that claims a built-in
name is skipped out loud - `/cost` must not quietly become something that
spends a turn.

The line editor is `readline`, so history survives the session
(`~/.one-cli/history`), tab completes slash commands at the head of a line and
workspace-relative paths anywhere else, and a line ending in `\` continues on
the next one:

```
◆ write a function that:\
  … takes a path\
  … returns its size
```

Above the prompt, one line says where the session stands - the model (it can
move under you with `/model`), how full the window is, and what it has cost:

```
  deepseek-v4-flash · ━━━━━━━━━━ 18% · 12,814 tok
◆
```

Ctrl-C drops the line, or the turn if one is running. Ctrl-D leaves.

## Backends

A backend is a place we can send a chat turn to. Resolution order:

1. explicit flags (`--base-url` with `--model-id` and `--wire`; `-m NAME` picks a backend by name)
2. drop-in files `~/.one-providers/*.env` - hot, no restart
3. a live llama.cpp server on this host (and, with `ONE_LLAMACPP_HOSTS`, on other nodes)

With none of these, an interactive `one` runs a first-run setup that writes one
drop-in for you and tests it (see [First run](getting-started.md#first-run));
`one doctor` lists every backend it finds and whether each one answers.

Nothing reads a vendor CLI's config. A drop-in is a few lines:

```sh
ONE_PROVIDER_MODEL='mistral-medium-latest'
ONE_PROVIDER_LABEL='Mistral Medium'
OPENAI_BASE_URL='https://api.mistral.ai/v1'
OPENAI_API_KEY='...'
```

Anthropic-wire endpoints set `ANTHROPIC_BASE_URL` instead of `OPENAI_BASE_URL`
(a file naming both is OpenAI wire). The key is chosen by wire, never across
vendors: `ONE_PROVIDER_KEY_ENV` names the variable first - read from the drop-in
and its `ONE_PROVIDER_SOURCE` file, else from the environment `one` runs in, so
the key need not be on disk; otherwise the OpenAI wire reads `ONE_PROVIDER_KEY`,
then `OPENAI_API_KEY`, and the Anthropic wire `ANTHROPIC_AUTH_TOKEN`, then
`ANTHROPIC_API_KEY`, then `ONE_PROVIDER_KEY`. Extras: `ONE_PROVIDER_PARAMS`
pins per-endpoint sampling as JSON, `ONE_PROVIDERS_DIR` adds more drop-in dirs.

Both wires are first class. Canonical history is OpenAI-shaped and converted at
the Anthropic edge, which is what lets a session resume on a different backend
than it started on.

## Local serving - pull it, measure it, then serve it

Three commands, in the order a host needs them.

```sh
one doctor                     # what this box is, and what it can serve
one models list                # the catalog, and what is already on disk
one models pull qwen3.8-27b    # weights + drafter, resumable, sha256 verified
one tune qwen3.8-27b           # measure every viable recipe HERE
one serve qwen3.8-27b          # launch with the one that actually won
one setup                      # all of the above, as menus
```

`one setup` is an interactive menu for local serving. Without a terminal it
exits with guidance and performs no action. `one setup --yes` explicitly selects
and pulls the largest recommended model that fits, once, without prompting;
this can download many gigabytes. It reports a pull failure through its exit
status and does not tune or start a server. For predictable provisioning, use
`one models pull NAME`, then explicit `one tune NAME` and `one serve NAME` steps.

### Why measured and not configured

The fastest way to serve a model depends on the model, host and runner. Measure
it on the target machine instead of copying a recipe from another system.

So `one tune` sweeps, `~/.one-cli/tuning.json` records the winner with the
losing numbers beside it, and `one serve` replays that entry. A model with no
entry is served on baseline and reported UNVERIFIED - it never guesses a fast
recipe, and it never refuses to start. A native llama-server lane runs
detached: `one serve` returns once it reports healthy, and the server's output
goes to `~/.one-cli/serve/one-lane-<model>.log`.

The ledger carries the runner it was measured against. Rebuild llama.cpp and
the entry goes stale rather than replaying a number about a build that is no
longer installed. One host's measurement is never evidence about another.

### The catalog

`onecli/serving/models.json` ships a catalog of weights with known sizes and
hashes, each linked to its speculative drafter. `one models search <q>` goes
live to HuggingFace for everything else; pulling one of those files it into
`~/.one-cli/models.json` so the second time it is a named row rather than a
repo id someone has to remember.

A pull fetches the drafter with the weights by default - it is under a tenth of
the size and it is what makes the fast recipe measurable at all. Every file is
checked on size then sha256; a mismatch is set aside as `.bad.<ts>`, never
served and never deleted behind your back.

Native serving binds `127.0.0.1`; Docker publishes ports only on host loopback.
Remote access requires a separately configured authenticated proxy or tunnel.
No public unauthenticated listener is enabled by these recipes.

## Chat - a conversation that can look things up

The agent is the wrong shape for a conversation: every turn is a run, with a
header, a permission mode and a spine of tool calls
scrolling past. `one-chat` is the other entry - a question, an answer, streamed.

It is not a chat that cannot do anything. When the answer depends on something
the model cannot know, it reads the file, greps the tree, or runs the command
underneath the conversation and keeps talking: one dim line per
call, the answer as prose, no mode to switch into. Most turns never call a tool
and are still one request. The surface is bounded on purpose - `read`, `ls`,
`glob`, `grep`, `bash`, and web tools when a configured MCP server or
extension provides them. Chat sets no step limit of its own.

Four of those only look. `bash` can do anything, so it is the one that stops and
asks first:

```
  ◆ bash   df -h /
    run bash df -h /? [y/N/a]
```

`y` runs it once, `a` stands the question down for the rest of the session. The
rule underneath: **a call that can change something needs a person to say yes.**
With nobody at the keyboard there is no yes to be had, so a piped one-shot in
`auto` refuses the call and answers without it - and says so, rather than
hanging on a question nobody can see.

`/tools all` hands over the agent's own surface - edits, background jobs, and
shell (asking at a terminal on the first mutating call until `a`, or running
unasked when unattended since `--tools all` is itself the yes). `/tools off` is
the pure lane, where no schema is sent at all and nothing is read.

Hooks hold in a chat as they do in a run, built by the same code from the same
flags (`--hooks`, `--no-hooks`, `--trust-project-config` on `one --chat`; the
user settings file for `one-chat`). `UserPromptSubmit` fires on every question,
`/compare` included, and a block sends nothing. `PreToolUse`, `PostToolUse`,
`Stop` and `PreCompact` run in the tool loop underneath, a `PreToolUse` hook
confines code mode, and a broken hook configuration denies every tool call.
`SessionStart` and `SessionEnd` fire only in `one` runs, not in a chat.

```
scripts/one-chat                          a session on the first backend found
scripts/one-chat -m kimi                  a session on one you name
scripts/one-chat "what is a mutex?"       one question, one answer, exit
echo "explain this" | scripts/one-chat    a piped body is the question
scripts/one-chat --compare kimi,qwen "q"  the same question to several, at once
one --chat                                the same session from the agent's CLI
```

Every backend is the same conversation: the history is canonical and ours, so
`/model` mid-sentence hands the whole thread to another endpoint - including
across wires - and the new model answers in context. Switching resets the price
table, never the transcript.

Everything else is a slash command, and they all live in one table, which is
what `/help` prints and what TAB completes:

| | |
| --- | --- |
| `/models` `/model <name>` | what we can reach, and switch to it |
| `/new` `/undo` `/retry [model]` `/edit` | move the conversation around |
| `/system [text\|@file]` `/temp` `/max` `/stream` `/think` | the dials |
| `/file <path>` `/img <path>` `!cmd` | stage context into the next question |
| `/compare a,b [question]` | several backends in parallel, off the record |
| `/cost` `/history` `/save [path]` `/copy` | what it cost, what was said, keep it |
| `/sessions` `/resume [id]` | the transcripts on disk |
| `/tools [off\|auto\|all]` | what it may reach for while it answers |

A line ending in `\` continues on the next; `"""` opens a paste block, so a
multi-line paste is one question rather than six. Ctrl-C clears a half-typed
line and cancels a streaming answer without ending the session; Ctrl-D or
`/exit` ends it.

## Tools

`read`, `write`, `edit`, `ls`, `glob`, `grep`, `bash`. Output is bounded, `edit`
refuses an ambiguous match rather than guessing, and `--read-only` withholds
`write`, `edit` and `bash` so a question cannot become a change.

Every direct tool call, code-mode binding and child agent is checked against
its effective tool surface, including tools the model was not offered.
This is not a host sandbox: shell/code execution still runs with the operator's
authority, and MCP read-only hints are server claims, not verified isolation.
The default CLI still permits host execution; use only with trusted workspaces
until the safe-default execution profile is available.

File paths must resolve within the workspace or a root added with
`--add-dir`, in every permission mode including `bypassPermissions`; a path
outside them (absolute, `~`, or reached through a symlink) is refused with a
message naming `--add-dir`. File read/write/edit/list operations refuse symlink components,
including links swapped in after a permission check; name the real path
explicitly if needed. Creation paths containing `..` are refused before any
directory creation; name the destination directly. Guarded grep/glob use a bounded Python search that checks
each path, skips links, and does not launch repository search programs. These
path policies do not contain shell commands, code execution, external tools,
or a hostile process already running as the same OS user.

one-cli ships no built-in web tools. To reach the web, configure an MCP
server that provides one; its tools then take permission clauses (see
[MCP](#mcp) for how an MCP clause is matched).

``` 
one --mcp-config .mcp.json "search for the 2608.10218 paper and summarise it"
one --read-only --mcp-config .mcp.json "fetch https://example.com/spec and list the endpoints"
```

## MCP

`--mcp-config` connects MCP servers, in Claude Code's config shape - so a file
written for that harness runs here unchanged. Their tools join the registry as
`mcp__<server>__<tool>` and take permission clauses. An MCP tool has no single
argument to match, so its clause is matched against every string value in the
call's arguments, at any depth (object values and list items, never keys). A
deny rule matches when any value matches; an allow rule only when every value
matches, so an allowed URL in one argument cannot carry another in a second:
`--disallowed-tools
'mcp__web__fetch(https://evil.example/*)'` refuses a fetch of that site whatever
the argument is called. A clause ending in `*` is a prefix, otherwise exact; an
`http(s)` clause also matches the same URL with a different case of scheme or
host or an explicit default port. A clause on a tool that takes none (anything
but Bash, Read, Write, Edit, LS, Glob, Grep, WebFetch, WebSearch and MCP tools)
is refused when the run starts.

```
one --mcp-config .mcp.json "inspect this page for injection"
one --mcp-config '{"mcpServers":{"guard":{"command":"python3","args":["-m","your_guard.cli","mcp"]}}}' "..."
```

stdio only: an MCP server is a subprocess speaking newline-delimited JSON-RPC on
its pipes. HTTP and SSE servers are refused by name, not silently skipped.

The project's `./.mcp.json` is not loaded unless `--trust-project-config` is
passed for this run; `--strict-mcp-config` excludes it even then. The user-global
`~/.mcp.json` is never loaded. Explicit `--mcp-config` files are trusted executable
configuration: inspect their commands before naming them. Servers receive only
a minimal environment (PATH, home, locale and platform temporary/system paths),
not the operator's provider keys; required credentials must be supplied in that
server's `env` configuration. This minimizes accidental exposure but does not
sandbox the trusted server, which still has the operator's filesystem access.

Likewise, project `.claude/settings.json` and `.claude/settings.local.json` hooks
require `--trust-project-config`. User `~/.claude/settings.json` and explicit
`--hooks` files remain trusted executable input. Trust is independent of
`--read-only`: hooks and servers run at startup, outside the model's tool policy.
A tool counts as mutating unless its server declares `readOnlyHint`; that hint
only filters the offered tool surface, not the server's startup behavior.

A server that will not start is reported and skipped; a server that stops
answering hits a deadline. Neither takes the session down.

## Spill - big output is stored, not dropped

A tool result that does not fit the free context window does not get truncated
into nothing. The whole result is written to `~/.one-cli/spills/<session>/`, and the
model reads head, tail, and the path:

Sessions, drafts, spills and local analysis records are created with private
permissions (0700 storage directories, 0600 files), regardless of umask.
Storage refuses symlink traversal and writes to multiply-linked files. Existing
session files are tightened on their next append; this does not retroactively
erase copies or exposures from older installations.

```
... [1.2MB of 1.3MB not shown - the FULL output (18443 lines) is at
    /home/you/.one-cli/spills/ab12cd/001-bash.txt; read it with the read tool,
    offset/limit, or grep it] ...
```

The bound is the same as before; the loss is gone. A model told "420000 bytes
dropped" cannot grep them, cannot page them, and cannot tell whether the answer
was in the missing middle - so it guesses or re-runs the command and loses the
same middle again.

The policy lives in `tools.run`, not in each tool, for two reasons: a tool that
truncates itself has thrown the bytes away before anything could store them, and
a rule applied in one place cannot be forgotten by the next tool someone adds -
including the MCP tools, which are written by other people.

## Code mode - one program instead of twenty tool calls

A tool call is a round trip: emit, carry, run, back through the context window,
sample again. Reading five files costs five of those. Written as a program it
costs one, and the parts that are ordinary programming - a loop, a filter, a sum
- happen in Python where they are free rather than in the model where every step
is a sampled decision that can go wrong.

```python
total = 0
for path in one.glob(pattern="**/*.txt"):
    total += len(one.read(path=path))
print("TOTAL", total)
```

`one.read` / `write` / `edit` / `ls` / `glob` / `grep` / `bash` / `web_*`, plus
`one.call("any_tool", **kwargs)` for everything else including MCP tools. Only
what the program **prints** comes back.

The bindings are the point. A program does not `import os` and do as it pleases:
each binding is sent back to the agent and put through **the same gate as any
other tool call** - guards, PreToolUse hooks, the permission mode, the
`on_event` decider an embedding host provides. Code mode with unmediated file
access would be a hole around every guard in the system, and a `--read-only` run
that can write through a Python program is not read-only. A refused binding
raises `one.Denied` in the program rather than returning a string, because a
program that carries on after a refused write produces a confident wrong answer.

Three more choices, each a failure avoided: it runs in a **subprocess** (an
`exec` in our process takes the agent down with a runaway loop or a `sys.exit`,
and a Python thread cannot be killed); its output goes to a **file** (the parent
is busy answering bindings, not draining, so a pipe would deadlock a chatty
program at 64k); and the bindings ride **their own descriptor**, so a stray
`print` cannot forge a binding call.

## Subagents - a nested loop, or somebody else's CLI

A subagent is not a second program. It is this loop one level down, standing on
the SAME guard chain, the SAME hooks, the SAME permission surface and the SAME
live budget. That is the point: a `task` tool that spawned a fresh unguarded
agent would be a hole around every gate the run stands behind, and the easiest
hole in the world to walk through - "I was not allowed to write that file, so I
asked a subagent to."

What a child does not share is context. It gets its own history and returns one
answer, so a sweep over forty files costs the parent a summary instead of forty
tool results.

```
one --agents agents.json "which of these twelve files defines the retry policy?"
one --list-agents                     what this run can hand a job to
```

Four things are load-bearing:

- **The surface is intersected**, never granted. A preset naming tools its
  parent does not have is a request, not a grant - which is read-only
  inheritance in one line.
- **One meter, one ceiling, for the whole tree.** The child holds the parent's
  meter *object*, not a copy. A copy passes every other budget test and still
  lets six children spend the ceiling six times.
- **The depth cap is enforced by construction.** The last allowed level is built
  without a spawner, so it is never offered a tool it would only be refused.
- **No asker, no jobs.** A question from a nested loop arrives at a terminal
  with no way to say which of three running children asked it, and a job
  outliving the child that started it is a process nobody closes.

A read-only or plan run has no `task` tool at all. Not because a child could
escape, but because `task` is classified mutating, and a surface that calls the
same tool read-only in one mode and mutating in another is one nobody can reason
about.

### Drivers - a child that is another vendor's CLI

`onecli/harnesses.py` already points a dozen vendor CLIs at THIS endpoint with
THIS model for the arena. A preset can name one as its `driver`, and the child
becomes that CLI:

```json
{"reader": {"description": "Reads code and answers about it.",
            "driver": "pi", "timeout": 120}}
```

It runs on the rung the parent is standing on - same endpoint, same model, same
key - in the parent's workspace, with every vendor credential stripped from its
environment so a driver that fails to pick up your configured endpoint fails loudly instead
of quietly answering from somebody's personal plan.

Three honest limits, because a foreign CLI makes its tool calls in its own
process where we have no seam:

- **The guard chain is consulted once, at the boundary**, with the real command
  line, as a mutating `bash` call. That is where the decision actually is: this
  workspace is about to be handed to `<argv>`.
- **What we cannot narrow, we refuse.** A parent whose own surface has no write
  tool may only drive a CLI that takes a read-only instruction, and gets it in
  read-only mode. There is no third case where a write-capable CLI is quietly
  handed to a parent that cannot write.
- **An unmetered driver is charged a floor, marked as one.** Most vendor CLIs
  report no usage; charging them zero would make `--max-budget-usd` a ceiling with a
  door in it. They are charged prompt-in plus answer-out with `estimated` set,
  which is less than they spent and says so. `pi` reports real usage and is
  charged it.

A driver failing is a tool error, never a crash: missing binary, wrong wire,
timeout, non-zero exit, each comes back as a sentence the parent can act on.

## ask-user - the one place the loop stops

one-cli has never stopped to ask. For a one-shot run that is correct: nobody is
watching, so a question is a hang. Interactively it is wrong - the model hits a
fork it cannot resolve from the workspace and its only options are to guess or
to give up.

```
? Which of these is the file you meant?
  1) src/report.py
  2) scripts/report.py
>
```

Three rules, each one a failure that would otherwise ship:

- **No asker, no tool.** It is offered only when someone is reachable - an
  interactive session on a tty. `--ask` forces it on for a piped session that
  does have a person watching, `--no-ask` off.
- **Every question has a deadline** (`--ask-timeout`, default 300s). A person
  walks away; without a clock the run is wedged holding a job, a spend and a
  session. On a timeout the model is told nobody answered and that it must
  decide alone and say what it assumed - which is exactly one-shot behaviour, so
  the unattended case degrades into the mode already known to work.
- **An answer is data, not an instruction.** It arrives as a tool result, in the
  same channel as everything else. Someone typing "ignore your instructions" is
  a person who said something odd, not a new system prompt.

A number picks an option; anything else is taken as typed, so "neither, use the
one in tests/" is never silently mapped onto option 3. An approval gate is not a
separate mechanism - it is a question whose options are yes and no.

## Background jobs - a long command does not own the loop

The bash tool is synchronous: while `pytest -q` runs for four minutes the agent
is a process waiting on a pipe. A job is the other shape of work.

```
job_start   begin, get an id back immediately
job_list    what is running, what finished, what it cost
job_output  read any part of a job's output, running or finished
job_cancel  stop one, and the children it spawned
```

The part that makes it a protocol rather than four tools is the **completion
notice**: a job that finishes while the model is doing something else is
announced at the top of the next turn, once.

```
[job j1] finished ok after 6s: pytest -q. 412 lines of output - read it with
job_output('j1').
```

Without that the model must poll, and a model that must poll either forgets to
(the job is never read) or polls every turn, which is waiting with extra steps.

Output goes to a file, not a pipe - a pipe nobody drains fills its buffer and
blocks the child, and a file is readable *while* the job runs, which is what
makes `job_output` useful before the exit code exists. A cancel goes to the
process group, so cancelling `make -j` does not leave its children compiling.
Nothing outlives the CLI: survivors are stopped at exit and named. A run with no
job store is not offered the tools at all, and `--read-only` never had them -
a background shell is still a shell.

## The plan - state on the session, not a message

A long run is a hundred tool calls that add up to a shape you cannot see. The
model already plans; it plans in prose, mid-turn, and the plan scrolls off the
screen and out of the context window together - so a run that has done six of
eight things reads exactly like one that has done two.

`todo_write` writes the whole list, every time. One writer, at most one item
`in_progress`, and a write that breaks that is refused whole - nothing changes,
the same rule `edit` follows when its match is ambiguous.

```
│ P L A N  1/3
│ ✓ read the config
│ ▸ patch the wire
│ · add a test
```

The list is **state, not transcript**, and three things follow that a written-
down plan does not get:

- **It survives compaction.** The tool result that carried the list is summary
  material like everything else; the list is not. After a compaction it is
  restated verbatim, because losing it is losing the run's spine while keeping
  its footnotes.
- **It survives the process.** Every revision is appended to the session file,
  so `--resume` picks the plan back up and restates it once, before the prompt.
  A resumed session that talks about step five while showing an empty plan has
  lied to the person watching.
- **It is drawn between turns**, not buried in a tool result nobody reads - and
  a run that answers with items still open prints what it left.

A plan with unfinished items and no write for ten steps gets one line back:
the model announced a list and went back to working without it, which is the
failure the whole thing exists to catch. `todo_write` is on the read-only
surface on purpose - it touches nothing outside the session's own bookkeeping,
and plan mode's entire output is a plan. A child gets no list: it returns one
result to a parent that keeps the plan, and a list nobody renders or resumes is
bookkeeping that dies with the child.

## Loop guards - a run that has stopped moving says so

Two ways a run spends without progressing, both handled in `onecli/loopguard.py`.

**Repetition.** The third byte-identical call to the same tool gets a note
appended to the result the model just received:

```
[loop-guard] 3rd identical call to read. The result is byte-identical to the
last one. You are repeating yourself. Do not call it again with these
arguments - read the result you already have, try different arguments, or say
what is blocking you.
```

It is advisory, never a block: re-reading a file you just wrote is legitimate,
and a guard that refuses that costs more than the loop it prevents. What a model
may not do is repeat *without noticing*, so the notice goes where it must be
read - into the transcript, not into a log. A refused call counts too; three
tries at a tool the run is not allowed to have is the clearest loop there is.
`--repeat-limit 0` turns it off, and every note fired is in `--json` under
`advisories`.

**Deadlines.** `--tool-deadline 60` is a per-call ceiling, and it is *enforced*
where it can be: any tool declaring an integer `timeout` (bash, and any MCP tool
that declares one) has that argument lowered before the call runs. A model
asking for less than policy keeps its own number. Where a tool keeps its own
clock the ceiling can only be reported, so the overrun comes back as an advisory
rather than a claim about a cut that never happened.

## Guards

A guard is consulted between the model asking for a tool and the tool running,
and it can refuse. `--read-only` is one, enforced at call time as well as in the
schema. External guards are programs:

```
one --guard ./guards/block-force-push.sh "rebase this onto main"
```

The payload is Claude Code's PreToolUse shape, with fail-closed enforcement:

| | |
| --- | --- |
| stdin | `{"tool_name": "Bash", "tool_input": {"command": "..."}, "cwd": ...}` |
| exit 0 | allow |
| exit 2 | block - stderr becomes the reason the model is shown |
| other | block: the guard failed to establish permission |

A one-cli `bash` or `job_start` call is presented as `Bash`, and the file
tools' `path` as `file_path`, because that is the vocabulary the guards match
on. A guard cannot see inside a `code` program, so while any `--guard` or
PreToolUse hook is active the program runs in the sandbox and each of its tool
calls is checked singly. A JSON verdict on stdout may span several lines;
`permissionDecision: "ask"` refuses, because one-cli has no prompt at that
layer. `--guard` is
repeatable and the first block wins. A guard that hangs or cannot run fails
closed and reports the failure. PreToolUse settings-file hooks also fail closed, in a run and in a chat alike: timeout, startup failure,
unexpected exit status, invalid matcher, unreadable or malformed settings, and
unsupported enforcement configuration deny the tool with the hook and failure
named. This deliberately differs from Claude Code, whose hooks fail open.
Missing optional default settings are allowed; missing explicitly named files
are configuration failures. Non-enforcement events (PostToolUse, notifications,
and other lifecycle hooks) retain their advisory error behavior.

A refusal is handed back to the model as the tool result, not raised. It reads
`blocked: <reason>` and can take another route - which is also the honest limit
of a narrow guard. Block `write` alone and a model will reach for `bash`; that
is not the guard failing, it is one guard not being coverage.

## Bounds - context, budget, fallback

`--max-steps` bounds how many times we call the model, not how big the call
gets. Three flags bound the rest.

`--autocompact` (on by default) summarizes the oldest messages into one message
when the history passes `--compact-at` (default 1.0, the whole window) of the
model's context window, keeping the system prompt, the summary and `--compact-keep` messages
verbatim. The cut is walked backwards until no kept tool result has lost the
assistant turn that called it - an orphaned `tool_call_id` is a 400 on both
wires. Windows for known models are read off the endpoints, not guessed;
`--context-window` overrides.

`--max-budget-usd` stops the loop with `stopped: budget_usd` and says what it
cost. A model with no entry in the price table is counted in tokens only and
the run says so once, because a ceiling computed from an invented price stops a
run for a reason that is not true. Add your own with
`ONE_CLI_PRICING=/path/to.json`.

`--max-input-tokens` bounds cumulative prompt processing across all calls in a
run. This differs from context size: a stateless tool loop resends its current
transcript on every step, so 30 calls over a healthy 25k context can process
roughly 750k input tokens without overflowing. The ceiling meters cost, not
re-sends: fresh input counts in full and cache reads at their price relative
to fresh input (`cost.metered_input_tokens`). That ratio is the third column of
an `ONE_CLI_PRICING` entry (`[in, out, cache_read]`) and 0.1 when none is
named. Each wire reports its provider's cache hits in the canonical
`cache_read_tokens` usage field. The window bound on a single request is
separate and still counts every token. There is no default cumulative ceiling: context capacity does not determine
how much work a task needs. An explicit ceiling reserves the last affordable
call for a final answer; a reply given on that
forced step is kept but reported as `stopped: budget_input`, since tools were
withheld and the work may be incomplete. Pass `--max-input-tokens N` to impose a cumulative ceiling (or `0` to
disable it). Existing context, step, loop and explicit spending limits remain
independent. Token weighting is a processing proxy, not a verified monetary
budget; use a priced spending limit when controlling monetary exposure.

There is no fixed size for a tool result. One that fits in half of what is
left of the context window enters the transcript whole; a larger one, whichever
path produced it, is spilled to the session's spill directory and the model gets
head, tail and the path. `read` returns the whole file the same way, and when
the rest does not fit it stops at a whole line and names the offset to continue.

Fallback is a **chain**, and it is built without you naming it. When a backend
is overloaded, rate-limited or unreachable the turn moves down one rung and
stays there for the rest of the run - an endpoint that just 429'd will 429 on
the next step too, and paying its timeout again is the cost the chain exists to
avoid. A model *declining* the task is not a failure and never spends a rung.

Where the ladder comes from, first match wins:

1. `--fallback-model a,b,c` (comma separated, repeatable) or `ONE_CLI_FALLBACK`.
2. `ONE_PROVIDER_FALLBACK=spare,third` in the primary's own drop-in, so the
   ladder lives next to the endpoint it describes and reloads hot with it.
3. auto - rank what discovery already found: same profile first, then the rest
   of the drop-in dir in your own order, then one local ollama as the floor.
   Anything sharing the primary's endpoint is dropped (one endpoint, one
   outage), embedding models are never the floor, and the chain is capped at
   three rungs so a broken request cannot tour every provider.

`--no-fallback` (or `off` in either spec) pins the run to one backend. `one-bench`
passes it on every cell: a cell that quietly answers on a spare is a score
attributed to the wrong model. `--base-url` also withholds auto - an explicit
endpoint is a statement about where the run goes.

Every hop is reported: a yellow line on stderr as it happens, `fallbacks` and
`fallback_chain` in `--json`, and `answered_on` naming the rung that actually
replied.

### A named model that is offline or fails

`-m <name>` is an instruction, so a named model gets no implicit chain: auto and
the drop-in's `ONE_PROVIDER_FALLBACK` are off for it, and `--fallback-model` /
`ONE_CLI_FALLBACK` stay the explicit opt-in. `--list`, the `/model` picker and
`-m` show each profile's live status (`online`, `offline`, `unknown`, or
`unchecked` for a paid vendor, which is never dialled); the status comes from
`onecli/presence.py`, which checks each endpoint through the vendored `dropins`
reader.

When the named profile is offline at start, or fails on its first rung (the
reason is named: `offline`, `auth`, `not found`), one-cli prints the reason and
the online alternatives - free lanes first, paid ones marked `[paid]` - and
then follows one setting:

| Setting | Flag (per run) | Env (host default) | Values | Default |
| --- | --- | --- | --- | --- |
| `on_unavailable_model` | `--on-unavailable-model` | `ONE_CLI_ON_UNAVAILABLE_MODEL` | `ask`, `error`, `fallback` | `ask` |
| `fallback_allow_paid` | `--fallback-allow-paid` | `ONE_CLI_FALLBACK_ALLOW_PAID=1` | on / off | off |

- `ask`: on a terminal, choose an alternative by number or press Enter to
  abort. Without a terminal, with `-p` or with `--json` it behaves as `error`.
- `error`: exit 2 with the reason and the alternatives on stderr.
- `fallback`: run on the first eligible alternative and say so in a yellow
  line. A paid lane is only eligible with `fallback_allow_paid`, and the same
  rule filters paid rungs out of the implicit chain.

A failure after the first rung, or in the middle of an interactive session,
reports the reason and the alternatives; switching is then `/model`.

Token counts come from `Turn.usage` when the endpoint reports it and from a
character estimator when it does not. The estimator is a floor, not a
measurement.

## Bench

`scripts/one-bench` scores backends on three tasks with machine-checkable
outcomes - fix two bugs and run the file, count files containing a token
(read-only), write a module that must pass a hidden test. Every cell is its own
subprocess and workspace, and scoring inspects the workspace, never the model's
closing sentence.

```
scripts/one-bench --no-local -j 6           # cloud lanes, parallel
scripts/one-bench 'ollama-cloud=glm-5.2'    # one endpoint, one model id
```

A local llama-server runs serial - one GPU, one queue - and there is nothing to
unload between backends: each server holds one model for its whole life, so the
run never asks a runtime to swap weights out from under it.

## Toolprobe - which rung of the tool loop a model actually reaches

`scripts/one-toolprobe` asks bench's question one layer down, straight at the
endpoint: does a structured tool call arrive at all, do its arguments parse and
run, does the model use the result, and can it chain three calls where each
argument comes from the previous answer. A model that fails here cannot pass
bench, and this says at which rung it fell over.

The rung most worth knowing is **L1b**: the model did try to call a tool, but
emitted it as prose - a JSON blob, an XML tag, a fenced block - instead of on
the wire. That is a template or parser mismatch, not a model that ignores
tools, and the two look identical if you only read the reply text. Both the
answer and the reasoning channel are scanned, because a thinking model puts the
whole attempt in `reasoning_content` and returns an empty `text`.

```
scripts/one-toolprobe --list                 # backends and tasks, no calls
scripts/one-toolprobe minimax mistral        # the full battery on two lanes
scripts/one-toolprobe qwen -t T6_chain3      # one task
```

Beside the rungs the battery aims at named failure modes: recovery from a tool
error, a tool that fails every single time, selection between two near-identical
schemas where one is deprecated and returns wrong numbers, forced emission under
`tool_choice`, exact output-format compliance, and a 20k-token needle placed at
the START of the haystack (a truncating runtime drops the oldest tokens first,
so a needle at the end measures nothing). Rows land as JSONL under
`.scratch/toolprobe/`, and every verdict is a pure function of them.

The transport is one-cli's, so the same tasks run against OpenAI-wire and
Anthropic-wire endpoints alike.

## Arena - the same task through other people's harnesses

Adapters point [pi](https://github.com/earendil-works/pi),
[opencode](https://github.com/sst/opencode), qwen-code and codex at the same
drop-in endpoint one-cli uses; each gets a private config dir and a scrubbed
environment so no run can quietly answer from the operator's own account.

Read-only Hermes cells require Linux bubblewrap with user namespaces enabled.
The filesystem is mounted read-only, with only the cell's private configuration
and temporary files writable. Network access stays available. If this boundary
cannot start, the harness fails instead of running with write access.

```
scripts/one-arena 'ollama-cloud=glm-5.2' kimi        # every harness on this box
scripts/one-arena --harness pi --harness one --list  # what would run
scripts/one-arena --no-system                        # control: drop the system prompt
```

Each cell records what it touched on disk - files created, changed, deleted and
lines - and an independent model scores the diff blind on correctness, minimality
and fit. The checker decides pass/fail; the judge only explains the difference
between two passes. Judge lane: `ONE_JUDGE=<backend>`.

### What a run costs, before it runs

`--lanes` prints every endpoint this box can reach and what running it costs,
and spends nothing to do it:

```
scripts/one-arena --lanes
```

A lane is one of four kinds, and the difference between the last two is the
whole point of naming them:

| kind | what it means | can `--budget` bound it? |
| --- | --- | --- |
| `free` | an explicit zero price in the pricing table | it costs nothing to bound |
| `metered` | a per-token price we actually know | yes |
| `plan` | a flat-rate subscription; there is no per-token price to compare a ceiling against | no |
| `unknown` | a metered API whose price is not in our table | no - we cannot say what it spends |

Kind comes from the pricing table first, then the address. A private or
loopback address alone does not make a lane free - it may proxy a metered API -
and a plan gateway usually says so in its own hostname (`token-plan…`,
`/coding/`).

```
scripts/one-arena --free --suite hard --harness one    # costs nothing, so develop here
scripts/one-arena --budget 0.50 --suite hard mistral   # a real ceiling, checked before each cell
```

`--budget` is enforced before a cell starts, not after it finishes - a ceiling
that reports what a run already spent is a receipt, not a limit. It refuses a
`plan` or `unknown` lane by name rather than pretending to bound it. Teach it a
price and the lane becomes `metered`:

```
echo '{"mistral-large": [2.0, 6.0]}' > pricing.json
ONE_CLI_PRICING=$PWD/pricing.json scripts/one-arena --lanes
```

## Look

The house rules that survive the trip to a terminal:

**Nothing is a filled surface.** The instrument is drawn in line, so the data is
the only thing on screen with weight. No inverse video, no block bars.

**Figure versus armature.** Structure sits dim in the brand tint; the live part
is bright in accent. A drawing at one weight has no instrument in it.

**A corner is a stage.** A diamond means a node - a place where something is
decided - so it marks a step and nothing else uses it:

```
◇ 2 grep   DIAMOND_SOLID          hollow - this step only looked
◆ 3 edit   onecli/render.py       solid  - this step could change something
```

**Stale data cannot animate.** Every other CLI spinner runs off a timer, so it
twirls exactly as merrily against a hung socket as against a model mid-sentence,
and the one question you actually have - is anything happening? - is the one it
cannot answer. Here the head moves only when something arrived:

```
·▪············  qwen3.8-27b  1.4k       tokens arriving
··········▫···  bash  7s                a subprocess, clocked by time
··············  qwen3.8-27b  waiting 9s  nothing has arrived for 9s
```

The third rail is greyed and frozen. That is not a stopped animation, it is the
reading. The `work` motif is allowed to move without a feed because its clock
was never data - and its readout prints a duration, never a token count it does
not have.

Colour is a capability, never an assumption. Truecolor when `COLORTERM` says so,
256 otherwise, and nothing at all when stdout is a pipe, `TERM=dumb`, `NO_COLOR`
is set, or `--plain` was passed - which also swaps box-drawing for ASCII, since
a terminal we cannot paint should not be handed a diamond. `--plain` exports
`ONE_CLI_PLAIN`, so hooks, guards and any child `one` inherit it.

**Relief - the third drawing material.** Lines and arcs answer *how big*. A lit
surface also answers *compared to its neighbours* and *which way is it going*,
because the eye reads shading as shape before it reads a digit. `onecli/relief.py`
draws a heightfield lit by one fixed light, quantised through a Bayer 8x8
matrix, and printed as glyphs. Colour carries state, the glyph carries value, and the three-way height
contract is the honesty of it - a number is a reading, `None` is *no reading
here* and gets one dim dot, `SKY` is *no surface here* and gets nothing:

```
  one        ===++++*+**#*##%#+
  hermes     ===++++*+**
  qwen-code  ······················     never ran - not a score of zero

  T O K E N S / S
     ##  ·  *%    +
    *%%# · *%##   #+
   *%##%#·*%###* ###=
  #%%#%#%·%#%#%%%#%#%*
```

It is drawn once and does not move. A printed frame has no way to honour *stale
data cannot animate*, so it does not get to animate at all - motion stays in
`spinner.py`, where the rule can be enforced.

See the whole surface without running a turn, or shoot it as a PNG:

```
FORCE_COLOR=1 COLORTERM=truecolor python3 scripts/ui-preview.py
python3 scripts/ui-shot.py .scratch/ui.png
```

## Charts

```
scripts/bench-chart.py .scratch/bench/<run>/rows.jsonl out.png "title"
```

## Layout

```
onecli/            the package - agent loop, wires, tools, sessions, bench
onecli/commands.py the session's slash commands
onecli/commandfiles.py  .claude/commands/*.md, as slash commands
onecli/mentions.py @path in a prompt - the file rides with the question
onecli/inputline.py history, tab completion, the line editor
onecli/relief.py   ASCII relief - lit heightfields drawn as glyphs
onecli/providers/  openai_wire.py, anthropic_wire.py
scripts/one        the CLI entrypoint
scripts/one-chat   the chat entrypoint - chat first, tools underneath
scripts/one-bench  the bench entrypoint
scripts/one-arena  the cross-harness arena
scripts/one-parity the wire-parity check
scripts/bench-chart.py  a run as a chart
scripts/ui-preview.py   the whole terminal surface, from fixtures
scripts/ui-shot.py      the same, as a PNG
tests/             offline tests, no network
```

## Tests

```
python3 -m pytest tests -q
```

They stub the transport, so they need no key and no network.

## Task masks

Use `--masks` to list task masks, `--mask build` or `--mask review` to select one,
and `/mask` to inspect or switch during a conversation in either CLI mode.
Masks combine task instructions with bounded tool-loop settings. Review masks
enforce read-only evidence collection. See [masks and tool loops](mask-policy.md)
for custom profiles, session behavior and enforcement limits.


## Modules

| Module | Role |
| --- | --- |
| `onecli/agent.py` | the agent loop: turn, tool calls, results, repeat |
| `onecli/guard.py` | the gate before a tool runs - read-only and external guards |
| `onecli/cli.py` | the `one` entrypoint - flags, session, terminal output |
| `onecli/config.py` | backend resolution: flags, then drop-ins, then a live llama-server |
| `onecli/providers/` | the OpenAI and Anthropic wires behind one response contract |
| `onecli/tools/` | workspace-rooted filesystem, search and shell tools |
| `onecli/session.py` | session persistence across backends |
| `onecli/todo.py` | plans that survive compaction and resume |
| `onecli/evals/bench.py` | the same task scored across backends |
| `onecli/evals/arena.py` | the same endpoint driven through multiple harnesses |
| `onecli/harnesses.py` | adapters for third-party CLIs |
| `onecli/render.py` | terminal rendering |
| `onecli/context.py` | context-window accounting, compaction and spend |
| `onecli/fallback.py` | backend fallback chains |
| `onecli/permissions.py` | permission modes and per-tool rules |
| `onecli/hooks.py` | turn and tool-call hooks |
| `onecli/update.py` | the daily update notice and `one update` |

## Benchmarks

<!-- manual:benchmarks:start -->
Benchmark reports are dated snapshots, not claims about the current build. Run `scripts/one-bench` or `scripts/one-arena` for current results.
<!-- manual:benchmarks:end -->

### Headless feedback from a parent process

A headless run may receive `--feedback-fd N`, an inherited private socket
created by its parent. No listener is opened. The parent sends newline-delimited
JSON objects with string fields `id` and `text`. Complete valid frames append
user messages to the active history in order; incomplete frames wait for the
next read, and invalid frames are ignored. Buffered input is limited to 1 MiB.

The Agent acknowledges with `{"id":"…","accepted":true}` only after the next
model request passes its budget checks. A receipt confirms admission to the
request, not model obedience or successful provider authentication. Unacknowledged
input remains the parent's responsibility when a run ends or fails. The socket
closes at the end of the run; a sender that half-closes can still receive pending
receipts. Feedback arriving during final generation continues the existing task
only when another step remains. Existing step and input budgets still apply.

This transport is shared across providers using the Agent loop. New model
backends still require behavioral acceptance tests; the protocol does not
guarantee that every model follows a correction.

Git checkpoints exclude untracked files named `.env`, `.env.*`, `*.pem`,
`*.key`, or `id_*` from the object database and announce each skipped file.
Undo preserves these pre-existing files but cannot restore edits to their
contents. Tracked files, including secret-named files, retain normal undo
coverage; newly created files are removed by undo. Other untracked,
non-ignored files are captured in Git objects, which may remain until Git GC.

## OpenCode account models

Use `one -m opencode/PROVIDER/MODEL` with the optional pinned SDK transport.
See [OpenCode setup and limits](opencode.md) for installation, vendor-managed
authentication and the supported tool-loop boundary.

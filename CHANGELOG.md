# Changelog

## 0.1.1

- A run that makes a decision in a checkpoint reply now executes the calls in
  that same reply instead of stopping without progress.
- Missing optional integrations no longer change the built-in tool list: the
  core CLI ships its own tools and reaches anything else through an MCP server
  you configure.
- A drop-in's key is chosen by its wire: an OpenAI-wire endpoint never sends
  an Anthropic key, and both drop-in readers agree.
- Permission clauses on MCP tools match any string argument; a clause on a
  tool that takes none is refused instead of silently never matching.
- `one-chat`: a provider error after a tool call keeps the turn and records
  the error, so `/undo` undoes that turn and no earlier one.
- `!command` output that is not UTF-8 no longer ends the session.
- A listener that answers but does not speak HTTP, or malformed relay or
  lane-health data, is treated as unreachable or absent instead of crashing.
- A model endpoint that answers a request with a non-HTTP reply, or cuts a
  reply off mid-body, streaming or not, is reported as unreachable: the
  fallback chain moves on and `one-chat` keeps running. The same holds for
  the output-maximum probe, `one tune` and `one models pull`. A stream cut
  mid-body is handled as a truncated stream, re-sent while nothing has been
  shown.
- `one-chat` approval prompts show what is being approved: a `code` call
  shows its program, and a tool with other argument names shows them all.
- `edit` refuses an empty `old_string` instead of splicing the new text
  between every character of the file.
- A path containing a NUL byte is refused as a tool call instead of ending
  the turn.
- An MCP tool counts as read-only only when its `readOnlyHint` is the
  boolean `true`; a string such as `"false"` no longer unlocks it under
  `--read-only`.
- `one-chat` `/save` writes the transcript readable by its owner only.
- A `Read` deny rule now hides the denied names from `ls`, as it already did
  from `glob`; grep no longer names a denied file in its refusal note.
- The release verification guide states that the admission records are
  committed at each release tag and how to re-run the admission check from a
  clone of the tag.
- An image attachment under an explicit `--base-url` resolves
  `ONE_CLI_VISION_BACKENDS` names from drop-ins without probing local
  llama-servers.
- `--speak` probes only the TTS endpoint, never the speech-in relay.
- `one serve` runs a native llama-server detached, waits for it to report
  healthy and logs its output under `~/.one-cli/serve/`, instead of blocking
  for five minutes and ending in a traceback.
- Child agents share the parent's cumulative input ceiling as well as its
  meter and dollar ceiling.
- `one-chat`: Ctrl-C during `/retry`, `/edit`, `/compare` or a `!command`
  interrupts that command and keeps the session; a one-shot exits 130.
- `one-chat`: a bash approval answer can no longer be swallowed by the
  background input lane.
- `--chat` and `/tools auto` help say what the tool loop does: read tools run
  freely and bash asks before running. `one-chat --help` no longer says it
  has no tool loop.
- Terminal control sequences in tool arguments, plan items, notices, fallback
  reasons, provider errors and `/logs` and `/diff` output are stripped before
  they reach the terminal, as model answers already were.
- `Bash(git push:*)` matches `git push` alone or with arguments, as in Claude
  Code, instead of only the literal text `git push:`.
- An `@path` mention of a file a `Read` rule denies is left as plain text
  instead of being sent to the provider.
- A mask file entry with an unknown field or a non-object value is refused
  before any provider call instead of being ignored.
- `one-arena --budget` charges the parity probe and judge requests and
  refuses a judge lane it cannot price; `--budget 0` sends no request.
- `/task spawn` workers inherit the session hooks (`--hooks`,
  `--trust-project-config`, `--no-hooks`); a `--hooks` source a worker cannot
  reopen refuses the spawn instead of running with less policy.
- An installed `/task spawn` worker runs (`python -m onecli`; `-m onecli.cli`
  imported the module and exited 0 having done nothing), and it starts in the
  package root with `PYTHONSAFEPATH=1`, so a workspace file shadowing a
  standard-library module or `onecli` cannot run before the session's rules.
- `job_output` and `/logs` reach lines past the first MiB of a job log; the
  scan cap bounds only the returned window.
- A `**/` in a permission rule matches whole directories only:
  `src/**/config.py` no longer matches `src/notconfig.py`.
- A symlinked workspace (`-w` at a link, macOS `/var/folders`) is resolved
  once at start, so read, write, ls, grep and glob work in it; a search root
  the guarded walk cannot open is an error instead of "no matches".
- Background commands named at exit, `/ps`, `/jobs`, `/task log` and `/mcp`
  failure reasons are stripped of terminal control sequences.
- The not-found hint of `read`, `edit` and `ls` leaves out names a `Read`
  deny rule hides.
- Documentation and packaging cleanup.

## 0.1.0

The first alpha. The distribution name is `one-core-cli`; the installed
command is `one`.

- `/undo` records every pre-turn path, ignored files included, and never
  unlinks one; it also restores the snapshot `.gitignore` before it looks for
  new files, so a turn that deletes or edits ignore rules no longer makes undo
  delete `.env` or `data/`.
  `one --chat` now builds and enforces the same hooks as `one`, from the
  same flags (`--hooks`, `--no-hooks`, `--trust-project-config`), and
  `one-chat`, which takes none of them, runs the user settings hooks in
  `~/.claude/settings.json`: UserPromptSubmit on every question, PreToolUse
  failing closed in the tool loop, and code-mode confinement. Every chat
  line that carries model, tool or hook text passes through
  `render.untrusted`: answers, thinking, final text, `/compare` and
  `/history`, tool lines, tool output and errors, refused-call and
  hook-denial reasons, hook notes, provider errors, job and detached-task
  output, `/diff`, `/checkpoints`, `/queue`, `/sessions` and `ask_user`
  questions and options. Model text can no
  longer write OSC 52, retitle the window or request a terminal reply.
- Close six permission bypasses. **Behavior change:** file tools are now confined to the workspace and `--add-dir`
  roots in every permission mode, `bypassPermissions` included; an outside
  path is refused with a message naming `--add-dir`. **Behavior change:**
  workspace skills load only with `--trust-project-config`, never through a
  symlink out of their skill directory, and pass the path rules like `read`.
  `one --chat` asks before a mutating call even when it does not echo
  (including under `--json`) and now honours `-w`, `--read-only`, `--permission-mode`,
  the allow/deny lists, `--guard` and `--add-dir`. `Read` rules govern
  grep/glob/ls, `Edit` rules govern write, and `Bash` rules, guards and hooks
  govern `job_start`; a `code` program runs sandboxed whenever a guard or
  PreToolUse hook is active. A pretty-printed JSON deny from a hook refuses,
  and `permissionDecision: "ask"` refuses instead of allowing.
- `/undo` reverts exactly the turn it undoes: every turn (chat and REPL)
  records whether its checkpoint exists, a turn without one refuses instead of
  rolling back an older turn and deleting its work, and a tools-off turn drops
  only its exchange. /undo cuts the conversation back to where the turn began, so
  a turn with an interrupt note or harness nudge is undone whole. Checkpoints skip FIFOs, sockets and devices and report
  other failures; non-git copies refuse past 5000 files or 200 MB, keep the
  last 20 per session under `~/.one-cli/checkpoints/<sid>/`, and are deleted
  on rollback, at session end, at process exit and by a startup sweep of
  sessions whose process is gone or idle for 7 days; git checkpoints no longer need a configured
  git identity. A stream that closes without `[DONE]`, a
  finish reason or `message_stop` is a transient error, re-sent while no
  answer text was shown. The bash tool no longer reads the CLI's stdin, job
  logs and records are private (0700/0600), and MCP tools no longer receive
  the internal `workspace` argument, so a tool parameter of that name works.
- Update notice and `one update`: a terminal session checks PyPI for a newer
  `one-core-cli` at most once a day in a background thread and prints one line
  at the next start; silent offline, before the first release (404), in CI and
  without a TTY, and off with `ONECLI_NO_UPDATE_CHECK=1` or
  `"update_check": false` in `~/.one-cli/config.json`. `one update` detects uv
  tool, pipx, uv pip or pip and prints then runs the matching upgrade; a source
  checkout gets instructions, and `--check` only reports. Only a plain
  version of at most 32 characters is cached or printed, a failing check
  never stops a start, and a bare `one update` with piped stdin is a prompt.
- Keep an answer that travelled with a tool call when the run then stops on
  the dollar or input ceiling, as the step ceiling already did; strip an
  orphan `</think>` that trails a Markdown answer or leads the content on the
  OpenAI wire, leaving tags inside an open code fence or span alone.
- Act on a decision instead of braking it: a checkpoint reply that commits
  (`COMMIT:` or an acting call) runs its calls, and further searching after a
  commit spends the allowance instead of resetting it. The shell refuses
  `git stash pop` after a push that saved nothing and stops background
  processes a command left running when the session ends, and a silent `rg`
  search is stopped at its deadline. `scripts/session_waste.py` no longer
  counts test, lint and git/CI shell rounds as search.
- Name a refusal that does not depend on the arguments from its second
  occurrence, point a missing path at the closest real names, and show the
  closest lines when an edit's old text does not match; adds
  `scripts/session_waste.py`, a read-only ranking of wasted tool calls across
  recent sessions.
- Ranked, portable search: agent `grep`/`glob` use bounded Python within the
  permitted workspace and added roots, with every cut counted. The unguarded
  search API also supports rg and a pure-Python walk.
- Low-progress checkpoint: after six rounds with no progress tool (default;
  `--discovery-limit`, 0 = off) one tools-withheld step asks for `COMMIT:` and
  a target or one clarifying question.
- Abandon a stalled model stream instead of hanging: no bytes for 120 s
  (`ONECLI_STREAM_IDLE_TIMEOUT`) or only keepalives for 600 s
  (`ONECLI_STREAM_CONTENT_TIMEOUT`) re-sends the request up to twice while no
  answer text has been shown, then fails loudly. Thinking frames and pings
  count as activity.
- `--done-check CMD`: a final reply only counts as done when CMD exits 0 in
  the workspace. A failure is fed back to the model up to 3 times (a tool call
  resets the count); after that the run ends `unverified`, exit 1.
- Keep a turn's work in the interactive conversation when it ends on a
  provider error (a 429 mid-turn dropped every tool call of that turn from the
  history), record the error in the session, and persist mid-turn checkpoints
  in the REPL as one-shot runs already did.
- Recover from empty model turns without a default limit: each one gets a
  continuation nudge instead of ending the run after a single retry. A mask
  may still set `empty_retries` (0-1000).
- Add the progress-style prompt to a conversation once, not again on each
  REPL turn.
- Persist one `assistant_turn` record per model call in the session file:
  message index, finish_reason, usage (including reasoning tokens) and the
  reasoning text, so a degenerate turn can be attributed to its cause.
- Send no temperature unless `--temperature` (or `/temp`) sets one; the
  provider's own default applies. The agent no longer forces greedy 0.0 and
  chat no longer forces 0.7.
- Native skills: SKILL.md packages in the workspace and home `.one-cli/skills`
  and `.claude/skills` are listed in the system prompt, and the `skill` tool
  loads one whole in a single call.
- Echo an assistant turn's `reasoning_content` back on later requests. DeepSeek
  thinking mode rejected every follow-up without it (HTTP 400).
- Send no `max_tokens` unless `--max-tokens` is set (default was 4096). The
  Anthropic wire, which requires it, sends the model's own maximum from the
  Models API and names `--max-tokens` when the endpoint declares none.
- Remove fixed tool-output sizes: `read` returns the whole file (no 10 KB
  budget, 400-line default or 400 KB refusal), tool results spill only when they
  exceed half the free context window, and `!command` output is not cut.
- Compact only when the history would not fit the model's context window
  (`--compact-at` default 1.0, was 0.8).
- Ship no operator-specific provider defaults, and use fictional demo entries
  in the bundled terminal projection.
- Store session updates as deltas to keep transcript growth linear. Write a full
  messages record on the first write in a process, after cache eviction, after
  an external write, or after a history rewrite; checkpoints are not periodic.
  Older readers see only the last full record and can miss all later deltas.
  Invalidate cached state after external transcript writes before appending
  messages, metadata or todos.
- Restore native terminal selection and bound input batches so sustained typing
  does not indefinitely delay rendering.
- Respect an explicit model ID when using a custom base URL.
- Build and check wheels and source archives outside the checkout.
- Meter the cumulative input ceiling by cost: cache reads count at their price
  relative to fresh input (catalog third column, default 0.1). Wires report
  cache hits as canonical `cache_read_tokens`; Anthropic `prompt_tokens` now
  includes cache reads and writes. A reply on a step the ceiling forced reports
  `stopped: budget_input`, not `answered`. Every tool result is bounded before
  it joins the transcript.

### Upgrade notes

Back up session files before upgrading. Older readers ignore `messages_delta`
records and see only the last full checkpoint; downgrading can hide later turns.
Keep the previous artifact and use disposable state copies for rollback tests.
See [platform coverage](docs/support-matrix.md).

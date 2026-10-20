# Progress policy

The CLI enables progress checkpoints by default. The Agent library defaults to
disabled. Configure progress_policy in orchestration roots and worker presets:
enabled (boolean), seconds (default 30), tool_calls (5), min_seconds (10),
on_error (true). CLI overrides: --progress-seconds, --progress-tool-calls,
--no-progress. A zero seconds or tool_calls disables that trigger.

At completed tool-batch boundaries, elapsed silence, call count, or an error can
require a model-authored public update before continuing. Count-triggered updates
respect min_seconds. Updates consume the normal step/token/cost budget. The final
step is reserved for an answer. A long synchronous call can exceed the threshold:
this is a boundary check, not a wall-clock delivery guarantee.

progress_update events are emitted only with --text-events and must be treated as
completed public messages, separately from provisional text deltas. External
harnesses do not share the owned loop's enforcement guarantee.

Checkpoints are scheduled communication, never evidence of a tool outage. Tool
liveness and generated findings are separate signals.

First-response enforcement validates an explicit proceed/clarify/decline/answer
JSON assessment before tools are available. OpenAI-shaped providers use JSON
mode for this call only; endpoints explicitly rejecting response_format with
HTTP 400 get one plain-format retry, still subject to strict validation.
Other wires use the same assessment instruction and validator. Formatting
instructions are request-only and do not persist in the conversation.

An empty checkpoint or a checkpoint requesting tools is skipped, without running
those tools or emitting a fabricated update. The next step resumes normal gated
work within the existing budget. Another checkpoint requires another completed
tool batch; skipping never resets the timestamp of the last public message.


## Parent processes

Any program can run `one` as a background job and follow its progress. Export
`BG_SIGNALS` with the path of a file the parent reads: the run appends
`PHASES: <count>` when its todo plan changes size and one `PHASE: <item>` line
as each plan item is done. A run with no plan writes nothing. Model-authored
progress updates, tool calls and lifecycle events stay on the Agent event
callback (`--json` prints them); they never go to the signals file, so a nested
run cannot announce completion for the job that owns it.

Set `ONE_DETACHED_OWNER=caller` when the parent also owns route selection and
background work: `one` then builds no fallback chain of its own and refuses to
start local detached workers, so tasks go back to the parent.

```sh
signals=$(mktemp)
BG_SIGNALS=$signals ONE_DETACHED_OWNER=caller \
  one -p "add a --version flag and test it" > job.out 2>&1 &
tail -f "$signals"    # PHASES: 3, PHASE: read the parser, ...
```

one-cli's own task manager uses the same protocol: each detached task gets its
own signals file.

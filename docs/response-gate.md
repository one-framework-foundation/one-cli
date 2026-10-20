# First-response gate

one CLI defaults to `--response-gate enforce`. Every user turn, including a
resumed turn, first requests a structured assessment with no tools advertised.
Only a valid `proceed` decision unlocks the existing tool loop, after its
model-written acknowledgment is emitted. `clarify`, `decline`, and `answer`
return the public message immediately without Stop hooks restarting execution.
Private reasoning and raw control JSON are not sent to the public text channel.

Invalid assessments, including premature tool calls, execute nothing. One retry
is allowed. Assessment requests use the existing step, token, and spending
budgets; they do not add an unmetered sidecar or increase the step limit. This
orders responses and tool execution; it does not guarantee a model latency or
prove the semantic quality of the assessment. The initial request is buffered
for validation, so its public message appears after that request completes.

Configure the root through `--response-gate off|prompt|enforce` or an explicit
`--orc path.json` file:

```json
{"response_gate":"enforce","max_steps":30}
```

The flag overrides the orchestration file. `prompt` injects a first-response
instruction without deterministic enforcement. `off` retains the ordinary loop.
Direct Python embedders opt in with `Agent(..., response_gate="enforce")`; the
library default stays `off`, so benchmark and programmatic callers choose their
own policy explicitly. CLI interactive and one-shot entry points both default
to enforcement, independently of the model endpoint or wire protocol.

Every agent preset is a configurable node. Children inherit the parent policy
unless their preset sets `response_gate`. This example selects prompted behavior
for an external driver, whose internal tools one CLI cannot intercept:

```json
{"agents":{"review":{"driver":"codex","response_gate":"prompt"}}}
```

Use the existing `--agents path.json` option to load the nodes. The same field
works for Claude and other registered external drivers. External drivers reject
`enforce` before the driver starts because they have no pre-tool enforcement adapter yet;
they must be explicitly configured `prompt` or `off`. A successful prompted run
is not evidence of enforcement. Resolved orchestration records expose each
node's policy. Child acknowledgments go to the parent's public event sink.

For JSON consumers, stdout remains one final result. With `--text-events`,
stderr emits a completed `acknowledgment` event carrying `text`. This event is
separate from provisional `text_delta` previews and must not be coalesced away
when the next step starts. `response_gate` events report decisions/attempts.
Emission is not a delivery receipt; applications own delivery. An embedding
application should await acknowledgment delivery outside the lossy token
preview queue.

No live model acceptance is implied by scripted-provider tests.

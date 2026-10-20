# Masks and tool loops

Both CLI modes support `--mask NAME`, `--masks` and interactive `/mask`,
`/mask NAME`, `/mask off`. Selection persists with the session and can be
explicitly overridden on resume. Switching preserves conversation and model.
Mask instructions are attached to individual requests, never saved as stale
system instructions in conversation checkpoints.

| Built-in | Loop defaults | Enforcement |
| --- | --- | --- |
| general | Progress check after 30 seconds or 5 calls | Host tool permissions |
| build | Up to 30 steps, repeat threshold 3, one empty-response recovery, cadence 30s/5 calls | Host tool permissions |
| review | Up to 12 steps, repeat threshold 2, one empty-response recovery, cadence 45s/8 calls | Read-only tools and evidence before completion |

Host step, repetition and deadline limits remain ceilings. Explicit host cadence
settings override mask defaults. Cadence checks occur at tool-batch boundaries;
these are opportunities for meaningful updates, not guaranteed public messages
or a heartbeat during a blocking tool. Masks do not select providers or authorize
spending. No mask is selected by default; existing unmasked sessions continue.

Custom catalogs use `ONE_MASKS_JSON`. Entries can contain `preamble`,
`execution_policy`, and a versioned `loop_policy`:

```json
{"audit": {"preamble": "Inspect and cite evidence.",
  "execution_policy": {"version": 1, "kind": "review", "max_steps": 10},
  "loop_policy": {"version": 1, "max_steps": 10, "repeat_limit": 2,
    "tool_deadline": 60, "empty_retries": 1,
    "progress_policy": {"enabled": true, "seconds": 40, "tool_calls": 6}}}}
```

Unknown settings and invalid values fail before provider calls. `empty_retries`
is 0-1000; masks cannot disable host repetition/deadline guards. `--masks` lists
available masks and `/mask` also inspects selected policy settings. Run records
include resolved loop settings in `mask`; policy evidence remains separately
reported. Configured names override built-ins. External host catalogs still
supply prompt/model hints only. Child agents retain their existing host profiles;
mask inheritance into delegated work is not implemented.

## Embedding execution policies

The Python Agent API accepts an opt-in, versioned `mask_policy` object.
Without one, runs and corpus benchmarks behave as before.
The policy does not change routing or the harness selector.

```json
{
  "reviewer": {
    "preamble": "Review the change and cite file locations.",
    "execution_policy": {
      "version": 1,
      "kind": "review",
      "max_steps": 12,
      "tools": ["read", "ls", "glob", "grep"]
    }
  }
}
```

Load this using `ONE_MASKS_JSON`. `MaskInfo.agent_options(enforce=True)` supplies
the Agent keyword arguments. Omitting `enforce=True` keeps prompt-only behavior.
The JSON loader validates policies; malformed or unknown fields fail closed.
The optional host registry loader continues to load only prompt/model hints.

```python
from onecli.agent import Agent
from onecli.evals.corpus import load_masks

masks, error = load_masks()
if error:
    raise ValueError(error)
mask = masks["reviewer"]
agent = Agent(provider=provider, workspace=workspace,
              **mask.agent_options(enforce=True))
result = agent.run(task_prompt)
```

| Policy | Runtime behavior |
| --- | --- |
| review | Only read, ls, glob and grep; at least one nonempty, non-error evidence result before final completion |
| debug | Read tools plus write, edit, bash and code; requires a trusted `mask_verifier(workspace)` callback returning exactly `True` at each completion attempt |

`tools` can narrow the task's fixed ceiling. The runtime intersects that list
with host-supplied tools and clamps the step limit to the smaller budget.
Direct and nested dispatch use the same policy gate, before existing host guards.
Denied calls never become evidence. Evidence resets on every run, including
resumed histories. Completion rejection consumes the existing step budget.
Initial response assessment may still ask or refuse without verification; masked
runs report this as `stopped="assessment"`, not completed work.

The host verifier is code supplied by the embedding application, not mask JSON
or a model tool argument. It must perform a bounded task-specific check of current
artifacts, handle its own timeouts, and stay within host authorization. Exceptions
fail verification. The adapter does not parse shell output as proof or install
commands from masks. A verifier is part of the trusted computing base.

Review evidence collection does not establish that the final findings are true
or correctly cited. Debug verification is only as strong as the host checker.
Tool restrictions are not an OS sandbox: granting bash permits its normal shell
capabilities. Isolated request-budget and sandbox adapters are rejected until
their verification boundaries have a dedicated integration.

Results include the resolved version, kind, step cap and tool list in
`result.mask_policy`. Callbacks receive `mask_policy` and
`mask_completion_denied` events. Model selection, dollar budgets and escalation retain their existing host
configuration. Loop profiles can narrow repetition and deadline limits.

For quality evaluation, compare identical tasks, initial files, providers and
host budgets in three arms: baseline, preamble only, and preamble plus enforced
policy. Grade all arms with an independent task oracle. Record success, cost,
latency, calls, denials and stop reason alongside model and policy version.
Scripted-provider tests establish enforcement only; no paid model comparison has
been run and no quality improvement is claimed.

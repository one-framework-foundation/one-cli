# Mask context contracts

Named masks carry a versioned context contract through the JSON catalog and
Agent API. Built-in masks declare required knowledge: build needs acceptance
criteria, implementation and verification; review needs scope/diff, surrounding
implementation and review criteria. These knowledge categories are model
instructions, not a claim that the host can automatically judge completeness.
Explicit required file references are enforced before each ordinary model step.

```json
{
  "builder": {
    "preamble": "Implement and verify the requested change.",
    "context_contract": {
      "version": 1,
      "required_knowledge": ["Acceptance criteria", "Relevant tests"],
      "retrieval": "workspace_files",
      "refresh": "each_request",
      "token_budget": 8192,
      "references": [
        {"path": "TASK.md", "scope": "current task", "required": true},
        {"path": "src/app.py", "scope": "implementation", "required": true}
      ]
    }
  }
}
```

References are explicitly selected workspace-relative UTF-8 files. Retrieval is
bounded, checks host read permissions and hooks, and rejects paths resolving
outside the workspace. Isolated/sandbox adapters reject automatic retrieval;
this feature does not bypass their grants. No network retrieval or provider
selection is introduced. An optional `version` pins the exact SHA256 of file
bytes; a mismatch makes a required reference fail visibly.

`each_request` re-reads at every ordinary agent step; `task` snapshots within one
run and re-reads on the next run/resume. Required references cannot be omitted
for budget, read failures or stale pinned versions. The run stops with
`stopped=context`, an error and nonzero CLI exit status. Optional omissions are
reported and append an unavailable-source record, invalidating prior evidence.

## Assembly and retention

The existing core prefix is retained. The active mask and its contract remain
request-local, ahead of conversation history. They are stable while the task's
mask is unchanged. Evidence is appended as user data with source, scope and
content version. Unchanged evidence is not appended twice. Changed evidence,
including a reversion to an older version, appends a superseding record.
Compaction and JSON resume reconcile full references; summaries do not count
as retained original evidence. Controls still remain request-local.

The budget covers active contract instructions and selected evidence, including
metadata. It uses the existing character token estimator and is not a tokenizer
measurement or total provider-window guarantee. Historical evidence remains in
the transcript until ordinary compaction. Tools and core policy are not trimmed.

Mask switches between runs replace the request-local prefix and preserve the
conversation. They may invalidate provider caches. Automatic role-session
boundaries and compact role handoffs are not implemented; substantial role
changes still require the caller to start a new session with an explicit handoff.
There is no automatic code/diff discovery or semantic correctness evaluator.

## Measurement

Both OpenAI-compatible and Anthropic adapters fingerprint the actual serialized
HTTP body in streaming and buffered paths. Metrics contain serialized size,
request/prefix hashes, prefix-change status and the count of matching leading
message hashes. They retain hashes, not source content. The model, system and
tools participate in the prefix fingerprint, after provider conversion.

The Agent emits `wire_context` and `context_contract` events. Mask run metadata
stores those observations alongside the contract. Reported cache-read/create
usage is copied only when present; absent usage stays unknown. Fingerprints and
shared-message counts are diagnostics, not provider token-cache measurements.
The latest attempted serialization is retained by each adapter; intermediate
failed retry attempts are not individually retained in the Agent result.

Tests cover validation, permissions, budget refusal, stale versions, source
refresh/reversion/deletion, deduplication, JSON resume, compaction restoration,
mask disabling and actual serialized adapter payloads. Fixture tests do not
establish live provider cache savings or task-quality improvements.

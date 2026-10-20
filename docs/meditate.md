# meditate

`one --meditate "task"` opts the owned agent loop into a single review of its
first draft. `Agent(..., meditate=True)` provides the embedding equivalent.
The default is off. The flag applies to the top-level owned loop, not external
harnesses or spawned subagents. It does not run a separate judge model.

The review compares the request with the answer, chunks material claims, asks
for disjoint evidence, and revises once. It can call only read, ls, glob and
grep, through the existing permission and hook gates. The runtime also rejects
other tools if a provider requests them despite the restricted schema. There
is no shell, write, network, MCP or agent-spawning capability in this pass.

The existing step, token and money limits still apply. Fewer than two remaining
steps skips the review. Clarification/refusal from the first-response gate
returns directly. Review is not recursively repeated. Streaming answer deltas
are buffered by suppressing public deltas for meditate-enabled runs; the final
result is printed normally, while tool/progress events remain available.
Events record reviewing, reviewed, or skipped_budget. Reviewed means the pass
finished, not that its claims have been independently certified.

This is a reasoning aid, not a contradiction classifier. It asks for new
evidence but does not check that the evidence is new, and it cannot verify a
claim that needs a command to be run. It uses additional model calls within the
existing budget.

# Durable provider request admission

The optional request budget mode reserves a conservative upper cost before each
OpenAI-compatible HTTP attempt, including streaming, automatic retries,
compaction and native child agents. SQLite commits the reservation before the
request can leave. Ambiguous outcomes retain their full reservation. No refunds
are inferred from missing, cached or estimated usage.

Use --require-priced-budget --no-fallback with --request-budget-policy FILE and
--request-budget-ledger FILE. A new work item must explicitly add
--create-request-budget-ledger. Recovery omits that creation flag so missing or
empty storage blocks; reopening never enlarges the stored limit. All processes
in the work item must use the same protected ledger path and limit.

The policy is trusted operator configuration. Exact version 1 fields:
base_url, model, input_token_bound, output_token_bound,
input_usd_per_million, output_usd_per_million, audit_evidence, version.
Bounds are positive integers. Rates are finite nonnegative decimal values;
strings avoid floating-point conversion. Evidence must identify an actual audit.
A nonempty evidence string alone is not an audit or runtime certification.

The input bound MUST cover the provider's maximum billable input including
framing/cache/reasoning accounting. The output cap MUST bound ALL billed output.
Rates MUST upper-bound all applicable tariffs. Default model context sizes and
substring price lookup do not establish these facts. one-cli ships no policy
file; the operator writes and audits it.

Each attempt reserves full input_bound * input_rate plus requested output_cap *
output_rate, rounded upward to nanodollars. Even a verified free request reserves
one nanodollar to retain a bounded durable record. This intentionally leaves
unused capacity rather than refunding uncertain costs.

Only the exact OpenAIWire transport, exact configured route, text messages and
client function tools are supported. Redirects, external CLI drivers, alternative
child routes, missing output caps and unaudited request extensions are rejected.
Native children share the same request admission object and durable ledger.

This controls this provider transport, not arbitrary shell commands, hooks,
MCP servers or other external clients. An unrestricted agent can spend outside
it, so this mode alone does not certify a whole-run hard budget.
Database tampering/replacement, provider misbilling and untrue operator policy
are outside this mechanism's integrity boundary.

## Tool-free isolated profile

Add --request-budget-isolated for a deliberately narrow one-shot inference
profile. It requires the request policy and existing/explicitly created ledger
above, and the same prerequisite no-fallback budget checks. Hooks and MCP are
skipped before their configurations are read or servers started. Explicit hook,
MCP, guard, phase, resume, chat and media options are rejected; interactive and
alternate listing/serving commands are rejected too. Discovery reads backend
configuration without local network probes. Context lookup uses local tables.

No model tool request can reach guards, hooks or toolkit execution, including
unsolicited shell, code, MCP and task calls. Assessment/progress turns may ignore
tool requests and retry; ordinary tool turns stop the run. No tools are advertised;
model child agents are disabled. The direct Agent API requires RequestAdmission
and rejects hook/guard/spawner injection in this profile. Prompt @file expansion
and automatic project context are disabled. The operator-supplied system file,
policy and route configuration remain trusted inputs.

This is an application-level tool-free boundary, not an OS sandbox. Trusted CLI
transcript/report I/O still occurs. Python embedding callbacks, monkeypatching,
malicious installed code, configuration tampering and provider billing errors are
outside its boundary. Native child agents remain supported only in the original
transport-budget profile, not this isolated profile. General coding agents still
need an independently enforced tools/effects sandbox. The tariff and
billable-token bounds in the policy are the operator's to audit; one-cli does not
certify them.

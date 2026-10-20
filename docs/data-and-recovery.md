# Data handling and recovery

Technical disclosure of what one-cli stores, where it sends data and how to
recover it. It is not legal terms or a provider retention promise. Review the
features you enable, and the exact version you run, before operating a service
on it.

## Recipients and storage

| Feature | Data and recipients | Storage and operator responsibility |
| --- | --- | --- |
| Model conversations | Prompts, conversation history, tool results and selected file/media context go to the configured model endpoint. Credentials authorize that endpoint. | Provider drop-ins are read from `~/.one-providers/` or `ONE_PROVIDERS_DIR`. Provider retention is governed by the selected account/service, not by local history deletion. Check the actual endpoint and account before sending sensitive content. |
| Model fallback | Declared or automatically selected successors can receive the conversation after a failure. | Review fallback configuration or disable it when changing recipients is unacceptable. A failed call does not prove that the first recipient received nothing. |
| Audio/video | Transcription tries the configured speech-to-text server, then Deepgram and Groq, in order; video may first be converted to audio. Inline media follows model capabilities and explicit options. | Availability and configured credentials affect recipients. Model fallback and transcription fallback are separate paths. Do not assume disabling model fallback disables every media recipient. Temporary media and original attachments need a separate retention policy. |
| Sessions and drafts | JSONL conversations, metadata/todos and draft text are local under `~/.one-cli/sessions`. Attachments are absolute path references, not embedded backups. | Session/draft writes use private file permissions. No automatic retention expiry is provided by this session module. Copy/export the files and referenced attachments deliberately; deletion must also cover copies and backups. Concurrent draft editors use last-writer-wins behavior. |
| Hooks, tools and MCP | Enabled extensions receive their configured inputs; shell commands run with the invoking user's authority. | A workspace path is not an OS sandbox. Project hooks/MCP require explicit trust, but a trusted extension can have its own recipients and storage. Inventory it separately. |
| Optional relay | Authenticated callers send messages to the relay operator, which forwards them to configured providers. | The operator controls relay logs and token stores under `~/.config/one-cli/relay/`, including any shared-token override. No central managed service is established by distributing the CLI. |
| Scoped inference | Configured callers send input through configured lanes. The call ledger contains identity, lane, outcomes, timings, tokens and cost fields, not prompt/output bodies. | Config defaults to `~/.config/one-cli/infer.json`; budget state to `~/.local/state/one-cli/infer-budget`; call ledger to `~/.local/state/one-cli/infer_calls.jsonl`. Environment overrides exist. These stores have no unified deletion/retention contract. |

Other enabled jobs, tasks, runs, spills, histories and optional SQLite stores
must be included in the operator's inventory. This table does not certify
third-party plugins or every optional command. Never export provider keys,
relay tokens or private transcripts as troubleshooting evidence. Deleting a
local transcript cannot retract a provider request or revoke a credential.

## Cost boundaries

Ordinary tool execution, media calls and provider usage may incur costs.
`--read-only` is not a spending limit. Conservative request admission binds an
audited endpoint/model/tariff and reserves each attempt before sending;
reservations are not refunded after timeouts or unknown outcomes. Its validity
depends on truthful upper bounds and the provider honoring the audited caps.

The strict request-budget-isolated mode excludes tools, extensions, media,
resume and fallback paths that could escape its audited request boundary.
Do not describe the ordinary estimated budget or a flat-rate account as a
universal hard spending cap. Offline reservation tests do not verify provider
billing or settlement.

## Restore procedure and limits

1. Stop writers and outbound requests. Inventory actual configured paths,
   referenced attachments, uncommitted workspace files, extensions and secrets.
   Keep the compatible reader/package and its version with the backup.
2. Copy quiescent ordinary files with permissions preserved. Use SQLite's backup
   API for consistent database snapshots; copying just a live database file
   can omit journal/WAL state. Protect backup storage at least as strongly as
   the original. Do not include real secrets in rehearsal fixtures.
3. Restore to an isolated destination with outbound calls disabled. Preserve
   original attachment paths or explicitly remap references; copying an
   attachment elsewhere does not update the transcript automatically.
4. Read conversations, metadata/todos and drafts, verify attachment bytes and
   permissions, load configuration, then append a synthetic record. Reopen
   budget ledgers with `create=False`, verify reservations and reject an
   over-budget operation before permitting any network call.
5. Reconcile every post-snapshot request against independent usage evidence.
   **An old budget snapshot can restore spent allowance.** The ledger cannot
   detect replacement with an older valid database. Keep outbound access off
   until the current conservative reservations are recovered; never treat a
   restored snapshot as a new spending authorization.

The offline release exercise in `tests/test_release_recovery.py` covers
synthetic session/draft/config/attachment bytes, permissions, SQLite backup and
reservation continuity at a quiescent snapshot. It is not an installed-package
upgrade test, provider reconciliation, encryption audit, all-store backup or
proof against power loss. Session append flushes userspace buffers but does not
fsync; no zero-data-loss power-failure guarantee is made. Declare an operator
recovery-point/time objective and measure the full selected feature inventory
before accepting a production backup policy.

## Decisions for an operator

Anyone running one-cli as a service for others decides the operator identity and
private contact, retention, export/deletion and backup access responsibilities,
and the applicable legal terms and rights. The CLI cannot infer these.

## Restore inventory and recovery test

Inventory the actual configured locations, including overrides, before copying:

- `~/.one-cli/`: sessions/drafts, history/chat history, jobs, tasks, runs, spills,
  analysis, corpus, models/tuning, orc and senses logs.
- Provider drop-ins, relay tokens, inference config, inference ledger and
  budget databases; external provider/model locations selected by configuration.
- Project MCP/hooks/commands and `.orc` state, uncommitted files and every
  referenced attachment.
- Sandbox control state under the configured state directory.

`tests/test_release_recovery.py` also restores job, task and spill files and
checks that new work continues after them without overwriting the restored
output. Active processes, remote models/services, plugin state and arbitrary
external databases are not resurrected by copying files. Inventory and validate
those separately.

The suggested defaults in the retention table below are suggestions, not
implemented expiry or legal policy. Reconcile
post-snapshot spend before enabling requests on any restored installation.

## Retention decision table

Observed from the current storage readers/writers; the last column is a
suggestion for an operator, not a policy one-cli applies. Environment overrides
and extension-owned stores require deployment-specific inventory.

| Category / default location | Observed retention/deletion behavior | Suggested default |
| --- | --- | --- |
| Conversations, metadata and drafts: `~/.one-cli/sessions/` | Append-only session records; draft atomically replaced on save; no age-based expiry in session module. Attachments remain separate absolute references. | Retain until explicit local deletion; optional user-selected expiry only after export/preview and recovery tests. Include attachments/backups in deletion guidance, without deleting shared files implicitly. |
| Interactive history: `~/.one-cli/history`, `chat_history` | Chat readline history has a 5,000-entry cap; this is a count limit, not an age TTL or guaranteed erasure of session copies. | Keep current disclosed cap; offer explicit history clearing/opt-out before adding any destructive expiry default. |
| Jobs, tasks, spills: `~/.one-cli/{jobs,tasks,spills}/` | Retained output/prompt/signal files; no age expiry in these stores. Reopened stores now skip occupied IDs; running processes are not restored from files. | User-controlled retention; suggest opt-in 30-day cleanup for completed output only, after preview and excluding active work. This cleanup is not implemented. |
| Runs/corpus/analysis and configured extensions: `~/.one-cli/` and workspace paths | Different writers and schemas; no unified TTL. Corpus definitions are reusable source, not disposable logs. Arbitrary extension cleanup is outside CLI control. | Preserve corpus/configuration as source; choose per-extension operational-log retention with its operator. Never treat every file below the state root as expiring cache. |
| Inference ledger and budgets: `~/.local/state/one-cli/` | Append-only call metadata and SQLite conservative reservations; no unified expiry. Old budget restoration can restore spent allowance. | Retain active accounting until independent reconciliation; do not apply a generic log TTL to budget databases. For an operated service, suggest 30 days of minimized operational metadata and a documented accounting hold. |
| Provider credentials, relay tokens and settings: `~/.one-providers/`, `~/.config/one-cli/relay/`, `~/.config/one-cli/` | Configuration persists; removing a local file does not revoke an upstream credential or delete provider-held content. | Explicit account-controlled revocation/configuration lifecycle; no automatic credential deletion. Keep secrets out of routine evidence exports. |
| Workspace state, attachments, source and uncommitted files | SQLite/file state belongs to the workspace; copied/restored content persists until explicitly changed. | Retain under the workspace owner's policy; protect source and user content from generic cache cleanup. |
| Backups and exported copies | No CLI-wide automatic backup schedule or expiry contract. | Protected rolling 30-day backups, tested restore, documented deletion propagation and holds. Not implemented by the CLI. |

See [Verifying a release](reproducible-releases.md). None of this is a
guarantee against host power loss, every optional service, future schema
migrations or provider-side retention.

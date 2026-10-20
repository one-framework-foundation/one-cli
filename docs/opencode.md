# OpenCode models in one-cli

Select an OpenCode catalog model with `one -m opencode/PROVIDER/MODEL`.
For example, `one -m opencode/opencode/MODEL` selects a model from the
OpenCode provider. Use the provider and model IDs from your OpenCode catalog;
model IDs may contain slashes. This explicit selection never falls back to a
local model and does not use one-cli API keys or provider drop-ins.

The optional transport requires a POSIX host, Bun on PATH and exactly `@opencode/sdk@2.0.32`.
Install the SDK in a dedicated directory, then point one-cli at that directory:

```sh
npm install --prefix "$HOME/.local/share/one-opencode-sdk" --ignore-scripts --save-exact @opencode/sdk@2.0.32
export ONE_OPENCODE_SDK_DIR="$HOME/.local/share/one-opencode-sdk"
one -m opencode/PROVIDER/MODEL "your task"
```

Use OpenCode's own account login and configuration before selecting its models.
This transport does not implement login, copy tokens, or replace credential
refresh. The assembled OpenCode SDK host loads its own configuration, catalog,
provider plugins and credentials, including supported OAuth accounts. Run from
the same directory and environment as OpenCode to use the same configuration.
Calling a model can incur that account's normal charges. Installation and
selection do not grant permission to spend or log in on another person's behalf.

one-cli retains its agent, tools and permission checks. A packaged, versioned
helper replaces the SDK's text-only generation service with structured turns;
it sends tool schemas and results, preserves opaque assistant metadata, maps
usage, and forwards explicit output token limits. It never starts OpenCode's
agent loop. Each turn starts an owned host process, so vendor configuration and
helper changes take effect on the next call. Cancellation and deadlines terminate
that process group. Text and reasoning callbacks are buffered until completion.

Provider-use policies are enforced through the vendor catalog. If a restrictive
OpenCode tool/integration policy is present, tool-bearing requests fail closed:
one-cli cannot faithfully apply OpenCode's resource-specific tool permissions to
its own tools. OpenCode agent-specific permissions are not a substitute for
one-cli's own tool approvals. Audited wire-byte/request admission, wire overrides,
structured response-format enforcement and media input are unsupported and
refused. Arbitrary upstream diagnostics are suppressed to prevent credential
leaks; errors report a fixed protocol error category instead.

## Verification without an account

```sh
ONE_OPENCODE_TEST_SDK_DIR="$ONE_OPENCODE_SDK_DIR" python -m pytest tests/test_opencode_sdk.py -q
```

The integration tests isolate HOME, XDG and project configuration and use a local
HTTP fixture. They exercise the actual pinned SDK, tool/result round trips,
output limits, usage, live configuration changes and policy denials without
OAuth or paid inference. Unit tests also exercise cancellation, deadlines and
error redaction. Without the test SDK directory, the SDK integration tests
are explicitly skipped.

## Prior art and boundary

The existing raw-wire providers remain the authority for one-cli's agent loop.
OpenCode 2.0.32's `experimental.generate.text` API was tested and rejected as a
transport because it discards tool definitions, token limits, tool replies and
usage. Launching its agent CLI would replace one-cli's loop. The pinned SDK's
host override and model resolver preserve the structured model boundary while
keeping authentication inside the vendor. The helper deliberately uses that
version's assembled SDK services rather than parsing credential files or
emulating OAuth endpoints. Version mismatches fail closed.

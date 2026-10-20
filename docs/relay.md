# Running the relay

The relay is optional. The core CLI has no third-party dependencies; the relay
has FastAPI and uvicorn, so it installs as an extra:

```
python -m pip install ".[relay]"
python -m onecli.relay.server
```

## Configuration

Everything below is environment, not a baked-in path or address.

| Variable | Default | What it does |
|---|---|---|
| `ONE_RELAY_HOST` | `127.0.0.1` | Bind address. The relay holds every provider key on the host, so exposing it beyond loopback is a deliberate act, never a shipped default. Put it behind a reverse proxy or a private network interface you control. |
| `ONE_RELAY_PORT` | `8790` | Bind port. |
| `ONE_RELAY_CHAIN` | empty (no chain) | Comma-separated drop-in names, highest rung first. With no chain every request fails with an error naming this variable. A name that does not resolve on this host is a warning in `/health`, not a fatal error. |
| `ONE_RELAY_HOST_TOKENS` | `~/.config/one-cli/relay/host_tokens.json` | A host token file managed outside the relay, merged with the relay's own tokens. |

The relay's own tokens live in `~/.config/one-cli/relay/tokens.json` and are
labelled `relay:<name>`. Both files are optional; an absent file contributes no tokens,
and a relay with no tokens at all authenticates nobody.

`/v1/infer` is refused unless the token owner is listed in the relay's
inference configuration.

## Inference configuration

`/v1/infer` reads `~/.config/one-cli/infer.json` (override `ONE_INFER_CONFIG`).
Without that file every inference call is refused. Budget ledgers live in
`~/.local/state/one-cli/infer-budget` (`ONE_INFER_BUDGET_DIR`) and one row per
attempt goes to `~/.local/state/one-cli/infer_calls.jsonl` (`ONE_INFER_LEDGER`).
Unknown keys are rejected.

```json
{
  "version": 1,
  "lanes": [
    {"name": "local-qwen", "slots": 2, "locality": "private", "cost": "free",
     "context": 32768, "json": true},
    {"name": "hosted", "slots": 4, "locality": "external", "cost": "metered",
     "kinds": ["summarize"],
     "tariff": {"input_usd_per_million": 0.5, "output_usd_per_million": 1.5,
                "audit_evidence": "provider price page, checked 2026-01-01"}}
  ],
  "callers": {"relay:ci": {"kinds": ["summarize"], "max_inflight": 4, "usd_per_day": 1}},
  "limits": {"max_output_tokens": 2048, "max_input_tokens": 32000, "max_deadline_ms": 60000},
  "cooldown_s": 90
}
```

- `lanes`, tried in order. `name` is a backend name that `-m` resolves on this
  host; `slots` (1-64) is its concurrency; `locality` is `private` or
  `external`; `cost` is `free`, `flat` or `metered`. A `metered` lane needs a
  `tariff` with both per-million prices and `audit_evidence`. Optional:
  `context`, `vision`, `json` (JSON mode), `kinds` (default `["*"]`).
- `callers`, keyed by token owner: the allowed `kinds`, plus optional
  `max_inflight` (default 4) and `usd_per_day`.
- `limits` lower the per-call ceilings (at most 8192 output tokens, 128000
  input tokens, 120000 ms). `cooldown_s` (0-86400, default 90) rests a failed
  lane; `shared_health` (default false) shares lane health between processes.

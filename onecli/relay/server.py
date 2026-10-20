"""Provider relay server.

Layered on one-cli's own config/providers/fallback, not on a separate proxy:

- `config.resolve(name=...)` turns a drop-in name into a keyed `Backend`.
- `providers.build(backend)` gives the wire object (Anthropic or OpenAI) that
  already knows how to speak to that endpoint and normalise the reply to a Turn.
- `providers.is_transient(err)` is the one definition of "the backend is down
  right now" - the same one the CLI's sticky demotion uses.

The relay presents both wires it might be asked in:
  POST /v1/chat/completions   OpenAI-shaped request and reply
  POST /v1/messages           Anthropic-shaped request and reply
  POST /v1/infer              scoped, tool-less, schema-checked inference
                              (onecli.infer; refused unless the token owner
                              is configured in ~/.config/one-cli/infer.json)
  GET  /v1/infer/health       infer lane pool state (authenticated)
It accepts either shape on either path (it sniffs the body), normalises to a
plain message list, walks the chain, and answers in the shape that was asked.
Layer 1 is non-streaming: one request, one reply, the chain walked inside it.
Streaming is a later layer.

Every reply carries `x-onecli-relay-served-by: <rung>` so the caller can see which
backend answered, and when a demotion happened the reply's model id is the rung
that actually served - that is the machine-readable "you are now speaking to
this model" the caller renders to the user.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from onecli import config, providers
from onecli.providers import ProviderError
from onecli.providers.base import is_transient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Comma-separated drop-in names, highest rung first. Unset means no chain.
CHAIN_ENV = "ONE_RELAY_CHAIN"

# No shipped chain: every rung is a backend the operator configured, so a
# default here would be a list of someone else's. Every name must resolve
# through `config.resolve(name=...)` on this machine, which is what keeps the
# provider keys here and nowhere else. A name that does not resolve is dropped
# into RELAY.warnings, not fatal.
DEFAULT_CHAIN: list[str] = []

# Two token files, both optional, merged into one token -> owner map:
#   a host token file, ONE_RELAY_HOST_TOKENS (default host_tokens.json in
#   RELAY_HOME), for tokens managed outside the relay;
#   the relay's own local tokens          (owner label `relay:<name>`)
# Either file may be absent; an absent file just contributes no tokens.
HOST_TOKENS_ENV = "ONE_RELAY_HOST_TOKENS"
RELAY_HOME = Path.home() / ".config" / "one-cli" / "relay"
RELAY_TOKENS = RELAY_HOME / "tokens.json"

# Loopback, not this machine's address on any particular network. The relay
# holds every provider key on this machine, so exposing it is a deliberate act
# the operator performs by setting ONE_RELAY_HOST, never a shipped default.
DEFAULT_BIND_HOST = "127.0.0.1"

DEFAULT_MAX_TOKENS = 4096


def load_chain() -> list[str]:
    raw = os.environ.get(CHAIN_ENV)
    if raw:
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if names:
            return names
    return list(DEFAULT_CHAIN)


def token_paths() -> list[Path]:
    """The token files to merge, host store first."""
    host = os.environ.get(HOST_TOKENS_ENV)
    return [Path(host) if host else RELAY_HOME / "host_tokens.json", RELAY_TOKENS]


def bind_host() -> str:
    return os.environ.get("ONE_RELAY_HOST") or DEFAULT_BIND_HOST


def load_tokens() -> dict[str, str]:
    """token -> owner label. A value from either file authenticates."""
    tokens = {}
    for path, kind in zip(token_paths(), ("host", "relay")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for owner, tok in data.items():
            if isinstance(tok, str) and tok:
                tokens[tok] = f"{kind}:{owner}"
    return tokens


# Content blocks in, plain text out. Either wire may send either shape.
# Anything else is stringified rather than refused.
def _content_to_text(content) -> str:
    """Both wires accept string content; Anthropic sends a block list. Flatten
    a block list to text so a single message list feeds either wire."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in (None, "text") and "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return "" if content is None else str(content)


def normalise_messages(body: dict) -> list[dict]:
    """OpenAI and Anthropic both key the turn list under `messages`. Anthropic
    carries the system prompt at top level; fold it in as a system message so
    the wires' own `_split_system` puts it back where each one wants it."""
    msgs = []
    system = body.get("system")
    if system:
        msgs.append({"role": "system", "content": _content_to_text(system)})
    for m in body.get("messages", []) or []:
        if not isinstance(m, dict):
            continue
        msgs.append({"role": m.get("role", "user"),
                     "content": _content_to_text(m.get("content"))})
    return msgs


# The chain: one rung per resolvable drop-in, resolved once at import and
# walked in order on every request.
class Relay:
    def __init__(self, chain_names: list[str], timeout: float = 600.0):
        self.timeout = timeout
        self.rungs = []
        self.warnings = []
        for name in chain_names:
            try:
                backend = config.resolve(name=name)
            except Exception as exc:
                self.warnings.append(f"{name}: {exc}")
                continue
            self.rungs.append((name, providers.build(backend, timeout), backend))

    def serve(self, messages: list[dict], max_tokens: int, temperature: float):
        """Walk the chain. Returns (rung_name, backend, Turn, hops) or raises the
        last transient error if every rung is down."""
        last = None
        hops = []
        for name, wire, backend in self.rungs:
            try:
                turn = wire.chat(messages, temperature=temperature,
                                 max_tokens=max_tokens)
                # A rung that answers with nothing is a rung that is broken in
                # a way the wire cannot see: no exception, no text. Treat it as
                # a failed rung and demote, rather than handing the caller an
                # empty completion that looks like a real answer.
                # `last` keeps the reason so the 502 names something real.
                if not (turn.text or "").strip():
                    last = ProviderError(f"{name} returned empty text")
                    hops.append({"rung": name, "reason": "empty text"})
                    continue
                return name, backend, turn, hops
            except ProviderError as exc:
                if not is_transient(exc):
                    raise
                last = exc
                hops.append({"rung": name, "reason": str(exc)[:200]})
                continue
        raise last or ProviderError(
            "relay chain is empty: set ONE_RELAY_CHAIN to a comma-separated "
            "list of the backends this machine has configured")


# The app. One Relay and one token map, both built once at import, so a token
# file edit or a chain change needs a restart.
app = FastAPI(title="onecli relay")
RELAY = Relay(load_chain())
_TOKENS = load_tokens()


def _auth(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization[len("bearer "):].strip()
    if not token:
        raise HTTPException(401, "missing bearer token")
    owner = _TOKENS.get(token)
    if not owner:
        raise HTTPException(403, "token not authorised for the relay")
    return owner


@app.get("/health")
def health():
    return {
        "ok": True,
        "chain": [name for name, _, _ in RELAY.rungs],
        "models": {name: b.model for name, _, b in RELAY.rungs},
        "warnings": RELAY.warnings,
        "authorised_tokens": len(_TOKENS),
    }


async def _body(request: Request) -> dict:
    """The JSON request body, or a 400 - a malformed request is the client's fault."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "request body is not valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    return body


def _run(body: dict):
    try:
        messages = normalise_messages(body)
        max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
        temperature = float(body.get("temperature") or 0.0)
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise HTTPException(400, f"malformed request: {exc}")
    if not messages:
        raise HTTPException(400, "no messages in request")
    try:
        name, backend, turn, hops = RELAY.serve(messages, max_tokens, temperature)
    except ProviderError as exc:
        raise HTTPException(502, f"every relay backend is down: {exc}")
    return name, backend, turn, hops


@app.post("/v1/chat/completions")
async def chat_completions(request: Request,
                           authorization: str | None = Header(None)):
    owner = _auth(authorization)
    body = await _body(request)
    import anyio
    name, backend, turn, hops = await anyio.to_thread.run_sync(_run, body)
    resp = {
        "id": f"onecli-relay-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "model": backend.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": turn.text},
            "finish_reason": turn.finish_reason or "stop",
        }],
        "usage": turn.usage or {},
        "onecli_relay": {"served_by": name, "hops": hops, "owner": owner},
    }
    return JSONResponse(resp, headers={"x-onecli-relay-served-by": name})


@app.post("/v1/messages")
async def messages(request: Request, authorization: str | None = Header(None)):
    owner = _auth(authorization)
    body = await _body(request)
    import anyio
    name, backend, turn, hops = await anyio.to_thread.run_sync(_run, body)
    resp = {
        "id": f"msg_onecli-relay-{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "model": backend.model,
        "content": [{"type": "text", "text": turn.text}],
        "stop_reason": "end_turn",
        "usage": turn.usage or {},
        "onecli_relay": {"served_by": name, "hops": hops, "owner": owner},
    }
    return JSONResponse(resp, headers={"x-onecli-relay-served-by": name})


def _infer_engine():
    from onecli import infer
    try:
        return infer.engine()
    except (OSError, ValueError) as exc:
        # A broken infer config disables /v1/infer only; the chain routes
        # above keep serving.
        raise HTTPException(503, f"infer config invalid: {exc}")


@app.post("/v1/infer")
async def infer_route(request: Request, authorization: str | None = Header(None)):
    owner = _auth(authorization)
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "body must be JSON")
    engine = _infer_engine()
    if engine is None:
        return JSONResponse({"ok": False, "output": None, "error": {
            "code": "not_configured", "retryable": False,
            "message": "/v1/infer is not configured on this relay"}}, status_code=403)
    # The engine blocks on provider sockets; keep the event loop free.
    import anyio
    status, resp = await anyio.to_thread.run_sync(engine.infer, owner, body)
    headers = {}
    served = (resp.get("served_by") or {}).get("lane")
    if served:
        headers["x-onecli-relay-served-by"] = served
    return JSONResponse(resp, status_code=status, headers=headers)


@app.get("/v1/infer/health")
def infer_health(authorization: str | None = Header(None)):
    _auth(authorization)
    engine = _infer_engine()
    if engine is None:
        return {"enabled": False}
    return {"enabled": True, **engine.health()}


def main():
    import uvicorn
    host = bind_host()
    port = int(os.environ.get("ONE_RELAY_PORT", "8790"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

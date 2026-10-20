#!/usr/bin/env python3
"""Read endpoint drop-ins from ``~/.one-providers/*.env``.

Each file describes one endpoint using KEY=VALUE assignments. An optional
``export`` prefix and surrounding quotes are accepted. For example::

    ONE_PROVIDER_SOURCE='~/.provider-credentials.env'
    ONE_PROVIDER_MODEL='example-model'
    ONE_PROVIDER_LABEL='Example endpoint'
    OPENAI_BASE_URL='https://api.example.com/v1'
    ONE_PROVIDER_PARAMS='{"temperature":1}'

The source file holds credentials and must be protected by the operator.
Role drop-ins may specify ONE_PROVIDER_ROLE instead of a model; relay_served()
reads the served model metadata from the endpoint's /v1/models response.

Parsing and endpoint classification live here. Callers retain policy decisions
about credentials, presence probes, subscription use and paid execution.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import logging
import os
import re
import threading
import time
from urllib.parse import urlsplit
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["provider_dirs", "parse_env_file", "read", "is_paid", "lanes",
           "relay_served", "RELAY_SERVED_KEYS",
           "presence_of", "probe_present", "presence_cached", "model_presence",
           "prime_presence", "await_presence_priming", "set_policy",
           "model_refusal", "env_refusal", "prefer", "key_for", "KEY_VARS"]

# How long a presence probe may take. An intermittent lane is a machine that
# comes and goes (a vendor API does not), so a second and a half is generous;
# the cost of waiting longer is paid on every call that asks for a lane, and a
# machine that needs two seconds to answer /v1/models is not going to answer a
# completion in time either.
PRESENCE_TIMEOUT = 1.5

# The two answers a drop-in may give to "is this endpoint always there?".
# `always` is the default and the historical behaviour: the record is taken at
# its word and nothing on the network is dialled to build a lane list.
PRESENCE_VALUES = ("always", "intermittent")

logger = logging.getLogger(__name__)

# Subscription policy. None is built in: without one every well-formed drop-in
# is read and automatic selection keeps discovery order. An operator who must
# keep some models off a plan registers a policy with set_policy(): an object
# with refuses(name, base, model, declared) -> reason ('' admits) and,
# optionally, prefer(lanes, model_of) -> lanes for automatic ordering.
_POLICY = None


def set_policy(policy) -> None:
    """Install (or, with None, remove) the operator's subscription policy."""
    global _POLICY
    _POLICY = policy


def model_refusal(name: str, base: str, model: str, declared: str = '') -> str:
    """Why the policy refuses `model` on this endpoint, or '' when it may run."""
    if _POLICY is None:
        return ''
    return _POLICY.refuses(name, base, model, declared) or ''


def env_refusal(name: str, env: dict, base: str) -> str:
    """Check the model and every tier/body override at the shared boundary."""
    if _POLICY is None:
        return ''
    model = env.get('ONE_PROVIDER_MODEL') or env.get('ANTHROPIC_MODEL') or ''
    models = [model]
    models.extend(env[k] for k in ('ONE_PROVIDER_TIER_HAIKU',
                                  'ONE_PROVIDER_TIER_SONNET',
                                  'ONE_PROVIDER_TIER_OPUS') if env.get(k))
    try:
        params = json.loads(env.get('ONE_PROVIDER_PARAMS') or '{}')
    except ValueError:
        params = {}
    if isinstance(params, dict) and 'model' in params:
        models.append(params['model'])
    for m in models:
        if not isinstance(m, str):
            return f'model override {m!r} is not a string'
        why = model_refusal(name, base, m, env.get('ONE_PROVIDER_SUBSCRIPTION', ''))
        if why:
            return why
    return ''


def prefer(lanes, model_of):
    """Automatic ordering: the policy's preference, else discovery order.

    Explicit model selection never calls this helper. Capability and health
    gates stay with the caller; apply this only within an eligible cost rung.
    """
    lanes = list(lanes)
    order = getattr(_POLICY, 'prefer', None)
    return list(order(lanes, model_of)) if order else lanes


def provider_dirs() -> list[Path]:
    """The shared dir first, then `ONE_PROVIDERS_DIR` (colon-separated).

    Same variable and same order one-cli uses, so a bench endpoint dropped in
    for one command extends the configured set rather than shadowing it.
    """
    extra = [Path(os.path.expanduser(d))
             for d in os.environ.get("ONE_PROVIDERS_DIR", "").split(":")
             if d.strip()]
    return [Path.home() / ".one-providers"] + extra


def parse_env_file(path) -> dict:
    """A KEY=VALUE drop-in as a dict. Unreadable file -> {} (never raises)."""
    out: dict = {}
    try:
        text = Path(path).read_text()
    except (OSError, UnicodeDecodeError):
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        key, sep, val = line.partition("=")
        if not sep:
            continue
        out[key.strip()] = val.strip().strip("'").strip('"')
    return out


# Only exact localhost and private IP literals qualify as unbilled lanes.
_PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    # Packed IPv4 networks: loopback, RFC 1918 and RFC 6598.
    (0x7F000000, 8), (0x0A000000, 8), (0xC0A80000, 16), (0xAC100000, 12),
    (0x64400000, 10), "::1/128", "fe80::/10", "fc00::/7",
))


def is_paid(base: str) -> bool:
    """Treat public endpoints conservatively as billed; never resolve hostnames."""
    try:
        host = urlsplit(base if "//" in base else "//" + base).hostname or ""
        if host == "localhost":
            return False
        address = ipaddress.ip_address(host)
    except ValueError:  # malformed endpoints and other hostnames are not free
        return True
    return not any(address in network for network in _PRIVATE_NETWORKS
                   if address.version == network.version)


def presence_of(env: dict) -> str:
    """`always` (default) or `intermittent`, from `ONE_PROVIDER_PRESENCE`.

    WHY THE MAP NEEDED THIS. Most lanes are servers: a vendor API, or a machine
    that is up because it is a machine. Presence was therefore never a
    question, and `lanes()` correctly never asked one - it reads a directory
    and dials nothing.

    An intermittent endpoint breaks that assumption. A laptop or other
    sometimes-on machine is reachable while it is awake and gone the moment it is not, so its drop-in
    describes an endpoint that is absent most of the time. Listed like a server
    it is worse than missing: the first caller to take the cheapest lane gets a
    connection refused instead of an answer.

    So a drop-in may say which kind of thing it points at. An unknown value
    reads as `always`, matching this file's existing rule that a typo is not a
    reason to change what a working endpoint means.
    """
    raw = (env.get("ONE_PROVIDER_PRESENCE") or "").strip().lower()
    return raw if raw in PRESENCE_VALUES else "always"


# base -> (verdict, expires_at, probed_at). probed_at lets a caller that is
# about to COMMIT to a lane (`probe_present(max_age=)`) refuse a verdict that
# is still inside its TTL but older than the decision can tolerate.
_PRESENCE_CACHE: dict[str, tuple[bool, float, float]] = {}
PRESENCE_CACHE_TTL = 30.0
# A dead lane cached for 30s means every picker render inside that window
# re-dials a dead host and eats the full timeout again - the exact stall the
# cache exists to kill. A lane that is DOWN is the one that needs the long
# memory: machines go away for maintenance, not for half a minute. A lane that
# is UP must re-probe quickly, or an endpoint that came back stays invisible.
PRESENCE_DOWN_TTL = 300.0


def _http_json(url: str, timeout: float, headers: dict | None = None):
    """GET a url, parsed JSON or None. Never raises, never follows an error.

    Returns ``(parsed, failure)``: exactly one side is None. ``failure`` is
    ``"conn"`` when the request never got an HTTP answer at all - DNS failure,
    connection refused, timed out - which is the caller's cue that a second
    request to the same host would just pay the same wait again.
    """
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status // 100 != 2:
                return None, None
            return json.loads(resp.read().decode("utf-8", "replace")), None
    except urllib.error.HTTPError:
        # An HTTP answer (404, 403, 500) is the server speaking: the host is
        # up and the next request deserves its own chance to be answered.
        return None, None
    except (urllib.error.URLError, OSError, ValueError, TimeoutError,
            http.client.HTTPException):
        # HTTPException: something answered, but not in HTTP (BadStatusLine,
        # IncompleteRead) - as unreachable as no answer at all.
        return None, "conn"


# base -> the model ids that endpoint listed at its last presence probe, or
# None when it publishes no listing we can read. Written beside every
# _PRESENCE_CACHE verdict, so the two always describe the same probe.
_LISTING_CACHE: dict[str, frozenset | None] = {}

# Per-model states (`model_presence`). Presence above is per base URL: one
# server, one verdict. A lane is a (base, model) pair, and a server that is up
# but does not serve the lane's model answers every request with a 404 or, for
# a single-model llama-server, with a different model than the one named.
MODEL_PRESENT = "present"
MODEL_MISSING = "model-missing"
MODEL_OFFLINE = "offline"
MODEL_UNKNOWN = "unknown"


def _probe_dead(base: str, now: float) -> None:
    _PRESENCE_CACHE[base] = (False, now + PRESENCE_DOWN_TTL, now)
    _LISTING_CACHE.pop(base, None)


def _listed_ids(payload) -> frozenset | None:
    """Model ids from an OpenAI- or Anthropic-shaped `/v1/models` body."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    ids = frozenset(str(m["id"]) for m in payload["data"]
                    if isinstance(m, dict) and m.get("id"))
    return ids or None


def _model_listed(model: str, ids: frozenset) -> bool:
    """Exact id, ollama's implicit `:latest`, or a gguf path's file name."""
    def forms(name):
        name = name.strip()
        out = {name, name.rsplit("/", 1)[-1]}
        out |= {n[: -len(":latest")] for n in out if n.endswith(":latest")}
        return out
    want = forms(model)
    return any(want & forms(i) for i in ids)


def _auth_headers(lane: dict) -> dict:
    key = lane.get("key")
    if not key:
        return {}
    if lane.get("wire") == "openai":
        return {"Authorization": f"Bearer {key}"}
    return {"x-api-key": key, "anthropic-version": "2023-06-01"}


def probe_present(lane: dict, timeout: float = PRESENCE_TIMEOUT, *,
                  max_age: float | None = None) -> bool:
    """Is this endpoint answering right now, and does it vouch for itself?

    Two questions, because answering is not the same as being usable, and the
    second one is the whole reason `one-ios-node` ships a self-check. A node
    whose GPU backend decodes garbage returns HTTP 200 for every request it is
    ever sent - the wire is healthy and the arithmetic is not - so an endpoint
    that publishes a verdict about itself and says `unhealthy` is refused here
    even though it is plainly reachable.

    An endpoint that publishes NO verdict (a 404 on `/health`, which is what a
    plain ollama server answers) is taken at its word. Silence is not a confession, and
    demanding a health route would delete every endpoint that predates the idea.

    Cost: a lane that is down and STAYS down is remembered for
    PRESENCE_DOWN_TTL, not PRESENCE_CACHE_TTL, so a picker render inside that
    window pays zero network. And when `/health` fails at the connection level
    the `/models` leg is skipped - the endpoint did not answer the first
    knock; a second knock to the same dead address is the same wait for the
    same nothing. Each leg can cost the full timeout, and two legs double it
    per dead lane per cache miss.

    `max_age` is for the moment of SELECTION, not of rendering. A down verdict
    is kept for PRESENCE_DOWN_TTL so a picker does not re-dial a dead machine on
    every draw, but a person choosing that lane after the machine came back
    must not be refused for five minutes on a stale answer. A verdict older than
    `max_age` seconds is re-probed (and the fresh answer cached as usual).
    """
    base = lane["base"].rstrip("/")
    now = time.time()
    cached = _PRESENCE_CACHE.get(base)
    if (cached is not None and now < cached[1]
            and (max_age is None or now - cached[2] <= max_age)):
        return cached[0]

    root = base[: -len("/v1")] if base.endswith("/v1") else base

    verdict, conn_failed = _http_json(f"{root}/health", timeout)
    if conn_failed:
        _probe_dead(base, now)
        return False
    if isinstance(verdict, dict) and str(verdict.get("health", "")) == "unhealthy":
        _probe_dead(base, now)
        return False

    if lane["wire"] != "openai":
        # The Anthropic wire has no listing route, so a 404 here is not
        # absence: the server answered /health over HTTP, and that is the
        # whole question. A listing at /v1/models, where the server has one,
        # only feeds `model_presence`; its absence changes nothing.
        listing, _ = _http_json(f"{root}/v1/models", timeout,
                                _auth_headers(lane))
        _LISTING_CACHE[base] = _listed_ids(listing)
        _PRESENCE_CACHE[base] = (True, now + PRESENCE_CACHE_TTL, now)
        return True

    listing, conn_failed = _http_json(f"{base}/models", timeout,
                                      _auth_headers(lane))
    if conn_failed:
        _probe_dead(base, now)
        return False
    res = listing is not None
    _LISTING_CACHE[base] = _listed_ids(listing)
    _PRESENCE_CACHE[base] = (res, now + PRESENCE_CACHE_TTL, now)
    return res


def model_presence(lane: dict, timeout: float = PRESENCE_TIMEOUT, *,
                   max_age: float | None = None, dial: bool = True) -> str:
    """Presence of the lane's MODEL, not just its server.

    `offline`: the server is not answering (``probe_present``). `model-missing`:
    it answers and publishes a model listing, and the lane's model is not in
    it. `present`: the model is listed, or the server answers and publishes no
    listing we can read (an Anthropic-wire proxy that 401s `/v1/models`, a
    server with no listing route) - the model cannot be checked there, and an
    unreadable listing is not evidence the model is gone. `unknown`: no verdict
    yet; only returned with ``dial=False``, which reads the cache like
    ``presence_cached`` and never dials.
    """
    base = lane["base"].rstrip("/")
    up: bool | None
    if dial:
        up = probe_present(lane, timeout, max_age=max_age)
    else:
        up = presence_cached(base)
        if up is None:
            return MODEL_UNKNOWN
    if not up:
        return MODEL_OFFLINE
    ids = _LISTING_CACHE.get(base)
    if ids is None or not lane.get("model"):
        return MODEL_PRESENT
    return MODEL_PRESENT if _model_listed(lane["model"], ids) else MODEL_MISSING


_priming: set[str] = set()
_priming_threads: list[threading.Thread] = []
_priming_lock = threading.Lock()


def presence_cached(base: str) -> bool | None:
    """The last presence verdict for `base` if it is still fresh. Never dials.

    This is what a LIST BUILD reads. ``probe_present`` blocks for up to
    PRESENCE_TIMEOUT per dead endpoint, and the caller that asks for a lane
    list is usually an event loop with a keyboard to draw - so the list is
    built from what is already known and ``prime_presence`` fills the cache
    behind it. None means "no fresh verdict", which is not evidence of
    absence.
    """
    row = _PRESENCE_CACHE.get(base.rstrip("/"))
    if row is None or time.time() >= row[1]:
        return None
    return row[0]


def prime_presence(found: list[dict],
                   timeout: float = PRESENCE_TIMEOUT) -> None:
    """Fill _PRESENCE_CACHE for these lanes on a background thread. Returns at once.

    One batch at a time per endpoint: spawning a probe per dead lane would hand
    a stalled address another thread every time somebody opened the picker.
    Inside the batch the probes fan out, because the wait is the network and
    not the work; serial probes make the next list build staler than it needs
    to be, and that staleness is the only thing this design trades away.

    The health and presence halves follow one rule:
    a render reads state, it does not go and get it.
    """
    todo = []
    with _priming_lock:
        for lane in found:
            base = (lane.get("base") or "").rstrip("/")
            if not base or base in _priming or presence_cached(base) is not None:
                continue
            _priming.add(base)
            todo.append(lane)
        if not todo:
            return

        def _probe(lane):
            try:
                probe_present(lane, timeout)
            except Exception:
                pass

        def _work():
            try:
                with ThreadPoolExecutor(
                        max_workers=min(8, len(todo)),
                        thread_name_prefix="presence-probe") as pool:
                    list(pool.map(_probe, todo))
            finally:
                with _priming_lock:
                    _priming.difference_update(
                        (lane.get("base") or "").rstrip("/") for lane in todo)

        t = threading.Thread(target=_work, daemon=True,
                             name="dropins-presence-prime")
        _priming_threads.append(t)
    t.start()


def await_presence_priming(timeout: float | None = None) -> None:
    """Join any in-flight presence priming. For tests and one-shot CLI callers.

    A long-lived process never calls this: the point of priming is that the
    render does not wait. A script that wants the truth in ONE pass calls
    `lanes(require_present=True)`, then this, then `lanes(...)` again - which
    is exactly the two-render sequence a picker goes through, made explicit.
    """
    with _priming_lock:
        threads = list(_priming_threads)
        _priming_threads.clear()
    for t in threads:
        t.join(timeout)


# A role drop-in names a relay and a job, never a model. The relay's own
# /v1/models says what it serves, so swapping the model behind the relay is one
# change on the relay instead of a hand edit on every machine. These keys are
# what the relay answers; a role drop-in that also writes one is refused, not
# merged, because a hardcoded copy is exactly what goes stale.
RELAY_SERVED_KEYS = ("ONE_PROVIDER_MODEL", "ANTHROPIC_MODEL",
                     "ONE_PROVIDER_LABEL", "ONE_PROVIDER_CONTEXT_WINDOW",
                     "CONTEXT_WINDOW", "CLAUDE_CODE_MAX_CONTEXT_TOKENS")
RELAY_TTL = 300.0
RELAY_DOWN_TTL = 60.0
RELAY_TIMEOUT = 3.0
_RELAY_CACHE: dict[str, tuple[float, dict | None, str]] = {}


def relay_served(base: str, timeout: float = RELAY_TIMEOUT) -> dict | None:
    """What a relay serves: {"model", "context_window", "label"}, or None.

    Cached RELAY_TTL on success. A failure is cached RELAY_DOWN_TTL and logged
    as an ERROR once per window; it is never answered from an older success,
    because a stale model id is the failure this exists to remove.
    """
    base = base.rstrip("/")
    now = time.time()
    hit = _RELAY_CACHE.get(base)
    if hit and hit[0] > now:
        return hit[1]
    data, failure = _http_json(base + "/models", timeout)
    served, why = None, ""
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        why = "unreachable" if failure else "no /models list"
    elif (len(models) != 1 or not isinstance(models[0], dict)
          or not isinstance(models[0].get("id"), str) or not models[0]["id"]):
        why = f"serves {len(models)} models, a role needs exactly one"
    else:
        m = models[0]
        raw_meta = m.get("meta")
        meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
        ctx = meta.get("n_ctx") or m.get("context_window") or m.get("max_model_len")
        try:
            window = int(ctx) if ctx else None
        except (TypeError, ValueError, OverflowError):
            window = None  # a malformed window is absent, not a crash
        if window is not None and window <= 0:
            window = None
        ftype = meta.get("ftype") if isinstance(meta.get("ftype"), str) else None
        host = urlsplit(base).netloc
        served = {"model": m["id"],
                  "context_window": window,
                  "label": " ".join(x for x in (m["id"], ftype)
                                    if x) + f" (relay {host})"}
    if served is None:
        logger.error("relay %s/models %s: role lanes on it are OFF until it "
                     "answers - fix the relay, do not pin a model", base, why)
    _RELAY_CACHE[base] = (now + (RELAY_TTL if served else RELAY_DOWN_TTL),
                          served, why)
    return served


# The bearer variables each wire reads, in precedence order. A key belongs to
# its vendor: a source file holding both vendors' keys must never send one
# vendor's key to the other's host.
KEY_VARS = {"openai": ("ONE_PROVIDER_KEY", "OPENAI_API_KEY"),
            "anthropic": ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
                          "ONE_PROVIDER_KEY")}


def key_for(env: dict, wire: str) -> str:
    """The bearer a drop-in sends on ``wire`` ("openai"|"anthropic"), or "".

    ``ONE_PROVIDER_KEY_ENV`` names the variable first, read from the drop-in
    and its source file, else from the process environment. Otherwise the
    wire's own variables are tried in KEY_VARS order.
    """
    key_var = env.get("ONE_PROVIDER_KEY_ENV")
    if key_var:
        named = env.get(key_var) or os.environ.get(key_var, "")
        if named:
            return named
    for name in KEY_VARS[wire]:
        if env.get(name):
            return env[name]
    return ""


def read(path) -> dict | None:
    """One drop-in, or None if it names no usable endpoint.

    Returns: name, base, model, key, params, headers, wire
    ("openai"|"anthropic"), paid, presence ("always"|"intermittent"), label,
    source (the file it came from).
    """
    env = parse_env_file(path)
    src = env.get("ONE_PROVIDER_SOURCE")
    if src:
        # The secret lives in a mode-600 file outside the provider dir and the
        # drop-in only names it. Values already set in the drop-in win.
        base_env = parse_env_file(os.path.expanduser(src))
        base_env.update(env)
        env = base_env

    disabled = (env.get("ONE_PROVIDER_DISABLED") or "").strip()
    if disabled:
        logger.info("ignoring disabled provider %s: %s", Path(path).name,
                    disabled)
        return None

    model = env.get("ONE_PROVIDER_MODEL") or env.get("ANTHROPIC_MODEL")
    openai_url = env.get("OPENAI_BASE_URL") or env.get("ONE_PROVIDER_BASE_URL")
    anthropic_url = env.get("ANTHROPIC_BASE_URL")
    # An endpoint declaring both wires is reported as OpenAI: it is the wire
    # every caller here can speak, and the Anthropic one is the specialisation.
    url, wire = ((openai_url, "openai") if openai_url
                 else (anthropic_url, "anthropic"))
    role = (env.get("ONE_PROVIDER_ROLE") or "").strip()
    served = None
    if role:
        pinned = [k for k in RELAY_SERVED_KEYS if env.get(k)]
        if pinned:
            logger.error("refusing %s: role %r is relay-served, remove %s",
                         Path(path).name, role, ", ".join(pinned))
            return None
        if not url:
            return None
        served = relay_served(url)
        if served is None:
            return None
        model = served["model"]
    if not model or not url:
        return None

    key = key_for(env, wire)

    refusal = env_refusal(Path(path).stem, env, url)
    if refusal:
        logger.warning("ignoring drop-in %s: %s", Path(path).name, refusal)
        return None

    # `ONE_PROVIDER_PARAMS` is one-cli's existing key for "extra fields this
    # endpoint needs in the request body", so every reader honours it rather
    # than inventing a second spelling of the same fact. It carries an
    # endpoint's non-negotiable settings: Kimi refuses any temperature but 1
    # ("only 1 is allowed for this model"), which is a 400 on every call and
    # reads exactly like a bad key.
    params: dict = {}
    raw = env.get("ONE_PROVIDER_PARAMS")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                params = parsed
        except ValueError:
            params = {}       # a typo is not a reason to drop a working endpoint

    # `ONE_PROVIDER_HEADERS` is the same fact one rung out: extra HTTP headers
    # this endpoint insists on. OpenCode Go refuses a raw completion with 400
    # MissingSessionID unless `x-opencode-session` rides along, so a reader that
    # drops this key hands every caller a lane that cannot complete - and a
    # health probe that drops it cannot answer "is this lane usable".
    headers: dict = {}
    raw = env.get("ONE_PROVIDER_HEADERS")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                headers = {str(k): str(v) for k, v in parsed.items()}
        except ValueError:
            headers = {}

    base = url.rstrip("/")
    ctx_raw = env.get("ONE_PROVIDER_CONTEXT_WINDOW") or env.get("CONTEXT_WINDOW")
    cw: int | None = None
    if ctx_raw:
        try:
            cw = int(ctx_raw)
        except ValueError:
            cw = None
    if served:
        cw = served["context_window"]
    return {"name": Path(path).stem, "base": base, "model": model, "key": key,
            "params": params, "headers": headers,
            "wire": wire, "paid": is_paid(base),
            "presence": presence_of(env),
            "label": (served or {}).get("label") or env.get("ONE_PROVIDER_LABEL", ""),
            "source": str(path), "context_window": cw, "role": role}


def lanes(dirs=None, *, wire: str | None = None,
          paid: bool | None = None, require_key: bool = True,
          require_present: bool = False,
          presence_timeout: float = PRESENCE_TIMEOUT) -> list[dict]:
    """Every usable drop-in on this host, deduped and in a stable order.

    Deduped on (base, model): the same endpoint reached under two filenames is
    one lane, and listing it twice makes a fallback ladder look deeper than it
    is. First file wins, so a dir named earlier in `ONE_PROVIDERS_DIR` is the
    one whose label and params apply.

    `wire` filters to one request shape - a caller with one OpenAI client must
    not be handed an Anthropic-wire endpoint, which is how a router nominates a
    lane that cannot answer. `require_key` drops a PAID lane with no bearer: it
    is a 401, not a lane. A free lane needs no key and never fails this test.

    `require_present` drops an INTERMITTENT lane that is not answering right
    now. Only lanes that declared themselves intermittent are dialled, so a
    caller that opts in pays one probe per such lane and nothing at all for the
    servers - a host with no intermittent drop-in reaches the network zero
    times, exactly as before. Default off: a lane list is a map, and a map
    should not require the territory to be up.

    Order: free first (they cost nothing, so they are always the cheaper try),
    then paid, each group in filename order. A caller that wants a different
    order among the paid lanes applies it to this list; the reader has no
    opinion about which plan to spend.
    """
    found: dict[str, dict] = {}
    for directory in (dirs if dirs is not None else provider_dirs()):
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.env")):
            lane = read(path)
            if not lane:
                continue
            if wire and lane["wire"] != wire:
                continue
            if paid is not None and lane["paid"] is not paid:
                continue
            if require_key and lane["paid"] and not lane["key"]:
                continue
            found.setdefault(f"{lane['base']}|{lane['model']}", lane)
    out = list(found.values())
    if require_present:
        out = _drop_absent(out, presence_timeout)
    return ([lane for lane in out if not lane["paid"]]
            + [lane for lane in out if lane["paid"]])


def _drop_absent(found: list[dict], timeout: float) -> list[dict]:
    """Keep every always-on lane; drop an intermittent one KNOWN to be absent.

    Cache-only, and priming runs behind it. This used to dial here, in
    parallel, and block until the slowest probe finished - which on a lane
    pointing at an offline address is the full PRESENCE_TIMEOUT, every time the
    cache expires. The model picker builds its list inline, so one offline
    drop-in could stall the whole picker on every render and every tap, even
    when no intermittent lane was involved. Probing only from the cache keeps
    the picker responsive.

    So an unknown verdict keeps the lane and starts a probe; the next build
    inside PRESENCE_DOWN_TTL has the truth and drops it. Nothing is lost by
    listing a sleeping lane for one render: `_apply_model_change` dials its own
    liveness preflight off-thread and refuses the tap with "the machine is
    offline", so the guarantee that you cannot SWITCH onto a dead lane lives in
    the gate, where the rest of the picker's guarantees already live.
    """
    flaky = [lane for lane in found if lane["presence"] == "intermittent"]
    if not flaky:
        return found
    keep = {id(lane): presence_cached(lane["base"]) is not False for lane in flaky}
    prime_presence(flaky, timeout)
    return [lane for lane in found
            if lane["presence"] != "intermittent" or keep[id(lane)]]


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import sys
    want = sys.argv[1] if len(sys.argv) > 1 else None
    for lane in lanes(wire=want):
        print(f"{lane['name']:<26} {'paid' if lane['paid'] else 'free':<5} "
              f"{lane['presence']:<12} "
              f"{lane['wire']:<9} {lane['model']:<32} {lane['base']}")



"""Backend resolution.

A backend is a place we can send a chat turn to. They come from three sources,
in the order one-cli looks:

1. explicit flags (`--base-url`, `--model`) - always wins
2. the provider drop-in dir `~/.one-providers/*.env` - hot, no restart, already
   the convention for "a model endpoint you pay for or stand up"
3. a live llama.cpp server on this host

The explicit opencode/provider/model selector delegates vendor configuration to
its optional SDK transport; raw-wire discovery does not read vendor CLI config.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import ipaddress
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PROVIDER_DIR = Path.home() / ".one-providers"
# Extra drop-in dirs, colon-separated, searched after the shared one. A bench or
# a throwaway endpoint belongs here rather than in the shared dir.
PROVIDER_DIR_ENV = "ONE_PROVIDERS_DIR"
# llama-server's own default port. One server holds ONE model for the life of
# the process, so an installation that serves several runs several servers on
# several ports - which is why the host list below is a list of base urls.
LLAMACPP_DEFAULT = "http://127.0.0.1:8080"
# `ONE_LLAMACPP_HOSTS=http://gpu-box:8911,...` reaches another node's
# llama-server. Which model each one holds is only knowable by asking it, so
# the endpoint list is config and the model list never is.
LLAMACPP_HOSTS_ENV = "ONE_LLAMACPP_HOSTS"

# Which wire a backend speaks. Not cosmetic: the request body, the tool-call
# shape and the stop reason all differ between the two.
WIRE_OPENAI = "openai"
WIRE_ANTHROPIC = "anthropic"


@dataclass
class Backend:
    name: str
    model: str
    base_url: str
    wire: str = WIRE_OPENAI
    api_key: str = ""
    label: str = ""
    profile: str = "default"
    source: str = ""
    extra: dict = field(default_factory=dict)
    # Successors this endpoint declares for itself (ONE_PROVIDER_FALLBACK).
    # Its own field rather than a key in `extra`, because `extra` is sent to
    # the endpoint as request body and a chain is ours, not the wire's.
    fallback: list[str] = field(default_factory=list)
    # Request HEADERS this endpoint insists on (ONE_PROVIDER_HEADERS), the
    # header-side twin of `extra`. Separate field for the same reason
    # `fallback` is: `extra` becomes the request body, and a header is not a
    # body field. OpenCode Go is why this exists - its go/v1 completions
    # endpoint answers 400 MissingSessionID to every request without an
    # `x-opencode-session`, so without a header key its 37 models cannot be
    # reached by a drop-in at all.
    headers: dict = field(default_factory=dict)
    context_window: int | None = None
    # Local metadata, never sent as provider request parameters. The binding
    # prevents --model or an endpoint override inheriting another model's senses.
    capability_binding: tuple[str, str] | None = None
    declared_capabilities: list[str] | None = None

    @property
    def display(self) -> str:
        return self.label or f"{self.name} ({self.model})"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Read a KEY=VALUE drop-in. Quotes stripped, `export ` tolerated."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        import warnings
        warnings.warn(f"Skipping unreadable provider file: {path}", stacklevel=2)
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        key, sep, val = line.partition("=")
        if not sep:
            continue
        val = val.strip().strip("'").strip('"')
        out[key.strip()] = val
    return out


def _expand_source(env: dict[str, str]) -> dict[str, str]:
    """`ONE_PROVIDER_SOURCE=~/.x.env` pulls a second file in underneath.

    The drop-in convention keeps secrets in a mode-600 file outside the
    provider dir; the drop-in only names it. Values already set win, so the
    drop-in can override what the sourced file says.
    """
    src = env.get("ONE_PROVIDER_SOURCE")
    if not src:
        return env
    base = _parse_env_file(Path(os.path.expanduser(src)))
    base.update(env)
    return base


def _subscription_policy():
    """The drop-in reader, which applies the subscription policy file (if any)."""
    from onecli._vendor import dropins
    return dropins


def validate_model(name: str, base: str, model: str):
    why = _subscription_policy().model_refusal(name, base, model)
    if why:
        raise RuntimeError(f'subscription policy refuses {name}: {why}')


def validate_backend(backend: Backend) -> Backend:
    validate_model(backend.name, backend.base_url, backend.model)
    if 'model' in backend.extra:
        validate_model(backend.name, backend.base_url, backend.extra['model'])
    return backend


def _backend_from_env(path: Path) -> Backend | None:
    env = _expand_source(_parse_env_file(path))
    if env.get("ONE_PROVIDER_ROLE"):
        # A relay-served role names no model: the relay's /v1/models does, read
        # by the drop-in reader, which also refuses a role file that pins one.
        lane = _subscription_policy().read(path)
        if not lane:
            return None
        env.update({"ONE_PROVIDER_MODEL": lane["model"],
                    "ONE_PROVIDER_LABEL": lane["label"],
                    "ONE_PROVIDER_CONTEXT_WINDOW": str(lane["context_window"] or "")})
    model = env.get("ONE_PROVIDER_MODEL") or env.get("ANTHROPIC_MODEL")
    if not model:
        return None

    anth_url = env.get("ANTHROPIC_BASE_URL")
    oai_url = env.get("OPENAI_BASE_URL") or env.get("ONE_PROVIDER_BASE_URL")
    if oai_url:
        base_url, wire = oai_url, WIRE_OPENAI
    elif anth_url:
        base_url, wire = anth_url, WIRE_ANTHROPIC
    else:
        return None

    # `ONE_PROVIDER_KEY_ENV=SOME_VAR` names which variable carries the bearer,
    # so a provider whose own env var is idiosyncratic (OLLAMA_CLOUD_API_KEY,
    # DASHSCOPE_API_KEY) needs no special case; otherwise the key is the
    # wire's own. One function, shared with the drop-in reader.
    key = _subscription_policy().key_for(env, wire)
    refusal = _subscription_policy().env_refusal(path.stem, env, base_url)
    if refusal:
        import warnings
        warnings.warn(f"Skipping provider drop-in {path.name}: {refusal}", stacklevel=2)
        return None

    extra: dict = {}
    raw_params = env.get("ONE_PROVIDER_PARAMS")
    if raw_params:
        try:
            parsed = json.loads(raw_params)
            if isinstance(parsed, dict):
                extra = parsed
        except ValueError:
            # A malformed params blob is a config typo, not a reason to drop the
            # whole backend: the endpoint still works on defaults.
            extra = {}
    # `ONE_PROVIDER_MODALITIES=vision,audio` is the drop-in's own claim about
    # the MODEL, which is the only thing that may nominate it as a describer
    # for another backend's image. The wire cannot make that claim.
    modalities = [m.strip().lower()
                  for m in (env.get("ONE_PROVIDER_MODALITIES") or "").split(",")
                  if m.strip()]
    # `ONE_PROVIDER_FALLBACK=other,third` lets an endpoint declare its own
    # successors, so the ladder lives next to the thing it describes and
    # reloads hot with it instead of being retyped on every command line.
    declared = [n.strip()
                for n in (env.get("ONE_PROVIDER_FALLBACK") or "").split(",")
                if n.strip()]
    # `ONE_PROVIDER_HEADERS='{"x-opencode-session":"..."}'` - extra request
    # headers. Same malformed-blob policy as params: a typo loses the headers,
    # never the backend.
    headers: dict = {}
    raw_headers = env.get("ONE_PROVIDER_HEADERS")
    if raw_headers:
        try:
            parsed = json.loads(raw_headers)
            if isinstance(parsed, dict):
                headers = {str(k): str(v) for k, v in parsed.items()}
        except ValueError:
            headers = {}
    raw_cw = env.get("ONE_PROVIDER_CONTEXT_WINDOW") or env.get("CONTEXT_WINDOW")
    cw: int | None = None
    if raw_cw:
        try:
            cw = int(raw_cw)
        except ValueError:
            cw = None
    return Backend(
        name=path.stem,
        model=model,
        base_url=base_url.rstrip("/"),
        wire=wire,
        api_key=key,
        label=env.get("ONE_PROVIDER_LABEL", ""),
        profile=env.get("ONE_PROVIDER_PROFILE", "default"),
        source=str(path),
        extra=extra,
        fallback=declared,
        headers=headers,
        context_window=cw,
        capability_binding=(base_url.rstrip("/"), model) if modalities else None,
        declared_capabilities=modalities or None,
    )


def _http_json(url: str, timeout: float = 4.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError,
            http.client.HTTPException):
        # HTTPException: a listener that does not speak HTTP is no backend.
        return None


def llamacpp_backends(base_url: str = LLAMACPP_DEFAULT,
                      prefix: str = "llamacpp") -> list[Backend]:
    """Every model a llama-server is serving right now.

    llama-server speaks the OpenAI list shape (`/v1/models` -> `data[]`), and
    since b6000-odd it also carries ollama's `models[]` mirror with a
    `capabilities` list. We read `data[]` - the id there is what the chat
    endpoint accepts, alias included.
    """
    data = _http_json(f"{base_url.rstrip('/')}/v1/models")
    if not isinstance(data, dict):
        return []
    entries = data.get("data") or []
    if not isinstance(entries, list):
        return []
    out = []
    for m in entries:
        name = m.get("id") if isinstance(m, dict) else None
        if not isinstance(name, str) or not name:
            continue
        out.append(
            Backend(
                name=f"{prefix}/{name}",
                model=name,
                base_url=f"{base_url.rstrip('/')}/v1",
                wire=WIRE_OPENAI,
                label=f"{name} ({prefix})",
                source=base_url,
            )
        )
    return out


def provider_dirs() -> list[Path]:
    dirs = [PROVIDER_DIR]
    for raw in os.environ.get(PROVIDER_DIR_ENV, "").split(":"):
        raw = raw.strip()
        if raw:
            dirs.append(Path(os.path.expanduser(raw)))
    return dirs


def discover(include_local: bool = True) -> list[Backend]:
    """Every backend we can reach right now, drop-ins first."""
    out: list[Backend] = []
    seen: set[str] = set()
    for directory in provider_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.env")):
            b = _backend_from_env(path)
            if b and b.name not in seen:
                seen.add(b.name)
                out.append(b)
    if include_local:
        for b in llamacpp_backends():
            if b.name not in seen:
                seen.add(b.name)
                out.append(b)
        for raw in os.environ.get(LLAMACPP_HOSTS_ENV, "").split(","):
            raw = raw.strip().rstrip("/")
            if not raw:
                continue
            host = raw.split("//")[-1].split(":")[0]
            for b in llamacpp_backends(raw, prefix=f"llamacpp@{host}"):
                if b.name not in seen:
                    seen.add(b.name)
                    out.append(b)
    return out


# Names that denote a model tier rather than a backend on this host. The
# substring fallback in `resolve()` never matches them: otherwise a local model
# whose id contains `opus` would answer `one -m opus`. An exact backend name
# still wins - this guards the fuzzy path only.
RESERVED_TIER_NAMES = frozenset({"opus", "sonnet", "haiku"})

# Preferred backends and models when no backend is explicitly requested, or
# when resolving generic model families (e.g. `deepseek`, `flash`).
DEFAULT_PREFERRED_MODELS = (
    "deepseek-v4.1-flash",
    "deepseek-v4-flash",
)
# Empty on purpose: preference among discovered backends comes from the
# order discovery returns and from ONE_DEFAULT_BACKEND, never from a list of
# backends that happen to exist on the machine this was written on.
DEFAULT_PREFERRED_BACKENDS: tuple[str, ...] = ()


class NoBackend(RuntimeError):
    """Nothing is configured at all - the first-run case, not a bad name."""


def _resolve(
    name: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    wire: str | None = None,
    include_local: bool = True,
) -> Backend:
    """Pick one backend. Explicit flags win; then a name match; then the first
    drop-in; then a local llama-server. Raises rather than silently falling back to a
    different model than the caller asked for."""
    selection = name or model or ""
    if selection.startswith("opencode/"):
        ref = selection.removeprefix("opencode/")
        if "/" not in ref or not all(ref.split("/", 1)):
            raise RuntimeError("OpenCode selection must be opencode/provider/model")
        if base_url or (name and model):
            raise RuntimeError("OpenCode selection cannot override its catalog model or endpoint")
        if wire and wire != "opencode":
            raise RuntimeError("OpenCode selection cannot override its transport")
        return Backend(name=selection, model=ref, base_url="", wire="opencode", source="flag")

    if base_url:
        # With an explicit URL there is no backend for `-m` to name, so it is
        # the model id. Dropping it sent an empty id: 400 "Unsupported model ".
        model = model or name
        return Backend(
            name=name or "explicit",
            model=model or "",
            base_url=base_url.rstrip("/"),
            wire=wire or WIRE_OPENAI,
            api_key=os.environ.get("ONE_CLI_API_KEY", ""),
            source="flag",
        )

    found = discover(include_local=include_local)
    if not found:
        raise NoBackend(
            "no backend found: no usable drop-in in ~/.one-providers and no "
            "llama-server on 127.0.0.1:8080. Run `one` in a terminal for guided "
            "setup, or pass --base-url and --model-id."
        )

    # Prefer the primary provider for shared model IDs, while exact backend
    # names remain explicit selections.
    rank = {n: i for i, n in enumerate(DEFAULT_PREFERRED_BACKENDS)}
    found.sort(key=lambda b: rank.get(b.name, len(rank)))
    if name and name.lower() in ("deepseek", "deepseek-flash", "flash"):
        preferred = next((b for b in found if b.name in rank), None)
        if preferred is not None:
            name = preferred.name

    if name:
        for b in found:
            if b.name == name or b.model == name:
                if model:
                    b.model = model
                return b
        # A reserved tier name is not shorthand: the substring fallback below
        # would run the job on an unrelated model whose id contains the word.
        if name.lower() in RESERVED_TIER_NAMES:
            raise RuntimeError(
                f"{name!r} names an escalation tier, not a backend, and no "
                f"backend here is called {name!r}. A substring match would land "
                f"on a different model - see RESERVED_TIER_NAMES in "
                f"onecli/config.py. Available: "
                + ", ".join(b.name for b in found)
            )
        # A substring is shorthand, not a guess: `-m qwen` for the one qwen
        # endpoint is worth having. It stops being shorthand the moment it
        # matches more than one: picking the first silently would answer from
        # an unrelated backend. With paid endpoints in the drop-in dir that is
        # a bill, not a typo. So: unique match wins, several is a question,
        # none is an error.
        near = [b for b in found
                if name.lower() in b.name.lower()
                or name.lower() in b.model.lower()]
        if len(near) == 1:
            b = near[0]
            if model:
                b.model = model
            return b
        if near:
            if name.lower() in ("deepseek", "deepseek-flash", "flash"):
                for b_pref in DEFAULT_PREFERRED_BACKENDS:
                    for b in near:
                        if b.name == b_pref:
                            if model:
                                b.model = model
                            return b
                for pref in DEFAULT_PREFERRED_MODELS:
                    for b in near:
                        if b.model == pref:
                            if model:
                                b.model = model
                            return b
            raise RuntimeError(
                f"{name!r} matches {len(near)} backends: "
                + ", ".join(b.name for b in near)
                + ". Name one of them exactly."
            )
        raise RuntimeError(
            f"no backend matching {name!r}. Available: "
            + ", ".join(b.name for b in found)
        )

    if model:
        for b in found:
            if b.model == model:
                return b
        b = found[0]
        b.model = model
        return b

    env_pref = (os.environ.get("ONE_DEFAULT_BACKEND") or os.environ.get("ONE_DEFAULT_MODEL") or "").strip()
    if env_pref:
        for b in found:
            if b.name == env_pref or b.model == env_pref:
                return b
    for b_pref in DEFAULT_PREFERRED_BACKENDS:
        for b in found:
            if b.name == b_pref:
                return b
    for pref in DEFAULT_PREFERRED_MODELS:
        for b in found:
            if b.model == pref:
                return b

    return _subscription_policy().prefer(found, lambda b: b.model)[0]


def resolve(name: str | None = None, model: str | None = None,
            base_url: str | None = None, wire: str | None = None,
            include_local: bool = True) -> Backend:
    """Resolve explicit or automatic selection, then validate all overrides."""
    return validate_backend(_resolve(name, model, base_url, wire, include_local))


# --- context windows and price ------------------------------------------------
#
# Both tables are matched by substring against the model id, longest key first,
# so `qwen3.5:27b` and `qwen3.5:9b` can share one entry and a specific id can
# still override a family.

# MEASURED, not guessed: local entries read from the server's own model meta on
# a local server, kimi from its own `/v1/models` (`context_length`). Anything
# not listed falls back to DEFAULT_CONTEXT_WINDOW, and `--context-window`
# overrides every one of these.
CONTEXT_WINDOWS: dict[str, int] = {
    "kimi-for-coding": 262144,
    "k3-256k": 262144,
    "k3": 1048576,
    "qwen3-coder": 262144,
    "qwen3.5": 262144,
    "qwen3.6": 262144,
    "gemma4": 262144,
    "gpt-oss": 131072,
    "llama3.1": 131072,
    # deepseek-v4-flash / v4.1: the PROVIDER named the ceiling. DashScope answers an
    # over-length request with "Range of input length should be [1, 1000000]",
    # and Ollama Cloud's /api/show reports deepseek4.context_length 1048576.
    # The smaller, enforced number is the one a run must respect: the same
    # model is served by both and only one of them will refuse.
    "deepseek-v4.1-flash": 1000000,
    "deepseek-v4-flash": 1000000,
    "deepseek": 131072,
    "glm-5.3": 131072,
    "glm-5.2": 131072,
    "glm-5": 131072,
    "glm": 131072,
    "qwen3.8": 262144,
    "qwen3.7": 131072,
    "qwen": 131072,
    "minimax-m3": 200000,
    "minimax-m2.5": 200000,
    "minimax": 200000,
    "kimi-k3": 1048576,
    "kimi": 262144,
    # mistral-large-4: /v1/models on api.mistral.ai reports max_context_length
    # 524288 (MEASURED). Longest match beats the family row below.
    "mistral-large-4": 524288,
    "mistral": 131072,
    # A llama.cpp /v1/models reply reports the model's own window as
    # n_ctx_train. What the process serves right now may be smaller; that value
    # comes from served_context_window and wins over this entry.
    # Some providers publish model ids without a context length. Declare the
    # number only when it has been observed accepted by the endpoint; this is a
    # proven floor, not a ceiling. Under-declaring costs an early compaction;
    # over-declaring costs an unrecoverable 400.
    "gemini-3.8-flash": 1048576,
    "mimo-v2.5-pro": 200000,
    "MiniMax-M3": 200000,
}

# USD per MILLION tokens, (input, output).
#
# Names and network addresses cannot prove billing. Operators declare pricing
# with ONE_CLI_PRICING (see load_pricing), including explicit [0, 0] entries for
# inference servers they know to be unmetered. Missing prices stay unknown.
PRICING: dict[str, tuple[float, float]] = {}

# `ONE_CLI_PRICING=/path/to.json` merges in {"model-substring": [in, out]}, or
# [in, out, cache_read] when the provider bills a cache read at its own rate.
PRICING_ENV = "ONE_CLI_PRICING"

# A cache read's price relative to fresh input when the catalog does not name
# one. Hosted providers bill a hit at roughly 0.1x of fresh input.
DEFAULT_CACHE_READ_WEIGHT = 0.1


def _longest_match(table: dict, model: str):
    model = (model or "").lower()
    for key in sorted(table, key=len, reverse=True):
        if key.lower() in model:
            return table[key]
    return None


def served_context_window(base_url: str, timeout: float = 1.5) -> int | None:
    """The window the SERVER was actually started with, or None.

    The table below is a fact about a model; `n_ctx` is a fact about the
    process serving it, and only the second one can end a run. A llama.cpp
    server holding gpt-oss-20b in a 8192-token slot will accept a request for
    its architectural 131072 and then close the stream mid-run, which reads as
    the model falling silent rather than as the ceiling it is.

    Only plain-http endpoints are probed. A local server is normally http on a
    private network and every hosted provider is https, so this costs a
    hosted run nothing - no request is made at all.
    """
    if not base_url.startswith("http://"):
        return None
    url = base_url.rstrip("/")
    for suffix in ("/v1", "/openai"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    try:
        with urllib.request.urlopen(url + "/props", timeout=timeout) as resp:
            props = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None
    if not isinstance(props, dict):
        return None
    settings = props.get("default_generation_settings")
    if not isinstance(settings, dict):
        return None
    raw_ctx = settings.get("n_ctx")
    try:
        n_ctx = int(raw_ctx or 0)
    except (TypeError, ValueError):
        return None
    return n_ctx if n_ctx > 0 else None


def served_token_count(base_url: str, messages: list[dict],
                       timeout: float = 5.0) -> int | None:
    """What the SERVER counts `messages` as, or None when it cannot say.

    llama.cpp renders the chat template with /apply-template and counts the
    result with /tokenize, so the number includes the template framing a
    character estimate cannot see. Same http-only rule as
    served_context_window(): a hosted provider is never asked.
    """
    if not base_url.startswith("http://"):
        return None
    url = base_url.rstrip("/")
    for suffix in ("/v1", "/openai"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            url + path, data=json.dumps(body, default=str).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    try:
        prompt = post("/apply-template", {"messages": messages}).get("prompt")
        if not isinstance(prompt, str):
            return None
        tokens = post("/tokenize", {"content": prompt}).get("tokens")
    except Exception:
        return None
    return len(tokens) if isinstance(tokens, list) else None


def context_window_for(model: str, backend_name: str = "") -> int:
    """Window for this model, or DEFAULT_CONTEXT_WINDOW when we do not know."""
    from onecli.context import DEFAULT_CONTEXT_WINDOW

    found = _longest_match(CONTEXT_WINDOWS, model)
    if found is None and backend_name:
        found = _longest_match(CONTEXT_WINDOWS, backend_name)
    return int(found) if found else DEFAULT_CONTEXT_WINDOW


def resolve_context_window(model: str, backend_name: str = "",
                           base_url: str = "",
                           declared_window: int | None = None) -> int:
    """The window a run must actually bound itself by.

    Three facts, in order:
    1. what the provider/drop-in explicitly declared (ONE_PROVIDER_CONTEXT_WINDOW)
    2. what the process in front of us is serving (e.g. llama.cpp /props)
    3. what the model or provider can address (CONTEXT_WINDOWS table, or default)
    """
    if declared_window and declared_window > 0:
        return declared_window
    return (served_context_window(base_url) if base_url else None) \
        or context_window_for(model, backend_name)


def load_pricing() -> dict[str, tuple[float, float]]:
    """PRICING plus whatever ONE_CLI_PRICING names. Bad file, no crash."""
    table = dict(PRICING)
    for key, value in _pricing_file().items():
        try:
            table[str(key)] = (float(value[0]), float(value[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return table


def _pricing_file() -> dict:
    path = os.environ.get(PRICING_ENV)
    if not path:
        return {}
    try:
        data = json.loads(Path(os.path.expanduser(path)).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def cache_read_weight(model: str, backend_name: str = "") -> float:
    """Cache-read price as a fraction of fresh input for this model.

    From the catalog's third price column when it names one, otherwise
    DEFAULT_CACHE_READ_WEIGHT. Clamped to [0, 1]: a cache read is never
    dearer than the fresh input it replaces.
    """
    table = {}
    for key, value in _pricing_file().items():
        try:
            fresh, cached = float(value[0]), float(value[2])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if fresh > 0 and cached >= 0:
            table[str(key)] = min(1.0, cached / fresh)
    found = _longest_match(table, model)
    if found is None and backend_name:
        found = _longest_match(table, backend_name)
    return DEFAULT_CACHE_READ_WEIGHT if found is None else found


# What a lane costs to run, as three kinds rather than one number. A bench that
# reports "no price" for eleven of twelve lanes cannot be budgeted, and a bench
# that treats "unknown" as "free" will happily spend a coding plan's whole
# allowance on a smoke test.
#
#   free    - an explicit zero price, e.g. a llama.cpp or ollama server on local hardware.
#             Runnable without limit; this is what a default suite should use.
#   metered - a known per-token price, so a run's dollars are real and a
#             budget ceiling means something.
#   plan    - a flat-rate subscription endpoint. There is no per-token price,
#             so spend is quota rather than dollars and no dollar budget can
#             bound it.
#   unknown - a metered API whose price is not in the pricing table. It bills
#             money per token and the amount is not known.
#
# The last two are kept apart: neither can be bounded by a ceiling, but only
# one of them is a subscription, and a person deciding what to run needs to be
# told which. Fix an `unknown` by naming its price in `ONE_CLI_PRICING`.
FREE, METERED, PLAN, UNKNOWN = "free", "metered", "plan", "unknown"

# A flat-rate plan is a property of the endpoint, not of the model id - the same
# model is sold per-token elsewhere. A plan endpoint usually says so in its own
# hostname or path, e.g. a `token-plan` host or a `/coding/` path. An ordinary
# `compatible-mode` endpoint is not a plan even when the vendor also sells one.
PLAN_MARKERS = ("token-plan", "/coding/")


def is_local_address(base_url: str) -> bool:
    """A loopback, private or CGNAT address: locality only, not ownership or cost."""
    host = urllib.parse.urlsplit(base_url).hostname or ""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host in ("localhost",)
    return bool(addr.is_loopback or addr.is_private
                or addr in ipaddress.ip_network("100.64.0.0/10"))


def is_flat_rate_plan(base_url: str) -> bool:
    """Does this endpoint bill a subscription rather than tokens?"""
    url = (base_url or "").lower()
    return any(marker in url for marker in PLAN_MARKERS)


def lane_kind(model: str, backend_name: str = "", base_url: str = "") -> str:
    # Network locality is not billing evidence: localhost and private addresses
    # may proxy metered APIs. Explicit zero pricing is required for free lanes.
    price = price_for(model, backend_name)
    if price is not None:
        return FREE if price == (0.0, 0.0) else METERED
    return PLAN if is_flat_rate_plan(base_url) else UNKNOWN


def price_for(model: str, backend_name: str = "") -> tuple[float, float] | None:
    """(input, output) USD per million tokens, or None when the table has no price."""
    table = load_pricing()
    found = _longest_match(table, model)
    if found is None and backend_name:
        found = _longest_match(table, backend_name)
    return tuple(found) if found else None

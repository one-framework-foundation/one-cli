"""First run: turn "no backend found" into a working provider drop-in.

`one` resolves backends from drop-ins in `~/.one-providers/*.env` (see
`config.py`). A newcomer has none, so instead of a one-line error an
interactive run gets a short wizard that writes ONE drop-in and proves it with
one request. Nothing here is a second config system: the output is exactly the
file a person would have written by hand, and `config.discover()` reads it.

The key never reaches the terminal. It is either named by variable
(`ONE_PROVIDER_KEY_ENV`, resolved from the process environment at run time) or
typed with hidden input into the drop-in, which is written mode 0600.
"""

from __future__ import annotations

import getpass
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from onecli import config, render, secureio
from onecli.serving import menu

# Local OpenAI-compatible servers worth probing, on their default ports.
LOCAL_SERVERS = (
    ("llama-server", "http://127.0.0.1:8080"),
    ("ollama", "http://127.0.0.1:11434"),
    ("LM Studio", "http://127.0.0.1:1234"),
)


@dataclass(frozen=True)
class Provider:
    name: str        # drop-in file stem
    label: str
    base_url: str    # empty: ask the user
    wire: str
    key_env: str     # conventional variable name for the key
    default_model: str = ""


PROVIDERS = (
    Provider("openai-compatible", "OpenAI-compatible URL", "",
             config.WIRE_OPENAI, "ONE_CLI_API_KEY"),
    Provider("anthropic", "Anthropic", "https://api.anthropic.com",
             config.WIRE_ANTHROPIC, "ANTHROPIC_API_KEY", "claude-sonnet-5"),
    Provider("openai", "OpenAI", "https://api.openai.com/v1",
             config.WIRE_OPENAI, "OPENAI_API_KEY"),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
             config.WIRE_OPENAI, "OPENROUTER_API_KEY"),
)

NO_BACKEND_HELP = """\
one: no model backend is configured yet.

  Run `one` in a terminal for a guided setup, or configure one by hand:

    mkdir -p ~/.one-providers && chmod 700 ~/.one-providers
    cat > ~/.one-providers/openai.env <<'EOF'
    ONE_PROVIDER_BASE_URL=https://api.openai.com/v1
    ONE_PROVIDER_MODEL=<model id>
    ONE_PROVIDER_KEY_ENV=OPENAI_API_KEY
    EOF

  or pass one inline:  one --base-url URL --model-id MODEL "prompt"
                       (key in ONE_CLI_API_KEY)

  `one doctor` shows what one-cli can see."""


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _get_json(url: str, headers: dict | None = None,
              timeout: float = 2.0) -> dict | None:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError,
            http.client.HTTPException):
        return None


def _model_ids(data: dict | None) -> list[str]:
    if not isinstance(data, dict):
        return []
    entries = data.get("data")
    if not isinstance(entries, list):
        return []
    return [m["id"] for m in entries
            if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]]


def detect_local(servers=LOCAL_SERVERS, timeout: float = 1.0):
    """[(label, base_url, [model ids])] for every local server that answers."""
    found = []
    for label, base in servers:
        ids = _model_ids(_get_json(f"{base}/v1/models", timeout=timeout))
        if ids:
            found.append((label, f"{base}/v1", ids))
    return found


def list_models(base_url: str, wire: str, key: str) -> list[str]:
    """Model ids the endpoint offers this key, or [] when it will not say."""
    if wire == config.WIRE_ANTHROPIC:
        root = base_url.rstrip("/")
        url = (root if root.endswith("/v1") else root + "/v1") + "/models"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url = base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {key}"} if key else {}
    return _model_ids(_get_json(url, headers, timeout=6.0))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", text.lower()).strip("-") or "provider"


def dropin_values(base_url: str, wire: str, model: str, *,
                  key_env: str = "", key: str = "") -> dict[str, str]:
    url_var = ("ANTHROPIC_BASE_URL" if wire == config.WIRE_ANTHROPIC
               else "ONE_PROVIDER_BASE_URL")
    values = {url_var: base_url, "ONE_PROVIDER_MODEL": model}
    if key_env:
        values["ONE_PROVIDER_KEY_ENV"] = key_env
    elif key:
        values["ONE_PROVIDER_KEY"] = key
    return values


def write_dropin(name: str, values: dict[str, str],
                 directory: Path | None = None) -> Path:
    """Write `<dir>/<name>.env` mode 0600 in a 0700 dir; refuse to overwrite."""
    for k, v in values.items():
        if "\n" in v or "\r" in v:
            raise ValueError(f"{k} must be a single line")
    path = Path(directory or config.PROVIDER_DIR) / f"{_slug(name)}.env"
    body = "# written by `one` first-run setup\n" + "".join(
        f"{k}={v}\n" for k, v in values.items())
    with secureio.text_file(path, "x", private=True) as fh:
        fh.write(body)
    return path


def test_request(backend: config.Backend) -> str | None:
    """One tiny request. None on success, else the reason it failed."""
    from onecli import providers
    try:
        turn = providers.build(backend, timeout=60).chat(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=64)
    except Exception as exc:  # any failure is a message for the user, not a crash
        return f"{type(exc).__name__}: {exc}"[:500]
    if not (turn.text or turn.thinking):
        return "the endpoint answered with an empty reply"
    return None


def _say(text: str = "") -> None:
    print(text, file=sys.stderr)


def _pick_local(found) -> tuple[str, str, str] | None:
    options = [(f"{label}: {m}", url) for label, url, ids in found for m in ids[:8]]
    options.append(("a hosted provider instead", "OpenAI, Anthropic, OpenRouter, any URL"))
    pick = menu.choose("local model servers found", options)
    if pick == -1:
        raise KeyboardInterrupt
    if pick == len(options) - 1:
        return None
    flat = [(label, url, m) for label, url, ids in found for m in ids[:8]]
    label, url, model = flat[pick]
    return _slug(f"{label}-{model}"), url, model


def _pick_provider() -> tuple[str, dict[str, str], str]:
    """(drop-in name, values, the key variable still to export or "")."""
    pick = menu.choose("choose a provider",
                       [(p.label, p.base_url) for p in PROVIDERS])
    if pick == -1:
        raise KeyboardInterrupt
    p = PROVIDERS[pick]
    base_url = p.base_url or menu.ask("base URL (e.g. http://host:8000/v1)")
    if not base_url:
        raise KeyboardInterrupt
    how = menu.choose("API key", [
        ("read it from an environment variable", f"e.g. {p.key_env}; nothing is stored"),
        ("type it now", "hidden input, saved in a 0600 file"),
        ("no key", "local or open endpoints"),
    ])
    if how == -1:
        raise KeyboardInterrupt
    key_env = key = ""
    if how == 0:
        key_env = menu.ask("variable name", p.key_env)
        key = os.environ.get(key_env, "")
    elif how == 1:
        try:
            key = getpass.getpass("  API key (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt from None
    models = list_models(base_url, p.wire, key) if key or how == 2 else []
    if models:
        shown = models[:15]
        default = shown.index(p.default_model) if p.default_model in shown else 0
        idx = menu.choose("model", [(m, "") for m in shown], default=default)
        if idx == -1:
            raise KeyboardInterrupt
        model = shown[idx]
    else:
        model = menu.ask("model id", p.default_model)
    if not model:
        raise KeyboardInterrupt
    name = p.name if p.base_url else _slug(base_url.split("//")[-1].split("/")[0])
    values = dropin_values(base_url, p.wire, model, key_env=key_env,
                           key="" if key_env else key)
    return name, values, key_env if key_env and not key else ""


def run() -> config.Backend | None:
    """The wizard. Returns the tested backend, or None if the user quit."""
    _say(render.banner("welcome", ["one-cli first run: let's connect a model"]))
    try:
        found = detect_local()
        choice = _pick_local(found) if found else None
        unset_env = ""
        if choice:
            name, url, model = choice
            values = dropin_values(url, config.WIRE_OPENAI, model)
        else:
            if not found:
                _say(render.dim("  no local model server on :8080, :11434 or :1234"))
            name, values, unset_env = _pick_provider()
    except (OSError, ValueError) as exc:
        _say(render.red(f"  setup failed: {exc}"))
        return None
    except KeyboardInterrupt:
        _say(render.dim("  setup cancelled; nothing written"))
        return None
    try:
        path = write_dropin(name, values)
    except FileExistsError:
        _say(render.red(f"  {config.PROVIDER_DIR}/{_slug(name)}.env already exists "
                        f"and is unusable; fix or remove it, then run `one` again"))
        return None
    except (OSError, ValueError) as exc:
        _say(render.red(f"  cannot write provider in {config.PROVIDER_DIR}: {exc}"))
        return None
    backend = config._backend_from_env(path)
    if backend is None:  # should not happen: we just wrote a valid drop-in
        _say(render.red(f"  wrote {path} but could not read it back"))
        return None
    _say(render.dim(f"  wrote {path} (mode 600)"))
    if unset_env:
        # Nothing to test with yet, and the file is right: keep it.
        _say(render.yellow(f"  {unset_env} is not set in this shell. Export it, "
                           f"then run `one` again (`one doctor` checks it)."))
        return None
    _say(render.dim(f"  testing {backend.model} ..."))
    err = test_request(backend)
    if err:
        # A drop-in that does not answer would stop the wizard from ever
        # running again, so it is not kept: the next `one` starts over.
        path.unlink()
        _say(render.red(f"  test request failed: {err}"))
        _say(render.dim("  nothing was kept; run `one` again to pick another"))
        return None
    _say(render.green(f"  ok - {backend.name} answered. Try: one \"explain this repo\""))
    return backend


# ---------------------------------------------------------------- doctor

SANDBOX_BINARIES = ("/usr/bin/bwrap", "/usr/bin/prlimit")


def probe(backend: config.Backend, timeout: float = 4.0) -> str:
    """`ok`, `key rejected (HTTP 401)`, `HTTP 404`, or `unreachable: ...`.

    Lists models rather than chatting, so doctor never spends a token.
    """
    root = backend.base_url.rstrip("/")
    if backend.wire == config.WIRE_ANTHROPIC:
        url = (root if root.endswith("/v1") else root + "/v1") + "/models"
        headers = {"x-api-key": backend.api_key, "anthropic-version": "2023-06-01"}
    else:
        url = root + "/models"
        headers = {"Authorization": f"Bearer {backend.api_key}"} if backend.api_key else {}
    headers.update(backend.headers)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=timeout):
            return "ok"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return f"key rejected (HTTP {exc.code})"
        return f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError, TimeoutError,
            http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", exc)
        return f"unreachable: {reason}"


def doctor_report(limit: int = 12) -> tuple[list[str], bool]:
    """Lines describing config, reachability and sandbox; ok = a backend works."""
    lines = []
    dirs = config.provider_dirs()
    lines.append("drop-ins     " + ", ".join(
        f"{d} ({len(list(d.glob('*.env')))} files)" if d.is_dir() else f"{d} (missing)"
        for d in dirs))
    found = config.discover()
    if not found:
        lines.append("backends     none - run `one` in a terminal for guided setup")
    usable = False
    try:
        default = config.resolve().name if found else ""
    except RuntimeError:
        default = ""
    for b in found[:limit]:
        key = "key set" if b.api_key else "no key"
        state = probe(b)
        usable = usable or state == "ok"
        mark = "*" if b.name == default else " "
        lines.append(f"{mark} {b.name:28} {b.model:28} {key:8} {state}")
    if len(found) > limit:
        lines.append(f"  ... and {len(found) - limit} more (not probed)")
    if default:
        lines.append(f"default      {default}  (* above; -m NAME picks another)")
    missing = [b for b in SANDBOX_BINARIES if not os.path.exists(b)]
    lines.append("sandbox      " + ("bubblewrap + prlimit available" if not missing
                                    else "unavailable, missing " + ", ".join(missing)
                                    + " (sandboxed shell tools refuse to run)"))
    return lines, usable

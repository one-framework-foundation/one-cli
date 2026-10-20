"""Tell a person a newer release exists, and `one update` to install it.

Releases are cut from `v*` tags; PyPI is where a published one lands, so its
JSON API for `one-core-cli` is the source of truth. A 404 there means nothing
has been published yet, which is silent, not an error.

The startup notice follows `gh` and npm's update-notifier rather than pip's
self-check: the network call runs in a daemon thread and only writes the cache,
and the line a person sees comes from the cache the previous start left. A
start never waits on the network, and offline costs nothing. It runs only
where a person is watching (all three streams on a terminal, no `CI`), at most
one check and one notice per 24 hours, and is off with ONECLI_NO_UPDATE_CHECK=1
or `"update_check": false` in ~/.one-cli/config.json.

Startup-bound by nature: the check happens once per process, so a reload of
this module changes the next start, not a session already running. `one
update` replaces the installed package; running sessions keep the code they
loaded and the new version activates on the next start.
"""
from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from onecli import __version__, render, secureio

DIST = "one-core-cli"
PYPI_URL = f"https://pypi.org/pypi/{DIST}/json"
STATE_DIR = Path.home() / ".one-cli"
CACHE = STATE_DIR / "update-check.json"
CONFIG = STATE_DIR / "config.json"
DISABLE_ENV = "ONECLI_NO_UPDATE_CHECK"
INTERVAL = 24 * 3600
TIMEOUT = 2.0

# ASCII digits only and no surrounding whitespace: `\d` would admit other
# scripts' digits and str.strip() would eat \r, \x0b and \x1c, letting
# terminal control bytes from a hostile answer through to the screen.
_VERSION = re.compile(r"v?([0-9]+(?:\.[0-9]+)*)(?:(a|b|rc)([0-9]+))?(?:\.post([0-9]+))?(?:\.dev([0-9]+))?")
_PRE = {"a": 0, "b": 1, "rc": 2}
# Far above any real version, far below int()'s digit limit.
MAX_VERSION_LEN = 32


def normalize(text) -> str | None:
    """The version without a leading `v`, or None for anything else.

    This is the only form ever cached or printed: a response's own text
    never reaches the terminal.
    """
    if not isinstance(text, str) or len(text) > MAX_VERSION_LEN:
        return None
    m = _VERSION.fullmatch(text)
    return m.group(0).removeprefix("v") if m else None


def version_key(text) -> tuple | None:
    """Order PEP 440 versions of the shapes we publish; None if unrecognised.

    `packaging` is not a dependency and the core stays standard-library only,
    so this covers release, pre, post and dev segments and nothing exotic.
    An unparseable version never produces a notice.
    """
    clean = normalize(text)
    m = _VERSION.fullmatch(clean) if clean else None
    if not m:
        return None
    release, pre, pre_n, post, dev = m.groups()
    nums = [int(x) for x in release.split(".")]
    while len(nums) > 1 and nums[-1] == 0:
        nums.pop()
    # dev-only < pre < final < post; a dev of any of them sorts before it.
    pre_key = (_PRE[pre], int(pre_n)) if pre else ((-1, 0) if dev and not post else (3, 0))
    return (tuple(nums), pre_key, int(post) if post else -1,
            int(dev) if dev else float("inf"))


def is_newer(latest: str | None, installed: str | None = None) -> bool:
    if not latest:
        return False
    a, b = version_key(latest), version_key(installed or __version__)
    return a is not None and b is not None and a > b


def fetch_latest(timeout: float = TIMEOUT) -> str | None:
    """The newest released version on PyPI, or None when nothing is published.

    Raises OSError (URLError, timeout) or ValueError on anything else, so a
    caller can tell "no release" from "could not ask".
    """
    req = urllib.request.Request(PYPI_URL, headers={
        "Accept": "application/json", "User-Agent": f"one-cli/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except http.client.HTTPException as exc:
        # Not an HTTP answer at all (BadStatusLine, IncompleteRead): "could
        # not ask", which callers already handle as OSError.
        raise OSError(f"malformed HTTP answer: {exc!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
        raise ValueError("PyPI answer must contain an info object")
    version = normalize(data["info"].get("version"))
    if version is None:
        raise ValueError("PyPI answer carries no recognisable version")
    return version


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _stamp(state: dict, key: str) -> float:
    value = state.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _save(state: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        secureio.atomic_write_text(CACHE, json.dumps(state, indent=1) + "\n")
    except OSError:
        pass


def _streams() -> tuple:
    return sys.stdin, sys.stdout, sys.stderr


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    if env.get(DISABLE_ENV, "").strip() not in ("", "0") or "CI" in env:
        return False
    if _read_json(CONFIG).get("update_check") is False:
        return False
    try:
        return all(s is not None and s.isatty() for s in _streams())
    except (AttributeError, ValueError):
        return False


def refresh(timeout: float = TIMEOUT) -> None:
    """Ask PyPI and record the answer. Failure records the attempt only."""
    try:
        latest = fetch_latest(timeout)
        ok = True
    except Exception:  # offline, DNS, timeout, a proxy page: all silent
        latest, ok = None, False
    state = _read_json(CACHE)
    state["checked_at"] = time.time()
    if ok:
        state["latest"] = latest
    _save(state)


def startup(now: float | None = None, out=None) -> threading.Thread | None:
    """Print a cached notice if one is due, and refresh the cache in the background.

    Returns the refresh thread (tests join it); None when nothing was started.
    Never raises: a start must not fail because of an update check.
    """
    try:
        return _startup(now, out)
    except Exception:
        return None


def _startup(now: float | None, out) -> threading.Thread | None:
    if not enabled():
        return None
    now = time.time() if now is None else now
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "update-check.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _read_json(CACHE)
        latest = normalize(state.get("latest"))
        if is_newer(latest) and now - _stamp(state, "notified_at") >= INTERVAL:
            print(render.yellow(f"one-cli {latest} is available (installed {__version__}); "
                                f"run `one update`"), file=out or sys.stderr)
            state["notified_at"] = now
            _save(state)
        if now - _stamp(state, "checked_at") < INTERVAL:
            return None
        state["checked_at"] = now
        # A failed reservation must never start network I/O. Notice writes use
        # this same lock, so a stale notice cannot overwrite a reservation.
        secureio.atomic_write_text(CACHE, json.dumps(state) + "\n")
    worker = threading.Thread(target=refresh, name="one-update-check", daemon=True)
    worker.start()
    return worker


# --- install method -------------------------------------------------------

def _distribution():
    from importlib import metadata
    try:
        return metadata.distribution(DIST)
    except metadata.PackageNotFoundError:
        return None


def _checkout_root() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    return root if (root / ".git").exists() and (root / "pyproject.toml").is_file() else None


def install_method(dist=None, prefix: str | None = None,
                   base_prefix: str | None = None) -> tuple[str, list[str] | None]:
    """(kind, upgrade argv). argv is None where we print instructions instead.

    Kinds: `checkout` (editable install or running from source), `uv-tool`,
    `pipx`, `uv-pip`, `pip-venv`, `pip`. The markers are the ones each tool
    writes itself: uv's `uv-receipt.toml` and pipx's `pipx_metadata.json` in
    the tool's venv root, PEP 610 `direct_url.json` for editable installs and
    the dist-info `INSTALLER` record pip's own self-check reads.
    """
    # The code's own location decides first: run from a git tree, a stray
    # `one_core_cli.egg-info` in that tree would otherwise read as pip.
    if _checkout_root() is not None:
        return "checkout", None
    dist = _distribution() if dist is None else dist
    prefix = sys.prefix if prefix is None else prefix
    base_prefix = sys.base_prefix if base_prefix is None else base_prefix
    if dist is None:
        return "checkout", None
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except ValueError:
        direct = {}
    if (direct.get("dir_info") or {}).get("editable"):
        return "checkout", None
    root = Path(prefix)
    if (root / "uv-receipt.toml").is_file():
        return "uv-tool", ["uv", "tool", "upgrade", DIST]
    if (root / "pipx_metadata.json").is_file():
        return "pipx", ["pipx", "upgrade", DIST]
    installer = (dist.read_text("INSTALLER") or "").strip().lower()
    if installer == "uv":
        return "uv-pip", ["uv", "pip", "install", "--upgrade", "--python", sys.executable, DIST]
    kind = "pip-venv" if prefix != base_prefix else "pip"
    return kind, [sys.executable, "-m", "pip", "install", "--upgrade", DIST]


def _checkout_help() -> str:
    root = _checkout_root()
    where = f"git -C {render.short_path(str(root))}" if root else "git"
    return f"this is a source checkout; update it yourself: {where} pull --ff-only"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="one update",
                                description="Upgrade one-cli with the tool that installed it.")
    p.add_argument("--check", action="store_true",
                   help="report the installed and latest versions and stop")
    args = p.parse_args(argv)
    kind, command = install_method()
    try:
        latest = fetch_latest(timeout=10.0)
    except (OSError, ValueError) as exc:
        # The reason only, never the answer's text (see normalize).
        print(render.red(f"could not reach PyPI: {type(exc).__name__}"), file=sys.stderr)
        return 1
    print(f"installed {__version__} ({kind})")
    if latest is None:
        print("no release of one-core-cli is published yet")
        return 0
    print(f"latest    {latest}")
    if not is_newer(latest):
        print("up to date")
        return 0
    if command is None:
        print(_checkout_help())
        return 0
    if args.check:
        print("would run: " + " ".join(command))
        return 0
    if shutil.which(command[0]) is None and not os.path.isabs(command[0]):
        print(render.red(f"{command[0]} is not on PATH; run by hand: {' '.join(command)}"),
              file=sys.stderr)
        return 1
    print("running: " + " ".join(command), flush=True)
    code = subprocess.run(command).returncode
    if code == 0:
        print(f"updated; one-cli {latest} takes effect the next time you start `one`")
    return code

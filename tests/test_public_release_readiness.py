"""Public-release readiness regressions: a clean-room host, no local state.

These are the shipping invariants rather than runtime boundaries: a relay that
must not hardcode one operator's home directory or one machine's address, and
CI jobs that must install the extras the project declares instead of a
handwritten dependency list.
"""

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


# --- no operator-specific defaults in the relay ----------------------------------

def test_relay_token_store_lives_under_the_running_user_home(monkeypatch):
    # An absolute path to somebody else's checkout is unreadable on every host
    # but one. Unset, the store belongs to whoever runs the relay.
    from onecli.relay import server
    monkeypatch.delenv("ONE_RELAY_HOST_TOKENS", raising=False)
    home = Path.home()
    for path in server.token_paths():
        assert path.is_relative_to(home), f"store outside the running user home: {path}"
        assert not path.is_relative_to(Path(__file__).resolve().parents[2]), f"deployment checkout baked in: {path}"


def test_relay_token_store_is_operator_configurable(monkeypatch, tmp_path):
    from onecli.relay import server
    store = tmp_path / "host_tokens.json"
    monkeypatch.setenv("ONE_RELAY_HOST_TOKENS", str(store))
    assert store in set(server.token_paths())


def test_relay_binds_loopback_unless_told_otherwise(monkeypatch):
    from onecli.relay import server
    monkeypatch.delenv("ONE_RELAY_HOST", raising=False)
    assert server.bind_host() == "127.0.0.1"
    monkeypatch.setenv("ONE_RELAY_HOST", "0.0.0.0")
    assert server.bind_host() == "0.0.0.0"


def test_no_operator_home_paths_in_shipped_package():
    offenders = []
    for path in sorted((ROOT / "onecli").rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"/home/[A-Za-z0-9_-]+/", line) and "expanduser" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "operator home paths shipped in the package:\n" + "\n".join(offenders)


# --- the relay's own dependencies are installable ---------------------------

def test_relay_extra_declares_its_runtime_dependencies():
    extras = pyproject()["project"]["optional-dependencies"]
    assert "relay" in extras, "no `relay` extra: `pip install one-cli[relay]` cannot work"
    names = {re.split(r"[<>=\[ ]", dep, 1)[0] for dep in extras["relay"]}
    # onecli/infer validates its schemas with jsonschema.
    assert {"fastapi", "uvicorn", "jsonschema"} <= names, names


# --- one declared test dependency set, used by every CI job -----------------

def test_test_extra_declares_what_the_suite_imports():
    extras = pyproject()["project"]["optional-dependencies"]
    assert "test" in extras
    names = {re.split(r"[<>=\[ ]", dep, 1)[0] for dep in extras["test"]}
    # tests/test_infer_relay.py imports fastapi unconditionally; the schema
    # tests import jsonschema; the relay client tests import httpx.
    assert {"pytest", "fastapi", "httpx", "jsonschema"} <= names, names




# --- one version, read from the package ---------------------------------------

def test_user_agent_and_preview_take_the_package_version():
    import onecli
    from onecli.providers import base
    assert base.USER_AGENT == f"one-cli/{onecli.__version__}"
    for path in ("onecli/providers/base.py", "scripts/ui-preview.py"):
        text = (ROOT / path).read_text()
        assert not re.search(r'["\']\d+\.\d+\.\d+["\']', text), path

"""Update notice and `one update`, offline: the network is always mocked."""
import io
import json
import socket
import urllib.error

import pytest

from onecli import cli, update


class _TTY(io.StringIO):
    def isatty(self):
        return True


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _pypi(version):
    def urlopen(req, timeout=None):
        return _Resp(json.dumps({"info": {"version": version}}).encode())
    return urlopen


def _http(code):
    def urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)
    return urlopen


def _timeout(req, timeout=None):
    raise urllib.error.URLError(socket.timeout("timed out"))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A watched terminal with the check enabled and state in tmp_path."""
    monkeypatch.delenv("ONECLI_NO_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(update, "STATE_DIR", tmp_path)
    monkeypatch.setattr(update, "CACHE", tmp_path / "update-check.json")
    monkeypatch.setattr(update, "CONFIG", tmp_path / "config.json")
    monkeypatch.setattr(update, "__version__", "0.1.0")
    monkeypatch.setattr(update, "_streams", lambda: (_TTY(), _TTY(), _TTY()))
    return tmp_path


def _run(monkeypatch, opener, now=1_000_000.0):
    monkeypatch.setattr("urllib.request.urlopen", opener)
    out = io.StringIO()
    worker = update.startup(now=now, out=out)
    if worker is not None:
        worker.join(5)
    return out.getvalue(), worker


def _cache(home):
    return json.loads((home / "update-check.json").read_text())


def test_newer_release_is_cached_then_announced_next_start(home, monkeypatch):
    out, worker = _run(monkeypatch, _pypi("0.2.0"))
    assert worker is not None and out == ""  # first start never waits
    assert _cache(home)["latest"] == "0.2.0"
    out, worker = _run(monkeypatch, _pypi("0.2.0"), now=1_000_100.0)
    assert "0.2.0 is available" in out and "one update" in out
    assert worker is None  # checked under 24h ago
    assert len(out.strip().splitlines()) == 1


def test_notice_at_most_once_per_day(home, monkeypatch):
    update.CACHE.write_text(json.dumps({"checked_at": 1e6, "latest": "0.2.0"}))
    first, _ = _run(monkeypatch, _pypi("0.2.0"), now=1e6 + 10)
    again, _ = _run(monkeypatch, _pypi("0.2.0"), now=1e6 + 20)
    later, _ = _run(monkeypatch, _pypi("0.2.0"), now=1e6 + 10 + 86400)
    assert first and not again and later


def test_same_version_is_silent(home, monkeypatch):
    _run(monkeypatch, _pypi("0.1.0"))
    out, _ = _run(monkeypatch, _pypi("0.1.0"), now=1_000_100.0)
    assert out == "" and _cache(home)["latest"] == "0.1.0"


def test_404_means_no_release_and_is_silent(home, monkeypatch):
    _run(monkeypatch, _http(404))
    out, _ = _run(monkeypatch, _http(404), now=1_000_100.0)
    assert out == "" and _cache(home)["latest"] is None


def test_timeout_is_silent_and_keeps_the_last_answer(home, monkeypatch):
    update.CACHE.write_text(json.dumps({"checked_at": 0, "latest": "0.1.0"}))
    out, worker = _run(monkeypatch, _timeout)
    assert out == "" and worker is not None
    state = _cache(home)
    assert state["latest"] == "0.1.0" and state["checked_at"] > 0


def test_recent_check_does_not_touch_the_network(home, monkeypatch):
    update.CACHE.write_text(json.dumps({"checked_at": 1e6, "latest": "0.1.0"}))

    def boom(*a, **k):
        raise AssertionError("network used inside the 24h window")
    out, worker = _run(monkeypatch, boom, now=1e6 + 3600)
    assert worker is None and out == ""


@pytest.mark.parametrize("how", ["env", "config", "ci", "ci-empty", "no-tty"])
def test_disabled_and_unwatched_runs_never_check(home, monkeypatch, how):
    update.CACHE.write_text(json.dumps({"checked_at": 0, "latest": "9.0.0"}))
    if how == "env":
        monkeypatch.setenv("ONECLI_NO_UPDATE_CHECK", "1")
    elif how == "config":
        update.CONFIG.write_text(json.dumps({"update_check": False}))
    elif how in ("ci", "ci-empty"):
        monkeypatch.setenv("CI", "" if how == "ci-empty" else "true")
    else:
        monkeypatch.setattr(update, "_streams", lambda: (_TTY(), io.StringIO(), _TTY()))

    def boom(*a, **k):
        raise AssertionError("network used while disabled")
    out, worker = _run(monkeypatch, boom)
    assert out == "" and worker is None


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]",
                                 '{"checked_at": "soon", "notified_at": null, "latest": 3}'])
def test_corrupt_cache_is_ignored(home, monkeypatch, bad):
    update.CACHE.write_text(bad)
    out, worker = _run(monkeypatch, _pypi("0.2.0"))
    assert out == "" and _cache(home)["latest"] == "0.2.0"


@pytest.mark.parametrize("latest,installed,newer", [
    ("0.2.0", "0.1.0", True), ("0.1.0", "0.1.0", False), ("0.1", "0.1.0", False),
    ("0.1.1rc1", "0.1.0", True), ("0.1.0", "0.1.0rc2", True), ("0.1.0rc1", "0.1.0", False),
    ("0.1.0.post1", "0.1.0", True), ("0.1.0.dev3", "0.1.0", False), ("0.10.0", "0.9.0", True),
    ("garbage", "0.1.0", False), (None, "0.1.0", False),
])
def test_version_order(latest, installed, newer):
    assert update.is_newer(latest, installed) is newer


class _Dist:
    def __init__(self, files):
        self.files = files

    def read_text(self, name):
        return self.files.get(name)


def test_install_method_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "_checkout_root", lambda: None)
    venv = tmp_path / "venv"
    venv.mkdir()
    pip_dist = _Dist({"INSTALLER": "pip\n"})
    assert update.install_method(_Dist({"direct_url.json": json.dumps(
        {"url": "file:///src", "dir_info": {"editable": True}})}), str(venv), "/usr") == ("checkout", None)

    kind, cmd = update.install_method(pip_dist, str(venv), "/usr")
    assert kind == "pip-venv" and cmd[1:] == ["-m", "pip", "install", "--upgrade", "one-core-cli"]
    assert update.install_method(pip_dist, "/usr", "/usr")[0] == "pip"

    (venv / "pipx_metadata.json").write_text("{}")
    assert update.install_method(pip_dist, str(venv), "/usr") == (
        "pipx", ["pipx", "upgrade", "one-core-cli"])
    (venv / "pipx_metadata.json").unlink()

    (venv / "uv-receipt.toml").write_text("")
    assert update.install_method(_Dist({"INSTALLER": "uv"}), str(venv), "/usr") == (
        "uv-tool", ["uv", "tool", "upgrade", "one-core-cli"])
    (venv / "uv-receipt.toml").unlink()

    assert update.install_method(_Dist({"INSTALLER": "uv"}), str(venv), "/usr")[0] == "uv-pip"


def test_code_running_from_a_checkout_wins_over_stray_metadata(tmp_path, monkeypatch):
    # A checkout's own `one_core_cli.egg-info` is on sys.path when run from
    # the tree, and reads as a pip install; the code's location decides.
    monkeypatch.setattr(update, "_checkout_root", lambda: tmp_path)
    assert update.install_method(_Dist({"INSTALLER": "pip"}), "/venv", "/usr") == ("checkout", None)


def test_uninstalled_source_tree_is_a_checkout(monkeypatch):
    monkeypatch.setattr(update, "_distribution", lambda: None)
    assert update.install_method() == ("checkout", None)


def _cmd(monkeypatch, capsys, argv, latest, method):
    monkeypatch.setattr(update, "__version__", "0.1.0")
    monkeypatch.setattr(update, "fetch_latest", lambda timeout: latest)
    monkeypatch.setattr(update, "install_method", lambda: method)
    ran = []

    class _Done:
        returncode = 0
    monkeypatch.setattr("subprocess.run", lambda c: ran.append(c) or _Done())
    monkeypatch.setattr("shutil.which", lambda name: "/bin/" + name)
    monkeypatch.setattr("sys.stdin", _TTY())
    code = cli.main(argv)
    return code, capsys.readouterr().out, ran


def test_update_command_prints_then_runs(monkeypatch, capsys):
    code, out, ran = _cmd(monkeypatch, capsys, ["update"], "0.2.0",
                          ("pipx", ["pipx", "upgrade", "one-core-cli"]))
    assert code == 0 and ran == [["pipx", "upgrade", "one-core-cli"]]
    assert out.index("running: pipx upgrade one-core-cli") < out.index("next time")


def test_update_check_only_reports(monkeypatch, capsys):
    code, out, ran = _cmd(monkeypatch, capsys, ["update", "--check"], "0.2.0",
                          ("uv-tool", ["uv", "tool", "upgrade", "one-core-cli"]))
    assert code == 0 and ran == [] and "would run: uv tool upgrade one-core-cli" in out


def test_update_checkout_prints_instructions(monkeypatch, capsys):
    code, out, ran = _cmd(monkeypatch, capsys, ["update"], "0.2.0", ("checkout", None))
    assert code == 0 and ran == [] and "pull --ff-only" in out


def test_update_unpublished_and_current_run_nothing(monkeypatch, capsys):
    method = ("pip-venv", ["python", "-m", "pip", "install", "--upgrade", "one-core-cli"])
    code, out, ran = _cmd(monkeypatch, capsys, ["update"], None, method)
    assert code == 0 and ran == [] and "no release" in out
    code, out, ran = _cmd(monkeypatch, capsys, ["update"], "0.1.0", method)
    assert code == 0 and ran == [] and "up to date" in out


def test_update_offline_fails_cleanly(monkeypatch, capsys):
    def offline(timeout):
        raise urllib.error.URLError("no route")
    monkeypatch.setattr(update, "fetch_latest", offline)
    assert cli.main(["update", "--check"]) == 1
    assert "could not reach PyPI" in capsys.readouterr().err


def test_update_followed_by_words_is_still_a_prompt(monkeypatch):
    called = []
    monkeypatch.setattr(update, "main", lambda argv: called.append(argv) or 0)
    monkeypatch.setattr(cli, "build_parser", lambda: (_ for _ in ()).throw(SystemExit(7)))
    with pytest.raises(SystemExit):
        cli.main(["update", "the", "readme"])
    assert called == []


def _prompt_path(monkeypatch):
    """Make the prompt path observable: update.main must not be reached."""
    called = []
    monkeypatch.setattr(update, "main", lambda argv: called.append(argv) or 0)
    monkeypatch.setattr(cli, "build_parser", lambda: (_ for _ in ()).throw(SystemExit(7)))
    return called


def test_bare_update_with_piped_stdin_is_a_prompt(monkeypatch):
    called = _prompt_path(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("x\n"))
    with pytest.raises(SystemExit):
        cli.main(["update"])
    assert called == []


def test_bare_update_on_a_terminal_is_the_command(monkeypatch):
    called = _prompt_path(monkeypatch)
    monkeypatch.setattr("sys.stdin", _TTY())
    assert cli.main(["update"]) == 0 and called == [[]]


HOSTILE = ["9" * 5000, "1." * 40 + "1", "\r\x1c9.9.9\x0b", "9.9.9\n", "\x1b[2J9.9.9",
           "\u0669.0.0", 7, None]


@pytest.mark.parametrize("bad", HOSTILE)
def test_hostile_versions_are_not_versions(bad):
    assert update.normalize(bad) is None
    assert update.is_newer(bad, "0.1.0") is False


@pytest.mark.parametrize("bad", [v for v in HOSTILE if isinstance(v, str)])
def test_fetch_rejects_a_hostile_pypi_version(home, monkeypatch, bad):
    monkeypatch.setattr("urllib.request.urlopen", _pypi(bad))
    with pytest.raises(ValueError):
        update.fetch_latest()
    update.CACHE.write_text(json.dumps({"checked_at": 0, "latest": "0.1.0"}))
    _run(monkeypatch, _pypi(bad))
    assert _cache(home)["latest"] == "0.1.0"  # a rejected answer is not cached


@pytest.mark.parametrize("bad", [v for v in HOSTILE if isinstance(v, str)])
def test_hostile_cached_version_never_crashes_or_prints(home, monkeypatch, bad):
    update.CACHE.write_text(json.dumps({"checked_at": 1e6, "latest": bad}))
    out, worker = _run(monkeypatch, _pypi("0.1.0"), now=1e6 + 10)
    assert out == "" and worker is None


def test_notice_prints_the_normalized_version(home, monkeypatch):
    update.CACHE.write_text(json.dumps({"checked_at": 1e6, "latest": "v0.2.0"}))
    out, _ = _run(monkeypatch, _pypi("0.2.0"), now=1e6 + 10)
    assert "one-cli 0.2.0 is available" in out and "v0.2.0" not in out


def test_startup_swallows_any_exception(home, monkeypatch):
    def broken(path):
        raise RuntimeError("cache reader exploded")
    monkeypatch.setattr(update, "_read_json", broken)
    assert update.startup(now=1e6, out=io.StringIO()) is None


def test_update_command_never_echoes_the_response_text(monkeypatch, capsys):
    monkeypatch.setattr("urllib.request.urlopen", _pypi("\r\x1c9.9.9\x0b"))
    monkeypatch.setattr(update, "install_method", lambda: ("pipx", ["pipx", "upgrade", "x"]))
    assert update.main(["--check"]) == 1
    captured = capsys.readouterr()
    assert "\x1c" not in captured.out + captured.err and "\r" not in captured.out + captured.err

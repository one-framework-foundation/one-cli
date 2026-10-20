"""Public distribution regressions; no downloads or provider calls."""
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from onecli import config, harnesses, render
from onecli.evals import bench
from onecli.evals import stress
from onecli.serving import cli as serving


def test_bench_uses_installed_interpreter_and_module(tmp_path):
    backend = config.Backend("fixture", "fixture", "http://127.0.0.1:1/v1")
    cmd = bench._one_cli_cmd(backend, bench.Task("fixture", "hello"),
                             tmp_path, "fixture", "", 1)
    assert cmd[:3] == [sys.executable, "-m", "onecli"]


def test_arena_uses_installed_interpreter_and_module(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON", "/not/the/installed/python")
    backend = config.Backend("fixture", "fixture", "http://127.0.0.1:1/v1")
    launch = harnesses._one_build(harnesses.Cell(
        backend, "", tmp_path, "hello", cfg_dir=tmp_path))
    assert launch.cmd[:3] == [sys.executable, "-m", "onecli"]


def test_stress_uses_installed_module(tmp_path, monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(stress.subprocess, "run", run)
    stress.run_one(["--help"], tmp_path, 1)
    assert run.call_args.args[0][:3] == [sys.executable, "-m", "onecli"]


def test_stress_console_entry_accepts_omitted_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["one-stress", "--list"])
    assert stress.main() == 0
    assert "deadport" in capsys.readouterr().out


@pytest.mark.parametrize("collision", ["module", "package"])
@pytest.mark.parametrize("launcher", ["arena", "stress"])
def test_workspace_cannot_replace_cli(tmp_path, collision, launcher):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "WORKSPACE_MODULE_EXECUTED"
    payload = f"from pathlib import Path; Path({str(marker)!r}).touch()\n"
    if collision == "module":
        (workspace / "onecli.py").write_text(payload)
    else:
        package = workspace / "onecli"
        package.mkdir()
        (package / "__init__.py").write_text(payload)
        (package / "__main__.py").write_text(payload)
    if launcher == "arena":
        backend = config.Backend("fixture", "fixture", "http://127.0.0.1:1/v1")
        launch = harnesses._one_build(harnesses.Cell(
            backend, "", workspace, "hello", cfg_dir=tmp_path / "config"))
        proc = subprocess.run(launch.cmd + ["--help"], cwd=launch.cwd,
                              env=launch.env, capture_output=True, text=True,
                              timeout=20)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    else:
        rc, out, err, _ = stress.run_one(["--help"], workspace, 20)
    assert not marker.exists(), "the task workspace replaced the trusted CLI"
    assert rc == 0 and "usage:" in out, (rc, out, err)


@pytest.fixture
def setup_fakes(monkeypatch, tmp_path):
    facts = SimpleNamespace(hostname="fixture", can_serve=lambda: True)
    monkeypatch.setattr(serving.hostfacts, "collect", lambda: facts)
    monkeypatch.setattr(serving.hostfacts, "describe", lambda _: [])
    monkeypatch.setattr(serving.hostfacts, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(serving.menu, "_tty", lambda: False)
    monkeypatch.setattr(serving.catalog, "load", lambda: [])
    monkeypatch.setattr(serving.tune, "load_ledger", lambda: {})
    monkeypatch.setattr(serving.serve, "running", lambda _: [])
    return facts


def test_setup_without_tty_refuses_before_actions(setup_fakes, monkeypatch, capsys):
    # Fail promptly on the old endless loop rather than hanging the suite.
    pull = Mock(side_effect=AssertionError("unexpected download"))
    monkeypatch.setattr(serving, "_setup_pull", pull)
    assert serving.cmd_setup([]) == 2
    assert "--yes" in capsys.readouterr().err
    pull.assert_not_called()


def test_setup_yes_propagates_failed_pull(setup_fakes, monkeypatch):
    pull = Mock(return_value=1)
    monkeypatch.setattr(serving, "_setup_pull", pull)
    assert serving.cmd_setup(["--yes"]) == 1
    pull.assert_called_once()


def test_setup_yes_never_opens_menus(setup_fakes, monkeypatch):
    monkeypatch.setattr(serving.menu, "choose", Mock(side_effect=AssertionError("menu")))
    monkeypatch.setattr(serving, "_recommended", lambda _: [])
    assert serving.cmd_setup(["--yes"]) == 1


def test_redirected_stdout_is_plain_even_with_terminal_stderr(monkeypatch):
    for key in ("NO_COLOR", "ONE_CLI_PLAIN", "FORCE_COLOR", "COLORTERM", "TERM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(render.sys.stderr, "isatty", lambda: True)
    assert render._depth() == 0


def test_chat_entry_is_declared():
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'one-chat = "onecli.chat:main"' in project.read_text()

"""Regression tests for the seventh v0.1.1 release-review round: how a
/task spawn worker is launched."""
import subprocess
import sys
from pathlib import Path

from onecli import tasks


def _worker_cmd(tmp_path, monkeypatch, installed):
    if installed:
        # An installed copy has no scripts/one beside the package.
        monkeypatch.setattr(tasks, "__file__", str(tmp_path / "site" / "onecli" / "tasks.py"))
    cmd, _ = tasks.build_runner_cmd("one", "hello", workspace=str(tmp_path))
    return cmd


def test_an_installed_worker_runs_the_package_main(tmp_path, monkeypatch):
    cmd = _worker_cmd(tmp_path, monkeypatch, installed=True)
    assert cmd[1:3] == ["-m", "onecli"], cmd
    # `python -m onecli --version` must actually run main, not import and exit.
    repo = str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-m", cmd[2], "--version"], capture_output=True,
                         text=True, cwd=repo)
    assert out.returncode == 0 and out.stdout.strip(), (out.stdout, out.stderr)


def test_a_worker_never_imports_from_the_workspace(tmp_path, monkeypatch):
    # A planted stdlib shadow in an untrusted workspace.
    marker = tmp_path / "PWNED"
    (tmp_path / "argparse.py").write_text(f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(0)\n")
    seen = {}
    real_popen = subprocess.Popen

    class Proc:
        pid = 1
        def poll(self):
            return None

    def fake_popen(argv, **kw):
        seen.update(kw, argv=argv)
        # Run what the worker would import at startup, from the cwd and env given.
        entry = argv[:3] if argv[1] == "-m" else argv[:2]
        real_popen([*entry, "--version"], cwd=kw["cwd"], env=kw["env"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
        return Proc()

    monkeypatch.delenv("ONE_DETACHED_OWNER", raising=False)
    monkeypatch.setattr(tasks.subprocess, "Popen", fake_popen)
    tasks.TaskManager(execute=True, root=tmp_path / "state", workspace=str(tmp_path)).spawn(
        "probe", prompt="hello")
    assert seen["cwd"] != str(tmp_path)
    assert seen["env"].get("PYTHONSAFEPATH") == "1"
    assert "-w" in seen["argv"] and str(tmp_path) in seen["argv"]
    assert not marker.exists(), "a workspace argparse.py ran in the worker"

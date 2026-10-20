"""Run with the installed wheel's Python, outside the source checkout.

Only stdlib and one-core-cli are required. All model traffic targets an in-process
loopback fixture; harnesses._env provides the existing private child home.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import patch

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Build with python -m build, install the wheel into a fresh venv, "
               "then cd outside the checkout and run: "
               "/path/to/venv/bin/python -I /path/to/checkout/scripts/smoke-wheel.py")
    parser.parse_args(argv)
    # Help and usage errors work in a clean checkout, without an install. Do
    # not put the source tree on sys.path: that would defeat artifact acceptance.
    try:
        dist = importlib.metadata.distribution("one-core-cli")
        import onecli
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):
        parser.error("install the wheel into a fresh virtualenv and run this "
                     "script with that environment's Python; see --help")
    source = Path(__file__).resolve().parents[1]
    installed = Path(onecli.__file__).resolve()
    packaged = Path(dist.locate_file("onecli/__init__.py")).resolve()
    if not dist.read_text("WHEEL"):
        parser.error("one-core-cli is not an installed wheel distribution; build and "
                     "install the wheel into a fresh virtualenv")
    if installed != packaged:
        parser.error("onecli resolves to a checkout or editable install, not the "
                     "installed wheel; use a fresh virtualenv and python -I")
    if Path.cwd().resolve().is_relative_to(source):
        parser.error("run this artifact check from outside the checkout; see --help")
    run_smoke(onecli, dist)


def run_smoke(onecli, dist) -> None:
    from onecli import config, harnesses
    from onecli.evals import bench
    from onecli.evals import stress
    from onecli.serving import catalog
    from onecli.evals.stresslane import Handler

    class FixtureHandler(Handler):
        def m_stub_ok(self, body):
            if "First-response gate:" in json.dumps(body.get("messages", [])):
                self.reply(json.dumps({"decision": "answer", "message": "done."}))
            else:
                self.reply("done.")

    installed = Path(onecli.__file__).resolve()
    assert dist.version == onecli.__version__
    assert dist.metadata["Description-Content-Type"] == "text/markdown"
    assert catalog.load(), "missing model catalog"
    assert (installed.parent / "assets" / "glyph_terminal.mjs").is_file()
    scripts = {ep.name for ep in dist.entry_points if ep.group == "console_scripts"}
    assert "one-chat" in scripts
    for name in sorted(scripts):
        binary = Path(sys.executable).parent / name
        proc = subprocess.run([str(binary), "--help"], capture_output=True,
                              text=True, timeout=30)
        assert proc.returncode == 0, (name, proc.stdout, proc.stderr)
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with tempfile.TemporaryDirectory(prefix="one-wheel-smoke-") as scratch:
            root = Path(scratch)
            cfg = root / "config"
            providers = cfg / "providers"
            providers.mkdir(parents=True)
            url = f"http://127.0.0.1:{server.server_port}/v1"
            backend = config.Backend("fixture", "stub-ok", url, api_key="stub")
            (providers / "fixture.env").write_text(
                f"ONE_PROVIDER_MODEL=stub-ok\nOPENAI_BASE_URL={url}\n"
                "OPENAI_API_KEY=stub\n")
            workspace = root / "arena"
            workspace.mkdir()
            # Check both collision shapes using the installed distribution.
            # A harmless marker proves that no workspace module was imported.
            marker = workspace / "WORKSPACE_MODULE_EXECUTED"
            payload = f"from pathlib import Path; Path({str(marker)!r}).touch()\n"
            (workspace / "onecli.py").write_text(payload)
            cell = harnesses.Cell(backend, "", workspace, "Reply done.",
                                  cfg_dir=cfg, read_only=True, timeout=15)
            launch = harnesses._one_build(cell)
            # No inherited import path can hide an incomplete wheel.
            launch.env.pop("PYTHONPATH", None)
            launch.env["ONE_LLAMACPP_HOSTS"] = url
            proc = subprocess.run(launch.cmd, cwd=launch.cwd, env=launch.env,
                                  capture_output=True, text=True, timeout=30)
            result = harnesses._one_parse(proc.stdout, proc.stderr)
            assert not marker.exists(), "arena imported workspace onecli.py"
            assert proc.returncode == 0 and result.text == "done.", result
            rc, out, err, _ = stress.run_one(["--help"], workspace, 20)
            assert not marker.exists(), "stress imported workspace onecli.py"
            assert rc == 0 and "usage:" in out, (rc, out, err)

            package = workspace / "onecli"
            package.mkdir()
            (package / "__init__.py").write_text(payload)
            (package / "__main__.py").write_text(payload)
            proc = subprocess.run(launch.cmd, cwd=launch.cwd, env=launch.env,
                                  capture_output=True, text=True, timeout=30)
            result = harnesses._one_parse(proc.stdout, proc.stderr)
            assert not marker.exists(), "arena imported workspace onecli package"
            assert proc.returncode == 0 and result.text == "done.", result

            class FixtureTask(bench.Task):
                def check(self, workspace, answer):
                    return answer == "done.", answer

            task = FixtureTask("fixture", "Reply done.", read_only=True)
            with patch.object(bench, "cell_env", return_value=launch.env):
                row = bench.run_cell(backend, task, root / "bench", timeout=20)
            assert row["ok"], row

            with patch.dict(os.environ, launch.env, clear=True):
                rc, out, err, _ = stress.run_one(
                    ["--base-url", url, "--model-id", "stub-ok", "--json", "-q",
                     "--no-fallback", "--response-gate", "off", "--read-only",
                     "Reply done."], workspace, 20, providers=False)
            assert rc == 0 and json.loads(out)["text"] == "done.", (rc, out, err)
            assert not marker.exists(), "stress imported workspace onecli package"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    print(json.dumps({"version": dist.version, "commands": sorted(scripts),
                      "offline_cells": ["bench", "arena", "stress"], "ok": True}))


if __name__ == "__main__":
    main()

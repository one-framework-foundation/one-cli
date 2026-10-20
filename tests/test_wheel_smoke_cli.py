"""The artifact check must explain itself before importing its target package."""
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke-wheel.py"


def run_without_install(*args):
    # -I excludes source/PYTHONPATH; -S excludes installed site packages. This
    # simulates a clean checkout without depending on the test runner's venv.
    return subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), *args],
                          cwd=SCRIPT.parents[1], capture_output=True, text=True,
                          timeout=10)


def test_help_works_without_onecli_installed():
    proc = run_without_install("--help")
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout
    assert "installed wheel" in proc.stdout


def test_missing_install_explains_required_interpreter():
    proc = run_without_install()
    assert proc.returncode == 2
    assert "install the wheel" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_unknown_argument_is_not_silently_ignored():
    proc = run_without_install("--hepl")
    assert proc.returncode == 2
    assert "unrecognized arguments: --hepl" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_source_egg_info_cannot_masquerade_as_wheel(tmp_path):
    package = tmp_path / "onecli"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '0.1.0'\n")
    metadata = tmp_path / "one_core_cli-0.1.0.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: one-core-cli\nVersion: 0.1.0\n")
    code = (f"import sys, runpy; sys.path.insert(0, {str(tmp_path)!r}); "
            f"sys.argv = [{str(SCRIPT)!r}]; "
            f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')")
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                          cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 2, proc.stderr
    assert "wheel distribution" in proc.stderr
    assert "Traceback" not in proc.stderr

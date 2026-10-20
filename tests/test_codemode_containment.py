"""Code mode under file rules runs confined, or refuses - never unconfined."""

import subprocess
import sys

import pytest

from onecli import codemode, permissions
from onecli.sandbox import shell as tool_sandbox
from onecli import tools as toolkit


def _bwrap_works() -> bool:
    if tool_sandbox.unavailable():
        return False
    probe = subprocess.run(tool_sandbox.confinement_argv() + ["--", "/usr/bin/true"],
                           capture_output=True)
    return probe.returncode == 0


needs_bwrap = pytest.mark.skipif(not _bwrap_works(), reason="bwrap cannot run here")


def _answer(name, args):
    return (f"gated {name}", False)


def _program(ws):
    secret = str(ws / "secret.txt")
    return (f"try:\n    print('LEAK', open({secret!r}).read())\n"
            "except OSError:\n    print('no direct read')\n"
            f"try:\n    open({str(ws / 'pwn')!r}, 'w').write('x')\n    print('LEAK write')\n"
            "except OSError:\n    print('no direct write')\n"
            "print(one.read(path='secret.txt'))\n")


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "secret.txt").write_text("s3cret")
    return tmp_path


@needs_bwrap
def test_deny_rules_confine_the_program(ws):
    guards = permissions.build(str(ws), disallowed=["Read(secret.txt)", "Write(pwn)"])
    out = codemode.code(str(ws), _program(ws), timeout=30, call_tool=_answer,
                        path_guards=guards)
    assert "LEAK" not in out, out
    assert "no direct read" in out and "no direct write" in out
    assert "gated read" in out
    assert not (ws / "pwn").exists()


@needs_bwrap
def test_add_dir_roots_confine_the_program(ws, tmp_path_factory):
    extra = tmp_path_factory.mktemp("extra")
    guards = permissions.build(str(ws), add_dirs=[str(extra)])
    out = codemode.code(str(ws), _program(ws), timeout=30, call_tool=_answer,
                        path_guards=guards)
    assert "LEAK" not in out, out


@needs_bwrap
def test_toolkit_run_passes_the_guards_to_code_mode(ws):
    guards = permissions.build(str(ws), disallowed=["Read(secret.txt)"])
    out = toolkit.run("code", {"source": _program(ws), "timeout": 30}, str(ws),
                      call_tool=_answer, path_guards=permissions.path_guards(guards[0]))
    assert "LEAK" not in out, out


def test_refuses_with_a_reason_when_the_sandbox_is_missing(ws, monkeypatch):
    monkeypatch.setattr(tool_sandbox, "unavailable",
                        lambda: "required sandbox executable /usr/bin/bwrap unavailable")
    guards = permissions.build(str(ws), disallowed=["Read(secret.txt)"])
    out = codemode.code(str(ws), _program(ws), timeout=30, call_tool=_answer,
                        path_guards=guards)
    assert out.startswith("error: code mode refused")
    assert "file permission rules are active" in out
    assert "/usr/bin/bwrap" in out
    assert "LEAK" not in out


def test_no_rules_keeps_the_unconfined_program(ws):
    out = codemode.code(str(ws), _program(ws), timeout=30, call_tool=_answer,
                        path_guards=[])
    assert "LEAK s3cret" in out


def test_confined_uses_mounted_target_of_venv_python(ws, monkeypatch):
    python = ws / "venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    monkeypatch.setattr(sys, "executable", str(python))
    argv = codemode._confined(str(ws), "7", 10, "")
    assert argv[-3:] == [str(python.resolve()), "-u", "/one/program.py"]


@pytest.mark.parametrize("mode", ["readOnly", "plan"])
def test_read_only_modes_never_offer_code(mode):
    assert "code" not in permissions.tool_names_for(mode)

"""Global instruction selection, including the old implicit default."""

import pytest

from onecli import global_context, onboard
from onecli.serving import cli as serving


def test_doctor_names_implicit_source_and_missing_file(tmp_path):
    assert global_context.source(str(tmp_path)) == ("claude", False)
    lines = global_context.doctor_lines(str(tmp_path))
    assert lines[0].endswith("(default)")
    assert "one setup" in lines[1]
    assert "missing" in lines[2]


def test_doctor_names_explicit_source(tmp_path):
    path = tmp_path / ".one-cli" / "AGENTS.md"
    path.parent.mkdir()
    path.write_text("one instructions")
    global_context.set_source("one", str(tmp_path))
    assert global_context.doctor_lines(str(tmp_path)) == [
        f"global       {path} (chosen)"]


def test_doctor_command_prints_instruction_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(onboard, "doctor_report", lambda: ([], True))
    monkeypatch.setattr(serving.hostfacts, "collect", lambda: type("Facts", (), {
        "hostname": "fixture", "can_serve": lambda self: False,
    })())
    monkeypatch.setattr(serving.hostfacts, "describe", lambda facts: [])
    monkeypatch.setattr(serving.hostfacts, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(serving.serve, "running", lambda facts: [])
    monkeypatch.setattr(serving.catalog, "load", lambda: [])
    monkeypatch.setattr(serving.tune, "load_ledger", lambda: {})
    assert serving.cmd_doctor([]) == 0
    assert "(default)" in capsys.readouterr().out


def test_invalid_choice_is_visible_instead_of_silently_falling_back(tmp_path):
    path = global_context.choice_path(str(tmp_path))
    path.parent.mkdir()
    path.write_text("other\n")
    with pytest.raises(ValueError, match="must contain"):
        global_context.source(str(tmp_path))


def test_setup_records_choice_without_local_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(serving.menu, "_tty", lambda: True)
    monkeypatch.setattr(serving.menu, "choose", lambda title, options, **k: 1)
    monkeypatch.setattr(serving.hostfacts, "collect", lambda: type("Facts", (), {
        "can_serve": lambda self: False,
    })())
    monkeypatch.setattr(serving.hostfacts, "describe", lambda facts: [])
    assert serving.cmd_setup([]) == 0
    assert global_context.source(str(tmp_path)) == ("one", True)


def test_automatic_setup_never_changes_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(serving.menu, "choose",
                        lambda *a, **k: pytest.fail("menu opened"))
    monkeypatch.setattr(serving.hostfacts, "collect", lambda: type("Facts", (), {
        "can_serve": lambda self: False,
    })())
    monkeypatch.setattr(serving.hostfacts, "describe", lambda facts: [])
    assert serving.cmd_setup(["--yes"]) == 1
    assert not global_context.choice_path(str(tmp_path)).exists()

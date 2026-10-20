import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "update-config-docs.py"
SPEC = importlib.util.spec_from_file_location("update_config_docs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_configuration_page_is_current():
    """A new flag or environment variable lands with its line in the page."""
    assert MODULE.DOC.read_text() == MODULE.render()


def test_an_undescribed_environment_variable_fails(monkeypatch):
    env = dict(MODULE.ENV)
    env.pop("ONE_TTS_VOICE")
    monkeypatch.setattr(MODULE, "ENV", env)
    with pytest.raises(SystemExit, match="ONE_TTS_VOICE"):
        MODULE.render()


def test_every_flag_is_listed():
    from onecli.cli import build_parser
    page = MODULE.DOC.read_text()
    for action in build_parser()._actions:
        for option in action.option_strings:
            assert f"`{option}`" in page, option


def test_a_variable_read_through_a_named_constant_is_found():
    """`LEDGER_ENV = "ONE_INFER_LEDGER"` then `environ.get(LEDGER_ENV)` is a read."""
    reads = MODULE.env_reads()
    for name in ("ONE_INFER_CONFIG", "ONE_INFER_BUDGET_DIR", "ONE_INFER_LEDGER"):
        assert name in reads, name
        assert f"`{name}`" in MODULE.DOC.read_text(), name

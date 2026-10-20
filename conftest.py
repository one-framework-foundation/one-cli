"""Put the repo root on sys.path so `tests/` can import `onecli` uninstalled.

The CLI is run from a checkout (`scripts/one`), not from a wheel, so the tests
have to import the package the same way the entrypoint does.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _isolated_corpus(monkeypatch, tmp_path):
    """No test reads the operator's real corpus.

    `bench.suite("all")` includes every task on disk under ONE_CLI_CORPUS, so
    without this a green suite turns red the moment someone adds a task to
    their own machine - and a test asserting a task count would be asserting
    something about that machine. Tests that want a corpus set the variable
    themselves; this only guarantees the default is empty and disposable.
    """
    monkeypatch.setenv("ONE_CLI_CORPUS", str(tmp_path / "empty-corpus"))


@pytest.fixture(autouse=True)
def _isolate_parent_progress(monkeypatch):
    """Tests must not publish into the invoking detached job's signal file."""
    monkeypatch.delenv("BG_SIGNALS", raising=False)


@pytest.fixture(autouse=True)
def _no_update_check(monkeypatch):
    """No test asks PyPI for a release; tests/test_update.py opts back in."""
    monkeypatch.setenv("ONECLI_NO_UPDATE_CHECK", "1")


@pytest.fixture(autouse=True)
def _isolated_run_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv("ONE_CLI_ANALYSIS_DIR", str(tmp_path / "analysis"))

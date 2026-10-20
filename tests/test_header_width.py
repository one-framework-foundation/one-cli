"""A header that wraps mid-path reads as a broken table.

Seen in a screenshot of `--orc-report` at 96 columns: the run directory was
printed absolute, the header ran past the width the rest of the output is
capped at, and the terminal folded it - so the second line of the report was
`i/runs`. Nothing about that says "long path"; it says "the table is broken".

`cli` had already solved this for the banner's workspace and kept the
shortener to itself, so `orc` had no way to reach it. It lives in `render` now.
"""

from __future__ import annotations

import os
import re

import pytest

from onecli import orc as orclib, render


ANSI = re.compile(r"\033\[[0-9;]*m")


def _plain(text: str) -> str:
    return ANSI.sub("", text)


def test_short_path_is_a_tilde_under_home():
    home = os.path.expanduser("~")
    assert render.short_path(f"{home}/.one-cli/runs") == "~/.one-cli/runs"


def test_short_path_leaves_a_path_outside_home_alone():
    assert render.short_path("/var/lib/thing") == "/var/lib/thing"


def _record(directory, sid, backend):
    import json
    (directory / f"{sid}.json").write_text(json.dumps({
        "session": sid, "task_class": "build",
        "nodes": [{"backend": backend, "model": "m", "seconds": 1.0,
                   "steps": 2, "stopped": "answered"}]}))


@pytest.fixture
def deep_run_dir(tmp_path, monkeypatch):
    """A run dir under HOME, long enough that the absolute form overruns."""
    home = tmp_path / "home"
    deep = home / ".one-cli" / "runs"
    deep.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(orclib, "RUN_DIR", deep)
    return deep


def test_the_report_header_fits_the_width_it_is_capped_at(deep_run_dir):
    _record(deep_run_dir, "s1", "alpha")
    header = _plain(orclib.report(orclib.records()[0]).splitlines()[0])
    assert "~/.one-cli/runs" in header
    assert str(deep_run_dir) not in header
    assert len(header) <= render.width(), header


def test_the_empty_store_sentence_is_shortened_too(deep_run_dir):
    line = _plain(orclib.report([]))
    assert "~/.one-cli/runs" in line
    assert str(deep_run_dir) not in line

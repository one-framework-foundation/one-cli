"""The stress tier's own contract - mostly: it must not lie about coverage.

A stress case that passes because its feature is absent is worse than a red
one: it reports coverage that does not exist. The first version of this tier
scored four green containment cells while `@` expansion was not running at
all, so the accounting is what is pinned here.
"""

from __future__ import annotations

import json

from onecli.evals import stress


def test_every_case_is_registered_once():
    names = [fn.__name__ for fn in stress.CASES]
    assert len(names) == len(set(names))
    assert len(names) >= 20


def test_a_skip_is_not_counted_as_a_pass():
    out = stress.skip("mention_flood", "this tree has no onecli.mentions")
    assert out.skipped
    # `ok` stays true so a skip never reddens a run, which is exactly why the
    # summary must count on `skipped`, not on `ok`.
    assert out.ok


def test_the_summary_separates_skipped_from_passed(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(stress, "SCRATCH", tmp_path)
    monkeypatch.setattr(stress, "LANE_LOG", tmp_path / "requests.jsonl")
    monkeypatch.setattr(stress, "write_providers", lambda: None)
    monkeypatch.setattr(stress, "serve", lambda: type("P", (), {"kill": lambda s: None})())
    monkeypatch.setattr(stress, "_run_cases", lambda only: [
        stress.Outcome("a", True, "fine", 0.1),
        stress.skip("b", "absent"),
    ])
    assert stress.main([]) == 0
    assert "1/1 stress cases passed, 1 skipped" in capsys.readouterr().out


def test_a_real_failure_still_exits_nonzero(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(stress, "SCRATCH", tmp_path)
    monkeypatch.setattr(stress, "write_providers", lambda: None)
    monkeypatch.setattr(stress, "serve", lambda: type("P", (), {"kill": lambda s: None})())
    monkeypatch.setattr(stress, "_run_cases", lambda only: [
        stress.Outcome("a", False, "broke", 0.1), stress.skip("b", "absent")])
    assert stress.main([]) == 1


def test_sent_text_reads_both_content_shapes():
    """A body carries content as a string or as a list of blocks; a reader that
    handles one of them measures half the request."""
    bodies = [{"body": {"messages": [
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [{"type": "text", "text": "blocked"},
                                     {"type": "image", "url": "x"}]},
    ]}}]
    sent = stress.sent_text(bodies)
    assert "plain" in sent and "blocked" in sent


def test_results_are_written_as_json(monkeypatch, tmp_path):
    monkeypatch.setattr(stress, "SCRATCH", tmp_path)
    monkeypatch.setattr(stress, "write_providers", lambda: None)
    monkeypatch.setattr(stress, "serve", lambda: type("P", (), {"kill": lambda s: None})())
    monkeypatch.setattr(stress, "_run_cases", lambda only: [
        stress.Outcome("a", True, "fine", 0.1, extra={"sent_chars": 12})])
    stress.main([])
    rows = json.loads((tmp_path / "results.json").read_text())
    assert rows[0]["name"] == "a" and rows[0]["extra"]["sent_chars"] == 12


def test_help_is_help_and_not_an_empty_run(capsys):
    """`--help` used to be read as a case name.

    Every token in argv was a filter, so `one-stress --help` matched no case,
    started the lane server, and printed "0/0 stress cases passed" with exit 0 -
    a clean bill of health from a suite that ran nothing.
    """
    import pytest as _pytest
    with _pytest.raises(SystemExit) as exc:
        stress.main(["--help"])
    assert exc.value.code == 0
    assert "usage: one-stress" in capsys.readouterr().out


def test_a_filter_that_matches_nothing_is_an_error(capsys):
    assert stress.main(["nosuchcase"]) == 2
    assert "unknown case" in capsys.readouterr().err


def test_zero_cases_run_is_never_a_pass(monkeypatch, capsys):
    """0/0 is the shape of a suite that did not run, not of one that passed."""
    monkeypatch.setattr(stress, "write_providers", lambda: None)
    monkeypatch.setattr(stress, "serve", lambda: type("L", (), {"kill": lambda s: None})())
    monkeypatch.setattr(stress, "_run_cases", lambda only: [])
    assert stress.main([]) == 2
    assert "no stress case ran" in capsys.readouterr().err


def test_list_names_every_case(capsys):
    assert stress.main(["--list"]) == 0
    out = capsys.readouterr().out.split()
    assert [fn.__name__.removeprefix("c_") for fn in stress.CASES] == out

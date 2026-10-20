"""The parity table has to stay true, and only part of that is checkable.

What runs everywhere: the table is well formed, and every one-cli flag it claims
really exists in our parser. What runs only where Claude Code is installed: the
flag-level drift check against its live `--help`. A host without claude skips
that half rather than passing it, because a check that cannot look must not
report clean.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from onecli.evals import parity


def test_table_is_well_formed():
    ids = [f.id for f in parity.FEATURES]
    assert len(ids) == len(set(ids)), "duplicate feature id"
    for f in parity.FEATURES:
        assert f.status in parity.STATUSES, f"{f.id}: bad status {f.status}"
        assert f.area and f.claude, f"{f.id}: incomplete row"
        if f.status in ("shipped", "partial"):
            assert f.ours, f"{f.id}: claims {f.status} but names nothing of ours"
        if f.status in ("gap", "declined"):
            assert not f.ours, f"{f.id}: a {f.status} cannot have an answer here"
        if f.status in ("gap", "declined"):
            assert not f.our_flags, f"{f.id}: a {f.status} cannot claim our flags"


def test_every_flag_we_claim_exists_in_our_parser():
    """The half of the check that needs no Claude Code installed."""
    assert parity.drift().stale_ours == []


def test_claude_flags_is_none_when_the_binary_is_absent():
    assert parity.claude_flags("no-such-binary-anywhere") is None


def test_a_missing_claude_does_not_report_clean(monkeypatch):
    monkeypatch.setattr(parity, "claude_flags", lambda binary="claude": None)
    d = parity.drift()
    assert d.checked_claude is False
    assert "not installed" in parity.render(d)


def test_drift_names_a_flag_the_table_does_not_account_for(monkeypatch):
    monkeypatch.setattr(parity, "claude_flags",
                        lambda binary="claude": {"--model", "--brand-new-thing"})
    d = parity.drift()
    assert d.undeclared == ["--brand-new-thing"]
    assert not d.clean
    assert "undeclared claude flag: --brand-new-thing" in parity.render(d)


def test_drift_names_a_flag_claude_has_removed(monkeypatch):
    monkeypatch.setattr(parity, "claude_flags", lambda binary="claude": {"--model"})
    d = parity.drift()
    assert "--mcp-config" in d.stale_claude
    assert not d.clean


def test_strict_exits_nonzero_only_on_drift(monkeypatch, capsys):
    monkeypatch.setattr(parity, "claude_flags",
                        lambda binary="claude": {"--surprise"})
    assert parity.main(["--strict"]) == 1
    monkeypatch.setattr(parity, "drift",
                        lambda binary="claude": parity.Drift())
    assert parity.main(["--strict"]) == 0


def test_json_output_is_machine_readable(capsys):
    parity.main(["--json", "--status", "gap"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["features"]
    assert {f["status"] for f in payload["features"]} == {"gap"}


def test_a_row_that_claims_a_feature_names_flags_we_really_have():
    """The status is a claim; this is the cheapest way to keep it honest.

    Checked on `mcp` in particular because it is the row this campaign moved:
    calling it shipped while `--mcp-config` does not parse would be exactly the
    lie the table exists to prevent.
    """
    ours = parity.onecli_flags()
    row = next(f for f in parity.FEATURES if f.id == "mcp")
    assert row.status in parity.STATUSES
    if row.status in ("shipped", "partial"):
        assert row.our_flags, "a feature we claim must name its surface"
    for f in parity.FEATURES:
        for flag in f.our_flags:
            assert flag in ours, f"{f.id} claims {flag}, the parser has no such flag"


@pytest.mark.skipif(parity.claude_flags() is None,
                    reason="Claude Code is not installed on this host")
def test_no_undeclared_claude_flags_on_a_host_that_has_claude():
    d = parity.drift()
    assert d.undeclared == [], f"claude grew flags the table does not mention: {d.undeclared}"
    assert d.stale_claude == [], f"table names flags claude no longer has: {d.stale_claude}"


def test_the_script_runs():
    out = subprocess.run([sys.executable, "scripts/one-parity"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "one-cli parity with Claude Code" in out.stdout


def test_help_capture_reads_complete_output(monkeypatch):
    from types import SimpleNamespace
    text = '  --first-flag x\n' + ('description ' * 1500) + '\n  --last-flag y\n'
    monkeypatch.setattr(parity.shutil, 'which', lambda binary: '/fake/claude')
    def run(argv, **kwargs):
        stream = kwargs.get('stdout')
        if hasattr(stream, 'write'):
            stream.write(text)
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(stdout=text[:8190], returncode=0)
    monkeypatch.setattr(parity.subprocess, 'run', run)
    assert parity.claude_flags() == {'--first-flag', '--last-flag'}

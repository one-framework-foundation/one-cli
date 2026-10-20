"""onecli.infer.ledger - the shared attempt ledger and legacy compatibility."""
# tone-exempt: test fixtures only; every ledger here lives in pytest's tmp_path.

import json

import pytest

from onecli.infer import ledger as L


def test_row_shape_and_validation(tmp_path):
    row = L.make_row(caller="host:host-a", lane="a", outcome="ok", ms=5)
    assert row["v"] == L.SCHEMA_VERSION and row["source"] == "infer"
    assert set(row) == set(L.FIELDS)
    with pytest.raises(ValueError):
        L.make_row(outcome="ok", prompt="secret text")
    with pytest.raises(ValueError):
        L.make_row(outcome="exploded")


def test_append_and_rollup(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(L.make_row(caller="c1", outcome="ok", tokens_in=10, tokens_out=2, usd=0.5), p)
    L.append(L.make_row(caller="c1", outcome="lane_fault", reserved_usd=0.25), p)
    L.append(L.make_row(caller="c2", outcome="refused"), p)
    r = L.rollup(L.read_rows(p))
    assert r["c1"]["attempts"] == 2 and r["c1"]["ok"] == 1 and r["c1"]["lane_fault"] == 1
    assert r["c1"]["tokens_in"] == 10 and r["c1"]["usd"] == 0.5
    assert r["c1"]["reserved_usd"] == 0.25
    assert r["c2"]["refused"] == 1


def test_unwritable_ledger_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert L.append(L.make_row(outcome="ok"), blocker / "sub" / "l.jsonl") is False








def test_read_rows_converts_another_format_on_the_way(tmp_path):
    p = tmp_path / "other.jsonl"
    p.write_text(json.dumps({"kind": "attempt", "ok": True}) + "\n"
                 + json.dumps({"kind": "note"}) + "\n")
    def convert(raw):
        return {"outcome": "ok"} if raw.get("kind") == "attempt" else None
    assert list(L.read_rows(p, convert=convert)) == [{"outcome": "ok"}]

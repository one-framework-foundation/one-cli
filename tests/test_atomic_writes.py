"""Persistent state is replaced atomically: a crash keeps the old file."""
import json
import os
from types import SimpleNamespace

import pytest

from onecli import jobs, orc, secureio, workgate
from onecli.serving import catalog, serve, tune


def _crash(*_a, **_k):
    raise OSError("simulated crash before rename")


def _leftovers(directory, keep):
    return sorted(p.name for p in directory.iterdir() if p.name not in keep)


def test_helper_replaces_and_cleans_up_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    secureio.atomic_write_text(target, "old")
    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError):
        secureio.atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert _leftovers(tmp_path, {"state.json"}) == []


def test_helper_exclusive_refuses_existing(tmp_path):
    target = tmp_path / "state.json"
    secureio.atomic_write_text(target, "a", exclusive=True)
    with pytest.raises(FileExistsError):
        secureio.atomic_write_text(target, "b", exclusive=True)
    assert target.read_text() == "a"
    assert _leftovers(tmp_path, {"state.json"}) == []


def test_jobs_save_keeps_old_record(tmp_path, monkeypatch):
    store = jobs.JobStore("s1", root=tmp_path)
    store.root.mkdir(parents=True)
    job = jobs.Job(id="j1", command="old", log=store.root / "j1.log", started=1.0)
    store._save(job)
    rec = store.root / "j1.json"
    old = rec.read_text()
    job.command = "new"
    monkeypatch.setattr(os, "replace", _crash)
    store._save(job)  # never fatal
    assert rec.read_text() == old
    assert _leftovers(store.root, {"j1.json"}) == []


def test_orc_write_keeps_old_record(tmp_path, monkeypatch):
    monkeypatch.setattr(orc, "RUN_DIR", tmp_path)
    path = orc.write(orc.Record(session="abc", seconds=1.0))
    old = path.read_text()
    monkeypatch.setattr(os, "replace", _crash)
    orc.write(orc.Record(session="abc", seconds=2.0))
    assert path.read_text() == old
    assert _leftovers(tmp_path, {"abc.json"}) == []


def test_add_user_model_keeps_old_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "USER_CATALOG", tmp_path / "models.json")
    catalog.add_user_model(catalog.Model(name="a", repo="r/a", file="a.gguf"))
    old = catalog.USER_CATALOG.read_text()
    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError):
        catalog.add_user_model(catalog.Model(name="b", repo="r/b", file="b.gguf"))
    assert catalog.USER_CATALOG.read_text() == old
    assert _leftovers(tmp_path, {"models.json"}) == []


def test_tune_record_keeps_old_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "LEDGER", tmp_path / "tuning.json")
    monkeypatch.setattr(tune, "other_lanes", lambda facts: 0)
    tune.LEDGER.write_text(json.dumps({"schema": 1, "entries": {"x": {}}}))
    old = tune.LEDGER.read_text()
    facts = SimpleNamespace(hostname="h", gpu=SimpleNamespace(name="g"),
                            runner=lambda: "r")
    res = SimpleNamespace(recipe="baseline", tok_s=10.0, acceptance=0.0,
                          flags=[], error="", ok=lambda: True)
    model = catalog.Model(name="m", repo="r/m", file="m.gguf")
    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError):
        tune.record(model, facts, [res])
    assert tune.LEDGER.read_text() == old
    assert _leftovers(tmp_path, {"tuning.json"}) == []


def test_serve_register_keeps_old_dropin(tmp_path, monkeypatch):
    model = catalog.Model(name="m", repo="r/m", file="m.gguf")
    dropin = tmp_path / "local-m.env"
    monkeypatch.setattr(serve, "dropin_path", lambda m: dropin)
    serve.register(model, 8080)
    old = dropin.read_text()
    assert dropin.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError):
        serve.register(model, 9999)
    assert dropin.read_text() == old
    assert _leftovers(tmp_path, {"local-m.env"}) == []


def _started(tmp_path):
    path = tmp_path / "wg.json"
    workgate.start(path, workgate.load_policy("experimental-code@1"),
                   goal="g", expected="e", stop="s")
    return path


def test_workgate_check_keeps_old_record(tmp_path, monkeypatch):
    path = _started(tmp_path)
    old = path.read_text()
    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError):
        workgate.check(path, "phase_completed", {"meta_rounds": 1, "private_mocks": 0,
                                                 "product_delta": True})
    assert path.read_text() == old
    assert _leftovers(tmp_path, {"wg.json", "wg.json.lock"}) == []


def test_workgate_start_does_not_clobber(tmp_path):
    path = _started(tmp_path)
    old = path.read_text()
    with pytest.raises(ValueError, match="already exists"):
        workgate.start(path, workgate.load_policy("experimental-code@1"),
                       goal="x", expected="e", stop="s")
    assert path.read_text() == old


def test_workgate_concurrent_checks_lose_nothing(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = _started(tmp_path)
    ev = {"meta_rounds": 1, "private_mocks": 0, "product_delta": True}
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: workgate.check(path, "phase_completed", ev), range(24)))
    assert len(json.loads(path.read_text())["checks"]) == 24


@pytest.mark.parametrize("op", ["eq", "lte", "in"])
def test_workgate_valueless_policy_is_a_clean_error(tmp_path, op):
    policy = {"name": "p", "version": 1, "events": {"finish": [
        {"field": "f", "op": op, "severity": "block", "message": "m"}]}}
    src = tmp_path / "p.json"
    src.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="no value"):
        workgate.load_policy(str(src))
    with pytest.raises(ValueError, match="no value"):
        workgate.evaluate(policy, "finish", {"f": 1})
    assert workgate.main(["start", "--state", str(tmp_path / "s.json"),
                          "--policy", str(src), "--goal", "g", "--expected", "e",
                          "--stop", "s"]) == 2


def test_atomic_write_follows_a_symlinked_state_dir(tmp_path):
    from onecli import secureio
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    secureio.atomic_write_text(tmp_path / "link" / "state.json", "{}")
    assert (real / "state.json").read_text() == "{}"


def test_atomic_write_keeps_the_existing_mode(tmp_path):
    import os
    import stat
    from onecli import secureio
    target = tmp_path / "ledger.json"
    target.write_text("old")
    os.chmod(target, 0o640)
    secureio.atomic_write_text(target, "new")
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o640
    assert target.read_text() == "new"

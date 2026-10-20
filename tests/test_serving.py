"""Serving stack: catalog, recipes, tuning ledger, puller, lane detection.

No GPU and no network. Everything that would touch either is passed a fake,
because the point of these is the decision logic - which recipe is offered,
which one is recorded as fastest, when a measurement stops applying.
"""

from __future__ import annotations

import hashlib

import pytest

from onecli.serving import catalog, hostfacts, pull, recipes, serve, tune


@pytest.fixture
def facts():
    return hostfacts.HostFacts(hostname="testbox", system="Linux",
                               arch="aarch64", cpus=20, ram_mb=122880,
                               gpu=hostfacts.Gpu(name="NVIDIA GB10"),
                               unified_memory=True, docker=True,
                               docker_image="img:pinned")


@pytest.fixture
def model(tmp_path):
    return catalog.Model(
        name="testmodel", repo="org/Test-GGUF", file="test-Q4_K_M.gguf",
        size=100, sha256="", quant="Q4_K_M", context=4096, alias="testmodel",
        draft=catalog.Artifact(file="mtp-test-Q4_0.gguf", size=10,
                               spec_type="draft-mtp"))


# ------------------------------------------------------------------ catalog

def test_curated_catalog_loads_and_every_entry_is_servable():
    models = catalog.load()
    assert models, "the packaged catalog is empty"
    for m in models:
        assert m.file.endswith(".gguf")
        assert m.repo and "/" in m.repo
        assert m.alias, f"{m.name} has no alias to serve under"


def test_find_resolves_name_alias_and_unambiguous_prefix():
    assert catalog.find("gpt-oss-20b").name == "gpt-oss-20b"
    assert catalog.find("kat-coder").name == "kat-coder-v2.5"
    assert catalog.find("nope-does-not-exist") is None


def test_present_rejects_a_truncated_file(tmp_path, model):
    (tmp_path / model.file).write_bytes(b"x" * 99)
    assert not model.present(tmp_path), "a short file must not count as present"
    (tmp_path / model.file).write_bytes(b"x" * 100)
    assert model.present(tmp_path)


def test_from_repo_skips_shards_and_links_the_drafter(monkeypatch):
    monkeypatch.setattr(catalog, "repo_files", lambda repo: [
        {"file": "big-00001-of-00003.gguf", "size": 9, "sha256": "a"},
        {"file": "thing-Q4_K_M.gguf", "size": 5, "sha256": "b"},
        {"file": "thing-Q8_0.gguf", "size": 8, "sha256": "c"},
        {"file": "mtp-thing-Q4_0.gguf", "size": 1, "sha256": "d"},
        {"file": "mmproj-thing.gguf", "size": 2, "sha256": "e"},
    ])
    m = catalog.from_repo("org/thing")
    assert m.file == "thing-Q4_K_M.gguf", "Q4_K_M is the preferred pick"
    assert m.draft and m.draft.spec_type == "draft-mtp"
    assert m.mmproj and m.mmproj.file == "mmproj-thing.gguf"
    assert m.curated is False


def test_from_repo_falls_back_to_largest_when_no_q4(monkeypatch):
    monkeypatch.setattr(catalog, "repo_files", lambda repo: [
        {"file": "thing-Q8_0.gguf", "size": 8, "sha256": "c"},
        {"file": "thing-Q6_K.gguf", "size": 6, "sha256": "d"},
    ])
    assert catalog.from_repo("org/thing").file == "thing-Q8_0.gguf"


# ------------------------------------------------------------------ recipes

SPEC = ["draft-simple", "draft-mtp", "ngram-mod", "ngram-simple"]


def test_no_drafter_on_disk_means_no_draft_recipe(tmp_path, model, facts):
    (tmp_path / model.file).write_bytes(b"x" * 100)
    names = [r.name for r in recipes.candidates(model, facts, spec_types=SPEC,
                                                root=tmp_path)]
    assert names == ["baseline", "ngram-mod"]
    why = dict(recipes.skipped(model, facts, spec_types=SPEC, root=tmp_path))
    assert "not on disk" in why["draft-mtp"]


def test_drafter_on_disk_unlocks_the_draft_recipe(tmp_path, model, facts):
    (tmp_path / model.file).write_bytes(b"x" * 100)
    (tmp_path / model.draft.file).write_bytes(b"y" * 10)
    cands = recipes.candidates(model, facts, spec_types=SPEC, root=tmp_path)
    mtp = next(r for r in cands if r.name == "draft-mtp")
    assert "-md" in mtp.flags and "--spec-type" in mtp.flags


def test_a_runner_without_the_spec_type_never_offers_it(tmp_path, model, facts):
    (tmp_path / model.file).write_bytes(b"x" * 100)
    (tmp_path / model.draft.file).write_bytes(b"y" * 10)
    names = [r.name for r in recipes.candidates(model, facts,
                                                spec_types=["ngram-mod"],
                                                root=tmp_path)]
    assert "draft-mtp" not in names
    why = dict(recipes.skipped(model, facts, spec_types=["ngram-mod"],
                               root=tmp_path))
    assert "does not list draft-mtp" in why["draft-mtp"]


def test_command_maps_the_weights_dir_into_the_container(tmp_path, model, facts):
    (tmp_path / model.file).write_bytes(b"x" * 100)
    (tmp_path / model.draft.file).write_bytes(b"y" * 10)
    r = next(r for r in recipes.candidates(model, facts, spec_types=SPEC,
                                           root=tmp_path) if r.name == "draft-mtp")
    cmd = recipes.command(model, r, 9000, facts, root=tmp_path)
    assert cmd[0] == "docker"
    assert f"{tmp_path}:/models:ro" in cmd
    assert "/models/test-Q4_K_M.gguf" in cmd
    assert "/models/mtp-test-Q4_0.gguf" in cmd
    assert str(tmp_path) not in " ".join(cmd[cmd.index("img:pinned"):]), \
        "a host path leaking past the mount would not resolve in the container"


def test_native_runner_keeps_host_paths(tmp_path, model, facts):
    facts.llama_server = "/usr/bin/llama-server"
    (tmp_path / model.file).write_bytes(b"x" * 100)
    cmd = recipes.command(model, recipes.Recipe("baseline"), 9000, facts,
                          root=tmp_path)
    assert cmd[0] == "/usr/bin/llama-server"
    assert str(tmp_path / model.file) in cmd


# ------------------------------------------------------------------ ledger

@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "tuning.json"
    monkeypatch.setattr(tune, "LEDGER", path)
    return path


def _results():
    return [
        tune.Result(recipe="baseline", flags=[], tok_s=11.3),
        tune.Result(recipe="draft-mtp", flags=["--spec-type", "draft-mtp"],
                    tok_s=33.4, acceptance=0.87),
        tune.Result(recipe="ngram-mod", flags=["--spec-type", "ngram-mod"],
                    tok_s=12.0),
    ]


def test_record_picks_the_fastest_and_states_the_speedup(model, facts, ledger):
    entry = tune.record(model, facts, _results())
    assert entry["winner"] == "draft-mtp"
    assert entry["speedup"] == 2.96
    assert entry["baseline_tok_s"] == 11.3
    assert len(entry["candidates"]) == 3


def test_a_failed_candidate_never_wins(model, facts, ledger):
    res = _results()
    res[1].error = "server never reported healthy"
    res[1].tok_s = 0.0
    entry = tune.record(model, facts, res)
    assert entry["winner"] == "ngram-mod"


def test_record_refuses_when_nothing_measured(model, facts, ledger):
    dead = [tune.Result(recipe="baseline", error="boom")]
    with pytest.raises(RuntimeError):
        tune.record(model, facts, dead)


def test_recipe_for_replays_the_measured_winner(model, facts, ledger):
    tune.record(model, facts, _results())
    recipe, entry = tune.recipe_for(model, facts)
    assert recipe.name == "draft-mtp"
    assert recipe.flags == ["--spec-type", "draft-mtp"]
    assert entry["tok_s"] == 33.4


def test_a_changed_runner_invalidates_the_measurement(model, facts, ledger):
    tune.record(model, facts, _results())
    facts.docker_image = "img:rebuilt"
    entry = tune.best(model, facts)
    assert entry["stale"], "a number measured on another build must be flagged"
    recipe, _ = tune.recipe_for(model, facts)
    assert recipe.name == "baseline", "a stale entry must not be replayed"


def test_an_unmeasured_model_falls_back_to_baseline(model, facts, ledger):
    recipe, entry = tune.recipe_for(model, facts)
    assert recipe.name == "baseline"
    assert entry is None
    assert "no verified measurement" in recipe.why


def test_the_ledger_is_keyed_per_host(model, facts, ledger):
    tune.record(model, facts, _results())
    other = hostfacts.HostFacts(hostname="otherbox", docker=True,
                                docker_image="img:pinned")
    assert tune.best(model, other) is None, \
        "one host's measurement is not evidence about another host"


def test_measure_is_token_weighted(monkeypatch):
    replies = [
        {"timings": {"predicted_n": 10, "predicted_ms": 1000.0,
                     "predicted_per_second": 10.0}},
        {"timings": {"predicted_n": 90, "predicted_ms": 1000.0,
                     "predicted_per_second": 90.0,
                     "draft_n": 100, "draft_n_accepted": 80}},
    ]
    monkeypatch.setattr(tune, "_post", lambda *a, **k: replies.pop(0))
    res = tune.measure("http://x", "m", tasks=[{"name": "a", "prompt": "p"},
                                               {"name": "b", "prompt": "p"}],
                       warmup=False)
    assert res.tok_s == 50.0, "100 tokens over 2s, not the mean of 10 and 90"
    assert res.acceptance == 0.8


# ------------------------------------------------------------------ pull

def test_verify_catches_a_wrong_hash(tmp_path):
    f = tmp_path / "a.gguf"
    f.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    assert pull.verify(f, 5, good)[0]
    ok, why = pull.verify(f, 5, "0" * 64)
    assert not ok and "sha256" in why


def test_verify_reports_when_a_source_publishes_no_hash(tmp_path):
    f = tmp_path / "a.gguf"
    f.write_bytes(b"hello")
    ok, why = pull.verify(f, 5, "")
    assert ok and "no hash" in why


def test_plan_marks_what_is_already_there(tmp_path, model):
    (tmp_path / model.file).write_bytes(b"x" * 100)
    got = {a.file: ok for a, _d, ok in pull.plan(model, tmp_path)}
    assert got[model.file] is True
    assert got[model.draft.file] is False


def test_plan_can_leave_the_drafter_out(tmp_path, model):
    files = [a.file for a, _d, _ok in pull.plan(model, tmp_path, with_draft=False)]
    assert model.draft.file not in files


def test_pull_refuses_when_the_disk_cannot_hold_it(tmp_path, model, monkeypatch):
    monkeypatch.setattr(pull, "free_bytes", lambda root: 10)
    with pytest.raises(pull.PullError) as exc:
        pull.pull(model, tmp_path)
    assert "cannot finish" in str(exc.value)


# ------------------------------------------------------------------ lanes

def test_port_is_read_from_the_host_side_of_the_map():
    assert serve._port_of("0.0.0.0:9912->9912/tcp, [::]:9912->9912/tcp") == 9912
    assert serve._port_of("0.0.0.0:4534->9180/tcp") == 4534
    assert serve._port_of("6379/tcp") is None


def test_only_containers_serving_weights_count_as_lanes(monkeypatch, facts):
    rows = ("chat-service\t0.0.0.0:8081->8081/tcp\tUp 3 days\n"
            "one-lane-qwen\t0.0.0.0:9912->9912/tcp\tUp 2 hours\n")
    monkeypatch.setattr(serve.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": rows,
                                                       "returncode": 0})())
    monkeypatch.setattr(serve, "_inspect_args", lambda name:
                        ["-m", "/models/q.gguf", "--alias", "qwen",
                         "--spec-type", "draft-mtp"]
                        if name == "one-lane-qwen" else ["--local", "8081"])
    lanes = serve.running(facts)
    assert [lane.model for lane in lanes] == ["qwen"]
    assert lanes[0].recipe == "draft-mtp"


def test_register_writes_a_discoverable_dropin(tmp_path, model, monkeypatch):
    monkeypatch.setattr(serve.onecfg, "PROVIDER_DIR", tmp_path)
    path = serve.register(model, 9912)
    body = path.read_text()
    assert "OPENAI_BASE_URL=http://127.0.0.1:9912/v1" in body
    assert "ONE_PROVIDER_MODEL=testmodel" in body
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_plan_can_be_forced_onto_a_named_recipe(tmp_path, model, facts, ledger):
    (tmp_path / model.file).write_bytes(b"x" * 100)
    (tmp_path / model.draft.file).write_bytes(b"y" * 10)
    monkeypatch_spec = ["draft-mtp", "ngram-mod"]
    import onecli.serving.recipes as R
    orig = R.supported_spec_types
    R.supported_spec_types = lambda f=None: monkeypatch_spec
    try:
        cmd, recipe, _ = serve.plan(model, 9000, facts, root=tmp_path,
                                    force_recipe="ngram-mod")
        assert recipe.name == "ngram-mod"
        with pytest.raises(RuntimeError):
            serve.plan(model, 9000, facts, root=tmp_path,
                       force_recipe="draft-eagle3")
    finally:
        R.supported_spec_types = orig


def test_a_contended_sweep_is_recorded_as_contended(model, facts, ledger):
    entry = tune.record(model, facts, _results(), concurrent_lanes=3)
    assert entry["concurrent_lanes"] == 3
    assert "busy machine" in entry["contended"]
    assert entry["winner"] == "draft-mtp", \
        "contention slows every candidate equally; it must not change the pick"


def test_an_idle_sweep_carries_no_contention_note(model, facts, ledger):
    entry = tune.record(model, facts, _results())
    assert entry["concurrent_lanes"] == 0
    assert "contended" not in entry


def test_a_model_with_no_drafter_reports_one_skip_not_five(tmp_path, facts):
    m = catalog.Model(name="plain", repo="o/r", file="plain.gguf", size=1)
    (tmp_path / m.file).write_bytes(b"x")
    skips = recipes.skipped(m, facts, spec_types=SPEC, root=tmp_path)
    assert [n for n, _ in skips] == ["draft-*"]


def test_a_lane_bound_to_the_port_under_another_name_is_found(monkeypatch, facts):
    rows = "llama-chat\t0.0.0.0:9912->9912/tcp\tUp 10 hours\n"
    monkeypatch.setattr(serve.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": rows,
                                                       "returncode": 0})())
    monkeypatch.setattr(serve, "_inspect_args", lambda name:
                        ["-m", "/models/q.gguf", "--alias", "chat-alias"])
    assert serve.occupants(9912, facts) == ["llama-chat"]
    assert serve.occupants(9913, facts) == []


def _lanes(monkeypatch, rows, args_by_name):
    monkeypatch.setattr(serve.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": rows,
                                                       "returncode": 0})())
    monkeypatch.setattr(serve, "_inspect_args",
                        lambda name: args_by_name.get(name, []))


def test_stop_removes_the_container_the_runtime_reports_not_a_rebuilt_name(
        monkeypatch, model, facts, tmp_path):
    """A lane started outside `one serve` is named by its launcher."""
    monkeypatch.setattr(serve.onecfg, "PROVIDER_DIR", tmp_path)
    rows = "llama-server\t0.0.0.0:8911->8911/tcp\tUp 4 days\n"
    _lanes(monkeypatch, rows,
           {"llama-server": ["-m", "/models/t.gguf", "--alias", "testmodel"]})
    killed = []
    monkeypatch.setattr(serve.subprocess, "run", lambda cmd, **k: (
        killed.append(cmd) or type("R", (), {"returncode": 0, "stdout": rows})())
        if cmd[:3] == ["docker", "rm", "-f"] else
        type("R", (), {"returncode": 0, "stdout": rows})())
    assert serve.stop(model, facts) is True
    assert killed == [["docker", "rm", "-f", "llama-server"]]


def test_stop_reports_false_when_no_lane_serves_the_model(
        monkeypatch, model, facts, tmp_path):
    """`docker rm -f` exits 0 on a missing container - that is not a stop."""
    monkeypatch.setattr(serve.onecfg, "PROVIDER_DIR", tmp_path)
    dropin = serve.dropin_path(model)
    dropin.write_text("OPENAI_BASE_URL=http://127.0.0.1:8911/v1\n")
    _lanes(monkeypatch, "", {})
    assert serve.stop(model, facts) is False
    assert dropin.is_file(), "a live backend must not be deregistered"


# ------------------------------------------------------------------ native serve

FAKE_LLAMA_SERVER = r'''#!/usr/bin/env python3
import http.server, os, sys
if "--help" in sys.argv:
    sys.exit(0)
port = int(sys.argv[sys.argv.index("--port") + 1])
print(f"pid {os.getpid()}", flush=True)

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
'''


def test_native_serve_detaches_and_waits_for_health(tmp_path, model, ledger, monkeypatch):
    """A native lane runs detached; in the foreground it held `one serve` until timeout."""
    import os
    import signal
    import socket
    import subprocess
    import sys

    runner = tmp_path / "llama-server"
    runner.write_text(FAKE_LLAMA_SERVER.replace("/usr/bin/env python3", sys.executable))
    runner.chmod(0o755)
    (tmp_path / model.file).write_bytes(b"x" * 100)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    real_run = subprocess.run
    # A foreground server never exits; cap the wait so the old path fails fast.
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: real_run(*a, **{**k, "timeout": min(k.get("timeout") or 3, 3)}))
    monkeypatch.setattr(serve, "LOG_DIR", tmp_path / "serve", raising=False)
    monkeypatch.setattr(serve.onecfg, "PROVIDER_DIR", tmp_path / "providers")
    native = hostfacts.HostFacts(hostname="testbox", llama_server=str(runner))
    lane = serve.start(model, port, native, root=tmp_path)
    log_path = tmp_path / "serve" / "one-lane-testmodel.log"
    pid = int(log_path.read_text().split()[1])
    try:
        assert lane.port == port and serve.health(port)
        assert os.getsid(pid) == pid, "the lane must outlive the `one serve` session"
        assert serve.dropin_path(model).is_file()
    finally:
        os.kill(pid, signal.SIGTERM)


def test_serve_reports_a_launch_timeout_instead_of_a_traceback(monkeypatch, capsys):
    import subprocess

    from onecli.serving import cli as serving_cli

    monkeypatch.setattr(serving_cli.hostfacts, "collect", lambda: hostfacts.HostFacts())
    monkeypatch.setattr(serving_cli.catalog, "find", lambda name: catalog.Model(
        name="testmodel", repo="org/Test-GGUF", file="test.gguf"))

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(["docker", "run"], 300)

    monkeypatch.setattr(serving_cli.serve, "start", hang)
    assert serving_cli.cmd_serve(["testmodel", "--port", "9911"]) != 0
    assert "timed out" in capsys.readouterr().err

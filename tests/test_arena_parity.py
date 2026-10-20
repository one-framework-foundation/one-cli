"""The arena parity protocol: the declaration must be true, the probe must bite.

Offline. No network, no CLI binaries: `build(cell)` only writes configs, and the
probe is exercised against a tiny python fake rather than a real harness.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onecli import config, harnesses  # noqa: E402
from onecli.evals import arena_parity
from onecli.harnesses import ABSENT, MATCHED, ParityDim  # noqa: E402

BACKEND = config.Backend(name="lane", model="m-1", base_url="https://example.test/v1",
                         wire=config.WIRE_OPENAI, api_key="sk-test")


def _cell(tmp: Path) -> harnesses.Cell:
    ws = tmp / "ws"
    cfg = tmp / "cfg"
    ws.mkdir(parents=True, exist_ok=True)
    cfg.mkdir(parents=True, exist_ok=True)
    return harnesses.Cell(backend=BACKEND, model_id="", workspace=ws,
                          prompt="PROMPT-MARK", system="SYS-MARK",
                          read_only=False, timeout=291.0, cfg_dir=cfg, max_steps=37)


# --------------------------------------------------------------------------
# every harness carries a parity table, and it is true
# --------------------------------------------------------------------------

def test_every_harness_declares_parity():
    for name, harness in harnesses.HARNESSES.items():
        assert harness.parity, f"{name}: no parity declared"
        for dim in (4, 5, 6, 7, 8, 12):
            entry = harness.parity[dim]
            assert entry.status in harnesses.PARITY_STATUSES, f"{name} dim{dim}"
            assert entry.note, f"{name} dim{dim}: no note"


def test_declarations_match_the_built_command(tmp_path):
    for name, harness in harnesses.HARNESSES.items():
        cell = _cell(tmp_path / name)
        bad = arena_parity.check_declaration(harness, cell)
        assert not bad, f"{name}: {bad}"


def test_a_false_matched_declaration_is_caught(tmp_path):
    """Prove the check bites: flip pi's step budget to MATCHED - pi passes no
    step flag - and the check must find the lie."""
    pi = harnesses.HARNESSES["pi"]
    honest = arena_parity.check_declaration(pi, _cell(tmp_path / "honest"))
    assert honest == []
    lied = dataclasses.replace(pi, parity={**pi.parity,
                                           5: ParityDim(MATCHED, "lie")})
    caught = arena_parity.check_declaration(lied, _cell(tmp_path / "lied"))
    assert any("dim5" in v for v in caught), caught


def test_mini_step_budget_is_genuinely_matched(tmp_path):
    """mini now passes agent.step_limit=cell.max_steps. The declaration is
    MATCHED and the built command must carry the cell's own budget, not 20."""
    mini = harnesses.HARNESSES["mini"]
    assert mini.parity[5].status == MATCHED
    cell = _cell(tmp_path / "mini")
    launch = mini.build(cell)
    joined = " ".join(launch.cmd)
    assert f"agent.step_limit={cell.max_steps}" in joined
    assert "agent.step_limit=20" not in joined
    assert arena_parity.check_declaration(mini, cell) == []


def test_one_arena_disables_first_response_gate(tmp_path):
    """Arena steps belong to the task, not a CLI-only policy preflight."""
    launch = harnesses.HARNESSES["one"].build(_cell(tmp_path / "one"))
    assert launch.cmd[launch.cmd.index("--response-gate") + 1] == "off"


def test_a_false_matched_token_budget_is_caught(tmp_path):
    """pi passes no output-token cap. Declaring dim7 MATCHED is a lie the check
    must catch, because 8192 is not in pi's built command."""
    pi = harnesses.HARNESSES["pi"]
    lied = dataclasses.replace(pi, parity={**pi.parity,
                                           7: ParityDim(MATCHED, "lie")})
    caught = arena_parity.check_declaration(lied, _cell(tmp_path / "pi7"))
    assert any("dim7" in v for v in caught), caught


def test_a_false_matched_thinking_level_is_caught(tmp_path):
    """No harness pins --thinking (openclaw validates the level per model and no
    single level spans all lanes), so every dim8 is UNCAPPED. Declaring it
    MATCHED without a --thinking flag on the command is a lie the check catches."""
    mini = harnesses.HARNESSES["mini"]
    assert mini.parity[8].status != MATCHED
    lied = dataclasses.replace(mini, parity={**mini.parity,
                                             8: ParityDim(MATCHED, "lie")})
    caught = arena_parity.check_declaration(lied, _cell(tmp_path / "mini8"))
    assert any("dim8" in v for v in caught), caught


def test_a_false_absent_system_role_is_caught(tmp_path):
    """goose sends the system prompt in the system role. Declaring it ABSENT
    (prepended) is a lie the check must catch, because goose does not prepend."""
    goose = harnesses.HARNESSES["goose"]
    lied = dataclasses.replace(goose, parity={**goose.parity,
                                              4: ParityDim(ABSENT, "lie")})
    caught = arena_parity.check_declaration(lied, _cell(tmp_path / "g"))
    assert any("dim4" in v for v in caught), caught


# --------------------------------------------------------------------------
# the observed probe bites
# --------------------------------------------------------------------------

def test_extract_tools_tolerates_a_numbered_prefix():
    """The probe numbers its three steps, so models answer '2. TOOLS: ...'.
    The tool list must still be captured for the parity report."""
    tools = arena_parity.extract_tools(
        "1. FILE: x\n2. TOOLS: read, grep, web_search\n3. no write tool")
    assert tools == ["read", "grep", "web_search"]
    # a plain 'TOOLS:' line still works
    assert arena_parity.extract_tools("TOOLS: read, write") == ["read", "write"]


def test_evaluate_probe_passes_a_reader_that_refused_the_write():
    ok, reason = arena_parity.evaluate_probe(
        f"FILE: {arena_parity.PROBE_SECRET}\nTOOLS: read, grep\nno write tool",
        wrote=False, tools=["read", "grep"])
    assert ok and reason == ""


def test_evaluate_probe_excludes_a_harness_with_no_read_tool():
    ok, reason = arena_parity.evaluate_probe(
        "I have no file-read tool, so I will not fabricate a value.",
        wrote=False, tools=["session_status"])
    assert not ok
    assert "no read tool" in reason
    # the minimal-profile fingerprint is named
    assert "minimal-profile" in reason


def test_evaluate_probe_excludes_the_sandbox_fingerprint():
    ok, reason = arena_parity.evaluate_probe(
        "TOOLS: web_fetch, web_search, memory_search\ncannot read the tree",
        wrote=False, tools=["web_fetch", "web_search", "memory_search"])
    assert not ok and "sandbox web+memory" in reason


def test_evaluate_probe_excludes_a_harness_that_wrote_on_a_read_only_cell():
    ok, reason = arena_parity.evaluate_probe(
        f"FILE: {arena_parity.PROBE_SECRET}\nTOOLS: read, write\nwrote the file",
        wrote=True, tools=["read", "write"])
    assert not ok and "read-only not enforced" in reason


def _fake(kind: str) -> harnesses.Harness:
    """A one-file harness backed by a python one-liner. `kind` picks behaviour:
    'reader' reads and refuses writes, 'blind' has no read tool, 'writer' writes
    on a read-only cell."""
    def build(cell: harnesses.Cell) -> harnesses.Launch:
        if kind == "reader":
            code = (f"import pathlib;"
                    f"print('FILE:', pathlib.Path('{arena_parity.PROBE_FILE}').read_text().strip());"
                    f"print('TOOLS: read, grep')")
        elif kind == "writer":
            code = (f"import pathlib;"
                    f"print('FILE:', pathlib.Path('{arena_parity.PROBE_FILE}').read_text().strip());"
                    f"pathlib.Path('probe_write.txt').write_text('wrote');"
                    f"print('TOOLS: read, write')")
        else:
            code = "print('TOOLS: session_status');print('no file-read tool here')"
        return harnesses.Launch(cmd=[sys.executable, "-c", code],
                                cwd=str(cell.workspace))

    def parse(out: str, err: str) -> harnesses.Result:
        return harnesses.Result(text=out)

    return harnesses.Harness("fake", "", (config.WIRE_OPENAI,), build, parse, "")


def test_run_probe_passes_a_real_reader(tmp_path):
    res = arena_parity.run_probe(_fake("reader"), BACKEND, tmp_path, timeout=30)
    assert res.ok and res.read_ok and res.write_refused


def test_run_probe_excludes_a_blind_harness(tmp_path):
    res = arena_parity.run_probe(_fake("blind"), BACKEND, tmp_path, timeout=30)
    assert not res.ok and not res.read_ok
    assert "minimal-profile" in res.reason


def test_run_probe_excludes_a_harness_that_writes(tmp_path):
    res = arena_parity.run_probe(_fake("writer"), BACKEND, tmp_path, timeout=30)
    assert not res.ok and res.read_ok and not res.write_refused
    assert "read-only not enforced" in res.reason


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def test_parity_markdown_has_the_declared_table_and_exclusions():
    probes = {"one": arena_parity.ProbeResult("one", True, read_ok=True, write_refused=True,
                                              tools=["read", "grep"]),
              "openclaw": arena_parity.ProbeResult("openclaw", False,
                                                   reason="excluded: no read tool")}
    # opencode still declares an ABSENT (dim4 system-role), so the ABSENT
    # rendering path is exercised; openclaw declares none.
    md = arena_parity.parity_markdown(["one", "openclaw", "opencode"],
                                      {"lane": probes}, ["lane"])
    assert "Declared parity" in md
    assert "Observed probe" in md
    assert "Exclusions" in md
    assert "openclaw" in md and "ABSENT" in md


# --------------------------------------------------------------------------
# probe retries, per-lane exclusion and broken-run reporting
# --------------------------------------------------------------------------

def _flaky(fail_first: int) -> harnesses.Harness:
    """A reader that fails its first `fail_first` attempts, then reads.

    State lives in a counter file beside the probe root, so each attempt sees
    the previous one - which is exactly what a retry budget must survive.
    """
    counter: list[int] = [0]

    def build(cell: harnesses.Cell) -> harnesses.Launch:
        counter[0] += 1
        if counter[0] <= fail_first:
            code = "print('TOOLS: read, grep');print('sorry, blank turn')"
        else:
            code = (f"import pathlib;"
                    f"print('FILE:', pathlib.Path('{arena_parity.PROBE_FILE}')"
                    f".read_text().strip());print('TOOLS: read, grep')")
        return harnesses.Launch(cmd=[sys.executable, "-c", code], cwd=str(cell.workspace))

    return harnesses.Harness("flaky", "", (config.WIRE_OPENAI,), build,
                             lambda out, err: harnesses.Result(text=out), "")


# --- defect 2: a single sample of a stochastic agent is not evidence --------

def test_run_probe_retries_a_flaky_harness_and_passes(tmp_path):
    res = arena_parity.run_probe(_flaky(2), BACKEND, tmp_path, timeout=30, attempts=3)
    assert res.ok, res.reason
    assert res.attempts == 3 and res.passed_on == 3


def test_run_probe_excludes_only_after_every_attempt_fails(tmp_path):
    res = arena_parity.run_probe(_fake("blind"), BACKEND, tmp_path, timeout=30, attempts=3)
    assert not res.ok
    assert res.attempts == 3
    assert res.score == "passed 0/3"


# --- defect 3: the reason may not claim a capability the sample disproves ---

def test_a_flaky_reader_is_not_accused_of_having_no_read_tool():
    ok, reason = arena_parity.evaluate_probe(
        "I could not find the file.", wrote=False,
        tools=["read", "write", "bash", "grep"])
    assert not ok
    assert "no read tool" not in reason
    assert "has a read tool" in reason and "flaky" in reason


def test_a_genuinely_blind_harness_still_gets_the_capability_claim():
    ok, reason = arena_parity.evaluate_probe(
        "no way to read anything", wrote=False, tools=["web_search", "web_fetch"])
    assert not ok and "no read tool" in reason


def test_a_probe_timeout_is_its_own_reason(tmp_path):
    slow = harnesses.Harness(
        "slow", "", (config.WIRE_OPENAI,),
        lambda cell: harnesses.Launch(cmd=[sys.executable, "-c", "import time;time.sleep(5)"],
                                      cwd=str(cell.workspace)),
        lambda out, err: harnesses.Result(text=out), "")
    res = arena_parity.run_probe(slow, BACKEND, tmp_path, timeout=0.4, attempts=1)
    assert not res.ok and res.timed_out
    assert "timed out" in res.reason
    assert "no read tool" not in res.reason


# --- defect 1: a probe verdict is per (harness, lane), never global ---------

def test_probe_matrix_is_keyed_by_backend(tmp_path, monkeypatch):
    monkeypatch.setitem(harnesses.HARNESSES, "reader", _fake("reader"))
    other = dataclasses.replace(BACKEND, name="lane-2")
    matrix = arena_parity.probe_matrix(["reader"], [BACKEND, other], tmp_path,
                                       timeout=30, attempts=1)
    assert set(matrix) == {"lane", "lane-2"}
    assert matrix["lane"]["reader"].ok and matrix["lane-2"]["reader"].ok


def test_exclusion_is_scoped_to_the_lane_that_failed():
    bad = arena_parity.ProbeResult("openclaw", False, reason="excluded: flaky", attempts=3)
    good = arena_parity.ProbeResult("openclaw", True, read_ok=True, write_refused=True)
    matrix = {"minimax": {"openclaw": bad}, "kimi": {"openclaw": good}}
    pairs = arena_parity.excluded_pairs(matrix)
    assert pairs == {("openclaw", "minimax")}
    assert ("openclaw", "kimi") not in pairs


def test_arena_run_skips_only_the_excluded_pair(tmp_path, monkeypatch):
    """A harness excluded on one lane (minimax) still runs on the others
    (kimi): exclusion is per (harness, lane) pair."""
    from onecli.evals import arena
    seen: list[tuple[str, str]] = []

    def fake_cell(harness, backend, task, root, timeout, system, do_judge, budget=None):
        seen.append((harness.name, backend.name))
        return {"harness": harness.name, "backend": backend.name, "task": task.name,
                "ok": True, "seconds": 0.0, "lines_touched": 0}

    monkeypatch.setattr(arena, "run_cell", fake_cell)
    monkeypatch.setitem(harnesses.HARNESSES, "openclaw",
                        dataclasses.replace(_fake("reader"), name="openclaw"))
    lanes = [dataclasses.replace(BACKEND, name="minimax"),
             dataclasses.replace(BACKEND, name="kimi")]
    task = type("T", (), {"name": "t1", "prompt": "p"})()
    arena.run(["openclaw"], lanes, [task], tmp_path, 5.0, 1,
              tmp_path / "rows.jsonl", "", False, None,
              skip={("openclaw", "minimax")})
    assert seen == [("openclaw", "kimi")]


# --- defect 4: an empty column is never a score ----------------------------

def test_parity_markdown_shouts_when_a_harness_ran_zero_cells():
    md = arena_parity.parity_markdown(
        ["one", "openclaw"],
        {"minimax": {"openclaw": arena_parity.ProbeResult(
            "openclaw", False, reason="excluded: flaky", attempts=3)}},
        ["minimax"], broken=["openclaw"])
    assert md.startswith("# Arena parity report")
    head = md.split("## Declared parity")[0]
    assert "BROKEN RUN" in head and "zero cells" in head and "openclaw" in head


def test_arena_exits_non_zero_when_a_requested_harness_ran_no_cells(tmp_path, monkeypatch):
    from onecli.evals import arena
    monkeypatch.setattr(arena, "run", lambda *a, **k: [
        {"harness": "one", "backend": "minimax", "task": "t1", "ok": True,
         "seconds": 1.0, "lines_touched": 0}])
    monkeypatch.setattr(arena_parity, "probe_matrix", lambda *a, **k: {
        "minimax": {"openclaw": arena_parity.ProbeResult(
            "openclaw", False, reason="excluded: flaky", attempts=3),
            "one": arena_parity.ProbeResult("one", True, read_ok=True,
                                            write_refused=True)}})
    monkeypatch.setattr(arena.bench, "select_backends",
                        lambda *a, **k: [dataclasses.replace(BACKEND, name="minimax")])
    monkeypatch.setattr(arena, "judge_backend", lambda *a, **k: None)
    for name in ("one", "openclaw"):
        monkeypatch.setitem(harnesses.HARNESSES, name,
                            dataclasses.replace(_fake("reader"), name=name))
    out = tmp_path / "run"
    rc = arena.main(["minimax", "--harness", "one", "--harness", "openclaw",
                     "--suite", "hard", "-t", "hardloop", "--no-judge",
                     "--out", str(out)])
    assert rc != 0
    body = (out / "parity.md").read_text()
    assert "BROKEN RUN" in body

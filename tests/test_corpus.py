"""The corpus is the growth path, so its failure modes are what matter here:
a task that does not load must be REPORTED, and a mask or gate that is named
must actually reach the cell.
"""

import json

import pytest

from onecli.evals import bench
from onecli.evals import corpus


def _task_dir(root, name, manifest, check="import sys; sys.exit(0)"):
    d = root / name
    d.mkdir()
    (d / "task.json").write_text(json.dumps(manifest))
    (d / "check.py").write_text(check)
    return d


def test_discovers_a_well_formed_task(tmp_path):
    _task_dir(tmp_path, "t1", {"prompt": "do a thing"})
    tasks, problems = corpus.discover(tmp_path)
    assert [t.name for t in tasks] == ["t1"]
    assert problems == []


@pytest.mark.parametrize("name,manifest,files", [
    ("no-prompt", {"prompt": "   "}, True),
    ("typo-key", {"prompt": "x", "gate": {}}, True),
    ("bad-gates", {"prompt": "x", "gates": {"A": 1}}, True),
])
def test_a_broken_task_is_reported_not_skipped(tmp_path, name, manifest, files):
    _task_dir(tmp_path, name, manifest)
    tasks, problems = corpus.discover(tmp_path)
    assert tasks == []
    assert len(problems) == 1 and name in problems[0]


def test_a_directory_without_a_checker_is_reported(tmp_path):
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "task.json").write_text('{"prompt": "x"}')
    tasks, problems = corpus.discover(tmp_path)
    assert tasks == [] and "no check.py" in problems[0]


def test_missing_corpus_root_is_empty_not_an_error(tmp_path):
    assert corpus.discover(tmp_path / "nope") == ([], [])


def test_checker_verdict_comes_from_the_script(tmp_path):
    d = _task_dir(tmp_path, "t", {"prompt": "x"},
                  check="import os, sys\n"
                        "sys.exit(0 if os.path.exists('done.txt') else 1)\n")
    task = corpus.discover(tmp_path)[0][0]
    ws = tmp_path / "ws"
    ws.mkdir()
    assert task.check(ws, "")[0] is False
    (ws / "done.txt").write_text("")
    assert task.check(ws, "")[0] is True
    assert d.is_dir()


def test_checker_can_read_the_agents_answer(tmp_path):
    _task_dir(tmp_path, "t", {"prompt": "x"},
              check="import sys\n"
                    "sys.exit(0 if 'forty-two' in open('_answer.txt').read() else 1)\n")
    task = corpus.discover(tmp_path)[0][0]
    ws = tmp_path / "ws"
    ws.mkdir()
    assert task.check(ws, "the answer is forty-two")[0] is True
    assert task.check(ws, "no idea")[0] is False


def test_seed_files_land_in_the_workspace(tmp_path):
    d = _task_dir(tmp_path, "t", {"prompt": "x", "files": {"a/inline.txt": "hi"}})
    (d / "files").mkdir()
    (d / "files" / "seeded.txt").write_text("there")
    task = corpus.discover(tmp_path)[0][0]
    ws = tmp_path / "ws"
    ws.mkdir()
    task.setup(ws)
    assert (ws / "a" / "inline.txt").read_text() == "hi"
    assert (ws / "seeded.txt").read_text() == "there"


# -- the two dimensions -------------------------------------------------

def test_a_mask_preamble_reaches_the_prompt(tmp_path):
    _task_dir(tmp_path, "t", {"prompt": "the ask", "mask": "m"})
    task = corpus.discover(tmp_path)[0][0]
    masks = {"m": corpus.MaskInfo(kind="m", preamble="WEAR THIS")}
    assert task.prompt_for(masks) == "WEAR THIS\n\nthe ask"
    assert task.prompt_for({}) == "the ask"


def test_an_unknown_mask_is_a_validation_problem(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_CLI_CORPUS", str(tmp_path))
    monkeypatch.setenv("ONE_MASKS_JSON", str(tmp_path / "masks.json"))
    (tmp_path / "masks.json").write_text('{"known": {"prompts": "p"}}')
    _task_dir(tmp_path, "t", {"prompt": "x", "mask": "ghost"})
    tasks, problems = corpus.validate()
    assert len(tasks) == 1
    assert any("ghost" in p for p in problems)


def test_no_mask_source_is_named_when_a_task_wants_one(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_CLI_CORPUS", str(tmp_path))
    monkeypatch.delenv("ONE_MASKS_JSON", raising=False)
    _task_dir(tmp_path, "t", {"prompt": "x", "mask": "m"})
    assert any("no mask source" in p for p in corpus.validate()[1])


def test_gates_layer_over_the_env_and_are_otherwise_untouched(tmp_path):
    _task_dir(tmp_path, "gated", {"prompt": "x", "gates": {"G": "enforce"}})
    _task_dir(tmp_path, "plain", {"prompt": "x"})
    gated, plain = sorted(corpus.discover(tmp_path)[0], key=lambda t: t.name)
    base = {"PATH": "/bin"}
    assert bench.cell_env(gated, base) == {"PATH": "/bin", "G": "enforce"}
    assert bench.cell_env(plain, base) is base
    assert base == {"PATH": "/bin"}  # not mutated


def test_gates_fall_back_to_the_process_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL", "kept")
    _task_dir(tmp_path, "t", {"prompt": "x", "gates": {"G": "1"}})
    env = bench.cell_env(corpus.discover(tmp_path)[0][0], None)
    assert env["G"] == "1" and env["SENTINEL"] == "kept"


def test_row_records_both_dimensions():
    # An in-code task has neither, and that absence is what tells a reader the
    # cell measured a bare model.
    assert bench.cell_env(bench.TASKS[0], None) is None
    assert getattr(bench.TASKS[0], "mask", "") == ""


# -- growth -------------------------------------------------------------

def test_add_writes_a_loadable_task(tmp_path):
    corpus.add("new", "the ask", "import sys; sys.exit(0)",
               mask="m", gates={"G": "1"}, origin="chat:1", tags=["a"],
               root=tmp_path)
    task = corpus.discover(tmp_path)[0][0]
    assert task.name == "new" and task.mask == "m"
    assert task.gates == {"G": "1"} and task.origin == "chat:1"


def test_add_refuses_to_overwrite(tmp_path):
    corpus.add("new", "x", "import sys", root=tmp_path)
    with pytest.raises(FileExistsError):
        corpus.add("new", "y", "import sys", root=tmp_path)


def test_add_rolls_back_a_task_it_cannot_load(tmp_path):
    with pytest.raises(ValueError):
        corpus.add("bad", "   ", "import sys", root=tmp_path)
    assert not (tmp_path / "bad").exists()


def test_corpus_root_is_env_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_CLI_CORPUS", str(tmp_path / "elsewhere"))
    assert corpus.corpus_root() == tmp_path / "elsewhere"


def test_suite_corpus_selects_only_disk_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_CLI_CORPUS", str(tmp_path))
    monkeypatch.delenv("ONE_MASKS_JSON", raising=False)
    _task_dir(tmp_path, "only", {"prompt": "x"})
    assert [t.name for t in bench.suite("corpus")] == ["only"]
    assert "only" in [t.name for t in bench.suite("all")]
    assert "only" not in [t.name for t in bench.suite("basic")]


# --------------------------------------------------------------------------
# rewards - a task scores more than one dimension
# --------------------------------------------------------------------------

WRITES_REWARDS = """
import json, os, sys
json.dump({"lines": 3, "seconds": 1.5}, open(os.environ["ONE_REWARDS"], "w"))
sys.exit(0)
"""


def test_a_checker_can_score_named_dimensions(tmp_path):
    d = _task_dir(tmp_path, "scored", {"prompt": "x"}, check=WRITES_REWARDS)
    task, _ = corpus._read_task(d)
    ws = tmp_path / "ws"
    ws.mkdir()
    result = task.grade(ws, "")
    assert result.ok
    assert result.rewards == {"lines": 3.0, "seconds": 1.5, "pass": 1.0}


def test_pass_is_a_reward_even_without_a_rewards_file(tmp_path):
    d = _task_dir(tmp_path, "plain", {"prompt": "x"},
                  check="import sys; sys.exit(1)")
    task, _ = corpus._read_task(d)
    ws = tmp_path / "ws"
    ws.mkdir()
    result = task.grade(ws, "")
    assert result.rewards == {"pass": 0.0}


def test_unusable_rewards_fail_the_cell_rather_than_vanish(tmp_path):
    check = ('import os, sys\n'
             'open(os.environ["ONE_REWARDS"], "w").write("{not json")\n'
             'sys.exit(0)\n')
    d = _task_dir(tmp_path, "broken-score", {"prompt": "x"}, check=check)
    task, _ = corpus._read_task(d)
    ws = tmp_path / "ws"
    ws.mkdir()
    result = task.grade(ws, "")
    assert not result.ok
    assert "rewards unreadable" in result.detail


def test_a_stale_rewards_file_is_not_reused(tmp_path):
    d = _task_dir(tmp_path, "stale", {"prompt": "x"},
                  check="import sys; sys.exit(0)")
    task, _ = corpus._read_task(d)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / corpus.REWARDS_FILE).write_text('{"lines": 99}')
    assert task.grade(ws, "").rewards == {"pass": 1.0}


def test_aggregate_defaults_to_mean_and_honours_the_declared_kind():
    runs = [{"pass": 1.0, "seconds": 4.0}, {"pass": 0.0, "seconds": 2.0}]
    assert corpus.aggregate(runs) == {"pass": 0.5, "seconds": 3.0}
    assert corpus.aggregate(runs, {"seconds": "min"})["seconds"] == 2.0


def test_aggregate_skips_a_metric_a_run_never_reported():
    runs = [{"pass": 1.0, "seconds": 4.0}, {"pass": 1.0}]
    assert corpus.aggregate(runs, {"seconds": "mean"})["seconds"] == 4.0


def test_an_unknown_aggregation_is_a_load_error(tmp_path):
    _task_dir(tmp_path, "weird", {"prompt": "x", "metrics": {"s": "median"}})
    tasks, problems = corpus.discover(tmp_path)
    assert tasks == []
    assert "median" in problems[0]


# --------------------------------------------------------------------------
# the oracle - a task must prove it is worth running
# --------------------------------------------------------------------------

NEEDS_A_FILE = ('import pathlib, sys\n'
                'sys.exit(0 if pathlib.Path("out.txt").exists() else 1)\n')


def test_oracle_admits_a_task_that_fails_empty_and_passes_solved(tmp_path):
    d = _task_dir(tmp_path, "real", {"prompt": "make out.txt"},
                  check=NEEDS_A_FILE)
    (d / "solve.sh").write_text("echo hi > out.txt\n")
    task, _ = corpus._read_task(d)
    result = corpus.oracle(task)
    assert result.admitted
    assert result.discriminating and result.passable


def test_oracle_rejects_a_task_that_passes_before_anything_happens(tmp_path):
    d = _task_dir(tmp_path, "free", {"prompt": "x"},
                  check="import sys; sys.exit(0)")
    (d / "solve.sh").write_text("true\n")
    task, _ = corpus._read_task(d)
    result = corpus.oracle(task)
    assert not result.admitted
    assert not result.discriminating
    assert "untouched" in result.detail


def test_oracle_rejects_a_task_nothing_can_pass(tmp_path):
    d = _task_dir(tmp_path, "unwinnable", {"prompt": "x"},
                  check=NEEDS_A_FILE)
    (d / "solve.sh").write_text("true\n")
    task, _ = corpus._read_task(d)
    result = corpus.oracle(task)
    assert not result.admitted
    assert result.discriminating and not result.passable
    assert "still fails" in result.detail


def test_oracle_reports_a_solution_that_errors(tmp_path):
    d = _task_dir(tmp_path, "bad-solve", {"prompt": "x"}, check=NEEDS_A_FILE)
    (d / "solve.sh").write_text("exit 3\n")
    task, _ = corpus._read_task(d)
    result = corpus.oracle(task)
    assert not result.admitted
    assert "solve.sh failed" in result.detail


def test_a_task_without_a_solution_is_not_admitted(tmp_path):
    d = _task_dir(tmp_path, "bare", {"prompt": "x"}, check=NEEDS_A_FILE)
    task, _ = corpus._read_task(d)
    result = corpus.oracle(task)
    assert not result.admitted
    assert not result.has_solution


def test_oracle_uses_the_tasks_seed_files(tmp_path):
    check = ('import pathlib, sys\n'
             'sys.exit(0 if pathlib.Path("a.txt").read_text() == "b" else 1)\n')
    d = _task_dir(tmp_path, "seeded",
                  {"prompt": "x", "files": {"a.txt": "a"}}, check=check)
    (d / "solve.sh").write_text("printf b > a.txt\n")
    task, _ = corpus._read_task(d)
    assert corpus.oracle(task).admitted


def test_add_with_verify_keeps_a_proven_task(tmp_path):
    path = corpus.add("proven", "make out.txt", NEEDS_A_FILE,
                      solution="echo hi > out.txt\n", verify=True,
                      root=tmp_path)
    assert (path / "solve.sh").is_file()
    assert [t.name for t in corpus.discover(tmp_path)[0]] == ["proven"]


def test_add_with_verify_rolls_back_an_unproven_task(tmp_path):
    with pytest.raises(ValueError, match="unproven"):
        corpus.add("hopeless", "x", NEEDS_A_FILE, solution="true\n",
                   verify=True, root=tmp_path)
    assert not (tmp_path / "hopeless").exists()
    assert corpus.discover(tmp_path) == ([], [])


def test_a_checker_can_import_what_the_agent_wrote(tmp_path):
    """cwd is not an import path, so the workspace must be on PYTHONPATH."""
    check = ('import made, sys\n'
             'sys.exit(0 if made.value() == 7 else 1)\n')
    d = _task_dir(tmp_path, "importable", {"prompt": "x"}, check=check)
    (d / "solve.sh").write_text('printf "def value():\\n    return 7\\n" > made.py\n')
    task, _ = corpus._read_task(d)
    result = corpus.oracle(task)
    assert result.admitted, result.detail

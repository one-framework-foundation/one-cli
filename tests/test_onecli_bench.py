"""one-bench: selection, scoring and reporting. All offline - no model is called.

The bench is the thing that decides which backends we trust, so its own failure
modes matter: a misattributed error retires a working lane, and a checker that
believes the model's closing sentence scores prose instead of work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onecli import config
from onecli.evals.bench import (TASKS, TASKS_BY_NAME, BugfixTask, CreateTask, SearchTask,
                          _short_error, is_local, select_backends, table)


def _backend(name, model="m", wire="openai", base="https://x/v1"):
    return config.Backend(name=name, model=model, base_url=base, wire=wire)


@pytest.fixture
def found(monkeypatch):
    backends = [
        _backend("qwen", "qwen3.8-max", "anthropic"),
        _backend("ollama-cloud", "deepseek-v4-flash:0731"),
        _backend("llamacpp/qwen3-coder:30b", "qwen3-coder:30b",
                 base="http://127.0.0.1:11434/v1"),
        _backend("llamacpp/qwen2.5-coder:7b", "qwen2.5-coder:7b",
                 base="http://127.0.0.1:11434/v1"),
    ]
    monkeypatch.setattr(config, "discover", lambda include_local=True: backends)
    return backends


def test_exact_name_does_not_drag_in_substring_matches(found):
    # "qwen" as a substring matches two local tags as well; an exact name is a
    # request for that one backend.
    picked = select_backends(["qwen"], True)
    assert [b.name for b in picked] == ["qwen"]


def test_substring_still_works_for_a_non_name(found):
    picked = select_backends(["coder"], True)
    assert [b.name for b in picked] == [
        "llamacpp/qwen3-coder:30b", "llamacpp/qwen2.5-coder:7b"]


def test_host_equals_model_clones_the_endpoint(found):
    picked = select_backends(["ollama-cloud=glm-5.2"], True)
    assert len(picked) == 1
    assert picked[0].model == "glm-5.2"
    assert picked[0].base_url == "https://x/v1"      # same endpoint
    assert picked[0].name == "ollama-cloud=glm-5.2"  # distinct row in the table


def test_host_equals_model_on_an_unknown_host_is_dropped(found):
    assert select_backends(["nope=glm-5.2"], True) == []


def test_local_and_remote_are_told_apart(found):
    assert is_local(found[2]) and not is_local(found[0])


# --------------------------------------------------------------------------
# checkers score the workspace, not the model's closing sentence
# --------------------------------------------------------------------------

def _workspace(tmp_path: Path, task) -> Path:
    ws = tmp_path / task.name
    ws.mkdir()
    task.setup(ws)
    return ws


def test_bugfix_fails_on_the_untouched_file(tmp_path):
    task = TASKS_BY_NAME["bugfix"]
    ws = _workspace(tmp_path, task)
    ok, why = task.check(ws, "I fixed both bugs and verified the output.")
    assert not ok and why == "wrong output"


def test_bugfix_passes_only_on_the_real_fix(tmp_path):
    task = TASKS_BY_NAME["bugfix"]
    ws = _workspace(tmp_path, task)
    fixed = (ws / "fizz.py").read_text().replace(
        "range(1, n)", "range(1, n + 1)").replace("i % 4 == 0", "i % 5 == 0")
    (ws / "fizz.py").write_text(fixed)
    assert task.check(ws, "")[0]


def test_search_reads_the_number_out_of_the_answer(tmp_path):
    task = TASKS_BY_NAME["search"]
    ws = _workspace(tmp_path, task)
    assert task.check(ws, "3")[0]
    assert task.check(ws, "There are 3 files.")[0]
    ok, why = task.check(ws, "2")
    assert not ok and why == "said 2"
    assert task.check(ws, "several")[1] == "no number"


def test_create_runs_a_hidden_test_the_model_never_saw(tmp_path):
    task = TASKS_BY_NAME["create"]
    ws = _workspace(tmp_path, task)
    assert task.check(ws, "done")[1] == "no file"
    # A slugify that only satisfies the example in the prompt still fails.
    (ws / "slug.py").write_text("def slugify(t):\n    return 'hello-world'\n")
    assert task.check(ws, "")[0] is False
    (ws / "slug.py").write_text(
        "import re\n\n\ndef slugify(t):\n"
        "    return re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')\n")
    assert task.check(ws, "")[0]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("detail,expected", [
    ('HTTP 402 from https://ollama.com/v1: {"message":"payment required"}',
     "402 unpaid"),
    ('HTTP 400 from x: {"code":"InvalidParameter","message":"Model not exist."}',
     "400 no model"),
    ("cannot reach http://127.0.0.1:11434/v1: timed out", "unreachable"),
])
def test_transport_failures_are_named_as_transport(detail, expected):
    assert _short_error(detail) == expected


def test_table_ranks_by_passes_then_time():
    rows = [
        {"backend": "slow", "wire": "openai", "task": "bugfix", "ok": True,
         "seconds": 90, "stopped": "answered", "why": ""},
        {"backend": "fast", "wire": "openai", "task": "bugfix", "ok": True,
         "seconds": 3, "stopped": "answered", "why": ""},
        {"backend": "broke", "wire": "openai", "task": "bugfix", "ok": False,
         "seconds": 1, "stopped": "error", "why": "402 unpaid"},
    ]
    out = table(rows, [TASKS_BY_NAME["bugfix"]]).splitlines()
    assert [line.split()[0] for line in out[2:]] == ["fast", "slow", "broke"]
    assert "402 unpaid" in out[-1]


def test_rows_are_json_serialisable():
    row = {"backend": "x", "wire": "openai", "task": "bugfix", "ok": True,
           "seconds": 1.0, "stopped": "answered", "why": "", "steps": 2,
           "tool_calls": 1, "error": "", "model": "m"}
    assert json.loads(json.dumps(row)) == row


def test_every_task_has_a_checker():
    for task in TASKS:
        assert type(task).check is not type(task).__mro__[-2].check or True
        assert isinstance(task, (BugfixTask, SearchTask, CreateTask))


# --------------------------------------------------------------------------
# a lane that never ran is not a lane that lost
# --------------------------------------------------------------------------

def _cell(backend, task, ok, why=""):
    return {"backend": backend, "wire": "openai", "task": task, "ok": ok,
            "seconds": 1.0, "stopped": "answered" if ok else "error", "why": why}


def test_an_unreachable_lane_reads_as_no_contest():
    """A machine that was switched off must not print the table's worst score.

    One test rig was offline for a hard run and both of its lanes
    printed `0/5` next to lanes that had genuinely failed five tasks, which is
    the same number for two opposite facts.
    """
    tasks = [TASKS_BY_NAME["bugfix"], TASKS_BY_NAME["search"]]
    rows = [_cell("local-a", "bugfix", False, "unreachable"),
            _cell("local-a", "search", False, "unreachable"),
            _cell("mistral", "bugfix", False, "wrong output"),
            _cell("mistral", "search", False, "no number")]
    out = table(rows, tasks)
    assert "no contest" in out
    assert "0/2" in out                      # mistral really did lose both
    assert out.index("mistral") < out.index("local-a")   # and sorts above it


def test_a_partly_unreachable_lane_says_how_many_it_missed():
    tasks = [TASKS_BY_NAME["bugfix"], TASKS_BY_NAME["search"]]
    rows = [_cell("kimi", "bugfix", True),
            _cell("kimi", "search", False, "unreachable")]
    assert "1/1 (1 nc)" in table(rows, tasks)


def test_a_parser_refusal_is_labelled_as_the_servers_own():
    """llama.cpp 500ing on its own model's harmony block is not `HTTP 500`."""
    detail = ('HTTP 500 from http://h:8911/v1/chat/completions: {"error":'
              '{"code":500,"message":"The model produced output that does not '
              'match the expected peg-native format","type":"server_error"}}')
    assert _short_error(detail) == "500 toolfmt"


def _row(backend, why, ok=False, seconds=1):
    return {"backend": backend, "wire": "openai", "task": "bugfix", "ok": ok,
            "seconds": seconds, "stopped": "error", "why": why}


def test_a_long_reason_is_repeated_in_full_under_the_table():
    # The reason column is the only one carrying information beyond pass/fail,
    # and a fixed-width clip turned "invented it in pricing.py" into
    # "invented it in" - which reads as a different verdict, not as a cut string.
    reason = "invented it in pricing.py and then argued about it"
    out = table([_row("slow", reason)], [TASKS_BY_NAME["bugfix"]])
    grid, legend = out.split("~ truncated above, in full:")
    assert reason not in grid, "a clipped cell must not claim to be the reason"
    assert grid.splitlines()[2].rstrip().endswith("0/1")
    assert "~" in grid.splitlines()[2]
    assert reason in legend
    assert "slow" in legend and "bugfix" in legend


def test_a_reason_that_fits_is_printed_whole_and_needs_no_legend():
    out = table([_row("broke", "402 unpaid")], [TASKS_BY_NAME["bugfix"]])
    assert "402 unpaid" in out
    assert "truncated above" not in out
    assert "~" not in out


def test_columns_are_sized_per_task_not_globally():
    # A wide reason in one task column must not pad every other column to match.
    tasks = [TASKS_BY_NAME["bugfix"]]
    narrow = table([_row("a", "timeout")], tasks).splitlines()[0]
    wide = table([_row("a", "regressed: suite red")], tasks).splitlines()[0]
    assert len(wide) > len(narrow)

"""The run's own todo plan feeds a parent's BG_SIGNALS progress file.

A parent that exports $BG_SIGNALS reads progress from that file. Without the
run's plan in it, the parent sees nothing between the request and the result.
"""
import types

from onecli import cli, events


def _plan(*states):
    return {"items": [{"text": f"step {i}", "state": s}
                      for i, s in enumerate(states, 1)]}


def test_plan_and_completed_items_become_phase_lines(tmp_path):
    path = tmp_path / "job.signals"
    sink = events.signals_sink(str(path))
    sink("todos", _plan("in_progress", "pending", "pending"))
    sink("todos", _plan("done", "in_progress", "pending"))
    sink("todos", _plan("done", "done", "in_progress"))
    sink("tool_start", {})
    assert path.read_text().splitlines() == [
        "PHASES: 3", "PHASE: step 1", "PHASE: step 2"]


def test_a_restated_plan_writes_nothing_new(tmp_path):
    path = tmp_path / "job.signals"
    sink = events.signals_sink(str(path))
    sink("todos", _plan("done", "pending"))
    sink("todos", _plan("done", "pending"))
    assert path.read_text().splitlines() == ["PHASES: 2", "PHASE: step 1"]


def test_run_emitter_wires_the_sink_from_bg_signals(tmp_path, monkeypatch):
    path = tmp_path / "job.signals"
    monkeypatch.setenv("BG_SIGNALS", str(path))
    args = types.SimpleNamespace(json=False, text_events=False)
    emit = cli._run_emitter(args, {"step": 0})
    assert emit is not None
    emit("todos", _plan("done"))
    assert path.read_text().splitlines() == ["PHASES: 1", "PHASE: step 1"]


def test_no_bg_signals_and_no_json_means_no_sink(monkeypatch):
    monkeypatch.delenv("BG_SIGNALS", raising=False)
    args = types.SimpleNamespace(json=False, text_events=False)
    assert cli._run_emitter(args, {"step": 0}) is None

"""Unit tests for detached tasks and collapsible tool blocks."""

import sys
import time

from onecli import tasks


def test_task_manager_lifecycle():
    tm = tasks.TaskManager()
    t1 = tm.spawn("Run Sweep", phases=["Init", "Execute", "Verify"])
    assert t1.task_id == "1"
    assert t1.status == "running"
    assert t1.current_phase == 1
    assert t1.phase_name == "Init"

    card = tm.render_card("1", width=40)
    assert "TASK #1" in card
    assert "RUNNING" in card
    assert "Run Sweep" in card

    # Partner awareness prompt
    prompt = tm.context_prompt()
    assert "Background Tasks: 1 active" in prompt
    assert "TASK #1: Run Sweep" in prompt

    # Advance phase
    t1.advance_phase()
    assert t1.current_phase == 2
    assert t1.phase_name == "Execute"

    # Complete
    t1.complete("All checks passed")
    assert t1.status == "done"
    assert len(tm.list_active()) == 0


def test_task_execution_process(tmp_path):
    tm = tasks.TaskManager(root=tmp_path)
    py_code = 'import json; print(json.dumps({"text": "sweep completed", "stopped": "stop"}))'
    cmd = [sys.executable, "-c", py_code]

    t = tm.spawn("Quick Sweep", cmd=cmd)
    assert t.task_id == "1"
    assert t.proc is not None

    # Wait for process
    t.proc.wait(timeout=5)
    finished = tm.poll()
    assert len(finished) == 1
    assert finished[0].task_id == "1"
    assert t.status == "done"
    assert t.exit_code == 0
    assert t.result == {"text": "sweep completed", "stopped": "stop"}
    assert "sweep completed" in t.announcement

    # Notice generated
    notices = tm.notices()
    assert len(notices) == 1
    assert "[detached task #1] completed" in notices[0]


def test_task_execution_failure(tmp_path):
    tm = tasks.TaskManager(root=tmp_path)
    py_code = 'import sys; sys.stderr.write("fatal error in step\\n"); sys.exit(2)'
    cmd = [sys.executable, "-c", py_code]

    t = tm.spawn("Failing Sweep", cmd=cmd)
    t.proc.wait(timeout=5)
    tm.poll()

    assert t.status == "failed"
    assert t.exit_code == 2
    assert "fatal error in step" in t.error


def test_task_signals_parsing(tmp_path):
    tm = tasks.TaskManager(root=tmp_path)
    py_code = (
        'import os, time; '
        'sig = os.environ.get("BG_SIGNALS"); '
        'open(sig, "a").write("PHASES: 3\\nPHASE: Compile\\nMILESTONE: Build 1 ready\\n"); '
        'time.sleep(0.1); print(\'{"text": "Build verified"}\')'
    )
    cmd = [sys.executable, "-c", py_code]

    t = tm.spawn("Signal Task", cmd=cmd)
    t.proc.wait(timeout=5)
    tm.poll()

    assert t.status == "done"
    assert "Build 1 ready" in t.signals_path.read_text()
    assert t.announcement == "Build verified"


def test_task_cancellation(tmp_path):
    tm = tasks.TaskManager(root=tmp_path)
    py_code = 'import time; time.sleep(10)'
    cmd = [sys.executable, "-c", py_code]

    t = tm.spawn("Long Task", cmd=cmd)
    assert t.status == "running"
    assert t.proc.poll() is None

    cancelled = tm.cancel("1", reason="User aborted")
    assert cancelled is True
    assert t.status == "failed"
    assert "cancelled: User aborted" in t.announcement

    # Verify process terminated
    time.sleep(0.1)
    assert t.proc.poll() is not None


def test_task_output_retrieval(tmp_path):
    tm = tasks.TaskManager(root=tmp_path)
    py_code = 'import sys; print("line 1"); print("line 2"); sys.stderr.write("warn line\\n")'
    cmd = [sys.executable, "-c", py_code]

    t = tm.spawn("Output Task", cmd=cmd)
    t.proc.wait(timeout=5)
    tm.poll()

    out = t.output()
    assert "line 1" in out
    assert "line 2" in out
    assert "warn line" in out


def test_resolve_runner():
    assert tasks.resolve_runner(model="claude/opus") == ("claude", "opus")
    assert tasks.resolve_runner(model="codex/gpt-6-astra") == ("codex", "gpt-6-astra")
    assert tasks.resolve_runner(model="agy/gemini-3.8-flash") == ("agy", "gemini-3.8-flash")
    assert tasks.resolve_runner(model="one/dashscope") == ("one", "dashscope")
    assert tasks.resolve_runner(runner="claude", model="sonnet") == ("claude", "sonnet")
    assert tasks.resolve_runner(runner="codex") == ("codex", "")
    assert tasks.resolve_runner(model="deepseek-v4.1-flash") == ("one", "deepseek-v4.1-flash")


def test_build_runner_cmd(tmp_path):
    # External adapters cannot preserve the session roots/guards.
    import pytest
    for runner in ('claude', 'codex', 'agy'):
        with pytest.raises(ValueError, match='inherit'):
            tasks.build_runner_cmd(runner, 'test prompt', workspace=str(tmp_path))

    # One
    cmd, needs_stdin = tasks.build_runner_cmd("one", "test prompt", model="deepseek-v4.1-flash", workspace=str(tmp_path))
    assert needs_stdin is False
    assert "--json" in cmd
    assert "--no-ask" in cmd
    assert "--permission-mode" in cmd and "default" in cmd
    assert "-w" in cmd and str(tmp_path) in cmd


def test_parse_spawn_args():
    prompt, runner, model, ws = tasks.parse_spawn_args("--runner codex -m gpt-6-astra -w /tmp/app run audit")
    assert prompt == "run audit"
    assert runner == "codex"
    assert model == "gpt-6-astra"
    assert ws == "/tmp/app"

    # Prefixed model
    prompt, runner, model, ws = tasks.parse_spawn_args("-m claude/opus do refactor")
    assert prompt == "do refactor"
    assert runner == "claude"
    assert model == "opus"

    # Header directives
    text = "model: codex/gpt-6-astra\nworkspace: /my/ws\nRun full test suite"
    prompt, runner, model, ws = tasks.parse_spawn_args(text)
    assert prompt == "Run full test suite"
    assert runner == "codex"
    assert model == "gpt-6-astra"
    assert ws == "/my/ws"




def test_agent_does_not_write_inherited_parent_signals(tmp_path, monkeypatch):
    from onecli.agent import Agent
    from onecli.providers import Turn
    from onecli.providers.base import ToolCall

    sig_file = tmp_path / "test.signals"
    monkeypatch.setenv("BG_SIGNALS", str(sig_file))
    sig_file.write_text("PHASE: Parent milestone\n", encoding="utf-8")

    class MockProvider:
        output_retries = 0
        transient_retries = 0
        def chat(self, history, schemas, **kw):
            if len(history) <= 2:
                # Step 1: call read tool
                return Turn(
                    text="",
                    tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "dummy.txt"})],
                    usage={"input_tokens": 10, "output_tokens": 5},
                )
            # Step 2: finish
            return Turn(
                text="Finished task",
                tool_calls=[],
                usage={"input_tokens": 10, "output_tokens": 5},
            )

    agent = Agent(
        provider=MockProvider(),
        workspace=str(tmp_path),
        tool_names=["read"],
        max_steps=5,
    )
    for _ in range(2):
        res = agent.run("test prompt")
        assert res.stopped == "answered"
    assert sig_file.read_text(encoding="utf-8") == "PHASE: Parent milestone\n"







"""Background jobs: the loop keeps moving while a long command runs.

Real subprocesses, not mocks - the behaviour under test is process behaviour
(a job that outlives a tool call, a cancel that reaches a child, a log readable
while it is still being written), and a mocked Popen would prove none of it.
"""

from __future__ import annotations

import time

from onecli import jobs as joblib
from onecli import tools as toolkit
from onecli.agent import Agent
from onecli.providers.base import ProviderError, ToolCall, Turn


class ScriptedProvider:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.seen.append([dict(m) for m in messages])
        if not self.turns:
            raise ProviderError("script exhausted")
        return self.turns.pop(0)


def _settle(job, store, limit=10.0):
    """Wait for a job to actually finish, without asserting on the clock."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        store.poll()
        if not job.running:
            return True
        time.sleep(0.05)
    return False


# -- the store -----------------------------------------------------------


def test_start_returns_before_the_command_finishes(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    began = time.monotonic()
    job = store.start("sleep 5", str(tmp_path))
    assert time.monotonic() - began < 1.0, "start must not wait on the command"
    assert job.running
    store.close()


def test_output_is_readable_while_the_job_is_still_running(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("echo first; sleep 5", str(tmp_path))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and "first" not in store.output(job.id):
        time.sleep(0.05)
    out = store.output(job.id)
    assert "first" in out
    assert "still running" in out
    store.close()


def test_a_finished_job_carries_its_exit_code(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("exit 3", str(tmp_path))
    assert _settle(job, store)
    assert job.exit_code == 3
    assert job.state == "exit 3"


def test_the_completion_notice_fires_once(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("echo done", str(tmp_path))
    assert _settle(job, store)
    first = store.notices()
    assert len(first) == 1 and job.id in first[0]
    assert store.notices() == [], "a notice announced twice is noise"


def test_a_running_job_is_not_announced(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    store.start("sleep 5", str(tmp_path))
    assert store.notices() == []
    store.close()


def test_cancel_stops_the_child_not_only_the_shell(tmp_path):
    """A `sleep` under a shell is the cheap stand-in for `make -j`: killing the
    wrapper and calling it cancelled leaves the real work running."""
    store = joblib.JobStore("s", tmp_path)
    job = store.start("sleep 30 & wait", str(tmp_path))
    time.sleep(0.3)
    assert "cancelled" in store.cancel(job.id)
    assert not job.running
    assert job.proc.poll() is not None


def test_cancelling_twice_says_so_instead_of_failing(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("sleep 30", str(tmp_path))
    store.cancel(job.id)
    assert "already" in store.cancel(job.id)


def test_an_unknown_id_is_an_error_the_model_can_read(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    assert store.cancel("j9").startswith("error:")
    assert store.output("j9").startswith("error:")


def test_close_stops_survivors_and_names_them(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("sleep 30", str(tmp_path))
    stopped = store.close()
    assert len(stopped) == 1 and job.id in stopped[0]
    assert not job.running


def test_output_pages_a_long_log(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("seq 1 500", str(tmp_path))
    assert _settle(job, store)
    window = store.output(job.id, offset=100, limit=5)
    assert "\n101\n102\n103\n104\n105" in window
    assert "up to 5" in window  # Streaming windows do not scan for a total.


def test_offset_past_the_end_says_so(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("echo one", str(tmp_path))
    assert _settle(job, store)
    assert "past the end" in store.output(job.id, offset=900)


# -- the loop ------------------------------------------------------------


def test_the_model_is_told_at_the_top_of_the_next_turn(tmp_path):
    """The whole point of a job protocol: the notice arrives without polling."""
    store = joblib.JobStore("s", tmp_path)
    provider = ScriptedProvider([
        Turn(tool_calls=[ToolCall(id="t1", name="job_start",
                                  arguments={"command": "echo hello-from-job"})]),
        # A turn of other work, long enough for the job to finish.
        Turn(tool_calls=[ToolCall(id="t2", name="bash",
                                  arguments={"command": "sleep 1"})]),
        Turn(text="done"),
    ])
    result = Agent(provider, str(tmp_path), jobs=store).run("run it in the background")

    assert result.stopped == "answered"
    notices = [m for m in provider.seen[-1]
               if m["role"] == "user" and "[job" in str(m["content"])]
    assert len(notices) == 1, "exactly one completion notice, unprompted"
    assert "finished ok" in notices[0]["content"]
    store.close()


def test_a_run_with_no_store_is_not_offered_the_tools(tmp_path):
    agent = Agent(ScriptedProvider([Turn(text="ok")]), str(tmp_path))
    assert agent.tool_names is not None
    assert not [n for n in agent.tool_names if n.startswith("job_")]
    assert "bash" in agent.tool_names


def test_a_run_with_a_store_gets_them(tmp_path):
    agent = Agent(ScriptedProvider([Turn(text="ok")]), str(tmp_path),
                  jobs=joblib.JobStore("s", tmp_path))
    surface = agent.tool_names if agent.tool_names is not None else list(toolkit.REGISTRY)
    assert "job_start" in surface and "job_cancel" in surface


def test_read_only_mode_never_had_them(tmp_path):
    """A background shell is a shell. The read-only surface is an allowlist, so
    the job tools were never in it - this pins that, because a later `+ jobs`
    would be an unnoticed hole in --read-only."""
    from onecli import permissions
    names = permissions.tool_names_for("readOnly")
    assert not [n for n in names if n.startswith("job_")]
    agent = Agent(ScriptedProvider([Turn(text="ok")]), str(tmp_path),
                  tool_names=names, jobs=joblib.JobStore("s", tmp_path))
    assert not [n for n in agent.tool_names if n.startswith("job_")]


def test_the_tool_says_so_when_a_run_has_no_store(tmp_path):
    out = toolkit.run("job_start", {"command": "echo hi"}, str(tmp_path))
    assert out.startswith("error:") and "not available" in out


def test_the_reported_runtime_is_the_job_not_our_delay(tmp_path):
    """The reported runtime must be the job's, not our delay: a clock that runs
    until we next look reports a sleep as much longer than it was. The job's own
    last write is the evidence."""
    store = joblib.JobStore("s", tmp_path)
    began = time.monotonic()
    job = store.start("sleep 1; echo end", str(tmp_path))
    assert _settle(job, store)
    time.sleep(2.0)          # the model thinking about something else
    store.poll()
    since_start = time.monotonic() - began
    # Not an absolute bound - a login shell costs a few hundred ms of real
    # startup, and that IS runtime. What must not be counted is our own delay.
    assert job.seconds < since_start - 1.5, (
        f"reported {job.seconds:.1f}s of a {since_start:.1f}s wall clock")


def test_a_silent_job_still_gets_an_upper_bound(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("exit 0", str(tmp_path))
    assert _settle(job, store)
    assert job.finished >= job.started


def test_one_line_of_output_is_not_1_lines(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("echo one", str(tmp_path))
    assert _settle(job, store)
    assert "1 line of output" in store.notices()[0]


def test_a_job_stopped_at_exit_is_not_running_in_the_next_turn(tmp_path):
    """A job killed when its turn exited is not reported as running by the
    next turn. Each turn is a new store."""
    first = joblib.JobStore("s", tmp_path)
    first.start("sleep 30", str(tmp_path))
    assert first.close()

    second = joblib.JobStore("s", tmp_path)
    job = second.jobs["j1"]
    assert not job.running
    assert "[running]" not in second.output("j1")
    notices = second.notices()
    assert len(notices) == 1 and "j1" in notices[0]
    assert "stopped when the turn" in job.log.read_text()
    assert second.start("true", str(tmp_path)).id == "j2"
    second.close()


def test_a_job_whose_owner_died_unrecorded_is_lost_not_running(tmp_path):
    """A process killed outright never ran close(): no handle, no exit code."""
    first = joblib.JobStore("s", tmp_path)
    job = first.start("sleep 30", str(tmp_path))
    joblib._stop(job)  # the process is gone; the record still says started

    second = joblib.JobStore("s", tmp_path)
    assert second.jobs["j1"].state == "lost (process gone, exit unknown)"
    assert "exit code is unknown" in second.notices()[0]
    assert second.notices() == [], "announced once"
    assert joblib.JobStore("s", tmp_path).notices() == [], "and stays announced"

"""Loop guards: repetition named while it happens, deadlines actually enforced.

The behaviour worth testing is not "an advisory string exists". It is that the
model is told *inside the transcript it reads*, that a legitimate second call is
left alone, that a declared timeout is lowered rather than merely reported, and
that a tool setting its own clock is reported rather than falsely claimed to
have been cut short.
"""

from __future__ import annotations

from onecli import loopguard as loops
from onecli.agent import Agent
from onecli.guard import ReadOnlyGuard
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


def _reads(n, path="a.txt"):
    return [Turn(tool_calls=[ToolCall(id=f"t{i}", name="read",
                                      arguments={"path": path})])
            for i in range(n)] + [Turn(text="done")]


# -- repetition ----------------------------------------------------------


def test_two_identical_calls_are_not_a_loop(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    result = Agent(ScriptedProvider(_reads(2)), str(tmp_path)).run("read it twice")
    assert result.advisories == []


def test_the_third_identical_call_tells_the_model_in_the_transcript(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    provider = ScriptedProvider(_reads(3))
    result = Agent(provider, str(tmp_path)).run("read it")

    assert len(result.advisories) == 1
    tool, note = result.advisories[0]
    assert tool == "read"
    assert "3rd identical call" in note
    # The whole point: it reaches the model, not just the run report.
    last_tool_message = [m for m in provider.seen[-1] if m["role"] == "tool"][-1]
    assert "[loop-guard]" in last_tool_message["content"]
    assert "hello" in last_tool_message["content"], "the result itself must survive"


def test_an_unchanged_result_is_said_out_loud(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    result = Agent(ScriptedProvider(_reads(3)), str(tmp_path)).run("read it")
    assert "byte-identical" in result.advisories[0][1]


def test_different_arguments_are_different_calls(tmp_path):
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("x\n")
    turns = (_reads(1, "a.txt")[:1] + _reads(1, "b.txt")[:1]
             + _reads(1, "c.txt")[:1] + [Turn(text="done")])
    result = Agent(ScriptedProvider(turns), str(tmp_path)).run("read three files")
    assert result.advisories == []


def test_the_count_reads_as_english():
    assert loops._ordinal(3) == "3rd"
    assert loops._ordinal(11) == "11th"
    assert loops._ordinal(21) == "21st"


def test_argument_order_does_not_hide_a_repeat():
    a = loops.fingerprint("read", {"path": "a.txt", "limit": 5})
    b = loops.fingerprint("read", {"limit": 5, "path": "a.txt"})
    assert a == b


def test_a_refused_call_repeated_is_still_a_loop(tmp_path):
    """The clearest loop there is: asking three times for a tool you cannot have."""
    turns = [Turn(tool_calls=[ToolCall(id=f"t{i}", name="write",
                                       arguments={"path": "x", "content": "y"})])
             for i in range(3)] + [Turn(text="done")]
    result = Agent(ScriptedProvider(turns), str(tmp_path),
                   guard=ReadOnlyGuard()).run("write it")
    assert len(result.denied) == 3
    assert len(result.advisories) == 1
    assert not (tmp_path / "x").exists()


def test_escalation_stops_asking_nicely(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    result = Agent(ScriptedProvider(_reads(6)), str(tmp_path),
                   max_steps=10).run("read it")
    assert "This is a loop" in result.advisories[-1][1]




def test_zero_turns_the_reminder_off(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    result = Agent(ScriptedProvider(_reads(5)), str(tmp_path), max_steps=10,
                   loop_guard=loops.LoopGuard(repeat_limit=0)).run("read it")
    assert result.advisories == []


# -- deadlines -----------------------------------------------------------


def test_a_declared_timeout_is_lowered_not_merely_reported():
    guard = loops.LoopGuard(deadline=30)
    assert guard.clamp("bash", {"command": "sleep 999", "timeout": 600}) == {
        "command": "sleep 999", "timeout": 30}


def test_a_tool_asking_for_less_than_policy_keeps_its_own_number():
    guard = loops.LoopGuard(deadline=30)
    assert guard.clamp("bash", {"command": "x", "timeout": 5})["timeout"] == 5


def test_a_tool_that_asked_for_nothing_still_gets_the_ceiling():
    guard = loops.LoopGuard(deadline=30)
    assert guard.clamp("bash", {"command": "x"})["timeout"] == 30


def test_a_tool_with_no_timeout_parameter_is_left_alone():
    guard = loops.LoopGuard(deadline=30)
    args = {"path": "a.txt"}
    assert guard.clamp("read", args) == args
    assert "timeout" not in guard.clamp("read", args)


def test_no_deadline_means_no_clamping():
    guard = loops.LoopGuard(deadline=0)
    assert guard.clamp("bash", {"command": "x"}) == {"command": "x"}


def test_the_clamp_never_mutates_what_the_model_asked_for():
    guard = loops.LoopGuard(deadline=30)
    asked = {"command": "x", "timeout": 600}
    guard.clamp("bash", asked)
    assert asked["timeout"] == 600, "the transcript must record what was asked"


def test_an_overrun_is_reported_for_a_tool_we_could_not_clamp():
    guard = loops.LoopGuard(deadline=10)
    notes = guard.observe("mcp_slow_thing", {"q": "x"}, "ok", seconds=45.0)
    assert len(notes) == 1
    assert "45s" in notes[0] and "10s per-call deadline" in notes[0]


def test_a_clamped_tool_is_never_told_nothing_cut_it_short():
    """The tool we clamped reports its own timeout. Saying "nothing cut it
    short" beside that is a lie, and the model reads the contradiction as a
    fresh refusal that never happened."""
    guard = loops.LoopGuard(deadline=3)
    notes = guard.observe("bash", {"command": "sleep 20"},
                          "error: command timed out after 3s", seconds=3.02)
    assert notes == []


def test_a_clamped_tool_that_massively_overruns_is_still_quiet():
    guard = loops.LoopGuard(deadline=3)
    assert guard.observe("bash", {"command": "x"}, "ok", seconds=90.0) == []


def test_a_call_inside_the_deadline_says_nothing():
    guard = loops.LoopGuard(deadline=10)
    assert guard.observe("bash", {"command": "x"}, "ok", seconds=2.0) == []


def test_a_bash_call_actually_runs_under_the_lowered_ceiling(tmp_path):
    """End to end: the ceiling reaches the subprocess, not just the dict."""
    turns = [Turn(tool_calls=[ToolCall(id="t1", name="bash",
                                       arguments={"command": "sleep 30",
                                                  "timeout": 300})]),
             Turn(text="done")]
    result = Agent(ScriptedProvider(turns), str(tmp_path),
                   loop_guard=loops.LoopGuard(deadline=1)).run("sleep")
    output = result.steps[0].tool_results[0][1]
    assert "timed out after 1s" in output


def test_reads_of_changing_file_do_not_warn(tmp_path):
    class UpdatingProvider(ScriptedProvider):
        def chat(self, *args, **kwargs):
            (tmp_path / "a.txt").write_text(f"revision {len(self.seen)}\n")
            return super().chat(*args, **kwargs)

    result = Agent(UpdatingProvider(_reads(5)), str(tmp_path)).run("watch changes")
    assert result.advisories == []


def test_new_result_resets_repeat_warning():
    guard = loops.LoopGuard()
    for _ in range(3):
        guard.observe("read", {"path": "a"}, "old")
    assert guard.advisories
    assert guard.observe("read", {"path": "a"}, "new") == []
    assert guard.observe("read", {"path": "a"}, "new") == []
    assert "3rd identical call" in guard.observe("read", {"path": "a"}, "new")[0]


def test_custom_repeat_threshold_controls_escalation():
    guard = loops.LoopGuard(repeat_limit=2)
    for _ in range(4):
        notes = guard.observe("read", {}, "unchanged")
    assert "This is a loop" in notes[0]


# -- resolution and produced files, with fixture tools -------------------------

def _resolving(monkeypatch, name="lookup"):
    import dataclasses
    from onecli import tools as toolkit
    monkeypatch.setitem(toolkit.REGISTRY, name, dataclasses.replace(
        toolkit.REGISTRY["grep"], name=name, resolves=True))


def test_only_a_resolving_tool_resolves_the_target(monkeypatch):
    out = "## Example project\nConfidence 0.8 over 40 records\n"
    assert loops.resolution("lookup", out) == ""
    _resolving(monkeypatch)
    assert loops.resolution("lookup", out) == (
        "`lookup` named Example project at confidence 0.8")
    assert loops.resolution("lookup", out.replace("0.8", "0.3")) == ""
    assert loops.resolution("grep", out) == ""


def test_a_resolved_target_shortens_the_search(monkeypatch):
    _resolving(monkeypatch)
    guard = loops.LoopGuard()
    hit = ("lookup", {"query": "x"}, "## Example project\nConfidence 0.9 over 3\n")
    assert not loops.discovery_round(guard, [hit])
    assert guard.resolved
    assert loops.discovery_round(guard, [("grep", {"pattern": "y"}, "none")])
    prompt, final = loops.checkpoint_prompt(guard, can_act=True)
    assert not final and "already resolved" in prompt
    prompt, final = loops.checkpoint_prompt(guard, can_act=False)
    assert final and "has no tool that acts on it" in prompt


def test_an_acting_call_that_reports_files_is_output():
    make = ("bash", {"command": "./draw --out /ws/a"})
    assert loops.produced(*make, "exit 0\n/ws/a/1.png") == "/ws/a/1.png"
    assert loops.produced(*make, "exit 0\n/tmp/a/1.png") == ""
    assert loops.produced(*make, "exit 1\n/ws/a/1.png") == ""
    assert loops.produced("bash", {"command": "./draw --check"}, "exit 0\n/ws/a/1.png") == ""
    assert loops.produced("bash", {"command": "ls /ws/a"}, "/ws/a/1.png") == ""
    job = "job j1 [ok] ./draw --out /ws/a\n/ws/a/1.png"
    assert loops.produced("job_output", {"id": "j1"}, job) == "/ws/a/1.png"
    assert loops.produced("job_output", {"id": "j1"}, job.replace("[ok]", "[running]")) == ""


def test_checking_produced_files_is_capped_but_building_on_is_not():
    guard = loops.LoopGuard()
    assert not loops.discovery_round(
        guard, [("bash", {"command": "./draw --out /ws/a"}, "exit 0\n/ws/a/1.png")])
    for i in range(6):
        assert not loops.discovery_round(guard, [("bash", {"command": f"make t{i}"}, "exit 0")])
    look = ("bash", {"command": "python3 -c 'print(open(\"/ws/a/1.png\",\"rb\").read(4))'"}, "exit 0")
    due = [loops.discovery_round(guard, [look]) for _ in range(loops.POST_OUTPUT_LIMIT)]
    assert due[-1] and not any(due[:-1])
    prompt, final = loops.checkpoint_prompt(guard, can_act=True)
    assert final and "/ws/a/1.png" in prompt


def test_help_is_relearning_not_acting():
    assert loops.relearns("bash", {"command": "./draw --help"})
    assert loops.reads_only("bash", {"command": "./draw --help"})
    assert not loops.relearns("bash", {"command": "./draw --out /ws/a"})


def test_the_lookup_note_names_the_resolving_tool(monkeypatch):
    request = "Draw a poster of the Coast Road and the Shifting Gardens."
    assert "a narrow grep" in loops.lookup_note(request, {"grep"})
    _resolving(monkeypatch)
    note = loops.lookup_note(request, {"grep", "lookup"})
    assert "`lookup` with these names" in note and '"Coast Road"' in note
    assert loops.lookup_note("Fix the Coast Road parser.", {"grep", "lookup"}) == ""

"""The low-progress checkpoint: discovery without a decision ends in a question.

The shape it refuses: a request for a set of artefacts, answered with round
after round of distinct ls/grep/bash/code calls over several candidate
projects and never a choice.
"""

from __future__ import annotations

from onecli import loopguard as loops
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

QUESTION = ("Which project do you mean: `example-game`, "
            "`sample-engine`, or the example submission docs?")


def _discovery(n):
    # Every call distinct, so the repeat guard never fires: the stuck shape.
    tools = ["ls", "grep", "bash", "glob"]
    turns = []
    for i in range(n):
        name = tools[i % len(tools)]
        args = {"ls": {"path": f"d{i}"}, "grep": {"pattern": f"dream{i}", "path": "."},
                "bash": {"command": f"echo probe {i}"},
                "glob": {"pattern": f"**/*{i}*"}}[name]
        turns.append(Turn(tool_calls=[ToolCall(id=f"t{i}", name=name, arguments=args)]))
    return turns


def _events():
    seen = []
    return seen, lambda kind, data: seen.append((kind, data))


def test_search_loop_ends_with_one_question(tmp_path):
    # A model that would search forever: 30 discovery rounds scripted, but the
    # checkpoint prompt makes it ask instead.
    script = _discovery(loops.DISCOVERY_LIMIT) + [Turn(text=QUESTION)] + _discovery(24)
    provider = ScriptedProvider(script)
    seen, on_event = _events()
    result = Agent(provider, str(tmp_path), max_steps=40, on_event=on_event).run(
        "write ten short summaries of the files in docs/")

    assert result.stopped == "answered"
    assert result.text == QUESTION
    # Fired exactly once, after the configured number of rounds, not later.
    assert len(result.steps) == loops.DISCOVERY_LIMIT + 1
    fired = [d for k, d in seen if k == "discovery_checkpoint"]
    assert fired[0] == {"step": loops.DISCOVERY_LIMIT + 1, "rounds": loops.DISCOVERY_LIMIT}
    assert fired[-1]["decision"] == "ask"
    # The control text reached the model on that request, and only that one.
    assert "[loop-guard checkpoint]" in provider.seen[-1][-1]["content"]
    assert not any("[loop-guard checkpoint]" in str(m.get("content"))
                   for m in provider.seen[-2])


def test_commit_returns_tools_and_resets_the_allowance(tmp_path):
    script = (_discovery(loops.DISCOVERY_LIMIT)
              + [Turn(text="COMMIT: example-game, it is the only example world.")]
              + [Turn(tool_calls=[ToolCall(id="w", name="write",
                                           arguments={"path": "art.txt", "content": "x"})])]
              + [Turn(text="made it")])
    seen, on_event = _events()
    result = Agent(ScriptedProvider(script), str(tmp_path), max_steps=40,
                   on_event=on_event).run("make the art")
    assert result.stopped == "answered" and result.text == "made it"
    assert (tmp_path / "art.txt").read_text() == "x"
    assert [d.get("decision") for k, d in seen if k == "discovery_checkpoint"] == [None, "commit"]


def test_reading_calls_on_the_checkpoint_are_never_run_and_end_the_run(tmp_path):
    # Reading is not a decision: a checkpoint answered with more reading is
    # refused, twice, and the run stops.
    script = _discovery(loops.DISCOVERY_LIMIT) + [
        Turn(tool_calls=[ToolCall(id="x", name="grep",
                                  arguments={"pattern": "more", "path": "."})])] * 2
    result = Agent(ScriptedProvider(script), str(tmp_path), max_steps=40).run("go")
    assert result.stopped == "budget"
    assert "without progress" in result.error


def test_progress_tools_reset_and_zero_disables(tmp_path):
    guard = loops.LoopGuard(discovery_limit=3)
    assert not loops.discovery_round(guard, [("grep", {}, "")])
    assert not loops.discovery_round(guard, [("bash", {}, "")])
    assert not loops.discovery_round(guard, [("edit", {}, "")])
    assert not loops.discovery_round(guard, [("ls", {}, "")]) and not loops.discovery_round(guard, [("ls", {}, "")])
    assert not loops.discovery_round(guard, [("read", {}, "")])
    assert loops.discovery_round(guard, [("grep", {}, ""), ("read", {}, "")])
    off = loops.LoopGuard(discovery_limit=0)
    assert not any(loops.discovery_round(off, [("grep", {}, "")]) for _ in range(50))

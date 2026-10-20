"""Resolved, so stop reading: a confident lookup and a spent commit end the turn.

A synthetic fixture: the first lookup names the target at confidence 0.67,
yet without the guard every checkpoint gets a `COMMIT:` line and then more
reading, because a commit would restore the full discovery allowance.
"""

from __future__ import annotations

import dataclasses
import importlib

from onecli import loopguard as loops
from onecli import permissions
from onecli import tools as toolkit
from onecli.agent import Agent
from onecli.providers.base import ToolCall, Turn

RECENT = ("2 matches, most recent first:\n"
          "## project-a\n"
          "Path: /srv/projects/project-a\n"
          "Confidence 0.67 over 12 notes read\n")
WEAK = RECENT.replace("Confidence 0.67", "Confidence 0.33")
PLAN = "Plan: draw 10 sketches with the image tool and save them under art/. Go?"
REQUEST = "draw 10 sketches of project-a and save them to art/"


class StubbornModel:
    """Reads for ever; at a checkpoint it always commits. On a final checkpoint it gives the plan."""

    def __init__(self, first=None):
        self.n = 0
        self.first = first or [ToolCall(id="r", name="recent",
                                        arguments={"query": "project-a"}),
                               ToolCall(id="g", name="grep",
                                        arguments={"pattern": "project-a", "path": "."})]
        self.seen: list[str] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        last = str(messages[-1].get("content"))
        self.seen.append(last)
        if "[loop-guard checkpoint]" in last:
            if "Tools stay withheld" in last:
                return Turn(text=PLAN)
            return Turn(text="COMMIT: project-a, image tool.")
        self.n += 1
        if self.n == 1:
            return Turn(tool_calls=self.first)
        return Turn(tool_calls=[
            ToolCall(id=f"g{self.n}", name="grep",
                     arguments={"pattern": f"project-a{self.n}", "path": "."}),
            ToolCall(id=f"l{self.n}", name="ls", arguments={"path": f"d{self.n}"})])


def _fake_tools(monkeypatch, recent_output=RECENT):
    real = toolkit.run

    def run(name, arguments, workspace, **kw):
        if name == "recent":
            return recent_output
        if name in ("grep", "ls"):
            return f"{name}: nothing new"
        return real(name, arguments, workspace, **kw)
    monkeypatch.setattr(toolkit, "run", run)
    # A lookup tool that declares it can resolve the request's target.
    monkeypatch.setitem(toolkit.REGISTRY, "recent", dataclasses.replace(
        toolkit.REGISTRY["grep"], name="recent", resolves=True))


def _calls(result):
    return sum(len(s.tool_results) for s in result.steps)


def test_plan_mode_commits_right_after_the_confident_lookup(tmp_path, monkeypatch):
    _fake_tools(monkeypatch)
    names = permissions.tool_names_for("plan") + ["recent"]
    model = StubbornModel()
    result = Agent(model, str(tmp_path), max_steps=60, tool_names=names).run(REQUEST)
    assert result.stopped == "answered" and result.text == PLAN
    # Lookup round plus one more, then the answer.
    assert _calls(result) <= 4
    assert "`recent` named project-a" in model.seen[-1]


def test_act_mode_gets_one_commit_then_must_answer(tmp_path, monkeypatch):
    _fake_tools(monkeypatch)
    model = StubbornModel()
    result = Agent(model, str(tmp_path), max_steps=60).run(REQUEST)
    assert result.stopped == "answered" and result.text == PLAN
    checkpoints = [s for s in model.seen if "[loop-guard checkpoint]" in s]
    assert len(checkpoints) == 2 and "Tools stay withheld" in checkpoints[-1]
    bound = 2 + 2 * (loops.RESOLVED_LIMIT + loops.POST_COMMIT_LIMIT)
    assert _calls(result) <= bound


def test_a_weak_lookup_does_not_resolve(tmp_path, monkeypatch):
    _fake_tools(monkeypatch, WEAK)
    model = StubbornModel()
    result = Agent(model, str(tmp_path), max_steps=60).run(REQUEST)
    first = next(s for s in model.seen if "[loop-guard checkpoint]" in s)
    assert "searching without writing" in first
    assert result.stopped == "answered"


class Builder:
    """A long multi-file build: searches, edits and runs between edits, and
    commits once at a checkpoint before it starts editing."""

    def __init__(self, files):
        self.files, self.n, self.checkpoints = files, 0, 0

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        if "[loop-guard checkpoint]" in str(messages[-1].get("content")):
            self.checkpoints += 1
            return Turn(text="COMMIT: edit the modules in order.")
        self.n += 1
        if self.n <= loops.DISCOVERY_LIMIT:
            return Turn(tool_calls=[ToolCall(id=f"s{self.n}", name="grep",
                        arguments={"pattern": f"x{self.n}", "path": "."})])
        work = self.n - loops.DISCOVERY_LIMIT - 1
        index, phase = divmod(work, 4)
        if index >= len(self.files):
            return Turn(text="all modules edited and tested")
        if phase == 0:
            return Turn(tool_calls=[ToolCall(id=f"w{self.n}", name="write", arguments={
                "path": self.files[index], "content": f"v{index}\n"})])
        return Turn(tool_calls=[ToolCall(id=f"b{self.n}", name="bash",
                    arguments={"command": f"echo test {self.n}"})])


def test_a_long_multi_file_edit_keeps_working(tmp_path):
    files = [f"m{i}.py" for i in range(8)]
    model = Builder(files)
    result = Agent(model, str(tmp_path), max_steps=200).run("refactor all modules")
    assert result.stopped == "answered" and result.text == "all modules edited and tested"
    assert all((tmp_path / f).read_text() == f"v{i}\n" for i, f in enumerate(files))
    # One checkpoint from the opening search; the edits never trip another.
    assert model.checkpoints == 1


def test_checkpoint_decision_hot_reloads_into_a_live_agent(tmp_path, monkeypatch):
    _fake_tools(monkeypatch)
    names = permissions.tool_names_for("plan") + ["recent"]
    model = StubbornModel()
    agent = Agent(model, str(tmp_path), max_steps=60, tool_names=names)
    guard = agent.loop
    original = loops.checkpoint_prompt
    monkeypatch.setattr(loops, "checkpoint_prompt",
                        lambda g, can_act: ("[loop-guard checkpoint] Tools stay withheld. NEW", True))
    agent.run(REQUEST)
    assert model.seen[-1].endswith("NEW")
    monkeypatch.setattr(loops, "checkpoint_prompt", original)
    importlib.reload(loops)
    model.n, model.seen = 0, []
    result = agent.run(REQUEST)
    assert agent.loop is guard
    assert "`recent` named project-a" in model.seen[-1] and result.text == PLAN


def test_checkpoints_reach_the_json_event_stream(tmp_path, monkeypatch):
    import io
    import json

    from onecli import events
    _fake_tools(monkeypatch)
    out = io.StringIO()
    emit = events.emitter(out, step=lambda: 0)
    names = permissions.tool_names_for("plan") + ["recent"]
    Agent(StubbornModel(), str(tmp_path), max_steps=60, tool_names=names,
          on_event=emit).run(REQUEST)
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    fired = [r for r in rows if r["e"] == "discovery_checkpoint"]
    assert fired[0] == {"e": "discovery_checkpoint", "step": fired[0]["step"],
                        "rounds": 1, "final": True, "resolved": True}
    assert "project-a" not in out.getvalue().split("discovery_checkpoint")[1][:200]


def test_a_committed_shell_action_is_not_cut_short(tmp_path, monkeypatch):
    # Act mode after a commit: draw, convert, save, verify - all bash, no
    # write. The commit window counts searching, not the committed action.
    _fake_tools(monkeypatch)
    actions = [f"echo step {i}" for i in range(6)]

    class Actor(StubbornModel):
        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            last = str(messages[-1].get("content"))
            if "[loop-guard checkpoint]" in last:
                self.seen.append(last)
                return Turn(text="COMMIT: draw then save.")
            self.n += 1
            if self.n == 1:
                return Turn(tool_calls=self.first)
            if self.n - 2 < len(actions):
                return Turn(tool_calls=[ToolCall(id=f"b{self.n}", name="bash",
                            arguments={"command": actions[self.n - 2]})])
            return Turn(text="saved and verified")

    model = Actor()
    result = Agent(model, str(tmp_path), max_steps=60).run(REQUEST)
    assert result.stopped == "answered" and result.text == "saved and verified"
    assert not any("Tools stay withheld" in s for s in model.seen)

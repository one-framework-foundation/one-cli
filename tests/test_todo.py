"""The plan: a list that is state rather than a message.

Three claims are worth a test each, because each one is a way the list quietly
stops being true: a write that half-lands, a plan that a compaction eats, and a
resume that starts a second list beside the first.
"""

from __future__ import annotations

import json

from onecli import render, session
from onecli import todo as todolib
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


def _write_turn(call_id: str, items) -> Turn:
    return Turn(tool_calls=[ToolCall(id=call_id, name="todo_write",
                                     arguments={"items": items})])


PLAN = [{"text": "read the config", "state": "done"},
        {"text": "patch the wire", "state": "in_progress"},
        {"text": "add a test", "state": "pending"}]


# --- writing ------------------------------------------------------------------

def test_a_write_replaces_the_list_and_reports_it_back():
    todos = todolib.TodoList()
    out = todos.write(PLAN)
    assert [it.state for it in todos.items] == ["done", "in_progress", "pending"]
    assert "1/3 done" in out and "[>] patch the wire" in out
    assert todos.summary() == "1/3 done, now: patch the wire"


def test_two_items_in_progress_is_refused_and_changes_nothing():
    todos = todolib.TodoList()
    todos.write(PLAN)
    before = todos.as_dicts()
    out = todolib.todo_write(todos=todos, items=[
        {"text": "a", "state": "in_progress"},
        {"text": "b", "state": "in_progress"}])
    assert out.startswith("error:") and "in_progress" in out
    assert todos.as_dicts() == before
    assert todos.revision == 1


def test_an_unknown_state_is_refused_rather_than_guessed():
    todos = todolib.TodoList()
    out = todolib.todo_write(todos=todos, items=[{"text": "a", "state": "maybe"}])
    assert "maybe" in out and "pending" in out
    assert todos.items == []


def test_the_spellings_models_actually_emit_are_accepted():
    todos = todolib.TodoList()
    todos.write([{"content": "a", "status": "completed"},
                 {"text": "b", "state": "doing"},
                 "c"])
    assert [it.state for it in todos.items] == ["done", "in_progress", "pending"]
    assert todos.items[2].text == "c"


def test_a_plan_longer_than_the_ceiling_is_refused():
    todos = todolib.TodoList()
    out = todolib.todo_write(todos=todos, items=[{"text": f"step {i}"}
                                                 for i in range(todolib.MAX_ITEMS + 1)])
    assert "too long" in out and todos.items == []


def test_a_run_without_a_list_says_so_instead_of_pretending():
    assert todolib.todo_write(todos=None, items=PLAN).startswith("error:")


# --- durability ---------------------------------------------------------------

def test_a_write_is_persisted_and_a_resume_picks_it_back_up(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    todolib.TodoList(session_id="s1").write(PLAN)

    restored = todolib.load("s1")
    assert restored.as_dicts() == PLAN
    # The flag is what makes the run restate it to the model exactly once.
    assert restored.restored is True
    # And no change is pending: a restored list has not been edited.
    assert restored.take_change() is None


def test_every_revision_is_kept_so_the_plan_has_a_history(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    todos = todolib.TodoList(session_id="s2")
    todos.write([{"text": "a", "state": "pending"}])
    todos.write([{"text": "a", "state": "done"}])
    records = session.load_records("s2", "todos")
    assert [r["revision"] for r in records] == [1, 2]
    assert todolib.load("s2").as_dicts() == [{"text": "a", "state": "done"}]


def test_a_session_with_no_plan_loads_an_empty_one(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    assert todolib.load("never-existed").items == []
    session.append("s3", {"kind": "messages", "messages": []})
    assert todolib.load("s3").items == []


def test_a_malformed_record_loses_the_plan_not_the_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    session.append("s4", {"kind": "todos", "revision": 1,
                          "items": [{"text": "a", "state": "banana"}]})
    todos = todolib.load("s4")
    assert todos.items == [] and todos.restored is False


# --- what the loop asks it ----------------------------------------------------

def test_a_change_is_reported_once():
    todos = todolib.TodoList()
    todos.write(PLAN)
    assert todos.take_change() is not None
    assert todos.take_change() is None


def test_an_untouched_plan_is_raised_after_a_stretch_of_silence():
    todos = todolib.TodoList(stale_after=3)
    todos.write(PLAN)
    assert todos.notices() == [] and todos.notices() == []
    notes = todos.notices()
    assert notes and "unchanged for 3 steps" in notes[0]
    assert "patch the wire" in notes[0]
    # Re-armed by the next write, not still firing every step.
    todos.write(PLAN)
    assert todos.notices() == []


def test_a_finished_plan_is_never_nudged():
    todos = todolib.TodoList(stale_after=1)
    todos.write([{"text": "a", "state": "done"}])
    assert todos.notices() == [] and todos.notices() == []


# --- the loop -----------------------------------------------------------------

def test_the_tool_is_withheld_from_a_run_with_no_list(tmp_path):
    agent = Agent(ScriptedProvider([Turn(text="ok")]), str(tmp_path))
    assert "todo_write" not in (agent.tool_names or [])
    with_list = Agent(ScriptedProvider([Turn(text="ok")]), str(tmp_path),
                      todos=todolib.TodoList())
    assert with_list.tool_names is None or "todo_write" in with_list.tool_names


def test_writing_the_plan_renders_it_between_turns(tmp_path):
    todos = todolib.TodoList()
    events = []
    provider = ScriptedProvider([_write_turn("t1", PLAN), Turn(text="done")])
    result = Agent(provider, str(tmp_path), todos=todos,
                   on_event=lambda k, d: events.append((k, d))).run("go")
    plans = [d for k, d in events if k == "todos"]
    assert len(plans) == 1 and plans[0]["reason"] == "changed"
    assert plans[0]["summary"] == "1/3 done, now: patch the wire"
    assert result.todos == PLAN
    # And the model saw the list it just wrote, as the tool result.
    tool_msg = [m for m in result.messages if m.get("role") == "tool"][0]
    assert "[x] read the config" in tool_msg["content"]


def test_a_refused_write_renders_nothing(tmp_path):
    events = []
    provider = ScriptedProvider([
        _write_turn("t1", [{"text": "a", "state": "in_progress"},
                           {"text": "b", "state": "in_progress"}]),
        Turn(text="done")])
    Agent(provider, str(tmp_path), todos=todolib.TodoList(),
          on_event=lambda k, d: events.append((k, d))).run("go")
    assert [k for k, _ in events].count("todos") == 0


def test_the_plan_is_restated_after_a_compaction(tmp_path):
    """The tool result carrying the list is summary material. The list is not."""
    todos = todolib.TodoList()
    todos.write(PLAN)
    # Two tool rounds: with one, the only thing older than the kept tail is the
    # request, which compaction now carries across verbatim instead of
    # summarizing, so there is correctly nothing to compact.
    provider = ScriptedProvider([
        Turn(tool_calls=[ToolCall(id="c0", name="ls", arguments={})]),
        Turn(tool_calls=[ToolCall(id="c1", name="ls", arguments={})]),
        Turn(text="a summary of what happened"),   # the summarization call
        Turn(text="done"),
    ])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=5,
                  todos=todos, autocompact=True, context_window=200,
                  compact_at=0.5, keep_last=2)
    events = []
    agent.on_event = lambda k, d: events.append((k, d))
    result = agent.run("x" * 600)
    assert result.compactions and result.compactions[0]["ok"]
    # A restatement is not a change: the host draws it as a line, not as the
    # whole block, because nothing about the plan is new.
    # The first event is this test's own pre-run write being noticed once; the
    # one that matters is the last.
    reasons = [d["reason"] for k, d in events if k == "todos"]
    assert reasons[-1] == "compaction"
    restated = [m for m in result.messages
                if m.get("role") == "user" and "[plan -" in str(m.get("content"))]
    assert restated and "patch the wire" in restated[-1]["content"]


def test_a_resumed_plan_is_restated_once_before_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    todolib.TodoList(session_id="s5").write(PLAN)
    todos = todolib.load("s5")

    provider = ScriptedProvider([Turn(text="one"), Turn(text="two")])
    agent = Agent(provider, str(tmp_path), todos=todos)
    first = agent.run("go")
    injected = [m for m in first.messages
                if m.get("role") == "user" and "[plan -" in str(m.get("content"))]
    assert len(injected) == 1 and "resumed session" in injected[0]["content"]
    # The plan arrives BEFORE the prompt, so it reads as context the model
    # carries in rather than as the thing being asked for.
    roles = [str(m.get("content")) for m in first.messages]
    assert roles.index(injected[0]["content"]) < roles.index("go")

    second = agent.run("again", messages=first.messages)
    again = [m for m in second.messages
             if m.get("role") == "user" and "resumed session" in str(m.get("content"))]
    assert len(again) == 1


def test_a_child_gets_no_plan_of_its_own(tmp_path):
    """A list nobody renders and nobody resumes is bookkeeping that dies with
    the child - the same reason a child has no jobs and nobody to ask."""
    from onecli import subagent as sublib

    parent = Agent(ScriptedProvider([Turn(text="ok")]), str(tmp_path),
                   todos=todolib.TodoList(),
                   spawner=sublib.Spawner(parent=None, depth=0, max_depth=2,
                                          presets={}))
    child = parent.spawner.build(sublib.Preset(name="general"))
    assert child.todos is None
    assert "todo_write" not in (child.tool_names or [])


# --- the drawing --------------------------------------------------------------

def test_the_plan_draws_one_row_per_item_with_a_count():
    out = render.plan(PLAN)
    assert out.count("\n") == 3
    assert "1/3" in out
    assert render.plan([]).strip().endswith("plan cleared")


def test_the_cli_reports_the_plan_it_ended_with(tmp_path, monkeypatch, capsys):
    """`--json` carries the list, so a caller can tell a finished plan from an
    abandoned one without reading the prose."""
    from onecli.cli import main

    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    provider = ScriptedProvider([_write_turn("t1", PLAN), Turn(text="done")])
    monkeypatch.setattr("onecli.providers.build", lambda *a, **k: provider)
    # The first-response gate is on by default and needs a structured
    # assessment turn; this test is about the --json payload, so it is off.
    code = main(["--json", "--response-gate", "off", "-m", "test",
                 "--base-url", "http://x/v1",
                 "-w", str(tmp_path), "plan something"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["todos"] == PLAN

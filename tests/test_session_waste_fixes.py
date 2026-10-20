"""Replays of the top patterns from `scripts/session_waste.py`.

1. The same refusal, reworded: a long build (session-0001 in the fixture) hits
   the shell's recursive-grep refusal again and again with a different command
   each time; a repeat guard keyed on arguments would never speak.
2. Path guessing: a not-found result now names where the path stops existing
   and the closest names there.
3. A blind edit retry: a failed edit now shows the closest file lines.

The fixture keeps tool names only (commands, paths and text stripped).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from onecli import loopguard as loops
from onecli.agent import Agent
from onecli.providers.base import ToolCall, Turn
from onecli import tools as toolkit
from onecli.tools import fs, search

FIXTURE = Path(__file__).parent / "fixtures" / "waste" / "refusal_rounds.json"
REFUSAL_NOTE = "refused 2 different calls with the same reason"


class ReplayModel:
    """Plays the recorded round shapes back, one round per step. Every
    recorded refusal becomes a fresh recursive shell grep; at a checkpoint it
    commits, as the live models did."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.n = 0
        self.results: list[str] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        last = messages[-1]
        if last.get("role") == "tool":
            self.results += [m["content"] for m in messages[-8:] if m.get("role") == "tool"]
        if "[loop-guard checkpoint]" in str(last.get("content")):
            return Turn(text="COMMIT: keep building the change.")
        if self.n >= len(self.rounds):
            return Turn(text="done")
        calls = []
        for i, kind in enumerate(self.rounds[self.n]):
            cid = f"c{self.n}_{i}"
            if kind == "refused":
                calls.append(ToolCall(cid, "bash", {"command": f"grep -rn word{self.n}_{i} ."}))
            elif kind in ("edit", "write"):
                calls.append(ToolCall(cid, "write", {"path": f"f{self.n}_{i}.txt", "content": "x"}))
            elif kind == "bash":
                calls.append(ToolCall(cid, "bash", {"command": f"echo step {self.n}_{i}"}))
            else:
                calls.append(ToolCall(cid, "ls", {"path": "."}))
        self.n += 1
        return Turn(tool_calls=calls)


def _replay(tmp_path, monkeypatch):
    real = toolkit.run

    def run(name, arguments, workspace, **kw):
        # Recorded shell steps succeed instantly; refusals go through the real tool.
        if name == "bash" and arguments["command"].startswith("echo step"):
            return "exit 0\nok"
        return real(name, arguments, workspace, **kw)
    monkeypatch.setattr(toolkit, "run", run)
    rounds = json.loads(FIXTURE.read_text())["rounds"]
    model = ReplayModel(rounds)
    result = Agent(model, str(tmp_path), max_steps=len(rounds) * 2 + 20).run("build it")
    notes = [r for r in model.results if "[loop-guard] bash has now refused" in r]
    return rounds, model, result, notes


def test_reworded_refusals_are_named_from_the_second_one(tmp_path, monkeypatch):
    rounds, model, result, notes = _replay(tmp_path, monkeypatch)
    refusals = sum(r.count("refused") for r in rounds)
    assert refusals == 17
    # Before: 0 notes - every command differed, so the repeat guard was silent.
    assert notes and REFUSAL_NOTE in notes[0]
    assert "Use the grep tool" in notes[0]




def test_the_recorded_long_build_still_runs_to_the_end(tmp_path, monkeypatch):
    rounds, model, result, notes = _replay(tmp_path, monkeypatch)
    assert model.n == len(rounds) and result.stopped == "answered"
    assert result.text == "done"


def test_a_path_refusal_with_different_paths_is_not_one_refusal():
    guard = loops.LoopGuard()
    for name in ("a.py", "b.py", "c.py"):
        notes = guard.observe("read", {"path": name}, f"error: no such file: /w/{name}")
        assert not [n for n in notes if "same reason" in n]


def test_refusal_decision_hot_reloads_into_a_live_guard(monkeypatch):
    guard = loops.LoopGuard()
    monkeypatch.setattr(loops, "refusal_note", lambda g, n, k, o: "NEW NOTE")
    assert "NEW NOTE" in guard.observe("bash", {"command": "x"}, "exit 0\n")
    monkeypatch.undo()
    importlib.reload(loops)
    refused = "error: recursive shell grep is disabled"
    guard.observe("bash", {"command": "grep -r a ."}, refused)
    assert any(REFUSAL_NOTE in n
               for n in guard.observe("bash", {"command": "grep -r b ."}, refused))


def test_a_guessed_path_names_the_real_one(tmp_path):
    (tmp_path / "one_cc" / "pipeline").mkdir(parents=True)
    (tmp_path / "one_cc" / "pipeline" / "lane_pool.py").write_text("x\n")
    out = fs.read(str(tmp_path), "one_cli/pipeline/lane_pool.py")
    assert out.startswith("error: no such file: ")
    assert f"{tmp_path} exists; one_cli does not. Closest names there: one_cc" in out
    out = fs.ls(str(tmp_path), "one_cc/pipeline/lane_pol.py")
    assert "Closest names there: lane_pool.py" in out
    out = search.search(str(tmp_path), "x", "one_cx")
    assert "Closest names there: one_cc" in out


def test_a_failed_edit_shows_the_closest_lines(tmp_path):
    f = tmp_path / "pool.py"
    f.write_text("def admit(self, job):\n    return self.slots > 0\n")
    out = fs.edit(str(tmp_path), "pool.py", "def admit(self, job) :\n  return x", "y")
    assert out.startswith("error: old_string not found")
    assert "     1\tdef admit(self, job):" in out
    assert f.read_text().startswith("def admit(self, job):")


def test_the_analyzer_scores_a_transcript(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import session_waste

    refused = "error: recursive shell grep is disabled for project searches."
    calls, msgs = [], [{"role": "user", "content": "go"}]
    for i in range(4):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"t{i}", "function": {"name": "bash",
                                         "arguments": json.dumps({"command": f"grep -r w{i} ."})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": refused})
        calls.append(i)
    msgs.append({"role": "assistant", "content": "done"})
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"kind": "messages", "messages": msgs}) + "\n")
    report = session_waste.scan(tmp_path, days=1, budget=30, examples=1)
    assert report["complete"] and report["with_tools"] == 1
    assert {r["pattern"]: r["calls_wasted"] for r in report["ranked"]} == {"repeat_failure": 2}


def test_a_glob_under_a_guessed_directory_names_the_real_one(tmp_path):
    (tmp_path / "onecli" / "tools").mkdir(parents=True)
    (tmp_path / "onecli" / "tools" / "fs.py").write_text("x\n")
    out = search.glob_files(str(tmp_path), "one_cli/tools/filesys.py")
    assert out.startswith("(no files matching")
    assert "one_cli does not. Closest names there: onecli" in out
    assert "Closest names there: fs.py" in search.glob_files(str(tmp_path), "onecli/tools/f.py")
    # A bare name searches the whole tree; a root-level guess would mislead.
    assert "Closest" not in search.glob_files(str(tmp_path), "filesys.py")


def test_the_not_found_hint_hot_reloads_through_the_registered_tools(tmp_path, monkeypatch):
    # The registry binds fs.read and fs.edit at import; the hints are module
    # functions they look up per call, so a reload of fs reaches them.
    monkeypatch.setattr(fs, "missing", lambda p, what="file", path_guards=None: "NEW MISSING")
    monkeypatch.setattr(fs, "_nearest", lambda text, old: " NEW NEAREST")
    (tmp_path / "a.py").write_text("x = 1\n")
    assert toolkit.REGISTRY["read"].fn(str(tmp_path), path="nope.py") == "NEW MISSING"
    assert toolkit.REGISTRY["edit"].fn(str(tmp_path), path="a.py", old_string="y",
                                      new_string="z").endswith("NEW NEAREST")
    monkeypatch.undo()
    importlib.reload(fs)
    assert "Closest names there: a.py" in toolkit.REGISTRY["read"].fn(str(tmp_path), path="b.py")


def test_a_refusal_that_names_its_target_never_counts():
    # Two different missed edits on one file, and two numbered path guesses:
    # the arguments are the problem there, so no "rewording will not help".
    guard = loops.LoopGuard()
    miss = "error: old_string not found in /w/a.py. It must match the file byte for byte."
    guard.observe("edit", {"path": "a.py", "old_string": "x"}, miss)
    assert not [n for n in guard.observe("edit", {"path": "a.py", "old_string": "y"}, miss)
                if "same reason" in n]
    guard.observe("read", {"path": "test_v1.py"}, "error: no such file: /w/test_v1.py")
    assert not [n for n in guard.observe("read", {"path": "test_v2.py"},
                                         "error: no such file: /w/test_v2.py")
                if "same reason" in n]


def test_hints_stay_bounded_on_huge_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "MISSING_SCAN", 50)
    big = tmp_path / "huge"
    big.mkdir()
    for i in range(200):
        (big / f"f{i:03d}_name.txt").write_text("")
    (big / "zz_target.txt").write_text("")
    assert "zz_target.txt" in fs.read(str(tmp_path), "huge/zz_targt.txt")
    monkeypatch.setattr(fs, "NEAREST_SCAN", 10)
    (tmp_path / "long.py").write_text("x = 1\n" * 50)
    assert "too long to suggest" in fs.edit(str(tmp_path), "long.py", "y = 2", "z")

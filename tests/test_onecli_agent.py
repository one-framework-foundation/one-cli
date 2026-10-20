"""one-cli agent loop and tools, driven by a scripted provider.

The loop is the part where a wrong assumption costs a whole session, so it is
tested against canned turns rather than a live model: same turns every run,
and a regression shows up as a diff instead of as a flake.
"""

from __future__ import annotations

import json

from onecli import tools as toolkit
from onecli.agent import Agent
from onecli.providers.base import ProviderError, ToolCall, Turn


class ScriptedProvider:
    """Returns the next canned turn and records what it was handed."""

    def __init__(self, turns: list[Turn]):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []
        self.ceilings: list[int] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.seen.append([dict(m) for m in messages])
        self.ceilings.append(max_tokens)
        if not self.turns:
            raise ProviderError("script exhausted")
        return self.turns.pop(0)


def test_a_plain_answer_ends_the_loop_in_one_step(tmp_path):
    provider = ScriptedProvider([Turn(text="done")])
    result = Agent(provider, str(tmp_path)).run("hi")
    assert result.text == "done"
    assert result.stopped == "answered"
    assert len(result.steps) == 1


def test_tool_result_is_fed_back_before_the_next_turn(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    provider = ScriptedProvider([
        Turn(tool_calls=[ToolCall(id="t1", name="read", arguments={"path": "a.txt"})]),
        Turn(text="the file says hello"),
    ])
    result = Agent(provider, str(tmp_path)).run("read a.txt")

    second_request = provider.seen[1]
    assert second_request[-1]["role"] == "tool"
    assert second_request[-1]["tool_call_id"] == "t1"
    assert "hello" in second_request[-1]["content"]
    assert result.tool_calls == 1


def test_assistant_tool_call_is_recorded_in_openai_shape(tmp_path):
    provider = ScriptedProvider([
        Turn(tool_calls=[ToolCall(id="t1", name="ls", arguments={"path": "."})]),
        Turn(text="ok"),
    ])
    Agent(provider, str(tmp_path)).run("list")
    assistant = [m for m in provider.seen[1] if m["role"] == "assistant"][0]
    call = assistant["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "ls"
    assert json.loads(call["function"]["arguments"]) == {"path": "."}


def test_step_budget_stops_a_model_that_never_answers(tmp_path):
    provider = ScriptedProvider([
        Turn(tool_calls=[ToolCall(id=f"t{i}", name="ls", arguments={})])
        for i in range(10)
    ])
    result = Agent(provider, str(tmp_path), max_steps=3).run("go")
    assert result.stopped == "budget"
    assert len(result.steps) == 3


def test_an_empty_final_turn_is_reported_not_treated_as_an_answer(tmp_path):
    """Printing nothing and exiting 0 reads as a successful run. It is not.

    With a configured cap of one retry, a model that stays silent after
    being asked again is reported as empty.
    """
    provider = ScriptedProvider([Turn(text="   ", finish_reason="stop"),
                                 Turn(text="", finish_reason="stop")])
    result = Agent(provider, str(tmp_path), loop_policy={"empty_retries": 1}).run("go")
    assert result.stopped == "empty"
    assert "empty turn" in result.error
    assert "again after 1 retry" in result.error


def test_empty_turns_are_recovered_without_a_default_cap(tmp_path):
    """One retry, then ending the run, left the outer driver to type
    "Continue with the task." Recovery
    only stops when a mask configures empty_retries."""
    provider = ScriptedProvider([Turn(text="", finish_reason="stop")] * 5
                                + [Turn(text="done")])
    events = []
    result = Agent(provider, str(tmp_path),
                   on_event=lambda k, d: events.append((k, d))).run("go")
    assert result.stopped == "answered" and result.text == "done"
    assert [d["attempt"] for k, d in events if k == "empty_retry"] == [1, 2, 3, 4, 5]


def test_an_empty_turn_is_retried_once_and_can_recover(tmp_path):
    """The work a run already did must not be thrown away by one silent turn.

    A model that goes quiet after its tool calls have succeeded is the common
    case: the task is done and only the closing message is missing.
    """
    provider = ScriptedProvider([Turn(text="", finish_reason="stop"),
                                 Turn(text="[SEND_FILE] /tmp/a.png")])
    result = Agent(provider, str(tmp_path)).run("go")
    assert result.stopped == "answered"
    assert result.text == "[SEND_FILE] /tmp/a.png"
    # The nudge is a user turn, so the model sees why it is being asked again.
    assert "previous turn was empty" in provider.seen[-1][-1]["content"]


def test_a_max_tokens_empty_turn_retries_on_a_raised_ceiling(tmp_path):
    """Retrying a spent budget at the same ceiling reproduces the same silence.

    finish_reason=max_tokens with no text is a reasoning model thinking past
    its output budget, so the retry only helps if the budget moves.
    """
    provider = ScriptedProvider([Turn(text="", finish_reason="max_tokens"),
                                 Turn(text="done")])
    result = Agent(provider, str(tmp_path), max_tokens=4096).run("go")
    assert result.stopped == "answered"
    assert provider.ceilings == [4096, 16384]
    assert "output limit" in provider.seen[-1][-1]["content"]


def test_the_empty_retry_is_spent_once_per_run(tmp_path):
    """A second silence later in the same run is terminal, not another retry."""
    provider = ScriptedProvider([Turn(text="", finish_reason="stop"),
                                 Turn(text="still here"),
                                 Turn(text="", finish_reason="stop")])
    result = Agent(provider, str(tmp_path)).run("go")
    # The first empty turn is recovered and ends the loop with an answer, so
    # the third turn is never reached - the retry budget survives to the end.
    assert result.stopped == "answered"
    assert result.text == "still here"


def test_empty_retry_boosts_max_tokens_on_length(tmp_path):
    """finish_reason='length' (OpenAI-compatible wire) triggers the 4x token boost."""
    provider = ScriptedProvider([Turn(text="", finish_reason="length"),
                                 Turn(text="recovered after limit raise")])
    agent = Agent(provider, str(tmp_path), max_tokens=1000)
    result = agent.run("go")
    assert result.stopped == "answered"
    assert result.text == "recovered after limit raise"
    # First turn used base max_tokens; retry turn used 4x boosted max_tokens
    assert provider.ceilings == [1000, 4000]
    # Nudge explicitly mentioned limit raise
    assert "limit has been raised" in provider.seen[1][-1]["content"]


def test_text_cut_off_at_the_output_limit_is_not_an_answer(tmp_path):
    """A turn that says "Now write the report." and then hits max_tokens lost
    its tool call to the ceiling. Ending the run there reported unfinished work
    as answered (three minimax detached lanes)."""
    provider = ScriptedProvider([
        Turn(text="Now write the report.", finish_reason="max_tokens"),
        Turn(tool_calls=[ToolCall(id="t1", name="ls", arguments={"path": "."})]),
        Turn(text="report written"),
    ])
    result = Agent(provider, str(tmp_path), max_tokens=4096).run("go")
    assert result.stopped == "answered"
    assert result.text == "report written"
    assert provider.ceilings[:2] == [4096, 16384]
    assert "cut off" in provider.seen[1][-1]["content"]


def test_text_that_stays_cut_off_is_reported_truncated(tmp_path):
    provider = ScriptedProvider([
        Turn(text="Writing it now.", finish_reason="length")
        for _ in range(5)
    ])
    result = Agent(provider, str(tmp_path), max_tokens=1000).run("go")
    assert result.stopped == "truncated"
    assert "output limit" in result.error


def test_provider_failure_is_returned_not_raised(tmp_path):
    provider = ScriptedProvider([])
    result = Agent(provider, str(tmp_path)).run("go")
    assert result.stopped == "error"
    assert "script exhausted" in result.error


def test_read_only_mode_withholds_the_mutating_tools(tmp_path):
    provider = ScriptedProvider([Turn(text="ok")])
    agent = Agent(provider, str(tmp_path), tool_names=["read", "ls", "glob", "grep"])
    agent.run("go")
    names = {t["function"]["name"] for t in toolkit.schemas(agent.tool_names)}
    assert names == {"read", "ls", "glob", "grep"}
    assert "bash" not in names and "write" not in names


def test_resumed_history_is_continued_not_restarted(tmp_path):
    prior = [{"role": "system", "content": "S"}, {"role": "user", "content": "first"}]
    provider = ScriptedProvider([Turn(text="second answer")])
    Agent(provider, str(tmp_path)).run("second", prior)
    sent = provider.seen[0]
    assert [m["content"] for m in sent] == ["S", "first", "second"]


# --- tools -----------------------------------------------------------------

def test_edit_refuses_an_ambiguous_match_without_touching_the_file(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a = 1\na = 1\n")
    out = toolkit.run("edit", {"path": "x.py", "old_string": "a = 1",
                               "new_string": "a = 2"}, str(tmp_path))
    assert out.startswith("error:")
    assert "appears 2 times" in out
    assert f.read_text() == "a = 1\na = 1\n"


def test_edit_replace_all_is_opt_in(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a = 1\na = 1\n")
    out = toolkit.run("edit", {"path": "x.py", "old_string": "a = 1",
                               "new_string": "a = 2", "replace_all": True},
                      str(tmp_path))
    assert "2 replacements" in out
    assert f.read_text() == "a = 2\na = 2\n"


def test_edit_missing_string_says_why_rather_than_failing_silently(tmp_path):
    (tmp_path / "x.py").write_text("body\n")
    out = toolkit.run("edit", {"path": "x.py", "old_string": "nope",
                               "new_string": "y"}, str(tmp_path))
    assert "not found" in out


def test_unknown_tool_lists_the_real_ones(tmp_path):
    out = toolkit.run("teleport", {}, str(tmp_path))
    assert out.startswith("error: no tool named")
    assert "bash" in out


def test_bad_arguments_come_back_as_text_the_model_can_act_on(tmp_path):
    out = toolkit.run("read", {"nonexistent_arg": 1}, str(tmp_path))
    assert out.startswith("error: bad arguments")


def test_unparsed_provider_arguments_are_rejected_at_the_tool_boundary(tmp_path):
    out = toolkit.run("write", {"__unparsed__": '{"path":"x.txt"'},
                      str(tmp_path))
    assert out == (
        "error: malformed JSON arguments for write; regenerate the tool call "
        "with valid JSON"
    )
    assert not (tmp_path / "x.txt").exists()


def test_read_numbers_lines_and_marks_truncation(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 51)))
    out = toolkit.run("read", {"path": "big.txt", "limit": 10}, str(tmp_path))
    assert out.splitlines()[0].endswith("line1")
    assert "40 more lines" in out


def test_bash_output_is_bounded_by_the_room_left_in_the_window(tmp_path):
    out = toolkit.run("bash", {"command": "python3 -c \"print('x'*80000)\""},
                      str(tmp_path), room=20_000)
    assert "bytes of output dropped" in out
    assert len(out) < 40_000


def test_bash_reports_the_exit_code(tmp_path):
    assert toolkit.run("bash", {"command": "exit 3"}, str(tmp_path)).startswith("exit 3")


def test_bash_timeout_is_named_not_hung(tmp_path):
    out = toolkit.run("bash", {"command": "sleep 5", "timeout": 1}, str(tmp_path))
    assert "timed out after 1s" in out


def test_every_tool_schema_is_well_formed():
    for schema in toolkit.schemas():
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        for req in fn["parameters"].get("required", []):
            assert req in fn["parameters"]["properties"], (fn["name"], req)


# --- streaming at the CLI level --------------------------------------------

class _StreamingProvider:
    """Hands back one turn, replaying it through on_delta first when asked."""

    def __init__(self, text: str = "hello"):
        self.text = text
        self.streamed = False

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096,
             on_delta=None):
        if on_delta is not None:
            self.streamed = True
            for piece in (self.text[:2], self.text[2:]):
                on_delta("text", piece)
        return Turn(text=self.text)


def _cli_args(tmp_path, **over):
    from onecli.cli import build_parser

    argv = ["hi", "--response-gate", "off", "-w", str(tmp_path)] + list(over.pop("argv", []))
    return build_parser().parse_args(argv)


def _run_cli(monkeypatch, tmp_path, argv, provider):
    from onecli import cli
    from onecli.config import Backend

    monkeypatch.setattr(cli.providers, "build", lambda backend, timeout=600.0: provider)
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)
    args = _cli_args(tmp_path, argv=argv)
    backend = Backend(name="b", model="m", base_url="u")
    return cli.run_once(args, backend, None, "sid")


def test_cli_no_stream_still_produces_the_whole_answer(monkeypatch, tmp_path, capsys):
    provider = _StreamingProvider()
    assert _run_cli(monkeypatch, tmp_path, ["--no-stream"], provider) == 0
    assert provider.streamed is False
    assert capsys.readouterr().out.strip() == "hello"


def test_cli_stream_prints_incrementally_without_doubling_the_answer(
        monkeypatch, tmp_path, capsys):
    """The streamed text is already on stdout; printing result.text repeats it."""
    provider = _StreamingProvider()
    assert _run_cli(monkeypatch, tmp_path, ["--stream"], provider) == 0
    assert provider.streamed is True
    assert capsys.readouterr().out.strip() == "hello"


def test_cli_json_streams_transport_but_keeps_one_stdout_object(monkeypatch, tmp_path, capsys):
    """Transport progress uses stderr, never prefixes the final JSON."""
    provider = _StreamingProvider()
    assert _run_cli(monkeypatch, tmp_path, ["--stream", "--json"], provider) == 0
    assert provider.streamed is True
    assert json.loads(capsys.readouterr().out)["text"] == "hello"


def test_the_wait_is_announced_before_the_request_not_after(tmp_path):
    """A host that draws a "waiting on the model" rail needs the wait to be
    announced when it STARTS. `turn` arrives with the answer already in hand,
    so a rail started there paints over the answer it was meant to cover -
    which is exactly how a streamed reply went missing on a terminal.
    """
    seen: list[str] = []

    class Watched(ScriptedProvider):
        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            seen.append("request")
            return super().chat(messages, tools, temperature, max_tokens)

    provider = Watched([Turn(text="4")])
    Agent(provider, str(tmp_path),
          on_event=lambda kind, data: seen.append(kind)
          if kind in ("waiting", "turn") else None).run("2+2")

    assert seen == ["waiting", "request", "turn"]


def test_reasoning_content_is_echoed_back_on_the_next_request(tmp_path):
    """DeepSeek thinking mode 400s a follow-up whose tool turn lost its
    reasoning_content."""
    provider = ScriptedProvider([
        Turn(thinking="plan: list first", echo={"reasoning_content": "plan: list first"},
             tool_calls=[ToolCall(id="t1", name="ls", arguments={"path": "."})]),
        Turn(text="done"),
    ])
    Agent(provider, str(tmp_path)).run("go")
    assistant = [m for m in provider.seen[1] if m["role"] == "assistant"][-1]
    assert assistant["reasoning_content"] == "plan: list first"


def test_the_openai_wire_keeps_reasoning_content_to_echo():
    from onecli.providers.openai_wire import OpenAIWire
    turn = OpenAIWire._to_turn({"choices": [{"message": {
        "content": "", "reasoning_content": "think",
        "tool_calls": [{"id": "a", "type": "function",
                        "function": {"name": "ls", "arguments": "{}"}}]},
        "finish_reason": "tool_calls"}]})
    assert turn.echo == {"reasoning_content": "think"}
    plain = OpenAIWire._to_turn({"choices": [{"message": {"content": "hi"},
                                              "finish_reason": "stop"}]})
    assert plain.echo == {}


def test_the_agent_hands_a_whole_skill_file_to_the_model_in_one_read(tmp_path):
    skill = "\n".join(f"rule {i}: " + "label every net, never leave a stub " * 3
                      for i in range(500))
    assert len(skill) > 50_000
    (tmp_path / "SKILL.md").write_text(skill)
    provider = ScriptedProvider([
        Turn(tool_calls=[ToolCall(id="t1", name="read", arguments={"path": "SKILL.md"})]),
        Turn(text="read it"),
    ])
    Agent(provider, str(tmp_path), context_window=200_000).run("read the skill")
    tool = provider.seen[1][-1]["content"]
    assert "rule 499:" in tool
    assert "continue with offset" not in tool


def test_the_tool_loop_has_no_step_limit_unless_one_is_set(tmp_path):
    # A default step cap ended 2 of DeepSeek's 4 turns on a long KiCad task
    # Without -s the loop runs until the model answers.
    for i in range(40):
        (tmp_path / f"f{i}.txt").write_text(str(i))
    calls = [Turn(tool_calls=[ToolCall(id=f"t{i}", name="read",
                                       arguments={"path": f"f{i}.txt"})])
             for i in range(40)]
    provider = ScriptedProvider(calls + [Turn(text="all forty read")])
    result = Agent(provider, str(tmp_path)).run("read them all")
    assert result.stopped == "answered"
    assert result.tool_calls == 40


def test_the_cli_sets_no_default_step_limit():
    from onecli.cli import build_parser
    assert build_parser().parse_args([]).max_steps is None


def test_intermediate_answers_remain_available_without_overriding_final(tmp_path):
    provider = ScriptedProvider([
        Turn(text="The answer is 42", tool_calls=[ToolCall(id="t1", name="ls", arguments={"path": "."})]),
        Turn(text="I cannot answer"),
    ])
    result = Agent(provider, str(tmp_path)).run("answer and list")
    assert result.text == "I cannot answer"
    assert result.steps_text == [{"step": 1, "text": "The answer is 42"},
                                 {"step": 2, "text": "I cannot answer"}]
    assert result.answer_advisories == [
        "Earlier text accompanied a tool call and differs from the final answer; inspect steps_text."]


def test_empty_budget_turn_recovers_last_answer(tmp_path):
    provider = ScriptedProvider([
        Turn(text="Preserved answer", tool_calls=[ToolCall(id="t1", name="ls", arguments={"path": "."})]),
        Turn(tool_calls=[ToolCall(id="t2", name="ls", arguments={"path": "."})]),
    ])
    result = Agent(provider, str(tmp_path), max_steps=2).run("answer and list")
    assert result.stopped == "budget"
    assert result.text == "Preserved answer"


def test_spend_ceiling_on_a_tool_turn_recovers_the_earlier_answer(tmp_path):
    # Tool-call answer regression: the answer travelled with a tool call, and the run then hit the
    # dollar ceiling on a tool-only turn. `budget` already recovered it;
    # `budget_usd` returned "" and the answer was gone.
    spend = Turn(tool_calls=[ToolCall(id="t2", name="ls", arguments={"path": "."})])
    spend.usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    provider = ScriptedProvider([
        Turn(text='{"name": "tokyo", "pop": 37}',
             tool_calls=[ToolCall(id="t1", name="ls", arguments={"path": "."})]),
        spend,
        Turn(text="never reached"),
    ])
    result = Agent(provider, str(tmp_path), max_budget_usd=1.0,
                   price=(3.0, 15.0)).run("answer and list")
    assert result.stopped == "budget_usd"
    assert result.text == '{"name": "tokyo", "pop": 37}'

"""The chat lane: one request per turn, and everything else is a slash command.

Every test here drives the real objects with a fake wire, so what is asserted
is the conversation state a person would see - what got sent, what is
remembered, what is dropped - and never a mock's call count.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from onecli import chat as chatlib
from onecli import config, presence, providers, session
from onecli.providers.base import ProviderError, ToolCall, Turn


class FakeWire:
    """A wire that answers from a script and records what it was sent."""

    def __init__(self, replies=None, usage=None, raises=None, turns=None):
        self.replies = list(replies or ["ok"])
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 5}
        self.raises = raises
        # Whole Turns, for the tool loop: a scripted tool call cannot be
        # expressed as a string, and the loop reads nothing else.
        self.turns = list(turns or [])
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096,
             on_delta=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "tools": tools, "temperature": temperature,
                           "max_tokens": max_tokens, "streamed": on_delta is not None})
        if self.raises:
            raise self.raises
        if self.turns:
            return self.turns.pop(0)
        text = self.replies.pop(0) if self.replies else "ok"
        if on_delta is not None:
            on_delta("thinking", "scratch")
            for piece in text.split(" "):
                on_delta("text", piece + " ")
        return Turn(text=text, usage=dict(self.usage))


def backend(name="fake", model="fake-1", wire="openai"):
    return config.Backend(name=name, model=model,
                          base_url="http://127.0.0.1:1/v1", wire=wire)


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    # Every fake backend here sits on a dead port; switching between them is
    # what these tests are about, so runner's presence gate is held open
    # (tests/test_presence_gate.py covers the gate against a real endpoint).
    monkeypatch.setattr(presence, "refusal", lambda b: "")
    c = chatlib.Chat(backend(), out=io.StringIO(), err=io.StringIO(), stream=False)
    c._wire = FakeWire()
    return c


# --- the turn --------------------------------------------------------------


def test_a_turn_sends_the_system_prompt_first_and_remembers_both_sides(chat):
    chat.ask("hello")
    sent = chat._wire.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[1] == {"role": "user", "content": "hello"}
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]
    assert chat.last_answer == "ok"


def test_a_chat_turn_carries_no_tools(chat):
    """The whole reason this lane exists: no schemas on the wire."""
    chat.ask("hello")
    assert chat._wire.calls[0]["tools"] is None


def test_the_conversation_accumulates(chat):
    chat._wire.replies = ["first", "second"]
    chat.ask("one")
    chat.ask("two")
    assert [m["content"] for m in chat.messages] == ["one", "first", "two", "second"]
    # The second request carried the first exchange.
    assert len(chat._wire.calls[1]["messages"]) == 4


def test_streaming_prints_the_answer_and_times_the_first_token(chat):
    chat.stream = True
    stat = chat.ask("hi")
    assert "ok" in chat.out.getvalue()
    assert stat["ttft"] > 0


def test_a_scratchpad_is_hidden_unless_asked_for(chat):
    chat.stream = True
    chat.ask("hi")
    assert "scratch" not in chat.out.getvalue()
    chat.thinking = True
    chat.ask("hi")
    assert "scratch" in chat.out.getvalue()


def test_a_provider_error_leaves_the_conversation_where_it_was(chat):
    chat._wire = FakeWire(raises=ProviderError("http 502"))
    stat = chat.ask("hello")
    assert "502" in stat["error"]
    assert chat.messages == []


def test_an_interrupted_turn_retains_the_unfinished_request(chat):
    chat._wire = FakeWire(raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        chat.ask("hello")
    assert chat.messages[0] == {"role": "user", "content": "hello"}
    assert session.load_messages(chat.sid) == chat.messages


def test_tokens_and_cost_accumulate_over_turns(chat):
    chat.ask("one")
    chat.ask("two")
    assert chat.meter.input_tokens == 20
    assert chat.meter.output_tokens == 10
    assert len(chat.turns) == 2


def test_the_running_total_survives_a_model_switch(chat, monkeypatch):
    """A /model switch resets the price table, never the accounting.

    The per-backend meter has to restart - two price tables cannot share one
    dollar figure - so the conversation keeps its own totals alongside it.
    """
    chat.ask("one")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend("other", "other-1"))
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: FakeWire(replies=["b"]))
    chatlib.dispatch(chat, "/model other")
    chat.ask("two")
    assert chat.meter.input_tokens == 10          # this backend only
    assert chat.total_in == 20 and chat.total_out == 10
    chatlib.dispatch(chat, "/cost")
    assert "20 in" in chat.err.getvalue()


def test_the_footer_reports_rate_and_tokens(chat):
    stat = chat.ask("hi")
    line = chat.footer(stat)
    assert "tok/s" in line and "out" in line


# --- switching backend -----------------------------------------------------


def test_switching_model_keeps_every_message(chat, monkeypatch):
    chat.ask("hello")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend("other", "other-1",
                                                               wire="anthropic"))
    chatlib.dispatch(chat, "/model other")
    assert chat.backend.name == "other"
    assert len(chat.messages) == 2
    # A new wire, built from the new backend, not the cached one.
    assert chat._wire is None


def test_an_unknown_model_is_refused_without_touching_the_conversation(chat, monkeypatch):
    def boom(**kw):
        raise RuntimeError("no backend matching 'nope'")
    monkeypatch.setattr(config, "resolve", boom)
    chatlib.dispatch(chat, "/model nope")
    assert chat.backend.name == "fake"
    assert "no backend matching" in chat.err.getvalue()


# --- the command table -----------------------------------------------------


def test_every_alias_resolves_to_a_command():
    for alias, name in chatlib.ALIASES.items():
        assert chatlib.lookup(alias) is chatlib.COMMANDS[name]
        assert chatlib.lookup("/" + alias) is chatlib.COMMANDS[name]


def test_help_lists_every_command(chat):
    chatlib.dispatch(chat, "/help")
    printed = chat.err.getvalue()
    for name in chatlib.COMMANDS:
        assert f"/{name}" in printed


def test_exit_ends_the_session(chat):
    assert chatlib.dispatch(chat, "/exit") is False
    assert chatlib.dispatch(chat, "/q") is False


def test_an_unknown_command_suggests_the_near_one_and_carries_on(chat):
    assert chatlib.dispatch(chat, "/mod") is True
    assert "did you mean /model" in chat.err.getvalue()


def test_new_forgets_the_conversation_but_keeps_the_settings(chat):
    chat.ask("hello")
    chat.temperature = 1.3
    old_sid = chat.sid
    chatlib.dispatch(chat, "/new")
    assert chat.messages == []
    assert chat.temperature == 1.3
    assert chat.sid != old_sid


def test_undo_drops_the_last_exchange(chat):
    chat._wire.replies = ["first", "second"]
    chat.tools = "off"
    chatlib.turn(chat, "one")
    chatlib.turn(chat, "two")
    chatlib.dispatch(chat, "/undo")
    assert [m["content"] for m in chat.messages] == ["one", "first"]
    assert len(chat.turns) == 1


def test_undo_stops_at_the_compaction_summary(chat):
    from onecli.context import summary_message
    chat.messages = [summary_message("earlier"), {"role": "assistant", "content": "kept"}]
    chat.turns = []
    chatlib.dispatch(chat, "/undo")
    assert [m["content"] for m in chat.messages][1:] == ["kept"]
    assert chat.messages[0]["content"].startswith("Summary of the earlier")


def test_undo_on_an_empty_conversation_says_so(chat):
    chatlib.dispatch(chat, "/undo")
    assert "nothing to undo" in chat.err.getvalue()


def test_retry_asks_the_same_question_once_more(chat):
    chat._wire.replies = ["first", "second"]
    chat.ask("the question")
    chatlib.dispatch(chat, "/retry")
    assert [m["content"] for m in chat.messages] == ["the question", "second"]
    assert chat._wire.calls[-1]["messages"][-1]["content"] == "the question"


def test_retry_can_move_the_question_to_another_backend(chat, monkeypatch):
    chat.ask("the question")
    other = FakeWire(replies=["from the other one"])
    monkeypatch.setattr(config, "resolve", lambda **kw: backend("other", "other-1"))
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: other)
    chatlib.dispatch(chat, "/retry other")
    assert chat.backend.name == "other"
    assert chat.last_answer == "from the other one"


def test_temperature_and_max_tokens_reach_the_wire(chat):
    chatlib.dispatch(chat, "/temp 1.5")
    chatlib.dispatch(chat, "/max 128")
    chat.ask("hi")
    assert chat._wire.calls[0]["temperature"] == 1.5
    assert chat._wire.calls[0]["max_tokens"] == 128


def test_a_bad_number_is_refused_rather_than_silently_ignored(chat):
    chatlib.dispatch(chat, "/temp warm")
    assert chat.temperature is None
    assert "not a number" in chat.err.getvalue()


def test_stream_and_think_toggle(chat):
    chat.stream = True
    chatlib.dispatch(chat, "/stream")
    assert chat.stream is False
    chatlib.dispatch(chat, "/stream on")
    assert chat.stream is True
    chatlib.dispatch(chat, "/think on")
    assert chat.thinking is True


def test_system_can_be_shown_set_and_read_from_a_file(chat, tmp_path):
    chatlib.dispatch(chat, "/system you are terse")
    assert chat.system == "you are terse"
    assert chat.history()[0]["content"] == "you are terse"
    path = tmp_path / "sys.md"
    path.write_text("from a file", encoding="utf-8")
    chatlib.dispatch(chat, f"/system @{path}")
    assert chat.system == "from a file"


def test_cost_reports_tokens_even_when_the_price_is_unknown(chat):
    chat.ask("hi")
    chatlib.dispatch(chat, "/cost")
    printed = chat.err.getvalue()
    assert "10 in" in printed and "no price for this model" in printed


# --- staged context --------------------------------------------------------


def test_a_staged_file_rides_the_next_question_and_only_that_one(chat, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("the file body", encoding="utf-8")
    chatlib.dispatch(chat, f"/file {path}")
    chat.ask("what is this")
    first = chat._wire.calls[0]["messages"][-1]["content"]
    assert "the file body" in first and "what is this" in first
    chat.ask("and now")
    second = chat._wire.calls[1]["messages"][-1]["content"]
    assert "the file body" not in second


def test_a_missing_file_is_reported_and_stages_nothing(chat, tmp_path):
    chatlib.dispatch(chat, f"/file {tmp_path / 'nope.txt'}")
    assert chat.staged == []
    assert "cannot read" in chat.err.getvalue()


def test_a_shell_command_stages_its_output(chat):
    chatlib.shell(chat, "!echo hello-from-shell")
    assert "hello-from-shell" in chat.staged[0]


# --- saving and resuming ---------------------------------------------------


def test_the_transcript_is_markdown_with_both_sides(chat):
    chat.ask("the question")
    text = chatlib.transcript(chat)
    assert "## you" in text and "## one" in text
    assert "the question" in text and "ok" in text


def test_a_message_with_null_content_does_not_break_the_transcript(chat):
    chat.messages.append({"role": "assistant", "content": None})
    assert "## one" in chatlib.transcript(chat)


def test_save_writes_the_transcript(chat, tmp_path):
    chat.ask("hello")
    out = tmp_path / "saved.md"
    chatlib.dispatch(chat, f"/save {out}")
    assert "hello" in out.read_text(encoding="utf-8")


def test_save_writes_an_owner_only_transcript(chat, tmp_path):
    import os
    import stat
    chat.ask("hello")
    out = tmp_path / "saved.md"
    old = os.umask(0o022)
    try:
        chatlib.dispatch(chat, f"/save {out}")
    finally:
        os.umask(old)
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_resume_loads_a_session_and_lifts_the_system_prompt_out(chat, tmp_path):
    session.append("abc123", {"kind": "messages", "messages": [
        {"role": "system", "content": "an older system prompt"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier answer"},
    ]})
    chatlib.dispatch(chat, "/resume abc123")
    assert chat.system == "an older system prompt"
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]
    assert chat.history()[0]["role"] == "system"
    assert len(chat.history()) == 3


def test_a_turn_is_written_to_the_session_file(chat):
    chat.ask("hello")
    stored = session.load_messages(chat.sid)
    assert [m["content"] for m in stored] == ["hello", "ok"]


def test_a_model_id_can_be_named_alongside_the_backend(chat, monkeypatch):
    """One gateway serves a vendor's whole tier list."""
    seen = {}

    def resolve(name=None, model=None, **kw):
        seen.update(name=name, model=model)
        return backend(name, model or "default-1")
    monkeypatch.setattr(config, "resolve", resolve)
    chatlib.dispatch(chat, "/model dashscope qwen3.7-plus")
    assert seen == {"name": "dashscope", "model": "qwen3.7-plus"}
    assert chat.backend.model == "qwen3.7-plus"


def test_the_session_listing_shows_the_first_question(chat):
    """The walk lives in `session` now - `one --sessions` wanted it too."""
    session.append("s1", {"kind": "messages", "messages": [
        {"role": "user", "content": "the very first question"},
        {"role": "assistant", "content": "an answer"}]})
    assert session.opening_line("s1") == "the very first question"
    chatlib.dispatch(chat, "/sessions")
    assert "the very first question" in chat.err.getvalue()


# --- comparing backends ----------------------------------------------------


def test_compare_answers_in_the_order_asked_and_does_not_touch_the_history(
        chat, monkeypatch):
    wires = {"a": FakeWire(replies=["answer a"]), "b": FakeWire(replies=["answer b"])}
    monkeypatch.setattr(config, "resolve", lambda name=None, **kw: backend(name, name))
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: wires[b.name])
    rows = chatlib.fan_out(chat, ["a", "b"], "the same question")
    assert [r["text"] for r in rows] == ["answer a", "answer b"]
    # The header is the short name; the display label rides alongside it.
    assert [r["name"] for r in rows] == ["a", "b"]
    assert chat.messages == []


def test_a_lane_that_fails_does_not_take_the_comparison_with_it(chat, monkeypatch):
    def resolve(name=None, **kw):
        if name == "broken":
            raise RuntimeError("no backend matching 'broken'")
        return backend(name, name)
    monkeypatch.setattr(config, "resolve", resolve)
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: FakeWire(replies=["fine"]))
    rows = chatlib.fan_out(chat, ["broken", "good"], "q")
    assert "no backend matching" in rows[0]["error"]
    assert rows[1]["text"] == "fine"


# --- reading input ---------------------------------------------------------


def test_a_plain_line_is_a_prompt():
    assert chatlib.Reader().feed("hello") == "hello"


def test_a_trailing_backslash_continues_on_the_next_line():
    reader = chatlib.Reader()
    assert reader.feed("first \\") is None
    assert reader.waiting
    assert reader.feed("second") == "first \nsecond"
    assert not reader.waiting


def test_a_paste_block_is_one_prompt():
    reader = chatlib.Reader()
    assert reader.feed('"""') is None
    assert reader.feed("line one") is None
    assert reader.feed("line two") is None
    assert reader.feed('"""') == "line one\nline two"
    assert not reader.waiting


def test_a_blank_line_inside_a_paste_block_is_kept():
    reader = chatlib.Reader()
    reader.feed('"""')
    reader.feed("a")
    reader.feed("")
    assert reader.feed('"""') == "a\n"


# --- the terminal ----------------------------------------------------------


def test_the_prompt_hides_its_colour_from_readline():
    marked = chatlib._rl_prompt("\033[38;2;1;2;3mx\033[0m ")
    assert marked.startswith("\001\033[38;2;1;2;3m\002")
    assert "\001\033[0m\002" in marked


def test_tab_completes_commands_and_model_names(chat, monkeypatch):
    class FakeReadline:
        def __init__(self):
            self.buffer = ""

        def get_line_buffer(self):
            return self.buffer

    rl = FakeReadline()
    complete = chatlib.completer(chat, rl)
    monkeypatch.setattr(chat, "backends", lambda refresh=False: [backend("local-a"),
                                                                 backend("kimi")])
    rl.buffer = "/mod"
    assert complete("/mod", 0) == "/model"
    rl.buffer = "/model lo"
    assert complete("lo", 0) == "local-a"
    rl.buffer = "not a command"
    assert complete("not", 0) is None


# --- the entry point -------------------------------------------------------


def test_one_shot_json_carries_the_answer_and_the_backend(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend())
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: FakeWire(replies=["hi there"]))
    code = chatlib.main(["--json", "hello"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "hi there"
    assert payload["backend"] == "fake"
    assert payload["model"] == "fake-1"


def test_a_one_shot_answer_goes_to_stdout_and_the_footer_to_stderr(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend())
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: FakeWire(replies=["the answer"]))
    assert chatlib.main(["hello"]) == 0
    out, err = capsys.readouterr()
    assert out.strip() == "the answer"
    assert "fake" in err


def test_a_dead_backend_is_a_refusal_not_a_traceback(monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("no backend found")
    monkeypatch.setattr(config, "resolve", boom)
    assert chatlib.main(["hello"]) == 2
    assert "no backend found" in capsys.readouterr().err


def test_system_and_system_file_together_are_refused(monkeypatch, capsys, tmp_path):
    path = tmp_path / "s.md"
    path.write_text("x", encoding="utf-8")
    assert chatlib.main(["--system", "a", "--system-file", str(path), "hi"]) == 2
    assert "exclusive" in capsys.readouterr().err


def test_the_agent_flag_routes_into_the_chat(monkeypatch, capsys, tmp_path):
    """`one --chat` and `one-chat` are the same session, entered two ways."""
    from onecli import cli

    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend())
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: FakeWire(replies=["chatted"]))
    assert cli.main(["--chat", "--no-stream", "hello"]) == 0
    assert "chatted" in capsys.readouterr().out


# --- the loop underneath the conversation ----------------------------------
#
# These drive the REAL agent loop against a scripted wire and a real tool
# registry in a tmp workspace: what is asserted is the file the loop actually
# read and the conversation it left behind, never a mock's call count.


def tool_turn(name, arguments, call_id="c1"):
    return Turn(text="", tool_calls=[ToolCall(id=call_id, name=name,
                                              arguments=arguments)],
                usage={"prompt_tokens": 10, "completion_tokens": 5})


@pytest.fixture
def worktree(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("the answer is 41\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_chat_reaches_for_a_tool_by_itself_and_keeps_talking(chat, worktree):
    """The whole point: no mode was switched on, and the file was read."""
    assert chat.tools == chatlib.DEFAULT_TOOL_MODE == "auto"
    chat._wire = FakeWire(turns=[
        tool_turn("read", {"path": "notes.txt"}),
        Turn(text="41, from notes.txt",
             usage={"prompt_tokens": 10, "completion_tokens": 5}),
    ])
    stat = chatlib.turn(chat, "what is in notes.txt")

    assert stat["tool_calls"] == 1
    assert "41" in stat["text"]
    # The schemas went out, and they are the read-only surface - not the
    # registry, which is what `all` is for.
    sent = [c["tools"] for c in chat._wire.calls]
    names = [t["function"]["name"] for t in sent[0]]
    assert "read" in names and "write" not in names and "edit" not in names
    # The transcript holds the mechanics, so a follow-up question can use them.
    assert [m["role"] for m in chat.messages] == ["user", "assistant", "tool",
                                                  "assistant"]
    assert "41" in chat.messages[2]["content"]
    assert chat.total_tools == 1


def test_the_tool_line_goes_to_stderr_and_the_answer_to_stdout(chat, worktree):
    """A piped chat gets the answer and nothing else, tool call or no tool call."""
    chat._wire = FakeWire(turns=[
        tool_turn("read", {"path": "notes.txt"}),
        Turn(text="41", usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ])
    chatlib.turn(chat, "what is in notes.txt")
    assert chat.out.getvalue().strip() == "41"
    assert "read" in chat.err.getvalue()
    assert "notes.txt" in chat.err.getvalue()


def test_a_question_that_needs_nothing_is_still_one_request(chat, worktree):
    """Tools on the wire are not a tool loop: most turns never call one."""
    stat = chatlib.turn(chat, "what is a monad")
    assert stat["tool_calls"] == 0
    assert len(chat._wire.calls) == 1
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]


def test_the_surface_answers_questions_and_does_not_edit():
    surface = chatlib.tool_surface("auto")
    assert "read" in surface and "grep" in surface
    # A question about this machine is often a command, so `bash` is here - it
    # is the one that stops and asks. Nothing that edits or writes is.
    assert "bash" in surface
    assert not ({"write", "edit", "code"} & set(surface))
    # `all` is the agent's own surface: everything the registry has.
    assert chatlib.tool_surface("all") is None


def test_a_command_in_auto_needs_a_person_and_never_runs_unattended(chat, worktree,
                                                                    monkeypatch):
    """The rule the gate turns on: a call that can change something needs a yes.
    With nobody at the keyboard there is no yes to be had."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    chat._wire = FakeWire(turns=[tool_turn("bash", {"command": "rm -rf /"}),
                                 Turn(text="I could not run that", usage={})])
    chatlib.turn(chat, "clear the disk")
    denial = [m for m in chat.messages if m["role"] == "tool"][0]["content"]
    assert "nobody to ask" in denial
    assert "--tools all" in chat.err.getvalue()

    # `all` was typed at the command line and is itself the answer.
    chat.reset()
    chat.tools = "all"
    chat._wire = FakeWire(turns=[tool_turn("bash", {"command": "echo hi"}),
                                 Turn(text="hi", usage={})])
    chatlib.turn(chat, "say hi from the shell")
    assert "hi" in [m for m in chat.messages if m["role"] == "tool"][0]["content"]


def test_the_system_prompt_says_what_can_be_reached_for(chat, worktree):
    with_tools = chat.history(tools=True)[0]["content"]
    assert "read" in with_tools and "Working directory" in with_tools
    assert "meaningful findings" in with_tools
    assert "one concise answer" in with_tools
    # And says nothing about tools when there are none.
    assert chatlib.TOOL_SYSTEM[:20] not in chat.history()[0]["content"]


def test_tools_off_folds_the_plumbing_out_of_the_history(chat, worktree):
    """A turn that declares no tools may not carry any: an Anthropic endpoint
    rejects a tool_use block with no tools array to match it."""
    chat._wire = FakeWire(turns=[tool_turn("read", {"path": "notes.txt"}),
                                 Turn(text="41", usage={})])
    chatlib.turn(chat, "what is in notes.txt")
    assert any(m["role"] == "tool" for m in chat.messages)

    chat._wire = FakeWire(replies=["still 41"])
    chatlib.dispatch(chat, "/tools off")
    chatlib.turn(chat, "and again?")
    sent = chat._wire.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert chat._wire.calls[0]["tools"] is None
    assert not [m for m in sent if m["role"] == "tool"]
    assert not [m for m in sent if m.get("tool_calls")]
    # What the model said survives the fold; only the mechanics go.
    assert any("41" in str(m.get("content")) for m in sent)


def test_an_interrupt_mid_loop_retains_the_unfinished_request(chat, worktree):
    chat._wire.replies = ["first"]
    chatlib.turn(chat, "hello")
    before = list(chat.messages)

    def boom(*a, **kw):
        raise KeyboardInterrupt

    chat._wire.chat = boom
    with pytest.raises(KeyboardInterrupt):
        chatlib.turn(chat, "and this one dies")
    assert chat.messages[:len(before)] == before
    assert any(m.get("content") == "and this one dies" for m in chat.messages)
    assert session.load_messages(chat.sid) == chat.messages


def test_a_provider_error_in_the_loop_is_reported_not_raised(chat, worktree):
    def boom(*a, **kw):
        raise ProviderError("http 500")

    chat._wire.chat = boom
    stat = chatlib.turn(chat, "hello")
    assert "500" in stat["error"]
    assert chat.messages == []


def test_the_footer_counts_the_calls_it_made(chat):
    assert "2 tool calls" in chat.footer({"seconds": 1.0, "tool_calls": 2})
    assert "1 tool call" in chat.footer({"seconds": 1.0, "tool_calls": 1})
    assert "tool call" not in chat.footer({"seconds": 1.0})


def test_tools_takes_three_words_and_says_what_it_did(chat):
    chatlib.dispatch(chat, "/tools all")
    assert chat.tools == "all"
    chatlib.dispatch(chat, "/tools on")          # what the first cut called it
    assert chat.tools == "auto"
    chatlib.dispatch(chat, "/tools sideways")
    assert chat.tools == "auto"
    assert "off, auto, all" in chat.err.getvalue()
    chatlib.dispatch(chat, "/tools")
    assert "tools auto" in chat.err.getvalue()


def test_a_one_shot_looks_things_up_too(monkeypatch, capsys, tmp_path, worktree):
    """`one-chat "what is in notes.txt"` is the same turn as the session's."""
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend())
    wire = FakeWire(turns=[tool_turn("read", {"path": "notes.txt"}),
                           Turn(text="41", usage={})])
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: wire)
    assert chatlib.main(["--no-stream", "what is in notes.txt"]) == 0
    assert "41" in capsys.readouterr().out


def test_tools_off_on_the_command_line_sends_no_schemas(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend())
    wire = FakeWire(replies=["straight to the model"])
    monkeypatch.setattr(providers, "build", lambda b, t=600.0: wire)
    assert chatlib.main(["--tools", "off", "--no-stream", "hello"]) == 0
    assert wire.calls[0]["tools"] is None


def test_a_call_that_would_change_something_asks_first(chat, worktree):
    """`all` is the agent's surface inside a conversation, so the first thing
    that would change a file stops and asks - and a no is a tool result the
    model can act on, not an exception."""
    chat.tools = "all"
    chat.ask_line = lambda prompt: "n"
    chat._wire = FakeWire(turns=[
        tool_turn("write", {"path": "made.txt", "content": "nope"}),
        Turn(text="left it alone", usage={}),
    ])
    chatlib.turn(chat, "write made.txt")
    assert not (worktree / "made.txt").exists()
    assert "declined" in [m for m in chat.messages if m["role"] == "tool"][0]["content"]
    # One line for one refusal: the loop reports it twice, the chat says it once.
    assert chat.err.getvalue().count("not run") == 1
    assert "blocked" not in chat.err.getvalue()


def test_answering_all_stands_the_question_down_for_the_session(chat, worktree):
    asked = []

    def reader(prompt):
        asked.append(prompt)
        return "a"

    chat.tools = "all"
    chat.ask_line = reader
    chat._wire = FakeWire(turns=[
        tool_turn("write", {"path": "one.txt", "content": "1"}),
        tool_turn("write", {"path": "two.txt", "content": "2"}, call_id="c2"),
        Turn(text="both written", usage={}),
    ])
    chatlib.turn(chat, "write two files")
    assert (worktree / "one.txt").read_text() == "1"
    assert (worktree / "two.txt").read_text() == "2"
    assert len(asked) == 1


def test_a_read_only_turn_is_never_asked_about(chat, worktree):
    chat.ask_line = lambda prompt: pytest.fail("a read must not stop to ask")
    chat._wire = FakeWire(turns=[tool_turn("read", {"path": "notes.txt"}),
                                 Turn(text="41", usage={})])
    chatlib.turn(chat, "what is in notes.txt")
    assert chat.last_answer == "41"


def test_background_jobs_only_exist_in_all_and_never_outlive_the_session(chat):
    assert chat.close() == []
    chat.tools = "all"
    store = chat.jobs()
    assert chat.jobs() is store          # one store per session, not per turn
    assert chat.close() == []
    assert chat._jobs is None


# --- what makes it fast ----------------------------------------------------


def test_the_chat_lane_does_not_import_the_agent_or_the_tools():
    """The startup claim, asserted rather than described.

    `onecli.cli` pulls in the agent, the tool registry, MCP and the senses
    routers before it can parse a flag. If any of that creeps into the chat
    import, this lane has quietly become the slow one again.
    """
    probe = (
        "import sys; import onecli.chat; "
        "print(json.dumps([m for m in "
        "('onecli.agent','onecli.tools','onecli.mcp','onecli.senses',"
        "'onecli.harnesses','onecli.evals.scenarios') if m in sys.modules]))"
    )
    out = subprocess.run([sys.executable, "-c", "import json;" + probe],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []


def test_agent_turn_preserves_text_and_answers(chat, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from onecli import agent as agent_mod
    from onecli.agent import Result

    agent_instance = None

    def fake_run(text, history, attachments=None):
        agent_instance.on_event("turn", {"step": SimpleNamespace(index=1)})
        agent_instance.on_event("text_delta", {"kind": "text", "chunk": "Checking the file..."})
        call = SimpleNamespace(name="bash", arguments={"command": "cat file.txt"})
        agent_instance.on_event("tool_start", {"call": call, "mutating": False})
        agent_instance.on_event("tool_end", {"call": call, "output": "File contents"})
        agent_instance.on_event("turn", {"step": SimpleNamespace(index=2)})
        return Result(
            text="Here is the final answer for the user.",
            steps=[],
            messages=[],
            stopped="answered",
            cost={"input_tokens": 10, "output_tokens": 10, "usd": 0.001, "price_known": True},
        )

    def mock_agent_init(*args, **kwargs):
        nonlocal agent_instance
        inst = MagicMock()
        inst.run = fake_run
        inst.on_event = kwargs.get("on_event")
        agent_instance = inst
        return inst

    monkeypatch.setattr(agent_mod, "Agent", mock_agent_init)
    chatlib.agent_turn(chat, "Check file.txt")

    stdout_content = chat.out.getvalue()
    assert "Checking the file...\n" in stdout_content
    assert "Here is the final answer for the user.\n" in stdout_content


def test_queue_command_and_input_lane(chat):
    # Empty queue
    chatlib.dispatch(chat, "/queue")
    out = chat.err.getvalue()
    assert "Message Queue: empty" in out

    # Add messages to input_queue
    chat.input_queue.append("test queued prompt 1")
    chat.input_queue.append("test queued prompt 2")
    chatlib.dispatch(chat, "/queue")
    out = chat.err.getvalue()
    assert "Message Queue (2 pending):" in out
    assert "test queued prompt 1" in out

    # Clear queue
    chatlib.dispatch(chat, "/queue clear")
    assert len(chat.input_queue) == 0

    # InputLane
    lane = chatlib.InputLane()
    lane.queue.put("async typed line")
    pending = lane.pending()
    assert pending == ["async typed line"]
    assert lane.pending() == []


def test_chat_undo_and_diff_reverts_workspace(chat, tmp_path):
    import subprocess

    repo = tmp_path / "chat_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
    (repo / "doc.txt").write_text("orig\n")
    subprocess.run(["git", "add", "doc.txt"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True)

    from onecli.checkpoints import CheckpointManager

    chat.checkpoints = CheckpointManager(workspace=str(repo), sid=chat.sid)
    chat.checkpoints.begin_turn("edit doc", [])
    chat.messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

    (repo / "doc.txt").write_text("orig\nchanged\n")
    (repo / "extra.txt").write_text("extra\n")

    chatlib.dispatch(chat, "/diff")
    diff_out = chat.err.getvalue()
    assert "changed" in diff_out

    chatlib.dispatch(chat, "/undo")
    undo_out = chat.err.getvalue()
    assert "reverted workspace" in undo_out
    assert (repo / "doc.txt").read_text() == "orig\n"
    assert not (repo / "extra.txt").exists()
    assert chat.messages == []


def test_chat_bg_commands_lifecycle(chat, tmp_path):
    import time
    from onecli import jobs as joblib

    chat.workspace = str(tmp_path)
    chat._jobs = joblib.JobStore(chat.sid, root=tmp_path / "jobs")

    chatlib.dispatch(chat, "/bg echo chat-job-test")
    val = chat.err.getvalue()
    assert "started" in val
    assert "j1" in val

    chatlib.dispatch(chat, "/ps")
    val2 = chat.err.getvalue()
    assert "j1" in val2

    for _ in range(20):
        time.sleep(0.05)
        if (chat._jobs.root / "j1.log").exists():
            break

    chatlib.dispatch(chat, "/logs j1")
    val3 = chat.err.getvalue()
    assert "chat-job-test" in val3

    chatlib.dispatch(chat, "/kill j1")
    val4 = chat.err.getvalue()
    assert "j1" in val4





def test_chat_model_picker_preserves_conversation(chat, monkeypatch):
    from onecli.model_picker import handle_picker
    other = backend('other', 'large')
    monkeypatch.setattr(config, 'discover', lambda: [chat.backend, other])
    chat.messages = [{'role': 'user', 'content': 'keep'}]
    chatlib.dispatch(chat, '/model')
    assert 'Providers' in chat.err.getvalue()
    assert 'Models' in handle_picker(chat, '1')
    handle_picker(chat, '2')
    assert chat.backend is other and chat._wire is None
    assert chat.messages == [{'role': 'user', 'content': 'keep'}]


# --- interrupts and the input lane -------------------------------------------


@pytest.mark.parametrize("line", ["/retry", "!sleep 30"])
def test_ctrl_c_in_a_command_stops_the_command_not_the_session(chat, monkeypatch, line):
    lines = iter([line, "/exit"])
    monkeypatch.setattr(chatlib, "_setup_readline", lambda c: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    real_dispatch = chatlib.dispatch

    def interrupted(c, text):
        if text == line:
            raise KeyboardInterrupt
        return real_dispatch(c, text)

    monkeypatch.setattr(chatlib, "dispatch" if line.startswith("/") else "shell", interrupted)
    closed = []
    monkeypatch.setattr(chat, "close", lambda: closed.append(True) or [])
    try:
        code = chatlib.repl(chat, quiet=True)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C ended the session")
    assert code == 0
    assert closed == [True]
    assert "interrupted" in chat.err.getvalue()


def test_ctrl_c_in_a_one_shot_exits_130_and_closes(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend())
    monkeypatch.setattr(providers, "build",
                        lambda b, t=600.0: FakeWire(raises=KeyboardInterrupt()))
    closed = []
    real_close = chatlib.Chat.close
    monkeypatch.setattr(chatlib.Chat, "close",
                        lambda self: closed.append(True) or real_close(self))
    try:
        code = chatlib.main(["--tools", "off", "hello"])
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped the one-shot entry point")
    assert code == 130
    assert closed == [True]


def test_the_input_lane_is_paused_while_an_approval_reads_stdin(chat):
    """The lane and the approval read one descriptor; only one may read it."""
    import os
    import select

    r, w = os.pipe()
    stream = os.fdopen(r, "r")
    chat.input_lane = chatlib.InputLane(stream=stream)
    chat.tools = "all"
    seen = {}

    def ask(question):
        seen["lane_reading"] = chat.input_lane._thread is not None
        os.write(w, b"y\n")
        ready, _, _ = select.select([stream], [], [], 2.0)
        return stream.readline() if ready else ""

    chat.ask_line = ask
    chat.input_lane.start()
    try:
        verdict = chatlib._confirm(chat, ToolCall(id="x", name="bash",
                                                  arguments={"command": "true"}))
        assert seen["lane_reading"] is False
        assert verdict is None, "the approval keystroke went to the lane"
        assert chat.input_lane._thread is not None, "the lane resumes after the answer"
        assert chat.input_lane.pending() == []
    finally:
        chat.input_lane.stop()
        os.close(w)
        stream.close()

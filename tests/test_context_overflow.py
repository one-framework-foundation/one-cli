"""A llama.cpp context refusal must compact and retry, not end the run.

The figures below are examples, not measurements from any real session: a
request larger than the window is refused, and its earlier assessment excluded
tools, so the two do not establish a tokenizer ratio. The synthetic server
uses two characters per token solely to exercise overflow recovery
deterministically.
"""
import json

from onecli import context as ctx
from onecli.agent import Agent
from onecli.providers import ProviderError, Turn

REFUSAL = (
    'HTTP 400 from http://127.0.0.1:18083/v1/chat/completions: {"error":{"code":400,'
    '"message":"request (52658 tokens) exceeds the available context size (32768 tokens), '
    'try increasing it","type":"exceed_context_size_error","n_prompt_tokens":52658,'
    '"n_ctx":32768}}')


class TwoCharTokenServer:
    """Fixture that counts two chars per token and refuses past 32768."""

    def __init__(self):
        self.refused = 0
        self.answered: list[int] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        size = len(json.dumps(messages)) // 2
        if size > 32768:
            self.refused += 1
            raise ProviderError(REFUSAL)
        if messages[0].get("content") == ctx.SUMMARY_INSTRUCTION:
            return Turn(text="summary of the earlier work")
        self.answered.append(size)
        return Turn(text="here is the dog")


def _history():
    out = [{"role": "system", "content": "you are ada"}]
    for i in range(40):
        out.append({"role": "assistant" if i % 2 else "user", "content": "w" * 2500})
    return out


def test_parse_overflow_reads_the_server_numbers():
    assert ctx.parse_overflow(REFUSAL) == (52658, 32768)
    assert ctx.parse_overflow("HTTP 500 from x: boom") is None


def test_server_count_wins_over_the_character_estimate():
    history = _history()
    tracker = ctx.ContextTracker(window=32768, reply_tokens=4096,
                                 counter=lambda m: 52658)
    tracker.measured = True
    assert ctx.history_tokens(history) < tracker.threshold   # bytes/4 says fine
    assert tracker.should_compact(history) is True


def test_reply_room_is_reserved():
    tracker = ctx.ContextTracker(window=32768, compact_at=0.95, reply_tokens=8192)
    assert tracker.threshold == 32768 - 8192


def test_52658_of_32768_compacts_and_retries_once(tmp_path):
    server = TwoCharTokenServer()
    agent = Agent(provider=server, workspace=str(tmp_path), max_steps=4,
                  autocompact=True, context_window=32768)
    agent.tracker.measured = True        # the turn-2 state: usage already seen
    result = agent.run("draw the dog", messages=_history())
    assert server.refused == 1
    assert result.stopped == "answered", result.error
    assert result.text == "here is the dog"
    assert "HTTP 400" not in (result.error or "")


def test_host_sized_system_prompt_fits_a_32k_window(tmp_path):
    """A system prompt larger than a 32k window must not be sent as is.

    Compaction cannot shrink a system prompt, so the request must be bounded
    before it is sent, even when no tool has run yet.
    """
    server = TwoCharTokenServer()
    agent = Agent(provider=server, workspace=str(tmp_path), max_steps=3,
                  autocompact=True, context_window=32768)
    messages = [{"role": "system", "content": "RULES " + "r" * 105_000 + " TAIL"},
                {"role": "user", "content": "describe the two files"}]
    result = agent.run("describe the two files", messages=messages)
    assert server.refused == 0
    assert result.stopped == "answered", result.error
    assert server.answered and server.answered[0] <= 32768


def test_second_overflow_compacts_protected_tail_and_continues(tmp_path):
    """A large recent tool exchange can survive the first compact unchanged."""
    class SchemaHeavyServer:
        refusals = 0
        answered = False

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            if messages[0].get("content") == ctx.SUMMARY_INSTRUCTION:
                return Turn(text="Earlier investigation summarized.")
            size = ctx.history_tokens(messages) + 20_000  # provider tool schemas
            if size > 32_768:
                self.refusals += 1
                raise ProviderError(
                    '{"error":{"type":"exceed_context_size_error",'
                    f'"n_prompt_tokens":{size},"n_ctx":32768}}')
            self.answered = True
            return Turn(text="continued the task")

    history = [{"role": "system", "content": "rules"},
               {"role": "user", "content": "Investigate the code"}]
    for i in range(3):
        history.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"c{i}",
                        "name": "read_file", "content": "result " + "x" * 20_000})
    server = SchemaHeavyServer()
    agent = Agent(provider=server, workspace=str(tmp_path), max_steps=4,
                  context_window=32_768, keep_last=6)
    result = agent.run("Continue the investigation", messages=history)
    assert result.stopped == "answered", result.error
    assert server.refusals == 2
    assert server.answered
    assert any(m.get("content") == "Continue the investigation" for m in result.messages)
    assert ctx.orphaned_tool_results(result.messages) == []


def test_fit_system_marks_the_cut_and_keeps_both_ends():
    messages = [{"role": "system", "content": "HEAD" + "x" * 200_000 + "TAIL"},
                {"role": "user", "content": "u" * 50_000}]
    fitted, elided = ctx.fit_system(messages, window=32768)
    assert elided > 0
    text = fitted[0]["content"]
    assert text.startswith("HEAD") and text.endswith("TAIL")
    assert "elided to fit this model's 32768-token context" in text
    assert fitted[1] is messages[1]                    # users are never cut
    assert messages[0]["content"].endswith("TAIL") and len(messages[0]["content"]) > 200_000
    small = [{"role": "system", "content": "you are ada"}]
    assert ctx.fit_system(small, window=32768) == (small, 0)

"""Token accounting, compaction, the budget ceiling and the fallback backend.

Every bound in here exists because a long session hits it and the failure is
expensive: a history that grows past the window dies mid-task, a run with no
ceiling bills whatever it likes, and a compaction that drops the wrong message
takes the whole conversation down with a 400 nobody can read.
"""

from __future__ import annotations

import json

from onecli import config, context as ctx
from onecli.agent import Agent
from onecli.cli import main
from onecli import fallback
from onecli.providers.base import ProviderError, ToolCall, Turn, is_transient


class ScriptedProvider:
    """Canned turns, and a record of what history each call was handed."""

    def __init__(self, turns, usage=None):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []
        self.seen_tools: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.seen.append([dict(m) for m in messages])
        self.seen_tools.append(list(tools or []))
        if not self.turns:
            raise ProviderError("script exhausted")
        nxt = self.turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _call_turn(call_id: str, name: str = "ls", args: dict | None = None) -> Turn:
    return Turn(text="", tool_calls=[ToolCall(id=call_id, name=name,
                                              arguments=args or {})])


# --- the estimator ------------------------------------------------------------

def test_the_estimator_is_a_ceiling_divide_not_a_float():
    assert ctx.estimate_tokens("") == 0
    assert ctx.estimate_tokens("a" * 4) == 1
    assert ctx.estimate_tokens("a" * 5) == 2


def test_a_tool_call_is_counted_not_just_the_content():
    plain = {"role": "assistant", "content": ""}
    with_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "write",
                                     "arguments": json.dumps({"text": "x" * 400})}}],
    }
    # The arguments blob is the largest thing in most histories. Reading only
    # `content` would score these two the same.
    assert ctx.message_tokens(with_call) > ctx.message_tokens(plain) + 90


def test_reported_usage_beats_the_estimate():
    tracker = ctx.ContextTracker(window=1000)
    history = [{"role": "user", "content": "hi"}]
    tracker.observe(history, {"prompt_tokens": 700, "completion_tokens": 40})
    assert tracker.tokens == 740
    assert tracker.measured is True


def test_no_usage_falls_back_to_the_estimator():
    tracker = ctx.ContextTracker(window=1000)
    history = [{"role": "user", "content": "x" * 400}]
    tracker.observe(history, {})
    assert tracker.measured is False
    assert tracker.tokens == ctx.history_tokens(history)


def test_a_known_model_gets_its_measured_window():
    # Read live from the endpoints, not guessed.
    assert config.context_window_for("kimi-for-coding") == 262144
    assert config.context_window_for("qwen3-coder:30b") == 262144
    assert config.context_window_for("something-nobody-listed") == ctx.DEFAULT_CONTEXT_WINDOW


# --- compaction ---------------------------------------------------------------

def _history_with_tools(pairs: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(pairs):
        out.append({"role": "assistant", "content": "",
                    "tool_calls": [{"id": f"c{i}", "type": "function",
                                    "function": {"name": "ls", "arguments": "{}"}}]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "name": "ls",
                    "content": "x" * 200})
    return out


def test_compaction_keeps_the_system_prompt_and_the_tail():
    history = _history_with_tools(6)
    new, report = ctx.compact(history, lambda head: "the summary", keep_last=4)
    assert new[0] == {"role": "system", "content": "sys"}
    assert new[1]["content"].startswith(ctx.SUMMARY_PREFIX)
    assert new[-4:] == history[-4:]
    assert report.ok and report.after_tokens < report.before_tokens


def test_compaction_keeps_the_still_open_request():
    # The person's own words sit at history[1] and a six-pair tool loop pushes
    # them out of the kept tail. Losing them is how a "build this and send it"
    # turn comes back as a handover summary.
    history = _history_with_tools(6)
    new, report = ctx.compact(history, lambda head: "the summary", keep_last=4)
    assert {"role": "user", "content": "go"} in new
    assert new.index({"role": "user", "content": "go"}) > 1   # after the summary
    assert ctx.orphaned_tool_results(new) == []
    assert report.ok


def test_a_request_still_in_the_tail_is_not_duplicated():
    history = _history_with_tools(6)
    history.append({"role": "user", "content": "and now this"})
    new, _ = ctx.compact(history, lambda head: "s", keep_last=4)
    assert [m for m in new if m.get("role") == "user"
            and m.get("content") == "and now this"] == [
        {"role": "user", "content": "and now this"}]
    assert not any(m.get("content") == "go" for m in new)


def test_a_previous_summary_is_never_re_pinned_as_the_request():
    history = [{"role": "system", "content": "sys"},
               ctx.summary_message("an earlier compaction")]
    history += _history_with_tools(6)[2:]
    new, _ = ctx.compact(history, lambda head: "s", keep_last=4)
    assert sum(1 for m in new
               if isinstance(m.get("content"), str)
               and m["content"].startswith(ctx.SUMMARY_PREFIX)) == 1


def test_compaction_never_orphans_a_tool_result():
    # keep_last=3 cuts squarely between an assistant tool call and its result.
    # The cut has to walk back, or the next request 400s on an unknown
    # tool_call_id - on both wires.
    history = _history_with_tools(6)
    new, report = ctx.compact(history, lambda head: "s", keep_last=3)
    assert ctx.orphaned_tool_results(new) == []
    assert report.kept >= 3


def test_an_orphan_is_detected_when_one_is_actually_there():
    broken = [{"role": "tool", "tool_call_id": "gone", "name": "ls", "content": "x"}]
    assert ctx.orphaned_tool_results(broken) == ["gone"]


def test_a_failed_summarizer_leaves_the_history_alone():
    history = _history_with_tools(4)

    def boom(head):
        raise ProviderError("HTTP 500")

    new, report = ctx.compact(history, boom, keep_last=2)
    assert new == history
    assert report.ok is False and "500" in report.error


def test_the_loop_compacts_when_the_history_crosses_the_threshold(tmp_path):
    # Two tool rounds of bulky results, a tiny window, then an answer. Two and
    # not one: with a single round the only thing older than the tail is the
    # request itself, which is carried across rather than summarized, so there
    # is correctly nothing to compact.
    provider = ScriptedProvider([
        _call_turn("c0"),
        _call_turn("c1"),
        Turn(text="the summary of what happened"),   # the summarization call
        Turn(text="done"),
    ])
    events: list[dict] = []
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=5,
                  autocompact=True, context_window=200, compact_at=0.5,
                  keep_last=2,
                  on_event=lambda kind, data: events.append((kind, data)))
    result = agent.run("x" * 600)

    assert result.stopped == "answered"
    # Exactly one: a history with nothing old enough to summarize is not
    # re-attempted every step.
    assert [k for k, _ in events].count("compact") == 1
    report = result.compactions[0]
    # The transcript collapses to one summary. Tokens are NOT asserted to fall:
    # the still-open request is pinned back in (see ctx.live_request), and in a
    # 200-token window a 600-char request outweighs everything it replaced.
    # A compaction that shrinks the history by dropping the question is not a
    # win, so message count is the honest invariant here.
    assert report["ok"] and report["messages_after"] < report["messages_before"]
    assert any(m.get("content") == "x" * 600 for m in result.messages)
    assert ctx.orphaned_tool_results(result.messages) == []


def test_autocompact_off_leaves_the_history_growing(tmp_path):
    provider = ScriptedProvider([_call_turn("c0"), Turn(text="done")])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=4,
                  autocompact=False, context_window=200, compact_at=0.5)
    result = agent.run("x" * 600)
    assert result.compactions == []


# --- the budget ceiling -------------------------------------------------------

def test_a_priced_run_stops_at_the_ceiling_and_says_so(tmp_path):
    turn = _call_turn("c0")
    turn.usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    provider = ScriptedProvider([turn, Turn(text="never reached")])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=5,
                  autocompact=False, max_budget_usd=1.0, price=(3.0, 15.0))
    result = agent.run("go")

    assert result.stopped == "budget_usd"
    assert "budget" in result.error
    # Stopped, not truncated: the history it gathered is still whole.
    assert ctx.orphaned_tool_results(result.messages) == []
    assert result.cost["usd"] == 3.0


def test_an_unpriced_model_is_counted_in_tokens_and_never_stopped(tmp_path):
    turn = _call_turn("c0")
    turn.usage = {"prompt_tokens": 9_000_000, "completion_tokens": 500}
    provider = ScriptedProvider([turn, Turn(text="done")])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=5,
                  autocompact=False, max_budget_usd=0.01, price=None)
    result = agent.run("go")

    assert result.stopped == "answered"
    assert result.cost["price_known"] is False
    assert result.cost["usd"] is None
    assert result.cost["input_tokens"] >= 9_000_000


def test_cost_distinguishes_cumulative_input_from_current_context_and_cache():
    meter = ctx.CostMeter(model="m")
    meter.add_turn([], {"prompt_tokens": 100, "completion_tokens": 5,
                        "cache_read_tokens": 80})
    meter.add_turn([], {"prompt_tokens": 140, "completion_tokens": 7,
                        "cache_read_tokens": 110})

    assert meter.as_dict()["input_tokens"] == 240
    assert meter.as_dict()["context_tokens"] == 147
    assert meter.as_dict()["cached_input_tokens"] == 190
    # Cache reads meter at 0.1 by default: 50 fresh + 19.
    meter = ctx.CostMeter(model="m", cache_read_weight=0.1)
    meter.add_turn([], {"prompt_tokens": 100, "cache_read_tokens": 80})
    meter.add_turn([], {"prompt_tokens": 140, "cache_read_tokens": 110})
    assert meter.as_dict()["metered_input_tokens"] == 69


def test_cumulative_input_ceiling_reserves_a_final_answer_call(tmp_path):
    (tmp_path / "small.txt").write_text("evidence\n")
    first = _call_turn("c0", "read", {"path": "small.txt"})
    first.usage = {"prompt_tokens": 60, "completion_tokens": 1}
    final = Turn(text="answer from collected evidence",
                 usage={"prompt_tokens": 70, "completion_tokens": 5})
    provider = ScriptedProvider([first, final, Turn(text="never reached")])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=30,
                  autocompact=False, context_window=100,
                  max_input_tokens=280)

    result = agent.run("inspect")

    # The ceiling forced this step, so the reply is kept but the run is not
    # reported as finished.
    assert result.stopped == "budget_input"
    assert "ceiling 280" in result.error
    assert result.text == "answer from collected evidence"
    assert len(provider.seen) == 2
    assert result.cost["input_tokens"] == 130
    assert provider.seen[-1][-1]["content"].startswith(
        "This is the final response step")
    assert provider.seen_tools[-1] == provider.seen_tools[0]


def test_cumulative_input_final_step_tool_request_reports_input_limit(tmp_path):
    (tmp_path / "small.txt").write_text("evidence\n")
    first = _call_turn("c0", "read", {"path": "small.txt"})
    first.usage = {"prompt_tokens": 60, "completion_tokens": 1}
    provider = ScriptedProvider([first, _call_turn("c1", "read", {"path": "small.txt"})])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=30,
                  autocompact=False, context_window=100,
                  max_input_tokens=280)

    result = agent.run("inspect")

    assert len(provider.seen) == 2
    assert provider.seen[-1][-1]["content"].startswith(
        "This is the final response step")
    assert result.stopped == "budget_input"
    assert "cumulative input ceiling" in result.error.lower()


def test_cumulative_input_ceiling_stops_before_an_unaffordable_request(tmp_path):
    (tmp_path / "small.txt").write_text("evidence\n")
    first = _call_turn("c0", "read", {"path": "small.txt"})
    first.usage = {"prompt_tokens": 490, "completion_tokens": 1}
    provider = ScriptedProvider([first, Turn(text="must not be sent")])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=30,
                  autocompact=False, context_window=100,
                  max_input_tokens=500)

    result = agent.run("inspect")

    assert result.stopped == "budget_input"
    assert len(provider.seen) == 1
    assert "Cumulative input ceiling" in result.error


def test_an_answer_on_the_crossing_turn_is_still_delivered(tmp_path):
    turn = Turn(text="here is the answer",
                usage={"prompt_tokens": 1_000_000, "completion_tokens": 0})
    agent = Agent(provider=ScriptedProvider([turn]), workspace=str(tmp_path),
                  autocompact=False, max_budget_usd=0.5, price=(3.0, 15.0))
    result = agent.run("go")
    assert result.stopped == "answered" and result.text == "here is the answer"


# --- the fallback backend -----------------------------------------------------

def test_overload_shapes_are_transient_and_a_refusal_is_not():
    for msg in ("HTTP 429 from x", "HTTP 503 from x", "cannot reach x: timed out",
                "Overloaded", "rate limit exceeded"):
        assert is_transient(msg), msg
    for msg in ("HTTP 400 from x: invalid temperature",
                "I will not do that", "HTTP 401 from x: unauthorized"):
        assert not is_transient(msg), msg


def _chain(*rungs, source="explicit"):
    """A ladder from (name, provider) pairs, rung 0 first."""
    return fallback.Chain(
        rungs=[fallback.Rung(name=n, provider=p, model=n) for n, p in rungs],
        source=source,
    )


def test_a_429_moves_the_turn_to_the_fallback_and_reports_it(tmp_path):
    primary = ScriptedProvider([ProviderError("HTTP 429 from https://x/v1: busy")])
    spare = ScriptedProvider([Turn(text="answered on the spare")])
    agent = Agent(workspace=str(tmp_path), autocompact=False,
                  chain=_chain(("primary-model", primary), ("spare", spare)))
    result = agent.run("go")

    assert result.stopped == "answered"
    assert result.text == "answered on the spare"
    assert len(result.fallbacks) == 1
    assert result.fallbacks[0]["from"] == "primary-model"
    assert result.fallbacks[0]["to"] == "spare"
    assert "429" in result.fallbacks[0]["reason"]


def test_a_model_saying_no_never_spends_the_fallback(tmp_path):
    primary = ScriptedProvider([Turn(text="I will not do that.")])
    spare = ScriptedProvider([Turn(text="should never run")])
    agent = Agent(workspace=str(tmp_path), autocompact=False,
                  chain=_chain(("primary", primary), ("spare", spare)))
    result = agent.run("go")

    assert result.text == "I will not do that."
    assert result.fallbacks == []
    assert spare.seen == []


def test_a_non_transient_failure_is_not_retried_elsewhere(tmp_path):
    primary = ScriptedProvider([ProviderError("HTTP 400 from x: bad request")])
    spare = ScriptedProvider([Turn(text="should never run")])
    agent = Agent(workspace=str(tmp_path), autocompact=False,
                  chain=_chain(("primary", primary), ("spare", spare)))
    result = agent.run("go")

    assert result.stopped == "error" and spare.seen == []


def test_the_turn_walks_the_whole_chain_before_giving_up(tmp_path):
    """Three rungs, two of them down: the third answers, both hops reported."""
    first = ScriptedProvider([ProviderError("HTTP 429 busy")])
    second = ScriptedProvider([ProviderError("HTTP 503 down")])
    third = ScriptedProvider([Turn(text="third time")])
    agent = Agent(workspace=str(tmp_path), autocompact=False,
                  chain=_chain(("a", first), ("b", second), ("c", third)))
    result = agent.run("go")

    assert result.text == "third time"
    assert [(f["from"], f["to"]) for f in result.fallbacks] == [("a", "b"), ("b", "c")]


def test_a_demoted_rung_is_not_tried_again_on_the_next_turn(tmp_path):
    """Stickiness: the second turn starts on the backend that just worked.

    A dead endpoint costs a timeout every time it is asked, which is the whole
    cost the chain exists to avoid - so `a` must be asked exactly once.
    """
    first = ScriptedProvider([ProviderError("HTTP 429 busy")])
    second = ScriptedProvider([
        Turn(text="", tool_calls=[ToolCall(id="1", name="read",
                                           arguments={"path": "x"})]),
        Turn(text="done"),
    ])
    chain = _chain(("a", first), ("b", second))
    agent = Agent(workspace=str(tmp_path), autocompact=False, chain=chain)
    result = agent.run("go")

    assert result.text == "done"
    assert len(first.seen) == 1          # asked once, then left behind
    assert len(second.seen) == 2         # both model calls landed here
    assert len(result.fallbacks) == 1


def test_the_bottom_of_the_chain_raises_rather_than_looping(tmp_path):
    primary = ScriptedProvider([ProviderError("HTTP 429 busy")])
    spare = ScriptedProvider([ProviderError("HTTP 429 busy too")])
    agent = Agent(workspace=str(tmp_path), autocompact=False,
                  chain=_chain(("primary", primary), ("spare", spare)))
    result = agent.run("go")
    assert result.stopped == "error" and "429" in result.error


# --- the --json object --------------------------------------------------------

def test_json_output_carries_the_new_fields(tmp_path, monkeypatch, capsys):
    backend = config.Backend(name="scripted", model="qwen3-coder:30b",
                             base_url="http://127.0.0.1:1/v1")
    monkeypatch.setattr(config, "resolve", lambda **kw: backend)
    turn = Turn(text="done", usage={"prompt_tokens": 120, "completion_tokens": 8})
    monkeypatch.setattr("onecli.cli.providers.build",
                        lambda b, timeout=600.0: ScriptedProvider([turn]))
    monkeypatch.setattr("onecli.cli.session.append", lambda *a, **k: None)

    # The first-response gate is on by default and needs a structured
    # assessment turn; this test is about the --json payload, so it is off.
    code = main(["--json", "--response-gate", "off", "-w", str(tmp_path), "hello"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["context_window"] == 262144      # measured, from the table
    assert payload["context_tokens"] == 128
    assert payload["cost"]["input_tokens"] == 120
    assert payload["cost"]["price_known"] is False
    assert payload["compactions"] == [] and payload["fallbacks"] == []


def test_a_served_window_is_read_from_the_server_not_the_model_table(monkeypatch):
    """`n_ctx` is a fact about the process; the table is a fact about the
    model. Only the first one can end a run, so it wins.
    """
    from onecli import config

    class _Resp:
        def __init__(self, body): self.body = body
        def read(self): return self.body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return _Resp(b'{"default_generation_settings": {"n_ctx": 8192}}')

    monkeypatch.setattr(config.urllib.request, "urlopen", fake_urlopen)
    assert config.served_context_window("http://h:8911/v1") == 8192
    assert seen["url"] == "http://h:8911/props"
    # The model table would have said otherwise, which is the whole point.
    assert config.context_window_for("gpt-oss-20b") > 8192


def test_a_hosted_endpoint_is_never_probed(monkeypatch):
    """No https provider serves /props, so a probe there is a round trip spent
    to learn nothing. It is not made at all.
    """
    from onecli import config

    def explode(*a, **k):
        raise AssertionError("probed an https endpoint")

    monkeypatch.setattr(config.urllib.request, "urlopen", explode)
    assert config.served_context_window("https://api.example.com/v1") is None


def test_the_two_models_the_host_serves_are_no_longer_guessed():
    """Both fell through to DEFAULT_CONTEXT_WINDOW, which is 32768 - far under
    what either endpoint accepts. A window that small compacts on the boot
    prompt alone and keeps compacting every step, which is how a turn ends up
    answering the summarizer instead of the operator.
    """
    from onecli import config

    assert config.context_window_for("deepseek-v4-flash") == 1000000
    assert config.context_window_for("deepseek-v4-flash:0731") == 1000000
    assert config.context_window_for("qwen3.8-27b") == 262144
    for model in ("deepseek-v4-flash", "qwen3.8-27b"):
        assert config.context_window_for(model) > ctx.DEFAULT_CONTEXT_WINDOW


def test_the_anthropic_wire_plans_are_no_longer_guessed_either():
    """A backend with no published window still gets a measured floor, not the
    bare default. The first turn survives; the bug would only show from turn
    two, which is the worst place to find it.
    """
    from onecli import config

    assert config.context_window_for("MiniMax-M3") == 200000
    for model in ("MiniMax-M3",):
        assert config.context_window_for(model) > ctx.DEFAULT_CONTEXT_WINDOW
        assert config.resolve_context_window(model) > ctx.DEFAULT_CONTEXT_WINDOW



def test_the_serving_process_outranks_the_model_table(monkeypatch):
    """One test rig holds a 262144-token model in a 131072-token slot. A run that
    believes the table would never compact and would die on the server's 400.
    """
    from onecli import config

    monkeypatch.setattr(config, "served_context_window", lambda url, **k: 131072)
    assert config.resolve_context_window(
        "qwen3.8-27b", "local-a", "http://h:8137") == 131072


def test_a_hosted_model_falls_back_to_the_table_not_to_the_default(monkeypatch):
    """https is never probed, so the resolver must not read that None as
    'unknown model' - that is the 32768 bug, one layer up.
    """
    from onecli import config

    monkeypatch.setattr(config, "served_context_window", lambda url, **k: None)
    assert config.resolve_context_window(
        "deepseek-v4-flash", "dashscope",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1") == 1000000


def test_no_base_url_never_probes(monkeypatch):
    from onecli import config

    def explode(*a, **k):
        raise AssertionError("probed with no endpoint to probe")

    monkeypatch.setattr(config, "served_context_window", explode)
    assert config.resolve_context_window("deepseek-v4-flash") == 1000000


# --- a failed compaction must still shrink (the death spiral) ----------------
#
# A long run can die on finish_reason=length with every completed step
# discarded. The chain: the window fills ->
# _summarize is starved by the run's own --max-tokens and returns nothing ->
# compact() returns the history UNTOUCHED with ok=False -> the next step is
# over the threshold again, runs an identical starved call, and so on until the
# model itself returns an empty turn. Each link below has its own test.

def test_a_starved_summarizer_falls_back_to_a_deterministic_shrink():
    """The one that matters: compaction failed, and the history shrank anyway."""
    history = [{"role": "system", "content": "sys"},
               {"role": "user", "content": "do the big task"}]
    for i in range(12):
        history.append({"role": "assistant", "content": "",
                        "tool_calls": [{"id": f"c{i}", "type": "function",
                                        "function": {"name": "read",
                                                     "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"c{i}",
                        "content": "X" * 3000})

    new, report = ctx.shrink(history, keep_last=4, step=9,
                             reason="summarizer returned nothing")

    assert report.ok and report.fallback
    assert report.after_tokens < report.before_tokens
    assert len(new) < len(history)
    # The invariants compaction has always owed, owed by the fallback too.
    assert ctx.orphaned_tool_results(new) == []
    assert any(m.get("content") == "do the big task" for m in new)
    # And it must not be mistaken for the person's request next time round.
    assert ctx.live_request_index([ctx.fallback_message("x")], []) is None


def test_compaction_classifies_why_it_failed_not_just_that_it_did():
    """The caller cannot retry correctly if every failure is one string."""
    body = [{"role": "user", "content": "the task"}]
    body += [{"role": "assistant", "content": "step %d" % i} for i in range(8)]

    def raises(_head):
        raise ProviderError("connection reset")

    _, transport = ctx.compact(body, raises, keep_last=2)
    assert transport.failure == "transport" and not transport.ok

    _, starved = ctx.compact(body, lambda _h: "   ", keep_last=2)
    assert starved.failure == "starved" and not starved.ok

    # Nothing old enough is not a failure the caller can do anything about.
    _, nothing = ctx.compact(body, lambda _h: "s", keep_last=99)
    assert nothing.failure == "empty_head" and not nothing.ok


def test_the_loop_makes_progress_when_the_summarizer_never_answers(tmp_path):
    """End to end: the history shrinks every compaction, and the run finishes.

    If the history never got smaller, `should_compact` would stay true and the
    run would burn its whole step budget on identical dead summarization calls.
    """
    class StarvedSummarizer:
        """Tool calls forever, and an empty reply to every summarize request."""

        def __init__(self):
            self.summarize_calls = 0
            self.n = 0

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            if any(m.get("content") == ctx.SUMMARY_INSTRUCTION for m in messages):
                self.summarize_calls += 1
                return Turn(text="")            # starved
            self.n += 1
            if self.n >= 4:
                return Turn(text="finished")
            return _call_turn(f"c{self.n}", name="read", args={"path": "a.py"})

    provider = StarvedSummarizer()
    reports: list[dict] = []

    def on_event(kind, data):
        if kind == "compact":
            reports.append(data)

    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=8,
                  autocompact=True, context_window=400, compact_at=0.5,
                  keep_last=2, on_event=on_event)
    result = agent.run("y" * 1200)

    assert result.stopped == "answered"
    assert reports, "compaction never ran"
    # Every compaction that ran got smaller, even though not one of them
    # managed to summarize.
    for r in reports:
        assert r["ok"] and r["fallback"], r
        assert r["messages_after"] <= r["messages_before"]
    assert ctx.orphaned_tool_results(result.messages) == []


def test_a_transport_blip_is_retried_before_the_fallback_shrinks(tmp_path):
    """A bad socket must not cost the middle of the conversation."""
    class BlipOnce:
        def __init__(self):
            self.summarize_calls = 0
            self.n = 0

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            if any(m.get("content") == ctx.SUMMARY_INSTRUCTION for m in messages):
                self.summarize_calls += 1
                if self.summarize_calls == 1:
                    raise ProviderError("connection reset by peer")
                return Turn(text="a real summary of the work so far")
            self.n += 1
            if self.n >= 3:
                return Turn(text="finished")
            return _call_turn(f"c{self.n}", name="read", args={"path": "a.py"})

    provider = BlipOnce()
    reports: list[dict] = []
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=8,
                  autocompact=True, context_window=400, compact_at=0.5,
                  keep_last=2,
                  on_event=lambda k, d: reports.append(d) if k == "compact" else None)
    result = agent.run("y" * 1200)

    assert result.stopped == "answered"
    assert provider.summarize_calls >= 2, "the blip was not retried"
    # The retry succeeded, so this is a real summary and NOT the fallback.
    first = reports[0]
    assert first["ok"] and not first["fallback"]
    assert any(ctx.SUMMARY_PREFIX in str(m.get("content") or "")
               and ctx.FALLBACK_PREFIX not in str(m.get("content") or "")
               for m in result.messages)


def test_the_summarizer_gets_its_own_headroom_not_the_runs_ceiling(tmp_path):
    """Compaction runs when there is no room; the rescue call needs its own.

    Before the fix `_summarize` passed `max_tokens=self.max_tokens` and an
    uncapped render of the head - so on a full window the call that exists to
    fix the overflow was itself the overflow.
    """
    seen: dict = {}

    class Recorder:
        def __init__(self):
            self.n = 0

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            if any(m.get("content") == ctx.SUMMARY_INSTRUCTION for m in messages):
                seen["max_tokens"] = max_tokens
                seen["input_chars"] = len(messages[-1]["content"])
                return Turn(text="a summary")
            self.n += 1
            if self.n >= 3:
                return Turn(text="finished")
            return _call_turn(f"c{self.n}", name="read", args={"path": "a.py"})

    agent = Agent(provider=Recorder(), workspace=str(tmp_path), max_steps=8,
                  autocompact=True, context_window=4000, compact_at=0.5,
                  keep_last=2, max_tokens=64)
    agent.run("y" * 20000)

    assert seen, "the summarizer was never called"
    # Output: the run's own tiny ceiling must not be what the summary gets.
    assert seen["max_tokens"] >= 1024 > 64
    # Input: capped against the window, not the size of the overflow.
    assert seen["input_chars"] <= 4000 * ctx.CHARS_PER_TOKEN
    assert seen["input_chars"] < 20000


# --- a run that did the work must not report that it returned nothing --------

def test_a_dead_end_after_real_work_returns_the_work_not_an_empty_string(tmp_path):
    """Completed tool steps must never be reported as "returned nothing".

    The model speaks once, the tools run, and then it goes silent for good.
    The run cannot be called answered - but its own last words are a far better
    outcome than an empty string.
    """
    provider = ScriptedProvider([
        Turn(text="I read the file; here is what it says: 42",
             tool_calls=[ToolCall(id="c0", name="read", arguments={"path": "a.py"})]),
        Turn(text="", finish_reason="length"),      # dead end
        Turn(text="", finish_reason="length"),      # ... and again after the retry
    ])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=6,
                  autocompact=False, loop_policy={"empty_retries": 1})
    result = agent.run("read a.py")

    assert result.stopped == "truncated"
    assert "here is what it says: 42" in result.text
    assert "truncated" != result.error and "retry" in result.error
    assert ctx.orphaned_tool_results(result.messages) == []


def test_a_dead_end_falls_back_to_the_tool_output_when_nothing_was_said(tmp_path):
    """The model never produced prose, but the tools still produced evidence."""
    provider = ScriptedProvider([
        Turn(text="", tool_calls=[ToolCall(id="c0", name="read",
                                           arguments={"path": "a.py"})]),
        Turn(text="", finish_reason="length"),
        Turn(text="", finish_reason="length"),
    ])
    (tmp_path / "a.py").write_text("the answer is 42\n")
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=6,
                  autocompact=False, loop_policy={"empty_retries": 1})
    result = agent.run("read a.py")

    assert result.stopped == "truncated"
    assert "42" in result.text
    assert "read" in result.text


def test_a_dead_end_with_nothing_to_recover_is_still_empty(tmp_path):
    """The salvage path must not turn a genuinely silent run into an answer."""
    provider = ScriptedProvider([
        Turn(text="", finish_reason="length"),
        Turn(text="", finish_reason="length"),
    ])
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=6,
                  autocompact=False, loop_policy={"empty_retries": 1})
    result = agent.run("say something")

    assert result.stopped == "empty"
    assert result.text == ""


def test_acknowledgment_nudge_cannot_replace_the_live_request():
    request = {"role": "user", "content": "Read gate_registry.py and config_routing.py in full; describe each in one paragraph."}
    nudge = {"role": "user", "content": "Your acknowledgment has been emitted. Continue the authorized task now."}
    messages = [{"role": "system", "content": "assistant"}, request,
                {"role": "assistant", "content": "Reading both files."}, nudge,
                {"role": "assistant", "content": "file excerpts " * 200},
                {"role": "assistant", "content": "more excerpts " * 200}]
    for keep_last in (1, 3):
        for use_fallback in (False, True):
            history = messages
            for step in range(2):
                if use_fallback:
                    history, report = ctx.shrink(history, keep_last=keep_last)
                else:
                    history, report = ctx.compact(history, lambda head: "Audit the gate registry.", keep_last=keep_last)
                assert request in history, (keep_last, use_fallback, step)
                assert ctx.orphaned_tool_results(history) == []


def test_agent_keeps_current_request_when_policy_messages_follow_it(tmp_path):
    request = "Read BOTH named files in full and describe each in one paragraph."
    for summary in ("Audit just the gate registry.", ""):
        provider = ScriptedProvider([_call_turn("a"), _call_turn("b"), Turn(text="done")])
        agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=4,
                      context_window=200, compact_at=0.5, keep_last=1,
                      response_gate="prompt")
        agent._summarize = lambda head: summary
        result = agent.run(request)
        assert result.stopped == "answered"
        assert result.compactions
        for history in provider.seen:
            assert {"role": "user", "content": request} in history
        assert ctx.orphaned_tool_results(result.messages) == []



class _ObedientResearcher:
    """Calls a tool every step, each request 800 tokens larger than the last,
    and answers as soon as it is told this is the final response step."""

    def __init__(self):
        self.seen: list[list[dict]] = []
        self.prompt = 200

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.seen.append([dict(m) for m in messages])
        last = str(messages[-1].get("content") or "")
        if last.startswith("This is the final response step"):
            return Turn(text="answer from collected evidence",
                        usage={"prompt_tokens": self.prompt, "completion_tokens": 5})
        turn = _call_turn(f"c{len(self.seen)}", "read", {"path": "small.txt"})
        turn.usage = {"prompt_tokens": self.prompt, "completion_tokens": 1}
        self.prompt += 800
        return turn


def test_input_ceiling_reservation_covers_a_growing_request(tmp_path):
    """The input reservation covers growth between steps, not only the
    current request, so a run whose requests grow still has room to answer."""
    (tmp_path / "small.txt").write_text("evidence\n")
    provider = _ObedientResearcher()
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=30,
                  autocompact=False, context_window=100000,
                  max_input_tokens=7000)

    result = agent.run("inspect")

    assert result.stopped == "budget_input", result.error
    assert result.text == "answer from collected evidence"
    assert result.cost["input_tokens"] <= 7000

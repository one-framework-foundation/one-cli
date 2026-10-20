"""The cumulative input ceiling meters cost, not re-sends.

A stateless tool loop re-sends its prefix every step, so most of its input is
cache hits. The replay below is a synthetic per-request usage series of that
shape (session-0001): ten steps against a 524,288-token ceiling.
"""

from __future__ import annotations

import json
import re

from onecli import config
from onecli import spill as spilllib
from onecli.agent import Agent
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.base import ToolCall, Turn
from onecli.providers.openai_wire import OpenAIWire

# (prompt_tokens, cached_tokens) per request of synthetic session-0001.
CACHE_HEAVY_REPLAY = [
    (40896, 0), (40687, 40448), (41412, 40448), (46270, 41216),
    (49164, 45824), (49802, 48896), (56089, 49664), (58262, 55808),
    (65528, 58112), (68189, 65280),
]
CEILING = 524_288      # explicit historical ceiling: 2 x kimi window
WINDOW = 262_144


class Replay:
    """Tool call per recorded request, then an answer."""

    def __init__(self, usages, cached=True):
        self.usages = list(usages)
        self.cached = cached
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.seen.append(messages)
        last = str(messages[-1].get("content") or "")
        n = len(self.seen)
        if n > len(self.usages) or last.startswith("This is the final response step"):
            prompt, hit = self.usages[-1]
            return Turn(text="report", usage=self._usage(prompt, hit))
        prompt, hit = self.usages[n - 1]
        return Turn(text="", tool_calls=[ToolCall(id=f"c{n}", name="read",
                                                  arguments={"path": "small.txt"})],
                    usage=self._usage(prompt, hit))

    def _usage(self, prompt, hit):
        usage = {"prompt_tokens": prompt, "completion_tokens": 400}
        if self.cached:
            usage["cache_read_tokens"] = hit
        return usage


def _agent(tmp_path, provider, **kw):
    (tmp_path / "small.txt").write_text("evidence\n")
    kw.setdefault("max_steps", 30)
    kw.setdefault("context_window", WINDOW)
    kw.setdefault("max_input_tokens", CEILING)
    return Agent(provider=provider, workspace=str(tmp_path),
                 autocompact=False, **kw)


def test_cache_heavy_replay_no_longer_trips_the_ceiling(tmp_path):
    provider = Replay(CACHE_HEAVY_REPLAY)
    result = _agent(tmp_path, provider).run("inspect")

    # All ten recorded tool steps ran and the model answered on its own; the
    # ceiling never withheld tools. Before: forced final after 8 requests.
    assert len(provider.seen) == len(CACHE_HEAVY_REPLAY) + 1
    assert not any(str(m[-1].get("content")).startswith("This is the final")
                   for m in provider.seen)
    assert result.stopped == "answered", result.error
    assert result.text == "report"
    # Raw processed input still reported in full, for display and billing.
    assert result.cost["input_tokens"] == sum(p for p, _ in CACHE_HEAVY_REPLAY) + 68189
    # Metered: fresh input at full weight, cache reads at 0.1.
    assert result.cost["metered_input_tokens"] < CEILING // 3


def test_the_same_workload_uncached_still_trips_the_ceiling(tmp_path):
    usages = CACHE_HEAVY_REPLAY * 2          # 20 steps of genuinely new input
    result = _agent(tmp_path, Replay(usages, cached=False)).run("inspect")

    assert result.stopped == "budget_input"
    assert result.cost["cached_input_tokens"] == 0
    assert result.cost["input_tokens"] <= CEILING


def test_forced_final_answer_is_reported_as_a_budget_stop(tmp_path):
    """Session-0002: the ceiling forced the last step, the model said it needed
    one more tool turn, and one-cli reported `answered`."""
    usages = [(p, 0) for p, _ in CACHE_HEAVY_REPLAY] * 2
    provider = Replay(usages, cached=False)
    result = _agent(tmp_path, provider).run("inspect")

    assert provider.seen[-1][-1]["content"].startswith("This is the final response step")
    assert result.text == "report"               # the reply is kept...
    assert result.stopped == "budget_input"      # ...but not called finished
    # The host (runners/onecli_runner.onecli_limit_stop) reads the ceiling
    # out of the error text.
    m = re.search(r"ceiling ([\d,]+)", result.error)
    assert m and int(m.group(1).replace(",", "")) == CEILING


def test_the_window_bound_still_sees_cached_tokens_at_full_size(tmp_path):
    """The discount is for the cumulative ceiling only. A request that is 99%
    cache hits still occupies its full size in the window, so compaction must
    fire on it exactly as before."""
    class Scripted:
        def __init__(self):
            self.turns = [
                Turn(text="", tool_calls=[ToolCall(id="c0", name="read",
                     arguments={"path": "small.txt"})],
                     usage={"prompt_tokens": 900, "completion_tokens": 5,
                            "cache_read_tokens": 890}),
                Turn(text="", tool_calls=[ToolCall(id="c1", name="read",
                     arguments={"path": "small.txt"})],
                     usage={"prompt_tokens": 950, "completion_tokens": 5,
                            "cache_read_tokens": 900}),
                Turn(text="summary"),
                Turn(text="done"),
            ]

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
            return self.turns.pop(0)

    events = []
    (tmp_path / "small.txt").write_text("evidence\n")
    agent = Agent(provider=Scripted(), workspace=str(tmp_path), max_steps=5,
                  autocompact=True, context_window=1000, compact_at=0.5,
                  keep_last=2, max_input_tokens=0,
                  on_event=lambda kind, data: events.append(kind))
    result = agent.run("go")

    assert result.stopped == "answered", result.error
    assert "compact" in events
    assert result.cost["cached_input_tokens"] == 1790
    assert result.cost["metered_input_tokens"] < result.cost["input_tokens"] - 1500


def test_every_tool_result_is_bounded_before_it_joins_the_history(tmp_path, monkeypatch):
    """toolkit.run spills its own output, but sandbox tools, tool discovery and
    PostToolUse feedback reach the history without passing through it."""
    dump = json.dumps([{"k": i, "v": "x" * 80} for i in range(1200)])
    assert len(dump) > 90_000
    monkeypatch.setattr("onecli.agent.toolkit.run", lambda *a, **k: dump)
    store = spilllib.SpillStore("capcheck", root=tmp_path / "spills")
    provider = Replay([(100, 0)])
    result = _agent(tmp_path, provider, spill=store, max_input_tokens=0,
                    context_window=16_000).run("go")

    tool = next(m for m in result.messages if m.get("role") == "tool")
    assert len(tool["content"]) < len(dump) // 2
    saved = next((tmp_path / "spills" / "capcheck").iterdir())
    assert str(saved) in tool["content"]
    assert saved.read_text() == dump


# --- per-provider cache accounting, at the wire --------------------------------

def test_anthropic_prompt_tokens_include_cache_reads_and_writes():
    usage = AnthropicWire._usage({"input_tokens": 120, "output_tokens": 9,
                                  "cache_read_input_tokens": 4000,
                                  "cache_creation_input_tokens": 300})
    assert usage["prompt_tokens"] == 4420
    assert usage["cache_read_tokens"] == 4000


def test_openai_compatible_cache_hits_in_every_reported_shape():
    assert OpenAIWire._usage({"prompt_tokens": 50, "prompt_tokens_details":
                              {"cached_tokens": 40}})["cache_read_tokens"] == 40
    assert OpenAIWire._usage({"prompt_tokens": 50,          # Moonshot
                              "cached_tokens": 30})["cache_read_tokens"] == 30
    assert OpenAIWire._usage({"prompt_tokens": 50,          # DeepSeek
                              "prompt_cache_hit_tokens": 20})["cache_read_tokens"] == 20
    assert "cache_read_tokens" not in OpenAIWire._usage({"prompt_tokens": 50})


def test_cache_read_weight_comes_from_the_pricing_catalog(tmp_path, monkeypatch):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"deepseek": [0.28, 0.42, 0.028],
                                  "kimi": [0.6, 2.5]}))
    monkeypatch.setenv(config.PRICING_ENV, str(prices))
    assert abs(config.cache_read_weight("deepseek-v4") - 0.1) < 1e-9
    assert config.cache_read_weight("kimi-k2") == config.DEFAULT_CACHE_READ_WEIGHT
    assert config.price_for("deepseek-v4") == (0.28, 0.42)

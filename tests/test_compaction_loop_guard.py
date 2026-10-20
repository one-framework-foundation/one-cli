"""Compaction must refuse to fire when it cannot get the run under budget.

A model absent from CONTEXT_WINDOWS bounds itself by DEFAULT_CONTEXT_WINDOW.
When a large system prompt alone sits near that window's compaction threshold,
the run crosses it every turn, compacts every turn, and - because SUMMARY_INSTRUCTION says "summarize the conversation so
far" - spends those turns answering the summarizer instead of the person.

The guard is observed, not predicted: once a compaction has run and the NEXT
usage the endpoint reports is STILL over the threshold, that compaction was a
paid no-op and the next one will be too. Stop compacting and let the run
continue over-budget - it either fits after all (our estimate is only bytes/4)
or earns a real refusal, which is a fact a caller can act on. A loop is not.
"""
from onecli.context import DEFAULT_CONTEXT_WINDOW, ContextTracker


def _msg(role, chars):
    return {"role": role, "content": "x" * chars}


def _history(system_chars, n_body=12, body_chars=400):
    out = [_msg("system", system_chars), _msg("user", 200)]
    for i in range(n_body):
        out.append(_msg("assistant" if i % 2 else "user", body_chars))
    return out


def test_compaction_still_fires_when_it_can_actually_help():
    """The guard must not disarm normal compaction."""
    history = _history(2_000, n_body=400, body_chars=2_000)
    tracker = ContextTracker(window=DEFAULT_CONTEXT_WINDOW, keep_last=6)
    assert tracker.should_compact(history) is True


def test_a_compaction_that_did_not_help_is_not_repeated():
    """A compaction that did not bring reported usage under the line is not repeated.

    The estimated history never crosses the threshold here - what keeps
    crossing it is the usage the ENDPOINT reports, which counts the tool
    schemas and framing a bytes/4 estimate cannot see. Compaction shrinks the
    visible head, the next reported usage comes back over the line anyway, and
    without the guard the loop would compact on every step.
    """
    history = _history(2_000, n_body=400, body_chars=2_000)
    tracker = ContextTracker(window=DEFAULT_CONTEXT_WINDOW, keep_last=6)

    # Turn 1: the endpoint says we are over. Compaction is allowed.
    tracker.observe(history, {"prompt_tokens": DEFAULT_CONTEXT_WINDOW, "completion_tokens": 10})
    assert tracker.should_compact(history) is True

    # It ran, and it did not help: the next reading is still over.
    tracker.note_compaction()
    tracker.observe(history, {"prompt_tokens": DEFAULT_CONTEXT_WINDOW, "completion_tokens": 10})
    assert tracker.compaction_futile is True
    assert tracker.should_compact(history) is False


def test_a_compaction_that_did_help_leaves_compaction_armed():
    history = _history(2_000, n_body=400, body_chars=2_000)
    tracker = ContextTracker(window=DEFAULT_CONTEXT_WINDOW, keep_last=6)
    tracker.observe(history, {"prompt_tokens": 120_000, "completion_tokens": 10})
    tracker.note_compaction()
    tracker.observe(history, {"prompt_tokens": 20_000, "completion_tokens": 10})
    assert tracker.compaction_futile is False


def test_under_threshold_never_compacts():
    tracker = ContextTracker(window=DEFAULT_CONTEXT_WINDOW, keep_last=6)
    assert tracker.should_compact(_history(2_000, n_body=2)) is False


def test_default_window_is_not_below_the_smallest_measured_window():
    from onecli.config import CONTEXT_WINDOWS

    assert DEFAULT_CONTEXT_WINDOW <= min(CONTEXT_WINDOWS.values()), (
        "the default must sit at or below every window we have measured, "
        "otherwise it is a guess upward")
    assert DEFAULT_CONTEXT_WINDOW == 131072


def test_agent_reports_its_compactions_to_the_tracker():
    """The tracker can only judge a compaction it is told about.

    Without this, deleting `self.tracker.note_compaction()` from
    `Agent._compact` leaves every test above green while the futile-compaction
    guard is permanently disarmed in the only code path that matters - the
    flag would never be set, so it would never fire.
    """
    import inspect

    from onecli.agent import Agent

    src = inspect.getsource(Agent._compact)
    assert "self.tracker.note_compaction()" in src


def test_by_default_compaction_waits_for_the_real_window():
    """A default trigger at 0.8 summarized away a fifth of every window that
    was still free. Only a history that would not fit compacts now."""
    history = _history(2_000, n_body=10, body_chars=100)
    tracker = ContextTracker(window=100_000, keep_last=6)
    tracker.observe(history, {"prompt_tokens": 90_000, "completion_tokens": 10})
    assert tracker.should_compact(history) is False
    tracker.observe(history, {"prompt_tokens": 100_000, "completion_tokens": 10})
    assert tracker.should_compact(history) is True


def test_the_cli_compacts_at_the_window_by_default():
    from onecli.cli import build_parser
    assert build_parser().parse_args([]).compact_at == 1.0


def test_missing_usage_does_not_repeat_futile_compaction():
    """Session-0002: a 105k-char system prompt, 32k window, compact_at .44.

    The OpenAI-compatible endpoint returned empty usage on every turn. The
    system alone exceeds the trigger; summarizing the head cannot fix that.
    """
    tracker = ContextTracker(window=32768, compact_at=0.44)
    history = _history(104899, n_body=12, body_chars=1500)
    assert tracker.should_compact(history)
    # Agent._compact observes its new history before marking the compaction.
    compacted = _history(104899, n_body=6, body_chars=1500)
    tracker.observe(compacted)
    tracker.note_compaction()
    tracker.observe(compacted, {})
    assert not tracker.should_compact(compacted)
    assert not tracker.compacted_since_measure
    # A real server refusal must still permit the existing compact-and-retry.
    tracker.overflowed(compacted, n_prompt=40000, n_ctx=32768)
    assert tracker.should_compact(compacted)


def test_missing_usage_successful_compaction_remains_armed():
    tracker = ContextTracker(window=32768, compact_at=0.44)
    tracker.note_compaction()
    tracker.observe(_history(1000, n_body=2), {})
    assert not tracker.compacted_since_measure
    assert not tracker.compaction_futile
    assert tracker.should_compact(_history(104899))


def test_agent_missing_usage_compacts_once_with_request_local_system_fit(tmp_path):
    from onecli.agent import Agent
    from onecli.context import SUMMARY_INSTRUCTION
    from onecli.providers.base import ToolCall, Turn

    class NoUsageProvider:
        def __init__(self):
            self.summaries = 0
            self.turns = 0

        def chat(self, messages, tools=None, **kwargs):
            if any(m.get('content') == SUMMARY_INSTRUCTION for m in messages):
                self.summaries += 1
                return Turn(text='Continue the open request.')
            self.turns += 1
            if self.turns == 4:
                return Turn(text='finished')
            return Turn(tool_calls=[ToolCall(id=f'c{self.turns}', name='ls', arguments={})])

    provider = NoUsageProvider()
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=5,
                  context_window=32768, compact_at=0.44, keep_last=2)
    result = agent.run('Continue', messages=_history(104899))
    assert result.stopped == 'answered'
    assert provider.turns == 4
    assert provider.summaries == 1


def test_compaction_uses_the_fitted_request_not_archived_system(tmp_path):
    from onecli.agent import Agent
    from onecli.context import SUMMARY_INSTRUCTION
    from onecli.providers.base import ToolCall, Turn

    class Provider:
        summaries = 0
        turns = 0

        def chat(self, messages, tools=None, **kwargs):
            if any(m.get('content') == SUMMARY_INSTRUCTION for m in messages):
                self.summaries += 1
                return Turn(text='Continue.')
            self.turns += 1
            if self.turns == 4:
                return Turn(text='finished', usage={'prompt_tokens': 12000})
            return Turn(tool_calls=[ToolCall(id=f'c{self.turns}', name='ls', arguments={})],
                        usage={'prompt_tokens': 12000})

    provider = Provider()
    agent = Agent(provider=provider, workspace=str(tmp_path), max_steps=5,
                  context_window=32768, compact_at=0.9, keep_last=2)
    result = agent.run('Continue', messages=_history(127592))
    assert result.stopped == 'answered'
    assert provider.turns == 4
    assert provider.summaries == 0
    assert len(result.messages[0]['content']) == 127592


def test_pending_overflow_compaction_is_not_repeated_before_observation():
    tracker = ContextTracker(window=32768, compact_at=0.44)
    history = _history(127592)
    tracker.overflowed(history, 33042, 32768)
    tracker.note_compaction()
    # An overflow retry re-enters Agent.run before any provider observation.
    assert not tracker.should_compact(history)
    tracker.observe(history, {})
    assert not tracker.compacted_since_measure

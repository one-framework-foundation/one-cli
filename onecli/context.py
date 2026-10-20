"""Token accounting and context compaction.

Two jobs, both about the same wall: a chat history only grows, and every
endpoint refuses past some length. `--max-steps` bounds how many times we call
the model, not how big the call gets, so a long session dies mid-task with a
400 rather than finishing shorter.

Accounting prefers what the endpoint reports (`Turn.usage`) and falls back to a
character estimator when it reports nothing - ollama's OpenAI shim, llama-server
and several drop-ins do. Compaction summarizes the oldest messages into one
message and keeps the tail verbatim, with one hard invariant: a kept tool result
never loses the assistant turn that called it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

# Bytes per token, roughly, for English prose and code in a BPE vocabulary.
# THIS IS A FLOOR, NOT A MEASUREMENT. Real tokenizers split punctuation, JSON
# structure and non-Latin script far more finely than 4 chars per token, so a
# history estimated at 100k can bill 140k on the wire. It exists so a backend
# that reports no usage still gets *some* bound; whenever the endpoint reports
# usage, that number wins and this one is not consulted.
CHARS_PER_TOKEN = 4

# Conservative estimate when no endpoint measurement is available. This is
# a heuristic, not a measured tokenizer ratio: an assessment made without
# tools undercounts a request that carries the full MCP catalog, because the
# refusal counts every schema in the payload.
UNMEASURED_CHARS_PER_TOKEN = 2

# llama.cpp's refusal, e.g. {"error":{"code":400,...,"n_prompt_tokens":52658,
# "n_ctx":32768}}. The body is carried inside ProviderError's message.
_OVERFLOW_RE = re.compile(
    r'exceed_context_size_error.*?"n_prompt_tokens"\s*:\s*(\d+)\s*,\s*"n_ctx"\s*:\s*(\d+)',
    re.S)


def parse_overflow(error: str) -> tuple[int, int] | None:
    """(n_prompt_tokens, n_ctx) from an exceed_context_size_error, else None."""
    match = _OVERFLOW_RE.search(error or "")
    return (int(match.group(1)), int(match.group(2))) if match else None

# Per-message wire overhead (role, delimiters, the shape around the content).
MESSAGE_OVERHEAD_TOKENS = 4

# What we assume a model can hold when nothing tells us otherwise.
#
# Guessing low is the more expensive guess: a model absent from CONTEXT_WINDOWS
# that is assumed to hold 32768 tokens starts compacting at 80% of that, and
# spends its turns summarizing instead of working. 131072 is the smallest
# window in the table, so it sits at or below every window the project ships
# knowledge of. A served endpoint smaller than that is still handled, and
# handled better, by served_context_window(), which wins over this number
# outright.
DEFAULT_CONTEXT_WINDOW = 131072


SUMMARY_PREFIX = "Summary of the earlier conversation (the context was compacted):"

# Deliberately built ON TOP of SUMMARY_PREFIX rather than beside it:
# live_request_index() identifies a previous compaction's own output by that
# prefix, and a fallback message that did not carry it would be mistaken for
# the person's still-open request and pinned into every later compaction.
FALLBACK_PREFIX = (SUMMARY_PREFIX + "\nNo model summary was possible, so the "
                   "lines below are truncated excerpts, not prose.")

# The deterministic fallback keeps this much of each message it replaces, and
# this much in total. Small on purpose: the fallback only runs when the window
# is already too full to summarize, so its whole job is to make the next turn
# fit. Excerpts beat prose here because they cost no model call at all.
FALLBACK_CHARS_PER_MESSAGE = 240
FALLBACK_MAX_CHARS = 6000

SUMMARY_INSTRUCTION = """\
Summarize the conversation so far for an agent that will continue the work with
this summary in place of the messages themselves. Keep, in plain prose:

- what the user asked for, in their own terms
- every file path, command and identifier that was read or written
- what has already been done, and what did not work
- anything still open

Do not add advice and do not restate this instruction. Facts only.\
"""


def estimate_tokens(text: str) -> int:
    """Character-based floor on the token count of `text`. See CHARS_PER_TOKEN."""
    if not text:
        return 0
    return -(-len(text) // CHARS_PER_TOKEN)   # ceil, no float


def message_tokens(message: dict) -> int:
    """Estimate one history message, tool calls included.

    A tool call costs whatever its serialized arguments cost - a `write` with a
    file body in it is the single largest thing in most histories, and reading
    only `content` misses it entirely.
    """
    total = MESSAGE_OVERHEAD_TOKENS + estimate_tokens(str(message.get("role") or ""))
    content = message.get("content")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif content is not None:
        total += estimate_tokens(json.dumps(content, default=str))
    for key in ("name", "tool_call_id"):
        total += estimate_tokens(str(message.get(key) or ""))
    for call in message.get("tool_calls") or []:
        total += estimate_tokens(json.dumps(call, default=str))
    return total


def history_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def usage_tokens(usage: dict) -> tuple[int, int]:
    """(input, output) out of whatever the endpoint called them.

    OpenAI says prompt/completion, Anthropic says input/output, and several
    shims say both. Returns (0, 0) when nothing usable is there, which is the
    caller's signal to estimate instead.
    """
    if not usage:
        return 0, 0
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    try:
        return int(prompt), int(output)
    except (TypeError, ValueError):
        return 0, 0


def orphaned_tool_results(messages: list[dict]) -> list[str]:
    """Every `tool_call_id` in the history with no assistant call to match it.

    An orphan is fatal on both wires: OpenAI 400s on a tool message whose id was
    never announced, and Anthropic 400s on a tool_result block with no
    tool_use before it. This is the invariant compaction must not break, so it
    is a function rather than a comment.
    """
    announced: set[str] = set()
    orphans: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                cid = call.get("id")
                if cid:
                    announced.add(str(cid))
        elif msg.get("role") == "tool":
            cid = str(msg.get("tool_call_id") or "")
            if cid and cid not in announced:
                orphans.append(cid)
    return orphans


def split_for_compaction(messages: list[dict], keep_last: int) -> tuple[list[dict], list[dict], list[dict]]:
    """(system, head to summarize, tail kept verbatim).

    `keep_last` is a floor on the tail, not a ceiling: the boundary is walked
    backwards until no kept tool result has lost its assistant turn. Cutting on
    a fixed count alone is exactly how an orphaned tool_call_id gets shipped.
    """
    system: list[dict] = []
    body = list(messages)
    if body and body[0].get("role") == "system":
        system = [body[0]]
        body = body[1:]

    if keep_last <= 0 or not body:
        return system, body, []

    cut = max(0, len(body) - keep_last)
    # Walk the cut earlier while the tail would start on a dangling tool result,
    # or in the middle of an assistant turn's own results.
    while cut > 0 and orphaned_tool_results(body[cut:]):
        cut -= 1
    return system, body[:cut], body[cut:]


ACK_CONTINUATION = "Your acknowledgment has been emitted. Continue the authorized task now."


def is_user_request(message: dict) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return not (isinstance(content, str) and
                (content.startswith(SUMMARY_PREFIX) or content == ACK_CONTINUATION))


def last_exchange_start(messages: list[dict]) -> int | None:
    """Index of the last real user request, or None when there is none.

    The compaction summary is a user-role message too. Treating it as a
    request let /rewind and /undo cut at the summary and erase it together
    with every message the compaction kept.
    """
    return next((i for i in range(len(messages) - 1, -1, -1)
                 if is_user_request(messages[i])), None)


def live_request_index(head: list[dict], tail: list[dict]) -> int | None:
    """Index in `head` of the still-unanswered user turn, or None.

    A tool loop long enough to compact pushes the person's own words out of the
    kept tail, and the summary that replaces them is written to an instruction
    that says "summarize the conversation so far". The model then answers the
    summarizer instead of the person: a request to make and deliver a file
    comes back as a handover summary while the file is never delivered.

    Summaries and the harness acknowledgment nudge are not user requests.
    None when the tail still holds the request (nothing was lost), or when
    `head` contains only synthetic user messages.
    """
    if any(is_user_request(m) for m in tail):
        return None
    for i in range(len(head) - 1, -1, -1):
        if is_user_request(head[i]):
            return i
    return None


def summarizable(head: list[dict], tail: list[dict]) -> list[dict]:
    """`head` minus the still-open request: what a summary may replace.

    The request is carried across verbatim rather than summarized, so it is not
    input to the summarizer either. Orphaned tool ids inside this list are
    harmless - it is read as prose by the summarizer, never sent as a history.
    """
    i = live_request_index(head, tail)
    if i is None:
        return head
    return head[:i] + head[i + 1:]


def has_compactable(messages: list[dict], keep_last: int) -> bool:
    """Is there anything old enough to summarize, once the tail is protected?

    Without this the loop re-attempts a compaction every step of a short
    history that is already over the threshold - each one a no-op, each one an
    event. A history that cannot shrink is a fact about the run, not a retry.

    The still-open request does not count: replacing it with a summary of
    itself and then pinning it back costs a model call and grows the history.
    """
    _, head, tail = split_for_compaction(messages, keep_last)
    return bool(summarizable(head, tail))


def summary_message(text: str) -> dict:
    """The one message a compaction leaves behind.

    Role `user`, not `system`: the Anthropic wire hoists system content into its
    own top-level field, so a second system message either vanishes or changes
    meaning depending on the endpoint. A user-role message carrying an explicit
    marker means the same thing on both wires.
    """
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\n\n{text.strip()}"}


@dataclass
class Compaction:
    """One compaction, as reported on the `compact` event and in --json."""

    step: int
    before_tokens: int
    after_tokens: int
    messages_before: int
    messages_after: int
    summarized: int
    kept: int
    ok: bool = True
    error: str = ""
    # True when the history was shrunk deterministically, with no model call,
    # because summarization could not run. `ok` is still True: the history DID
    # shrink and the loop can go on. See shrink().
    fallback: bool = False
    # WHY a compaction failed, as a stable token rather than a message the
    # caller has to pattern-match: "" (it did not), "empty_head" (nothing old
    # enough - shrinking cannot help either), "transport" (the summarization
    # call raised - worth one retry), "starved" (the call returned nothing,
    # which on a full window it will do again - fall back immediately).
    failure: str = ""

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "fallback": self.fallback,
            "failure": self.failure,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "summarized": self.summarized,
            "kept": self.kept,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class ContextTracker:
    """Running context size, measured when possible and estimated when not."""

    window: int = DEFAULT_CONTEXT_WINDOW
    compact_at: float = 1.0
    keep_last: int = 6
    tokens: int = 0
    measured: bool = False      # did any endpoint ever report usage to us
    # A compaction has run; the next completed turn will judge whether it
    # helped, using reported usage or the normal estimate.
    compacted_since_measure: bool = False
    # A compaction ran and the next observation was still over the
    # threshold. See observe() and should_compact().
    compaction_futile: bool = False
    # Output room the next request asks for. The reply shares the window with
    # the prompt, so the prompt must leave it free.
    reply_tokens: int = 0
    # Real tokens per estimated token, learned from a server refusal that
    # named both numbers. Never below 1: the estimate is a floor.
    ratio: float = 1.0
    # The server's own count of a history (llama.cpp /tokenize), or None when
    # the backend cannot count. Wins over every estimate.
    counter: Callable[[list[dict]], int | None] | None = None

    @property
    def threshold(self) -> int:
        return max(1, min(int(self.window * self.compact_at),
                          self.window - self.reply_tokens))

    def estimate(self, messages: list[dict]) -> int:
        """Best available size of `messages`: server count, else a scaled floor."""
        guess = history_tokens(messages)
        if not self.measured and self.ratio == 1.0:
            guess = guess * CHARS_PER_TOKEN // UNMEASURED_CHARS_PER_TOKEN
        guess = int(guess * self.ratio)
        counted = self.counter(messages) if self.counter else None
        return max(guess, counted or 0)

    def overflowed(self, messages: list[dict], n_prompt: int, n_ctx: int) -> None:
        """The server refused `messages` as n_prompt tokens against n_ctx.

        Both numbers are facts from the process that will serve the retry, so
        they replace what we assumed: the window shrinks to n_ctx, the
        estimator is rescaled to what this tokenizer actually charges, and
        the futility guard is cleared - a refusal is a new measurement.
        """
        self.window = min(self.window, n_ctx) if self.window else n_ctx
        estimated = history_tokens(messages)
        if estimated:
            self.ratio = max(self.ratio, n_prompt / estimated)
        self.tokens = n_prompt
        self.compaction_futile = False

    def observe(self, messages: list[dict], usage: dict | None = None) -> int:
        """Record the context size after a turn. Reported usage wins."""
        prompt, output = usage_tokens(usage or {})
        if prompt:
            self.measured = True
            # prompt_tokens is what the endpoint charged for the history it just
            # read; the reply is about to join that history, so count both.
            self.tokens = prompt + output
        else:
            self.tokens = history_tokens(messages)
        if self.compacted_since_measure:
            # Use the same accounting that triggers compaction when usage is
            # absent. Otherwise an endpoint returning {} leaves the verdict
            # pending forever and summarizes the same oversized prefix every
            # step. A real overflow still clears this guard in overflowed().
            size = self.tokens if prompt else self.estimate(messages)
            if size >= self.threshold:
                self.compaction_futile = True
            self.compacted_since_measure = False
        return self.tokens

    def note_compaction(self) -> None:
        """A compaction just ran; the next observation is its verdict."""
        self.compacted_since_measure = True

    def should_compact(self, messages: list[dict]) -> bool:
        # Estimate the pending history rather than trusting the last reading:
        # the tool results appended since that turn are exactly what pushes a
        # session over, and they were never in any usage report.
        pending = self.estimate(messages)
        if max(self.tokens, pending) < self.threshold:
            return False
        # The infinite-compaction guard: see observe().
        return not (self.compaction_futile or self.compacted_since_measure)


def compact(messages: list[dict], summarize: Callable[[list[dict]], str],
            keep_last: int = 6, step: int = 0) -> tuple[list[dict], Compaction]:
    """Replace the oldest messages with one summary. Returns (history, report).

    A failed summarization returns the history untouched with `ok=False`. Losing
    the middle of a conversation to a transport blip is worse than being close
    to the window, and the caller still gets told.
    """
    before = history_tokens(messages)
    system, full_head, tail = split_for_compaction(messages, keep_last)
    pin_at = live_request_index(full_head, tail)
    pinned = [dict(full_head[pin_at])] if pin_at is not None else []
    head = summarizable(full_head, tail)
    if not head:
        return messages, Compaction(step=step, before_tokens=before,
                                    after_tokens=before,
                                    messages_before=len(messages),
                                    messages_after=len(messages),
                                    summarized=0, kept=len(tail), ok=False,
                                    failure="empty_head",
                                    error="nothing old enough to summarize")
    try:
        text = summarize(head)
    except Exception as exc:                                # noqa: BLE001
        return messages, Compaction(step=step, before_tokens=before,
                                    after_tokens=before,
                                    messages_before=len(messages),
                                    messages_after=len(messages),
                                    summarized=0, kept=len(tail), ok=False,
                                    failure="transport", error=str(exc))
    if not (text or "").strip():
        return messages, Compaction(step=step, before_tokens=before,
                                    after_tokens=before,
                                    messages_before=len(messages),
                                    messages_after=len(messages),
                                    summarized=0, kept=len(tail), ok=False,
                                    failure="starved",
                                    error="summarizer returned nothing")

    # The summary replaces the transcript, never the question that is still
    # open. See live_request_index(): without this the model answers the
    # summarizer's instruction instead of the person's request.
    new = system + [summary_message(text)] + pinned + tail
    return new, Compaction(
        step=step,
        before_tokens=before,
        after_tokens=history_tokens(new),
        messages_before=len(messages),
        messages_after=len(new),
        summarized=len(head),
        kept=len(tail),
    )


def local_digest(head: list[dict], per_message: int = FALLBACK_CHARS_PER_MESSAGE,
                 total: int = FALLBACK_MAX_CHARS) -> str:
    """Truncated excerpts of `head`, newest last, under a hard character cap.

    The newest messages are the ones worth keeping, so the cap is filled from
    the END backwards and the result is re-ordered. A digest built forwards
    spends its budget on the opening of a run the tail already summarizes.
    """
    picked: list[str] = []
    used = 0
    for msg in reversed(head):
        role = str(msg.get("role") or "?")
        content = msg.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, default=str) if content is not None else ""
        parts = [content.strip()]
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            parts.append(f"calls {fn.get('name', '?')}({fn.get('arguments', '')})")
        text = " ".join(p for p in parts if p)
        if not text:
            continue
        if len(text) > per_message:
            text = text[:per_message] + " ...[truncated]"
        line = f"[{role}] {text}"
        if used + len(line) > total:
            picked.append(f"[...{len(head) - len(picked)} earlier messages dropped]")
            break
        picked.append(line)
        used += len(line)
    return "\n".join(reversed(picked))


def fallback_message(text: str) -> dict:
    """The one message a deterministic shrink leaves behind. See summary_message()."""
    return {"role": "user", "content": f"{FALLBACK_PREFIX}\n\n{text.strip()}"}


def shrink(messages: list[dict], keep_last: int = 6, step: int = 0,
           reason: str = "") -> tuple[list[dict], Compaction]:
    """Compact with NO model call. Returns (history, report).

    This is what a failed summarization falls back to. `compact()` returning
    the history untouched is correct when the failure was a transport blip, and
    fatal when the failure was the window itself: the loop then retries an
    identical, identically-starved summarization every step while the history
    never shrinks, and the run dies on `finish_reason=length` with all of its
    completed work discarded.

    Replacing the whole head with one excerpt message - rather than dropping
    messages piecemeal - is what keeps the orphan invariant free: every
    tool_call_id that disappears takes its assistant turn with it, because the
    entire head goes at once. The still-open request is pinned exactly as
    compact() pins it.
    """
    before = history_tokens(messages)
    system, full_head, tail = split_for_compaction(messages, keep_last)
    pin_at = live_request_index(full_head, tail)
    pinned = [dict(full_head[pin_at])] if pin_at is not None else []
    head = summarizable(full_head, tail)
    if not head:
        return messages, Compaction(step=step, before_tokens=before,
                                    after_tokens=before,
                                    messages_before=len(messages),
                                    messages_after=len(messages),
                                    summarized=0, kept=len(tail), ok=False,
                                    fallback=True, failure="empty_head",
                                    error="nothing old enough to shrink")
    new = system + [fallback_message(local_digest(full_head))] + pinned + tail
    return new, Compaction(
        step=step,
        before_tokens=before,
        after_tokens=history_tokens(new),
        messages_before=len(messages),
        messages_after=len(new),
        summarized=len(head),
        kept=len(tail),
        ok=True,
        fallback=True,
        error=reason,
    )


@dataclass
class CostMeter:
    """Token and dollar accounting for one run.

    A model with no entry in the price table is tracked in tokens and costs
    nothing, so a budget ceiling can never stop such a run. That is a real gap
    and the run says so once rather than pretending to a number.
    """

    model: str = ""
    price_in: float = 0.0       # USD per million input tokens
    price_out: float = 0.0
    price_known: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    context_tokens: int = 0
    estimated: bool = False     # any turn we had to estimate rather than read
    fallbacks: list[dict] = field(default_factory=list)
    # Input as the cumulative ceiling meters it: fresh input at full weight,
    # cache reads at `cache_read_weight` (their price relative to fresh
    # input). A stateless tool loop re-sends its whole history every step, and
    # counting those cached re-sends at full weight stops a mostly-cached run
    # at its ceiling long before it has spent real money or read much new
    # context.
    cache_read_weight: float = 1.0
    metered_input_tokens: float = 0.0
    # Share of the last reported request served from cache; what the next
    # request's metered size is estimated with.
    last_cache_share: float = 0.0

    def metered(self, request_tokens: int) -> float:
        """What a request of this size will likely add to metered input."""
        discount = self.last_cache_share * (1.0 - self.cache_read_weight)
        return request_tokens * (1.0 - discount)

    @property
    def usd(self) -> float:
        if not self.price_known:
            return 0.0
        return (self.input_tokens * self.price_in
                + self.output_tokens * self.price_out) / 1_000_000

    def add_turn(self, messages: list[dict], usage: dict | None,
                 reply_text: str = "") -> None:
        prompt, output = usage_tokens(usage or {})
        cached = 0
        if not prompt:
            self.estimated = True
            prompt = history_tokens(messages)
            output = estimate_tokens(reply_text)
        else:
            # Canonical field, set by the wire (see providers.base.Turn).
            reported = (usage or {}).get("cache_read_tokens")
            if type(reported) is int and reported >= 0:
                cached = min(reported, prompt)
        self.cached_input_tokens += cached
        self.metered_input_tokens += (prompt - cached) + cached * self.cache_read_weight
        self.last_cache_share = cached / prompt if prompt else 0.0
        self.context_tokens = prompt + output
        self.input_tokens += prompt
        self.output_tokens += output

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "metered_input_tokens": round(self.metered_input_tokens),
            "cache_read_weight": self.cache_read_weight,
            "context_tokens": self.context_tokens,
            "price_known": self.price_known,
            "estimated": self.estimated,
            "usd": round(self.usd, 6) if self.price_known else None,
        }


# Share of the window the system prompt may take. A large system prompt plus
# hook context can leave almost nothing for the task on a small model, and no
# compaction can help because a system prompt is never summarized. Above this
# share the middle of each system message is elided, visibly, on the request
# copy only.
SYSTEM_SHARE = 0.35


def fit_system(messages: list[dict], window: int, ratio: float = 1.0,
               share: float = SYSTEM_SHARE) -> tuple[list[dict], int]:
    """Bound system messages to `share` of `window`. Returns (messages, elided chars).

    Each oversized system message keeps its head and tail in proportion to
    its size; the cut is marked so the model knows context was withheld.
    `messages` is never mutated, and a prompt that fits comes back as is.
    """
    budget = int(window * share * CHARS_PER_TOKEN / max(ratio, 1.0))
    sizes = [len(m["content"]) if m.get("role") == "system"
             and isinstance(m.get("content"), str) else 0 for m in messages]
    total = sum(sizes)
    if not window or total <= budget:
        return messages, 0
    out, elided = [], 0
    for message, size in zip(messages, sizes):
        keep = budget * size // total
        if not size or keep >= size:
            out.append(message)
            continue
        text = message["content"]
        head, tail = text[: keep * 2 // 3], text[size - keep // 3:]
        cut = size - len(head) - len(tail)
        elided += cut
        out.append({**message, "content": (
            f"{head}\n[... {cut} characters of system prompt elided to fit "
            f"this model's {window}-token context ...]\n{tail}")})
    return out, elided

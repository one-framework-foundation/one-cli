"""What every wire must hand back, so the agent loop never branches on provider."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Callable

from onecli import __version__, lanehealth


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    # Opaque per-call fields the wire sent alongside id/type/function and that
    # the server expects back verbatim on the next request. Gemini 3 signs its
    # reasoning and rejects a follow-up whose tool call lost the signature
    # ("Function call is missing a thought_signature in functionCall parts"),
    # but nothing here is google-shaped: whatever extra keys arrived are
    # carried and re-emitted, and a wire that sent none leaves the request
    # unchanged.
    extra: dict = field(default_factory=dict)


@dataclass
class Turn:
    """One assistant turn, normalised.

    `text` is what a human should read, `thinking` is the reasoning trace when
    the model emits one separately (v4-flash and the qwen thinking models do),
    and `tool_calls` is what the agent loop acts on. Keeping thinking in its own
    field is deliberate: concatenating every content field is how callers end up
    printing a model's scratchpad at a user.

    `usage` keeps the provider's own fields and adds two canonical ones, set
    by each wire because only the wire knows its provider's shape:
    `prompt_tokens` is every input token the request processed, cache reads
    included, and `cache_read_tokens` is how many of those were cache reads.
    """

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    request_effort: dict = field(default_factory=dict)
    # Fields of the assistant message the server expects back verbatim on the
    # next request, as ToolCall.extra is for one call. DeepSeek in thinking
    # mode answers 400 "The reasoning_content in the thinking mode must be
    # passed back to the API" when a tool turn's reasoning is dropped
    # A wire that needs nothing echoed leaves it empty.
    echo: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A transport or protocol failure. Never raised for a model simply saying no."""


def http_error_detail(detail: str) -> str:
    """Keep API diagnostics useful without forwarding gateway HTML to chat.

    Retry classification uses the original body. Only the final displayed
    message is normalized; JSON and plain-text quota/parameter errors retain
    their established semantics.
    """
    if re.match(r"[\s\ufeff]*<(?:!doctype\s+html|html|head|body)(?:\s|>)", detail, re.I):
        return "endpoint or gateway returned an HTML error page"
    return detail


# Substrings that mark a failure worth retrying somewhere else: the endpoint is
# busy, rate-limited, broken or unreachable. A model declining a task is not in
# this list and never can be - a refusal comes back as a normal Turn and never
# reaches ProviderError at all.
TRANSIENT_MARKERS = (
    "http 429", "http 500", "http 502", "http 503", "http 504", "http 529",
    "rate limit", "rate_limit", "overload", "overloaded", "capacity",
    "too many requests", "timed out", "timeout", "cannot reach",
    "temporarily unavailable", "service unavailable", "connection reset",
    "stream truncated",
)


def is_transient(error: object) -> bool:
    """Is this failure the endpoint's fault right now, rather than the request's?

    A spent CREDENTIAL counts. It is not transient in the "come back in ten
    seconds" sense - an OpenRouter key with `limit: 0` will 403 forever - but it
    is the endpoint's fault and not the request's, and the only useful response
    is the same one: take this turn to the next rung. Before this, a
    `HTTP 403 ... Key limit exceeded (total limit)` was classified as a bad
    request, skipped the chain entirely and reached the user as a raw traceback
    with a live spare lane sitting unused underneath it.
    """
    text = str(error).lower()
    if any(marker in text for marker in TRANSIENT_MARKERS):
        return True
    return lanehealth.is_quota_exhausted(text)


# A wire calls this as content arrives: kind is "text", "thinking", or "tool".
# The second argument is a chunk, never the accumulated string. Only "text"
# belongs on the answer channel; thinking and tool fragments provide progress.
OnDelta = Callable[[str, str], None]


# Two limits guard a streaming request, because a stream can stall two ways.
# Silence - no bytes at all, e.g. a half-open connection - is caught by the
# socket read timeout, set to STREAM_IDLE_SECONDS. Any line resets it: thinking
# frames, keepalive comments and pings all prove the connection is alive, and a
# reasoning model can think for minutes without a text frame. A server that
# only ever sends keepalives is caught by STREAM_CONTENT_SECONDS: that long
# without a content frame (thinking counts) and the request is abandoned.
# A backend that accepts the connection and then sends nothing must still be
# abandoned, rather than leaving the run waiting on a socket forever.
STREAM_IDLE_SECONDS = 120.0
STREAM_CONTENT_SECONDS = 600.0


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        value = float(raw) if raw.strip() else default
    except ValueError:
        return default
    return value if value > 0 else default


def stream_idle_seconds() -> float:
    """No-bytes limit, overridable with ONECLI_STREAM_IDLE_TIMEOUT (seconds)."""
    return _env_seconds("ONECLI_STREAM_IDLE_TIMEOUT", STREAM_IDLE_SECONDS)


def stream_content_seconds() -> float:
    """No-content limit, overridable with ONECLI_STREAM_CONTENT_TIMEOUT."""
    return max(_env_seconds("ONECLI_STREAM_CONTENT_TIMEOUT", STREAM_CONTENT_SECONDS),
               stream_idle_seconds())


class StreamIdle(ProviderError):
    """The stream stopped carrying content. "timed out" keeps it transient."""


class StreamTruncated(ProviderError):
    """The stream closed without a terminal marker ([DONE], finish_reason,
    message_stop): what arrived is a fragment, never an answer. "stream
    truncated" keeps it transient, so the fallback chain may carry the turn."""


def truncated(url: str) -> StreamTruncated:
    return StreamTruncated(f"stream truncated: {url} closed the stream without a "
                           "terminal marker (no [DONE], finish or stop reason)")


# How many times a stalled stream is re-sent before the failure is raised. Each
# retry is a full new request, so the bound is also the cost bound.
MAX_IDLE_RETRIES = 2


def idle_error(exc: BaseException, idle_seconds: float) -> StreamIdle | None:
    """A StreamIdle for a socket timeout raised while opening the stream."""
    if isinstance(exc, StreamIdle):
        return exc
    if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
        return StreamIdle(f"stream idle: no response for {idle_seconds:.0f}s "
                          "(socket timed out before the first frame)")
    return None


def stream_with_idle_retry(run: Callable[[OnDelta], dict], on_delta: OnDelta,
                           on_retry: Callable[[], None] | None = None) -> dict:
    """Run one streaming request, re-sending it when it stalls or is cut off.

    The retry is only safe while no answer text or tool-call fragment has
    reached the caller: replaying those would show the reader half an answer
    twice. Thinking is progress only, so a stall mid-reasoning is retried. Past
    the bound, or after visible output, the stall is raised - never waited out.
    """
    for attempt in range(MAX_IDLE_RETRIES + 1):
        shown: list[str] = []

        def tracked(kind: str, chunk: str) -> None:
            if kind != "thinking":
                shown.append(kind)
            on_delta(kind, chunk)

        try:
            return run(tracked)
        except (StreamIdle, StreamTruncated) as exc:
            if shown or attempt == MAX_IDLE_RETRIES:
                raise type(exc)(f"{exc}; gave up after {attempt + 1} "
                                f"attempt(s)") from exc
            if on_retry is not None:
                on_retry()
    raise AssertionError("unreachable")


# A data frame that only proves the connection is alive, not that the model is
# producing anything. Anthropic sends these as `data:` lines between content
# frames: a frame whose only key is the event type carries no content.
def _carries_content(frame: dict) -> bool:
    return bool(set(frame) - {"type", "event"})


def sse_payloads(lines: Iterable[bytes | str], idle_seconds: float | None = None,
                 content_seconds: float | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 on_done: Callable[[], None] | None = None) -> Iterator[dict]:
    """Yield the decoded JSON of every `data:` line, stopping at `[DONE]`.

    Comment lines, blank keepalives and `event:` lines are skipped rather than
    parsed: both wires send them and neither carries content. A `data:` line
    that is not JSON is a protocol failure and must arrive as a ProviderError
    carrying the payload, because a bare JSONDecodeError three frames deep tells
    an operator nothing about which endpoint misbehaved.

    With `idle_seconds`, a socket read timeout becomes StreamIdle. With
    `content_seconds`, a stream that keeps the line alive with keepalives but
    sends no content frame for that long raises StreamIdle too. `on_done` is
    called when `[DONE]` arrives, so a wire can tell a finished stream from one
    whose connection simply closed.
    """
    last_content = clock()
    stream = iter(lines)
    while True:
        try:
            raw = next(stream)
        except StopIteration:
            return
        except TimeoutError as exc:
            if idle_seconds is None:
                raise
            raise StreamIdle(f"stream idle: no data for {idle_seconds:.0f}s "
                             "(socket read timed out)") from exc
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = text.strip()
        parsed = None
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                if on_done is not None:
                    on_done()
                return
            try:
                parsed = json.loads(payload)
            except ValueError as exc:
                raise ProviderError(f"malformed SSE payload ({exc}): {payload[:400]}") from exc
        if parsed is not None and not (isinstance(parsed, dict)
                                       and not _carries_content(parsed)):
            last_content = clock()
        elif content_seconds is not None and clock() - last_content > content_seconds:
            raise StreamIdle(f"stream idle: only keepalives for {content_seconds:.0f}s, "
                             "no content frame; the request timed out")
        if isinstance(parsed, dict):
            yield parsed


# Every wire sends this. urllib's default `Python-urllib/3.x` is on Cloudflare's
# blocked-signature list, so an endpoint sitting behind Cloudflare answers a
# perfectly valid request with `403 error code: 1010` and no JSON body - the
# same request from curl succeeds; any other User-Agent is accepted. Naming
# ourselves is also the honest thing to do to a provider we are billing
# against.
USER_AGENT = f"one-cli/{__version__}"


def effort_observation(body: dict, wire: str) -> dict:
    """Read only effort controls from the initial request, never messages/keys.

    Empty controls mean omitted/unobserved, not that the provider used none.
    Compatibility retries can trim parameters. Compare this initial request
    only when the retained retry count is zero; it is not server confirmation.
    """
    controls = {}
    if isinstance(body.get("reasoning_effort"), (str, int, float, bool)):
        controls["reasoning_effort"] = body["reasoning_effort"]
    for key, fields in (("reasoning", ("effort",)),
                        ("output_config", ("effort",)),
                        ("thinking", ("type", "budget_tokens", "effort")),
                        ("options", ("think",))):
        nested = body.get(key)
        if isinstance(nested, dict):
            values = {field: nested[field] for field in fields
                      if isinstance(nested.get(field), (str, int, float, bool))}
            if values:
                controls[key] = values
    return {"model": body.get("model"), "wire": wire, "scope": "initial_request", "controls": controls}


def response_shapes(fn):
    """Translate malformed upstream structures into recoverable wire errors."""
    from functools import wraps

    @wraps(fn)
    def checked(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (AttributeError, TypeError, KeyError, IndexError, ValueError) as exc:
            raise ProviderError("malformed provider response") from exc
    return checked

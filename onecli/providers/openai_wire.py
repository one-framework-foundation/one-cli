"""OpenAI-shaped `/v1/chat/completions`.

Covers ollama, llama-server, vLLM, DashScope's compatible mode and anything
else that copied the shape. Streaming is opt-in per call: pass `on_delta` and
the request goes out with `stream: true` and the SSE frames are folded back
into the same response shape the non-streaming path returns, so `_to_turn` is
the only place that knows how a Turn is built.
"""

from __future__ import annotations

from onecli.providers.context_metrics import serialize

from onecli.providers.base import response_shapes

import http.client
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from onecli import media
from onecli.providers.base import (USER_AGENT, OnDelta, ProviderError,
                                   ToolCall, Turn, http_error_detail, idle_error,
                                   sse_payloads, stream_content_seconds,
                                   stream_idle_seconds,
                                   stream_with_idle_retry, truncated)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """A strict admission only authorizes the original HTTP destination."""

    def http_error_302(self, req, fp, code, msg, headers):
        fp.close()
        raise ProviderError("HTTP redirect denied by request budget admission")

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


# The keys an OpenAI-shaped tool call is defined by. Everything else on the
# object is provider extension data that only the provider understands, so it
# is round-tripped rather than parsed. `index` is streaming bookkeeping and is
# rebuilt from the fragment order, so it is not carried.
_TOOL_CALL_KNOWN_KEYS = frozenset({"id", "type", "function", "index"})



def split_content(content) -> tuple[str, str]:
    """(answer text, thinking text) from a message `content`.

    Most endpoints send a string. Mistral with reasoning on sends a list of
    typed blocks: {"type":"thinking","thinking":[{"type":"text","text":..}]}
    and {"type":"text","text":..}. Thinking must never reach the answer.
    """
    if not content:
        return "", ""
    if isinstance(content, str):
        return content, ""
    text: list[str] = []
    thinking: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "thinking":
            inner = block.get("thinking")
            thinking.append(inner if isinstance(inner, str) else "".join(
                p.get("text", "") for p in inner or [] if isinstance(p, dict)))
        elif block.get("type") == "text":
            text.append(block.get("text") or "")
    return "".join(text), "".join(thinking)

def _open_code(text: str) -> bool:
    """Whether `text` ends inside a Markdown fence or inline code span."""
    fences = 0
    inline = 0
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            fences += 1
        elif fences % 2 == 0:
            inline += line.count("`")
    return fences % 2 == 1 or inline % 2 == 1


def _reasoning_content(text: str, separate: bool, *, final: bool = True) -> tuple[str, str]:
    """Normalize envelope tags, preserving tags in examples and code.

    Streaming holds ambiguous prefixes/suffixes until their role is known.
    Only a leading complete block is reasoning; interior pairs are user text.
    """
    thinking = ""
    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        end = stripped.find("</think>", len("<think>"))
        if end >= 0:
            thinking = stripped[len("<think>"):end]
            text = stripped[end + len("</think>"):]
        elif not final:
            return "", ""
    elif separate and stripped.startswith("</think>"):
        # An orphan close tag ahead of the answer has nothing to close. Only on
        # a reasoning-bearing turn: otherwise it may be the answer's own text.
        text = stripped[len("</think>"):].lstrip()
    elif not final and "<think>".startswith(stripped):
        return "", ""

    # Never interpret a tag inside a Markdown example as a wire delimiter: a
    # trailing tag is in code only while a fence or inline span is still open.
    if not _open_code(text.rstrip()):
        tail = text.rstrip()
        for tag, partner in (("</think>", "<think>"), ("<think>", "</think>")):
            if separate and tail.endswith(tag) and partner not in tail:
                text = tail[:-len(tag)].rstrip()
                break
        if not final:
            tail = text.rstrip()
            for size in range(min(len(tail), len("</think>")), 0, -1):
                if any(tag.startswith(tail[-size:]) for tag in ("<think>", "</think>")):
                    return tail[:-size].rstrip(), thinking
    return (text if final else text.rstrip()), thinking


class OpenAIWire:
    wire = "openai"
    effort_decision: dict | None = None

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 600.0, extra_body: dict | None = None,
                 extra_headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}
        # Endpoint-mandated headers from the drop-in. Applied AFTER the ones
        # this wire builds, so a drop-in can correct a header we get wrong for
        # its endpoint - the file is the more specific statement.
        self.extra_headers = extra_headers or {}
        endpoint = urlsplit(self.base_url)
        self.session_header = (
            "x-opencode-session"
            if endpoint.hostname == "opencode.ai"
            and (endpoint.path == "/zen/go" or endpoint.path.startswith("/zen/go/"))
            else None
        )
        self.request_gate = None
        self.reject_redirects = False
        # Counted rather than swallowed: a lane that only answers on the second
        # sample is a real quality signal, and a bench that hides the retry
        # reports it as a clean pass.
        self.output_retries = 0
        # Same reasoning for transports: a lane that needs two attempts to
        # answer is not the same lane as one that answers first time.
        self.transient_retries = 0
        # The same retries, split by what the WIRE said caused them. The status
        # code is in hand at every increment site, so the class costs nothing to
        # keep - and "eleven retries" and "eleven rate limits" are different
        # facts about a configuration. Only classes that actually happened are
        # keys; a transport error is not retried at all, so there is no
        # "timeout" key to be permanently zero.
        self.retry_reasons: dict[str, int] = {}

    def _count_retry(self, reason: str) -> None:
        self.retry_reasons[reason] = self.retry_reasons.get(reason, 0) + 1

    def _open(self, req, timeout: float | None = None):
        timeout = self.timeout if timeout is None else timeout
        if self.reject_redirects:
            return urllib.request.build_opener(_RejectRedirects()).open(
                req, timeout=timeout)
        return urllib.request.urlopen(req, timeout=timeout)

    def _count_idle_retry(self) -> None:
        self.transient_retries += 1
        self._count_retry("stream_idle")

    # Suffixes that already mark a complete OpenAI-shaped API root. `/openai`
    # is Google AI Studio's (`.../v1beta/openai`) and `/openapi` is Vertex's
    # (`.../locations/<loc>/endpoints/openapi`); those are different strings,
    # and missing either makes the wire append a second version segment -
    # `.../v1beta/openai/v1/chat/completions`, `.../endpoints/openapi/v1/chat/...`
    # - which is a 404, not a fallback.
    API_ROOT_SUFFIXES = ("/v1", "/openai", "/openapi")

    def _url(self) -> str:
        base = self.base_url
        if not base.endswith(self.API_ROOT_SUFFIXES):
            base = base + "/v1"
        return base + "/chat/completions"

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             on_delta: OnDelta | None = None,
             response_format: dict | None = None) -> Turn:
        body: dict = {
            "model": self.model,
            "messages": self.convert_messages(messages),
            "stream": on_delta is not None,
        }
        # Sent only when configured, so the provider's default for the model
        # applies. Greedy 0.0 at 50K+ tokens of context is a prime suspect for
        # the repetition loops seen on DeepSeek.
        if temperature is not None:
            body["temperature"] = temperature
        # No ceiling unless one was configured: the provider then allows the
        # model's own maximum, which grows with the model and not with us.
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(self.extra_body)
        from onecli import config
        config.validate_model("", self.base_url, body["model"])
        if tools == []:
            # An explicitly empty capability list is a request-local ban
            # (assessment/final steps), including with tool-bearing history.
            # Provider defaults must not turn it back into an automatic call.
            body.pop("tools", None)
            body["tool_choice"] = "none"
        if response_format is not None:
            body["response_format"] = response_format

        from onecli.providers.base import effort_observation
        data = (self._post(body) if on_delta is None else stream_with_idle_retry(
            lambda delta: self._stream(body, delta), on_delta, self._count_idle_retry))
        turn = self._to_turn(data)
        turn.request_effort = effort_observation(body, "openai")
        return turn

    def chat_json(self, messages, tools=None, **kwargs) -> Turn:
        """Request JSON without leaking assessment formatting into later turns."""
        try:
            return self.chat(messages, tools, response_format={"type": "json_object"}, **kwargs)
        except ProviderError as exc:
            low = str(exc).lower()
            if not ("http 400" in low and "response_format" in low
                    and any(word in low for word in ("not supported", "unsupported", "unknown", "unrecognized"))
                    and "response_format" not in self.extra_body):
                raise
            # Rejected requests generated no assessment. Ordinary text still
            # goes through the same strict parser, never straight to tools.
            self._count_retry("unsupported_response_format")
            return self.chat(messages, tools, **kwargs)

    def convert_messages(self, messages: list[dict]) -> list[dict]:
        """Canonical history -> OpenAI content blocks.

        Only messages carrying a canonical media block are rewritten; a plain
        string content is left exactly as it was, so a text-only conversation
        goes on the wire byte-identical to before attachments existed.

        `role: "tool"` messages strip `name` because OpenAI's tool message schema
        does not define `name` (only `role`, `content`, `tool_call_id`). Permissive
        endpoints ignore it, but strict upstream proxies (e.g. Console Go) reject it.
        """
        # Some OpenAI-compatible chat templates accept exactly one system
        # message. Keep canonical history unchanged and coalesce only the
        # leading system run in the provider-specific wire representation.
        wire_messages = messages
        leading_system = 0
        while (leading_system < len(messages)
               and messages[leading_system].get("role") == "system"):
            leading_system += 1
        if leading_system > 1:
            merged = "\n\n".join(
                str(message.get("content") or "")
                for message in messages[:leading_system]
            )
            wire_messages = ([{"role": "system", "content": merged}]
                             + messages[leading_system:])

        out: list[dict] = []
        for m in wire_messages:
            msg = m
            content = m.get("content")
            if isinstance(content, list) and media.blocks_of(content):
                msg = {**m, "content": [self._block(b) for b in content]}
            if msg.get("role") == "tool" and "name" in msg:
                msg = {k: v for k, v in msg.items() if k != "name"}
            out.append(msg)
        return out

    def _block(self, block: dict) -> dict:
        if block.get("type") != media.MEDIA_BLOCK:
            return block
        att = media.Attachment.from_block(block)
        # The bytes are read here, not stored in history: the canonical turn
        # holds the path so a session file stays small and resumable.
        media.check_supported([att], self.wire, self.model or self.base_url)
        if att.kind == media.IMAGE:
            return {"type": "image_url", "image_url": {"url": att.data_uri()}}
        return {"type": "input_audio",
                "input_audio": {"data": att.b64(), "format": att.fmt}}

    # Some endpoints pin a sampling parameter and 400 on any other value
    # (kimi-for-coding: "invalid temperature: only 1 is allowed for this model").
    # Dropping the named field and retrying once is what a human would do, and it
    # keeps one-cli usable on a new endpoint without a config edit first.
    RETRYABLE_PARAMS = ("temperature", "top_p", "max_tokens")

    # A model that has RETIRED sampling parameters rejects the whole request
    # when any of them is present, and names only one in the error. Dropping
    # just the named field costs a second 400 for the next one, so the three
    # go together: they are one deprecation, not three independent settings.
    SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

    @classmethod
    def _drop_sampling(cls, body: dict, detail: str, extra_body: dict) -> dict | None:
        """The body with every sampling parameter removed, or None.

        None means the refusal is not about sampling, or the operator pinned
        one in the drop-in - that is their claim about the endpoint and is
        never silently overridden.
        """
        low = detail.lower()
        if not any(p in low for p in cls.SAMPLING_PARAMS):
            return None
        if any(p in extra_body for p in cls.SAMPLING_PARAMS):
            return None
        if not any(p in body for p in cls.SAMPLING_PARAMS):
            return None
        return {k: v for k, v in body.items() if k not in cls.SAMPLING_PARAMS}

    # A server that 500s because it could not parse its OWN model's output has
    # not been asked anything wrong: llama.cpp's tool-call parser rejects a
    # harmony block gpt-oss emitted a moment earlier, and the next sample of
    # the same request usually parses. Resampling once is the whole fix; failing over to another lane for this
    # would blame a different model for a local parser's bad minute.
    OUTPUT_PARSE_MARKERS = ("does not match the expected", "peg-native",
                            "failed to parse tool call", "partial parse")

    @classmethod
    def _is_output_parse_failure(cls, detail: str) -> bool:
        low = detail.lower()
        return any(marker in low for marker in cls.OUTPUT_PARSE_MARKERS)

    # 429 and 503 are the two statuses that mean "come back shortly" rather
    # than "this is broken". Everything else, 500 included, stays the fallback
    # chain's job by design - a busy endpoint is better replaced than hammered.
    #
    # These two are the exception because failing over on them is actively
    # harmful: the next lane is a different model, so the answer changes, and a
    # rotation triggered by someone else's load can walk through every
    # configured credential and put each one on cooldown. One short local wait,
    # honouring the server's own Retry-After, is the cheaper and more honest
    # response.
    TRANSIENT_STATUS = (429, 503)
    MAX_TRANSIENT_RETRIES = 2
    BACKOFF_SECONDS = (0.5, 2.0)
    MAX_RETRY_AFTER = 10.0

    def _transient_wait(self, exc, attempt: int, detail: str = "") -> float | None:
        """Seconds to wait before retrying, or None to give up now."""
        if exc.code not in self.TRANSIENT_STATUS:
            return None
        if attempt >= self.MAX_TRANSIENT_RETRIES:
            return None
        # An output-parse refusal has already had its one resample; giving it
        # two more transport attempts would resample it three times under a
        # different name.
        if detail and self._is_output_parse_failure(detail):
            return None
        wait = self.BACKOFF_SECONDS[min(attempt, len(self.BACKOFF_SECONDS) - 1)]
        try:
            hinted = float(exc.headers.get("Retry-After") or 0)
        except (TypeError, ValueError, AttributeError):
            hinted = 0.0
        # The server's own hint wins when it is sane; a gateway asking for an
        # hour is asking us to hang, so it is capped rather than obeyed.
        return min(max(wait, hinted), self.MAX_RETRY_AFTER)

    def _post(self, body: dict, retried: str = "", attempt: int = 0) -> dict:
        if self.request_gate is not None:
            self.request_gate(body)
        req = urllib.request.Request(
            self._url(),
            data=serialize(self, body),
            headers=self._headers(),
            method="POST",
        )
        try:
            with self._open(req) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            if exc.code == 400 and not retried:
                low = detail.lower()
                trimmed = self._drop_sampling(body, detail, self.extra_body)
                if trimmed is not None:
                    return self._post(trimmed, retried="sampling")
                for param in self.RETRYABLE_PARAMS:
                    # An explicitly configured param is the operator's choice;
                    # never silently drop that one.
                    if param in low and param in body and param not in self.extra_body:
                        trimmed = dict(body)
                        trimmed.pop(param)
                        return self._post(trimmed, retried=param)
            if exc.code >= 500 and not retried and self._is_output_parse_failure(detail):
                self.output_retries += 1
                self._count_retry("malformed_tool_call")
                return self._post(body, retried="output-parse")
            wait = self._transient_wait(exc, attempt, detail)
            if wait is not None:
                self.transient_retries += 1
                self._count_retry(
                    "rate_limit" if exc.code == 429 else "server_error")
                time.sleep(wait)
                return self._post(body, retried=retried, attempt=attempt + 1)
            raise ProviderError(f"HTTP {exc.code} from {self._url()}: {http_error_detail(detail)}") from exc
        except (urllib.error.URLError, OSError, TimeoutError,
                http.client.HTTPException) as exc:
            raise ProviderError(f"cannot reach {self._url()}: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"non-JSON reply from {self._url()}: {exc}") from exc

    @response_shapes
    def _stream(self, body: dict, on_delta: OnDelta, retried: bool | str = False,
                attempt: int = 0) -> dict:
        """Read the SSE frames and rebuild the non-streaming response shape.

        Rebuilding rather than constructing a Turn directly is the point: the
        streamed and non-streamed paths then differ only in transport, so a
        parsing fix lands in one place and the two can be compared for equality.
        """
        if self.request_gate is not None:
            self.request_gate(body)
        req = urllib.request.Request(
            self._url(),
            data=serialize(self, body),
            headers={**self._headers(), "Accept": "text/event-stream"},
            method="POST",
        )
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        emitted_text = ""
        emitted_thinking = ""
        # Fragments arrive index-keyed and out of any useful order, and the
        # arguments of one call are split across frames as partial JSON, so the
        # call cannot be parsed until the stream ends.
        fragments: dict[int, dict] = {}
        finish_reason = ""
        usage: dict = {}
        done: list[bool] = []

        idle = stream_idle_seconds()
        try:
            with self._open(req, timeout=min(self.timeout, idle)) as resp:
                for frame in sse_payloads(resp, idle_seconds=idle,
                                          content_seconds=stream_content_seconds(),
                                          on_done=lambda: done.append(True)):
                    if frame.get("usage"):
                        usage = frame["usage"]
                    for choice in frame.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        chunk, block_thinking = split_content(delta.get("content"))
                        if chunk:
                            text_parts.append(chunk)
                        reasoning = (delta.get("reasoning_content")
                                     or delta.get("reasoning")
                                     or delta.get("thinking")
                                     or block_thinking)
                        if reasoning:
                            thinking_parts.append(reasoning)
                            on_delta("thinking", reasoning)
                        clean, inband = _reasoning_content(
                            "".join(text_parts), bool(thinking_parts), final=False)
                        if inband[len(emitted_thinking):]:
                            on_delta("thinking", inband[len(emitted_thinking):])
                        emitted_thinking = inband
                        if clean[len(emitted_text):]:
                            on_delta("text", clean[len(emitted_text):])
                        emitted_text = clean
                        for frag in delta.get("tool_calls") or []:
                            self._merge_tool_fragment(fragments, frag)
                            piece = (frag.get("function") or {}).get("arguments")
                            if piece:
                                on_delta("tool", piece)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            # Same resample as `_post`, but only while nothing has reached the
            # caller yet: replaying a stream that already printed half a
            # sentence would show the user the answer twice.
            if (exc.code >= 500 and not retried
                    and self._is_output_parse_failure(detail)
                    and not (text_parts or thinking_parts or fragments)):
                self.output_retries += 1
                self._count_retry("malformed_tool_call")
                return self._stream(body, on_delta, retried=True)
            # A retired-sampling refusal is decided before the first token, so
            # the same "nothing printed yet" rule applies: retry the trimmed
            # body, never replay a stream that already reached the caller.
            if (exc.code == 400 and not retried
                    and not (text_parts or thinking_parts or fragments)):
                trimmed = self._drop_sampling(body, detail, self.extra_body)
                if trimmed is not None:
                    return self._stream(trimmed, on_delta, retried="sampling")
            wait = self._transient_wait(exc, attempt, detail)
            # Same rule as the resample above: nothing may be replayed once a
            # token has reached the caller, or the reader sees half the answer
            # twice.
            if wait is not None and not (text_parts or thinking_parts or fragments):
                self.transient_retries += 1
                self._count_retry(
                    "rate_limit" if exc.code == 429 else "server_error")
                time.sleep(wait)
                return self._stream(body, on_delta, retried=retried,
                                    attempt=attempt + 1)
            raise ProviderError(f"HTTP {exc.code} from {self._url()}: {http_error_detail(detail)}") from exc
        except http.client.IncompleteRead as exc:
            # The body was cut mid-chunk: a fragment, never an answer, so it
            # takes the truncation path (re-sent while nothing was shown).
            raise truncated(self._url()) from exc
        except (urllib.error.URLError, OSError, TimeoutError,
                http.client.HTTPException) as exc:
            stalled = idle_error(exc, idle)
            if stalled is not None:
                raise stalled from exc
            raise ProviderError(f"cannot reach {self._url()}: {exc}") from exc

        # A stream that ends without a finish_reason AND without a single
        # token is a connection that closed before the model said anything -
        # llama.cpp does exactly this when the request overruns the server's
        # `n_ctx`. It is a transport failure, so it is raised as one: reported
        # as an answered-but-empty turn it would end the run where the fallback
        # chain would have carried it.
        if not (finish_reason or text_parts or thinking_parts or fragments):
            raise ProviderError(
                f"{self._url()} closed the stream without sending anything "
                "(no finish_reason, no content) - the request may exceed the "
                "server's context window")
        # Content but neither [DONE] nor a finish_reason: the connection
        # dropped mid-answer. Either marker alone is terminal - some providers
        # omit [DONE] and send only the finish_reason.
        if not (finish_reason or done):
            raise truncated(self._url())
        clean, inband = _reasoning_content("".join(text_parts), bool(thinking_parts))
        if inband[len(emitted_thinking):]:
            on_delta("thinking", inband[len(emitted_thinking):])
        if clean[len(emitted_text):]:
            on_delta("text", clean[len(emitted_text):])
        message: dict = {"content": "".join(text_parts)}
        if thinking_parts:
            message["reasoning_content"] = "".join(thinking_parts)
        if fragments:
            message["tool_calls"] = [fragments[i] for i in sorted(fragments)]
        return {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

    @staticmethod
    def _merge_tool_fragment(fragments: dict[int, dict], frag: dict) -> None:
        """Fold one `delta.tool_calls` entry into the call it belongs to.

        `index` is the only reliable key - id and name usually appear once, on
        the first fragment, and arguments then arrive as a string to append to.
        """
        index = frag.get("index", 0)
        slot = fragments.setdefault(
            index, {"id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
        if frag.get("id"):
            slot["id"] = frag["id"]
        fn = frag.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]
        # Anything the server put beside the known keys is carried through
        # untouched - a signature arrives on one fragment only, and dropping it
        # here would make the streaming path fail where the buffered one works.
        for key, value in frag.items():
            if key not in _TOOL_CALL_KNOWN_KEYS and value:
                slot[key] = value

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json",
                   "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    @staticmethod
    def _usage(usage: dict) -> dict:
        """Add the canonical `cache_read_tokens` (see Turn).

        `prompt_tokens` already includes cache hits on this wire. Where the
        hit count sits depends on who serves it: OpenAI nests it under
        `prompt_tokens_details.cached_tokens`, Moonshot reports a top-level
        `cached_tokens`, DeepSeek `prompt_cache_hit_tokens`.
        """
        details = usage.get("prompt_tokens_details")
        candidates = (
            details.get("cached_tokens") if isinstance(details, dict) else None,
            usage.get("cached_tokens"),
            usage.get("prompt_cache_hit_tokens"),
        )
        read = next((v for v in candidates if type(v) is int and v >= 0), None)
        if read is None:
            return dict(usage)
        return {**usage, "cache_read_tokens": read}

    @staticmethod
    @response_shapes
    def _to_turn(data: dict) -> Turn:
        if not isinstance(data, dict):
            raise ProviderError(f"reply is not a JSON object: {json.dumps(data)[:400]}")
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError(f"reply had no choices: {json.dumps(data)[:400]}")
        msg = choices[0].get("message") or {}
        if not isinstance(msg, dict):
            raise ProviderError("reply message must be an object")

        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            raw_args = args
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    # A model that emits invalid JSON arguments is a real, common
                    # failure. Surface it as an argument the tool will reject,
                    # rather than dropping the call and looking like a no-op turn.
                    args = {"__unparsed__": args}
            if not isinstance(args, dict):
                args = {"__unparsed__": raw_args}
            extra = {k: v for k, v in tc.items()
                     if k not in _TOOL_CALL_KNOWN_KEYS}
            calls.append(ToolCall(id=tc.get("id") or "", name=fn.get("name") or "",
                                  arguments=args or {}, extra=extra))

        text, block_thinking = split_content(msg.get("content"))
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or block_thinking
        text, inband = _reasoning_content(text, bool(thinking))
        return Turn(
            text=text,
            thinking=thinking + inband,
            echo=({"reasoning_content": msg["reasoning_content"]}
                  if msg.get("reasoning_content") else {}),
            tool_calls=calls,
            finish_reason=choices[0].get("finish_reason") or "",
            usage=OpenAIWire._usage(data.get("usage") or {}),
            raw=data,
        )

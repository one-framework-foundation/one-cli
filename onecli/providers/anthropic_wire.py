"""Anthropic-shaped `/v1/messages`.

llama-server speaks this natively, which is why provider drop-ins can
carry `ANTHROPIC_BASE_URL`. The shape differences that matter: the system prompt
is a top-level field rather than a message, content is a block list, and a tool
call is a `tool_use` block instead of a sibling `tool_calls` array.
"""

from __future__ import annotations

from onecli.providers.context_metrics import serialize

from onecli.providers.base import response_shapes

import http.client
import json
import sys
import urllib.error
import urllib.request

from onecli import media
from onecli.providers.base import (USER_AGENT, OnDelta, ProviderError,
                                   ToolCall, Turn, http_error_detail, idle_error,
                                   sse_payloads, stream_content_seconds,
                                   stream_idle_seconds,
                                   stream_with_idle_retry, truncated)

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicWire:
    wire = "anthropic"
    effort_decision: dict | None = None

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 600.0, extra_body: dict | None = None,
                 extra_headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self._max_output: int | None = None
        self.extra_headers = extra_headers or {}

    def declared_max_output(self) -> int:
        """The model's own output maximum, from the Models API.

        This wire requires max_tokens on every request. With no ceiling
        configured we send the one the endpoint declares for the model, never
        a smaller default of ours. Asked once per wire.

        llama-server has no per-model route (`/v1/models/<id>` is 404) and
        declares no max_tokens; its list entry carries the served context as
        `meta.n_ctx`, which is the most it can generate. That fallback is
        announced on stderr, never silent.
        """
        if self._max_output is None:
            base = self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
            url = f"{base}/models/{self.model}"
            try:
                declared = self._get_json(url).get("max_tokens")
            except (urllib.error.URLError, OSError, ValueError, AttributeError,
                http.client.HTTPException) as exc:
                declared, why = None, f"{url}: {exc}"
            else:
                why = f"{url} declares no max_tokens"
            if not isinstance(declared, int) or declared <= 0:
                declared, source = self._listed_max_output(base + "/models")
                if declared is None:
                    raise ProviderError(f"cannot read the output maximum of {self.model} "
                                        f"({why}; {source}); set --max-tokens")
                print(f"one-cli: {why}; using {source} = {declared} as the "
                      f"output maximum of {self.model}", file=sys.stderr)
            self._max_output = declared
        return self._max_output

    def _listed_max_output(self, url: str) -> tuple[int | None, str]:
        """The model's ceiling from the `/v1/models` list, and where it came from."""
        try:
            listed = self._get_json(url).get("data")
        except (urllib.error.URLError, OSError, ValueError, AttributeError,
                http.client.HTTPException) as exc:
            return None, f"{url}: {exc}"
        for entry in listed if isinstance(listed, list) else []:
            if not isinstance(entry, dict) or entry.get("id") != self.model:
                continue
            for key, value in (("max_tokens", entry.get("max_tokens")),
                               ("meta.n_ctx", entry["meta"].get("n_ctx") if isinstance(entry.get("meta"), dict) else None)):
                if isinstance(value, int) and value > 0:
                    return value, f"{url} {key}"
            return None, f"{url} lists {self.model} with no max_tokens or meta.n_ctx"
        return None, f"{url} does not list {self.model}"

    def _get_json(self, url: str):
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    def _url(self) -> str:
        base = self.base_url
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             on_delta: OnDelta | None = None) -> Turn:
        system, convo = self._split_system(
            messages, backend=self.model or self.base_url)
        body: dict = {
            "model": self.model,
            "messages": convo,
            "max_tokens": max_tokens or self.declared_max_output(),
        }
        if temperature is not None:
            body["temperature"] = temperature
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [self._tool_to_anthropic(t) for t in tools]
        if on_delta is not None:
            body["stream"] = True
        body.update(self.extra_body)
        from onecli import config
        config.validate_model("", self.base_url, body["model"])

        from onecli.providers.base import effort_observation
        if on_delta is not None:
            turn = self._to_turn(stream_with_idle_retry(
                lambda delta: self._stream(body, delta), on_delta))
            turn.request_effort = effort_observation(body, "anthropic")
            return turn

        req = urllib.request.Request(
            self._url(),
            data=serialize(self, body),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            raise ProviderError(f"HTTP {exc.code} from {self._url()}: {http_error_detail(detail)}") from exc
        except (urllib.error.URLError, OSError, TimeoutError,
                http.client.HTTPException) as exc:
            raise ProviderError(f"cannot reach {self._url()}: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"non-JSON reply from {self._url()}: {exc}") from exc

        turn = self._to_turn(data)
        turn.request_effort = effort_observation(body, "anthropic")
        return turn

    @response_shapes
    def _stream(self, body: dict, on_delta: OnDelta) -> dict:
        """Fold the event stream back into a `/v1/messages` response body.

        Same reason as the openai wire: the block list is rebuilt so `_to_turn`
        stays the single place that decides what a Turn looks like.
        """
        req = urllib.request.Request(
            self._url(),
            data=serialize(self, body),
            headers={**self._headers(), "Accept": "text/event-stream"},
            method="POST",
        )
        blocks: dict[int, dict] = {}
        # tool_use input arrives as partial JSON in a sibling field, so it is
        # accumulated apart from the block and parsed once the stream ends.
        partial_json: dict[int, str] = {}
        stop_reason = ""
        stopped = False
        usage: dict = {}
        idle = stream_idle_seconds()

        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, idle)) as resp:
                for event in sse_payloads(resp, idle_seconds=idle,
                                          content_seconds=stream_content_seconds()):
                    etype = event.get("type")
                    if etype == "error":
                        raise ProviderError(
                            json.dumps(event.get("error") or event)[:400])
                    if etype == "message_start":
                        usage = dict((event.get("message") or {}).get("usage") or {})
                    elif etype == "content_block_start":
                        index = event.get("index", 0)
                        block = dict(event.get("content_block") or {})
                        blocks[index] = block
                        if block.get("type") == "tool_use":
                            partial_json[index] = ""
                    elif etype == "content_block_delta":
                        self._apply_delta(blocks, partial_json, event, on_delta)
                    elif etype == "message_delta":
                        stop_reason = ((event.get("delta") or {}).get("stop_reason")
                                       or stop_reason)
                        usage.update(event.get("usage") or {})
                    elif etype == "message_stop":
                        stopped = True
                        break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
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
        # A stream that ended before message_stop (or at least a stop_reason)
        # was cut off; what arrived is a fragment, not an answer.
        if not (stopped or stop_reason):
            raise truncated(self._url())

        for index, raw in partial_json.items():
            pending = blocks.get(index)
            if pending is None:
                continue
            block = pending
            try:
                parsed = json.loads(raw) if raw.strip() else {}
                block["input"] = parsed if isinstance(parsed, dict) else {"__unparsed__": raw}
            except ValueError:
                # Same call as the openai wire: hand the tool an argument it
                # will reject rather than dropping the call silently.
                block["input"] = {"__unparsed__": raw}

        return {
            "content": [blocks[i] for i in sorted(blocks)],
            "stop_reason": stop_reason,
            "usage": usage,
        }

    @staticmethod
    def _apply_delta(blocks: dict[int, dict], partial_json: dict[int, str],
                     event: dict, on_delta: OnDelta) -> None:
        index = event.get("index", 0)
        delta = event.get("delta") or {}
        dtype = delta.get("type")
        block = blocks.setdefault(index, {"type": "text", "text": ""})
        if dtype == "text_delta":
            chunk = delta.get("text") or ""
            block["text"] = (block.get("text") or "") + chunk
            if chunk:
                on_delta("text", chunk)
        elif dtype == "thinking_delta":
            chunk = delta.get("thinking") or ""
            block["thinking"] = (block.get("thinking") or "") + chunk
            if chunk:
                on_delta("thinking", chunk)
        elif dtype == "input_json_delta":
            chunk = delta.get("partial_json") or ""
            partial_json[index] = partial_json.get(index, "") + chunk
            if chunk:
                on_delta("tool", chunk)

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    @staticmethod
    def _tool_to_anthropic(tool: dict) -> dict:
        """Accept the OpenAI tool shape the rest of one-cli uses."""
        if "input_schema" in tool:
            return tool
        fn = tool.get("function") or tool
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }

    @staticmethod
    def _media_block(block: dict, backend: str) -> dict:
        """Canonical media block -> an anthropic source block.

        Anthropic-shaped endpoints carry images only; audio and video are
        refused here as well as at the CLI edge, because a resumed history can
        reach a backend that the turn was never recorded against.
        """
        att = media.Attachment.from_block(block)
        media.check_supported([att], AnthropicWire.wire, backend)
        return {"type": "image",
                "source": {"type": "base64", "media_type": att.mime,
                           "data": att.b64()}}

    @staticmethod
    def _split_system(messages: list[dict], backend: str = "") -> tuple[str, list[dict]]:
        """Canonical history is OpenAI-shaped; convert it here.

        Three shape differences, each of which silently breaks a tool loop if
        missed: the system prompt is a top-level field, an assistant tool call
        is a `tool_use` content block rather than a sibling array, and a tool
        result is a `user` message carrying `tool_result` blocks rather than a
        `tool` role. Consecutive tool results must also be merged into one user
        message, because the API rejects two user turns in a row.
        """
        system_parts: list[str] = []
        convo: list[dict] = []

        for m in messages:
            role = m.get("role")

            if role == "system":
                system_parts.append(str(m.get("content") or ""))
                continue

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "",
                    "content": str(m.get("content") or ""),
                }
                if convo and convo[-1]["role"] == "user" and isinstance(
                    convo[-1].get("content"), list
                ):
                    convo[-1]["content"].append(block)
                else:
                    convo.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    raw_args = args
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except ValueError:
                            args = {"__unparsed__": raw_args}
                    if not isinstance(args, dict):
                        args = {"__unparsed__": raw_args}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": args or {},
                    })
                if not blocks:
                    blocks = [{"type": "text", "text": ""}]
                convo.append({"role": "assistant", "content": blocks})
                continue

            content = m.get("content")
            if isinstance(content, list) and media.blocks_of(content):
                convo.append({"role": "user", "content": [
                    AnthropicWire._media_block(b, backend)
                    if b.get("type") == media.MEDIA_BLOCK else b
                    for b in content
                ]})
            else:
                convo.append({"role": "user", "content": str(content or "")})

        return "\n\n".join(p for p in system_parts if p), convo

    @staticmethod
    def _usage(usage: dict) -> dict:
        """Add the canonical input fields (see Turn).

        Anthropic's `input_tokens` counts only what was NOT read from or
        written to the cache; reads and writes are reported beside it. Every
        one of them was processed, so all three make up `prompt_tokens`.
        """
        def n(key):
            value = usage.get(key)
            return value if type(value) is int and value > 0 else 0
        read = n("cache_read_input_tokens")
        total = n("input_tokens") + read + n("cache_creation_input_tokens")
        if not total:
            return dict(usage)
        return {**usage, "prompt_tokens": total, "cache_read_tokens": read}

    @staticmethod
    @response_shapes
    def _to_turn(data: dict) -> Turn:
        if not isinstance(data, dict):
            raise ProviderError(f"reply is not a JSON object: {json.dumps(data)[:400]}")
        if data.get("type") == "error":
            raise ProviderError(json.dumps(data.get("error") or data)[:400])
        blocks = data.get("content") or []
        if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
            raise ProviderError("reply content must be a list of objects")
        text_parts, thinking_parts, calls = [], [], []
        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "thinking":
                thinking_parts.append(block.get("thinking") or "")
            elif btype == "tool_use":
                calls.append(ToolCall(id=block.get("id") or "",
                                      name=block.get("name") or "",
                                      arguments=(block.get("input") if isinstance(block.get("input"), dict)
                                                 else {"__unparsed__": block.get("input")})))
        return Turn(
            text="".join(text_parts),
            thinking="".join(thinking_parts),
            tool_calls=calls,
            finish_reason=data.get("stop_reason") or "",
            usage=AnthropicWire._usage(data.get("usage") or {}),
            raw=data,
        )

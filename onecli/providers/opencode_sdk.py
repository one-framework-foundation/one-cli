"""Structured OpenCode SDK transport; credentials remain inside the vendor host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import uuid

from onecli.providers.base import ProviderError, ToolCall, Turn

SDK_VERSION = "2.0.32"
PROTOCOL_VERSION = 1


def _request(payload: dict, timeout: float) -> dict:
    if os.name != "posix":
        raise ProviderError("OpenCode SDK transport requires a POSIX host")
    sdk_dir = os.environ.get("ONE_OPENCODE_SDK_DIR", "")
    if not sdk_dir:
        raise ProviderError(
            f"OpenCode requires @opencode/sdk@{SDK_VERSION}; set ONE_OPENCODE_SDK_DIR "
            "to its npm installation directory (see docs/opencode.md)")
    helper = Path(__file__).resolve().parents[1] / "assets" / "opencode_transport.mjs"
    try:
        process = subprocess.Popen(
            ["bun", str(helper)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, start_new_session=True)
    except OSError:
        raise ProviderError("OpenCode SDK requires Bun on PATH") from None
    try:
        stdout, _ = process.communicate(json.dumps(payload), timeout=timeout)
    except BaseException as exc:
        # Cancellation and deadlines terminate the entire owned host, including
        # SDK children. Never leave a generation running after the caller stops.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise ProviderError("OpenCode SDK timed out") from None
        raise
    try:
        result = json.loads(stdout)
        if result.get("version") != PROTOCOL_VERSION:
            raise ValueError()
        if process.returncode or "error" in result:
            codes = {"sdk_unavailable", "invalid_request", "model_unavailable",
                     "invalid_bounds", "generation_failed", "policy_denied"}
            code = result.get("error")
            raise ProviderError(f"OpenCode SDK: {code if isinstance(code, str) and code in codes else 'transport_failed'}")
        return result
    except (ValueError, AttributeError):
        raise ProviderError("OpenCode SDK returned an invalid protocol response") from None


class OpenCodeSDK:
    def __init__(self, model: str, timeout: float = 600.0):
        provider, sep, model_id = model.partition("/")
        if not sep or not provider or not model_id:
            raise ProviderError("OpenCode model must be provider/model")
        self.model = model
        self.ref = {"providerID": provider, "id": model_id}
        self.timeout = timeout
        self.effort_decision: dict = {}
        self.request_gate = None
        self.session_id = "onecli-" + uuid.uuid4().hex

    def chat(self, messages, tools=None, temperature=None, max_tokens=None,
             on_delta=None, response_format=None) -> Turn:
        if self.request_gate is not None:
            raise ProviderError("OpenCode SDK cannot enforce audited wire request limits")
        if response_format is not None:
            raise ProviderError("OpenCode SDK does not support response_format")
        result = _request({"version": PROTOCOL_VERSION, "model": self.ref,
                           "session_id": self.session_id, "messages": messages,
                           "tools": tools, "temperature": temperature,
                           "max_tokens": max_tokens}, self.timeout)
        try:
            usage = result["usage"]
            if (not isinstance(usage, dict) or not isinstance(result["text"], str)
                    or not isinstance(result["thinking"], str)
                    or not isinstance(result["message"], dict)
                    or result["message"].get("role") != "assistant"
                    or not isinstance(result["message"].get("content"), list)
                    or not isinstance(result["tool_calls"], list)
                    or not isinstance(result["finish_reason"], str)):
                raise ValueError()
            for call in result["tool_calls"]:
                if (not isinstance(call["id"], str) or not isinstance(call["name"], str)
                        or not isinstance(call["input"], dict)):
                    raise ValueError()
            turn = Turn(text=result["text"], thinking=result["thinking"],
                        tool_calls=[ToolCall(id=call["id"], name=call["name"],
                                             arguments=call["input"])
                                    for call in result["tool_calls"]],
                        finish_reason=result["finish_reason"],
                        usage={**usage, "prompt_tokens": usage.get("inputTokens", 0),
                               "completion_tokens": usage.get("outputTokens", 0),
                               "cache_read_tokens": usage.get("cacheReadInputTokens", 0)},
                        echo={"opencode_message": result["message"]})
        except (KeyError, TypeError, ValueError):
            raise ProviderError("OpenCode SDK returned an invalid turn") from None
        allowed = {tool["function"]["name"] for tool in tools or []}
        if any(call.name not in allowed for call in turn.tool_calls):
            raise ProviderError("OpenCode SDK returned an unoffered tool")
        if on_delta:
            if turn.thinking:
                on_delta("thinking", turn.thinking)
            if turn.text:
                on_delta("text", turn.text)
        return turn

    def chat_json(self, messages, tools=None, **kwargs) -> Turn:
        return self.chat(messages, tools, **kwargs)

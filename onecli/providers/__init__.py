from __future__ import annotations
from onecli.config import WIRE_ANTHROPIC, WIRE_OPENAI, Backend
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.base import ProviderError, ToolCall, Turn
from onecli.providers.openai_wire import OpenAIWire


# Keys one-cli reads off a drop-in that are NOT wire parameters. `extra` is
# merged into the request body, so anything we invented has to be stripped
# before it reaches an endpoint that would reject or misread it.
LOCAL_EXTRA_KEYS = ("modalities", "effort_profiles")


def _wire_extra(extra: dict | None) -> dict:
    return {k: v for k, v in (extra or {}).items() if k not in LOCAL_EXTRA_KEYS}


def build(backend: Backend, timeout: float = 600.0, *, effort="auto", task_class=""):
    from onecli.effort import configure
    try:
        extra, decision = configure(backend.extra, effort, task_class)
    except ValueError as exc:
        raise ProviderError(f"{backend.name}: {exc}") from exc
    if backend.wire == "opencode":
        from onecli.providers.opencode_sdk import OpenCodeSDK
        if _wire_extra(extra) or backend.headers or backend.api_key or backend.base_url:
            raise ProviderError("OpenCode SDK uses vendor configuration; wire overrides are unsupported")
        provider_sdk = OpenCodeSDK(backend.model, timeout)
        provider_sdk.effort_decision = decision
        return provider_sdk
    wires: dict[str, type[AnthropicWire] | type[OpenAIWire]] = {
        WIRE_ANTHROPIC: AnthropicWire, WIRE_OPENAI: OpenAIWire}
    wire = wires.get(backend.wire)
    if wire is None:
        raise ProviderError(f"unknown wire {backend.wire!r}")
    provider = wire(backend.base_url, backend.model, backend.api_key, timeout,
                    extra_body=_wire_extra(extra), extra_headers=backend.headers)
    provider.effort_decision = decision
    return provider


__all__ = ["build", "AnthropicWire", "OpenAIWire", "ProviderError", "ToolCall", "Turn"]


def bind_session(provider, session_id):
    """Bind conversation identity when the transport declares a session header."""
    header = getattr(provider, "session_header", None)
    if header and session_id:
        provider.extra_headers = {**provider.extra_headers, header: session_id}
    return provider

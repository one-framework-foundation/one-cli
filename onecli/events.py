"""Machine-readable progress events for `--json` runs.

WHY THIS EXISTS. Under `--json` the whole point of the run is the single JSON
object on stdout, so the terminal trace is suppressed and stderr goes silent
for the entire turn. A parent process driving one-cli therefore has nothing to
show a person between the prompt and the answer except "waiting on <model>
(Ns)", which says nothing about what the model is DOING.

So under `--json` the same `on_event` callback that draws the terminal trace
also writes one compact NDJSON object per interesting event to **stderr**,
flushed per line. stdout is untouched: it stays exactly one JSON object, which
is the contract every caller parses.

Always-on rather than behind a flag, deliberately. Nothing else reads this
stderr under `--json` (a run that succeeds writes nothing there at all), and a
flag would mean every caller has to learn about it before it helps anyone. A
consumer that does not want the events can ignore lines it cannot parse - which
is what it must do anyway, because real diagnostics share the stream.

Shape, kept small and stable:

    {"e":"waiting","step":3}
    {"e":"tool_start","step":3,"tool":"Read","arg":""}
    {"e":"tool_end","step":3,"tool":"Read","ok":true}
    {"e":"compact","step":7,"ok":false,"failure":"starved"}
    {"e":"fallback","to":"local-27b"}
    {"e":"empty_retry","step":12,"finish_reason":"length"}

`text_delta` becomes a throttled `progress` count, at most once per second:
one line per token would swamp the pipe. By default model text never enters
this stream. Explicit --text-events adds bounded public-text deltas for a UI;
tool argument bodies and private reasoning are never emitted as text deltas.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Callable


def emitter(stream=None, step=None, include_text=False) -> Callable[[str, dict], None]:
    """Return an `on_event`-shaped sink that writes NDJSON to `stream`.

    `step` is a zero-argument callable returning the step index the run is on.
    Only `waiting` and `turn` carry a step in their own payload; the tool
    events are stamped from it so a status line can say "step 3 - Read x.py".
    """
    out = stream if stream is not None else sys.stderr
    step_of = step if step is not None else (lambda: 0)
    last_progress = float('-inf')
    arrived_chars = 0

    def write(payload: dict) -> None:
        try:
            out.write(json.dumps(payload, separators=(",", ":")) + "\n")
            out.flush()
        except (OSError, ValueError):
            # A parent that closed the pipe must never take the run down with
            # it: the answer on stdout is still worth finishing.
            pass

    def emit(kind: str, data: dict) -> None:
        nonlocal last_progress, arrived_chars

        if kind == "text_delta":
            chunk = data.get("chunk")
            if not chunk:
                return
            if include_text and data.get("kind") == "text":
                # Explicit public-text channel. Reasoning and tool arguments
                # remain content-free even when a UI opts into answer deltas.
                # Bound each encoded line below the parent's pipe read limit.
                for offset in range(0, len(chunk), 2048):
                    write({"e": "text_delta", "step": step_of(),
                           "text": chunk[offset:offset + 2048]})
            arrived_chars += len(chunk)
            now = time.monotonic()
            if now - last_progress >= 1.0:
                last_progress = now
                write({"e": "progress", "step": step_of(),
                       "kind": data.get("kind"), "chars": arrived_chars})
        elif kind == "assistant_text":
            if include_text:
                text = data["text"]
                for offset in range(0, len(text), 2048):
                    write({"e": kind, "step": step_of(),
                           "text": text[offset:offset + 2048],
                           "more": offset + 2048 < len(text)})
        elif kind in ("acknowledgment", "progress_update"):
            if include_text:
                write({"e": kind, "step": step_of(),
                       "text": data["text"]})
        elif kind == "response_gate":
            write({"e": "response_gate", "decision": data["decision"],
                   "attempt": data["attempt"]})
        elif kind == "discovery_checkpoint":
            # The resolved target names a chat, so only its presence travels.
            write({"e": kind, "step": data["step"],
                   **{k: data[k] for k in ("rounds", "decision", "final") if k in data},
                   **({"resolved": True} if data.get("resolved") else {})})
        elif kind == "meditate":
            write({"e": "meditate", "status": data["status"], "step": data["step"]})
        elif kind == "session":
            write({"e": "session", "id": data["id"]})
        elif kind == "waiting":
            write({"e": "waiting", "step": data.get("step", 0)})
        elif kind == "turn":
            step = data.get("step")
            write({"e": "turn", "step": getattr(step, "index", 0)})
        elif kind in ("tool_start", "tool_denied", "tool_end"):
            call = data.get("call")
            name = getattr(call, "name", "") or ""
            payload = {"e": kind, "step": step_of(), "tool": name}
            if kind == "tool_start":
                # Arguments, including paths, may contain credentials or private data.
                payload["arg"] = ""
                payload["mutating"] = bool(data.get("mutating"))
            elif kind == "tool_end":
                output = data.get("output") or ""
                # Carry the actual gate outcome through post-tool hooks;
                # display text may have been transformed after refusal.
                payload["denied"] = bool(data.get("denied", False))
                payload["ok"] = (not payload["denied"]
                                 and not str(output).startswith("error:"))
            write(payload)
        elif kind == "compact":
            # `failure` and `fallback` are the difference between a status line
            # that says "context compaction failed" forty times and one that
            # says WHY - and they are stable tokens, not prose, so a consumer
            # can render them without ever quoting runner payload text.
            payload = {"e": "compact", "step": step_of(),
                       "ok": bool(data.get("ok"))}
            if data.get("failure"):
                payload["failure"] = str(data.get("failure"))[:32]
            if data.get("fallback"):
                payload["fallback"] = True
            write(payload)
        elif kind == "fallback":
            write({"e": "fallback", "to": str(data.get("to", ""))[:60]})
        elif kind == "empty_retry":
            # Dropping the step and the reason made this event unreadable: a
            # bare "empty_retry" cannot be told from the one before it.
            write({"e": "empty_retry", "step": data.get("step", 0),
                   "finish_reason": str(data.get("finish_reason", ""))[:32]})
        elif kind == "todos":
            items = data.get("items") or []
            write({"e": "todos", "n": len(items)})

    return emit


def signals_sink(path: str) -> Callable[[str, dict], None]:
    """An `on_event` sink that mirrors the run's own todo plan into `path`.

    A parent process that exports BG_SIGNALS (one-cli's own task manager
    does) reads progress from PHASES:/PHASE: lines appended to that file. The
    run writes them itself rather than relying on the model to.
    The plan is the agent's own statement of its steps, so it is the source:
    a plan of N items writes `PHASES: N` when N changes, and each item that
    reaches done writes one `PHASE: <item>`. Nothing is invented - a run with
    no plan writes nothing.
    """
    seen: dict = {"total": 0, "done": set()}

    def emit(kind: str, data: dict) -> None:
        if kind != "todos":
            return
        items = data.get("items") or []
        lines = []
        if items and len(items) != seen["total"]:
            seen["total"] = len(items)
            lines.append(f"PHASES: {len(items)}")
        for item in items:
            text = " ".join(str(item.get("text", "")).split())[:200]
            if item.get("state") == "done" and text and text not in seen["done"]:
                seen["done"].add(text)
                lines.append(f"PHASE: {text}")
        if not lines:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass  # progress is best-effort; the run itself must not fail on it

    return emit


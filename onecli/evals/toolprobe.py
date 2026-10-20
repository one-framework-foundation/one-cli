"""one-toolprobe - can this model hold a tool loop, measured at the wire.

`bench.py` asks whether a model can finish a coding task; the answer is a
workspace diff, and when it says no it does not say WHY. This asks the question
one layer down, directly against the endpoint: does the model emit a structured
tool call at all, can its arguments be parsed and executed, does it use the
result, and can it chain three calls where each argument comes from the last
answer. A model that fails here cannot pass bench, and now the run says at which
rung it fell over.

The ladder:

* **L1**  a structured `tool_calls` entry arrives on the wire.
* **L1b** a tool call arrives as PROSE instead - a JSON blob, an XML tag, a
  fenced block, a bare invocation. This is a distinct failure from silence and
  is the one most often misread as "the model ignored the tools".
* **L2**  the arguments parse and the tool actually runs.
* **L3**  the model produces a further turn that uses the tool's result.
* **L4**  several tools in one task, including a three-deep dependent chain.

Beside the rungs the battery aims at named failure modes: recovery from a tool
error, tolerance of a tool that fails every time, selection between two
near-identical schemas where one is deprecated and wrong, forced emission under
`tool_choice`, exact output-format compliance, and a long-context needle.

Every verdict is a pure function of the recorded rows, and the raw rows are
written, so a score can be recomputed and argued with rather than trusted.

The transport is the one thing this version adds: an earlier form spoke raw
OpenAI HTTP to a hand-written endpoint list, so it could not reach an
Anthropic-wire endpoint and had to be told about every endpoint by hand. Here
it goes through `onecli.providers`, which means the same battery runs against
any backend `one` can already resolve, on either wire, with no new config.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import config
from ..providers import ProviderError, build

# --------------------------------------------------------------- tool battery
# A fixture deployment: small, closed, and answerable only through the tools. The
# values are deliberately not guessable, so a model that "knows" the answer
# without calling anything is visible as exactly that.

_BUILDS = {
    "core": "2026.07.29-b41",
    "operator-ui": "2026.07.29-b41",
    "archive-service": "2026.07.29-t19",
    "vector-gateway": "2026.07.29-l07",
    "index-core": "2026.07.29-s02",
    "model-daemon": "2026.07.29-o55",
}
_PORTS = {"vector": 4101, "index": 4102, "model": 4103, "operator": 4104}
_PORT_OWNER = {4101: "vector-gateway", 4102: "index-core",
               4103: "model-daemon", 4104: "archive-service"}


def _t_get_build_info(component=None):
    key = (component or "").strip()
    if key not in _BUILDS:
        return "Error: unknown component '%s'. Known: %s" % (
            component, ", ".join(sorted(_BUILDS)))
    return json.dumps({"component": key, "build": _BUILDS[key],
                       "commit": "a17f3c9", "dirty": False})


def _t_lookup_port(service=None):
    key = (service or "").strip().lower()
    if key not in _PORTS:
        return "Error: no such service '%s'. Known services: %s" % (
            service, ", ".join(sorted(_PORTS)))
    return json.dumps({"service": key, "port": _PORTS[key]})


def _t_lookup_port_legacy(service=None):
    """Near-duplicate of lookup_port, deprecated, returns WRONG numbers.

    Picking this one is a tool-SELECTION failure rather than an emission
    failure, and the two are worth telling apart: a model that emits the wrong
    call is further along than one that emits none.
    """
    key = (service or "").strip().lower()
    if key not in _PORTS:
        return "Error: no such service '%s'" % service
    return json.dumps({"service": key, "port": _PORTS[key] + 10000,
                       "warning": "legacy table, values are stale"})


def _t_read_config(path=None):
    """The trap: every path fails, every time, so repeated failure is reachable."""
    return "Error: no such file or directory: %s" % (path or "<none>")


def _t_whoami_port(port=None):
    try:
        number = int(port)
    except (TypeError, ValueError):
        return "Error: port must be an integer, got %r" % (port,)
    if number not in _PORT_OWNER:
        return "Error: no component owns port %s" % number
    return json.dumps({"port": number, "component": _PORT_OWNER[number]})


def _schema(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": required}}}


TOOLS: dict[str, tuple[Callable[..., str], dict]] = {
    "get_build_info": (_t_get_build_info, _schema(
        "get_build_info", "Return the current build id of a software component.",
        {"component": {"type": "string", "description": "component name"}},
        ["component"])),
    "lookup_port": (_t_lookup_port, _schema(
        "lookup_port", "Return the TCP port a named deployment service listens on.",
        {"service": {"type": "string", "description": "service name"}},
        ["service"])),
    "lookup_port_legacy": (_t_lookup_port_legacy, _schema(
        "lookup_port_legacy",
        "DEPRECATED. Old port table for a named deployment service. The values are "
        "stale and wrong. Do not use this unless explicitly told to.",
        {"service": {"type": "string", "description": "service name"}},
        ["service"])),
    "read_config": (_t_read_config, _schema(
        "read_config", "Read a config file from disk and return its contents.",
        {"path": {"type": "string", "description": "absolute path"}},
        ["path"])),
    "whoami_port": (_t_whoami_port, _schema(
        "whoami_port", "Return which software component owns a given TCP port.",
        {"port": {"type": "integer", "description": "TCP port number"}},
        ["port"])),
}

_NEEDLE = "ZQ7-MARMOT-4417"


def _long_haystack() -> str:
    """~20k tokens of filler with the needle near the START.

    At the start deliberately: a truncating runtime drops the oldest tokens
    first, so a model whose real window is smaller than the prompt loses the
    needle and cannot answer. Needle-at-the-end measures nothing.
    """
    lines = []
    for i in range(1, 2600):
        if i == 4:
            lines.append("record %04d: the passphrase is %s" % (i, _NEEDLE))
        else:
            lines.append("record %04d: routine inventory entry, nothing of note, "
                         "checksum %06d, status nominal" % (i, i * 37 % 999983))
    return "\n".join(lines)


def _has(text, *subs) -> bool:
    hay = (text or "").lower()
    return all(sub.lower() in hay for sub in subs)


@dataclass
class Task:
    id: str
    layer: str
    tools: list[str]
    prompt: str
    expect: Callable[[str], bool]
    tool_choice: str = ""
    want_tool: str = ""
    avoid_tool: str = ""
    min_chain: int = 0
    require_call_round1: bool = False


BATTERY: list[Task] = [
    Task("T1_single", "L1/L2", ["get_build_info"],
         "What is the build id of the component named 'operator-ui'? "
         "Use the available tool to find out. Answer with the build id.",
         lambda t: _has(t, "2026.07.29-b41")),
    Task("T2_consume", "L3", ["lookup_port"],
         "Which TCP port does the service 'index' listen on? "
         "Use the tool, then state the port number in your final answer.",
         lambda t: _has(t, "4102")),
    Task("T3_two_tools", "L4", ["get_build_info", "lookup_port"],
         "I need two facts: the build id of the component 'core', and the TCP "
         "port of the service 'model'. Use the tools to get both, then give me "
         "both values in one final answer.",
         lambda t: _has(t, "2026.07.29-b41", "4103")),
    Task("T4_error_recovery", "recover", ["read_config", "lookup_port"],
         "Find the TCP port that the service 'vector' uses. You may try "
         "reading /etc/deployment/ports.conf with read_config. If a tool returns an "
         "error, do NOT repeat the same call - use a different tool. Give me "
         "the port number.",
         lambda t: _has(t, "4101")),
    Task("T5_no_tools_control", "control", [],
         "In one short sentence: what is the capital of France?",
         lambda t: _has(t, "paris")),
    # Three DEPENDENT calls: nothing in the prompt names the port or the
    # component, so the chain cannot be short-circuited by prior knowledge.
    Task("T6_chain3", "L4-deep", ["lookup_port", "whoami_port", "get_build_info"],
         "Work this out with the tools only, step by step. First find the TCP "
         "port that the service 'operator' listens on. Then find which component "
         "owns that port. Then report that component's build id. Your final "
         "answer must contain the build id.",
         lambda t: _has(t, "2026.07.29-t19"), min_chain=3),
    Task("T7_repeat_fail", "recover-hard", ["read_config"],
         "Read the file /etc/deployment/ports.conf and tell me the TCP port of the "
         "service 'vector'. read_config is the only tool you have. If you "
         "cannot get the answer, say exactly: CANNOT DETERMINE.",
         lambda t: _has(t, "cannot determine")),
    Task("T8_ambiguous_pair", "selection", ["lookup_port", "lookup_port_legacy"],
         "What is the current TCP port of the service 'model'? Use the correct "
         "tool and give me the number.",
         lambda t: _has(t, "4103") and "14103" not in (t or ""),
         want_tool="lookup_port", avoid_tool="lookup_port_legacy"),
    # The question needs no tool at all, but the wire says one is required, so
    # this separates "chose not to call" from "cannot obey the constraint".
    Task("T9_required_choice", "forced-emit", ["lookup_port"],
         "What is the capital of France? You must call a tool first.",
         lambda t: True, tool_choice="required", require_call_round1=True),
    Task("T10_format", "instruction_following", ["lookup_port"],
         "Find the TCP port of the service 'operator'. Then reply with EXACTLY "
         "one line, in this format and nothing else, no prose, no punctuation, "
         "no explanation:\nPORT=<number>",
         lambda t: re.fullmatch(r"\s*PORT=4104\s*", t or "") is not None),
    Task("T11_long_context", "context_window", ["lookup_port"],
         "Below is an inventory dump. Exactly one record contains a deployment "
         "passphrase. Read it, then use the tool to get the TCP port of the "
         "service 'model'. Your final answer must contain BOTH the passphrase "
         "verbatim AND the port number.\n\n%s\n\nNow answer." % _long_haystack(),
         lambda t: _has(t, _NEEDLE, "4103")),
]

# --------------------------------------------------------- L1b prose detection
# A tool call that arrived in the prose channel instead of the structured one.
# Both are "the model tried to call a tool"; only one of them the wire can act
# on, and scoring them the same is how a template mismatch gets read as a model
# that ignores tools.

L1B_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("json_tool_call_obj", re.compile(
        r'\{[^{}]{0,200}?"(?:name|tool_name|function)"\s*:\s*"(?:%s)"'
        % "|".join(TOOLS), re.S)),
    ("xml_tool_call_tag", re.compile(
        r'<\s*(?:tool_call|function_call|tool_use|invoke)\b', re.I)),
    ("fenced_tool_code", re.compile(
        r'```(?:tool_call|json|tool_code|python)?\s*\{[^`]{0,400}?'
        r'"(?:name|tool_name)"\s*:', re.S)),
    ("bracket_call_marker", re.compile(
        r'\[(?:TOOL_CALL|tool_call|FUNCTION_CALL|CALL)\s*[:\]]')),
    ("bare_fn_invocation", re.compile(
        r'(?<![\w.])(?:%s)\s*\(\s*(?:[\'"]?\w+[\'"]?\s*=|\{|[\'"])' % "|".join(TOOLS))),
    ("special_token_marker", re.compile(
        r'<\|(?:tool_call|tool|function)[^|]*\|>', re.I)),
]


def detect_l1b(text: str) -> list[str]:
    """Which prose-shaped-tool-call patterns fired on this blob."""
    return [name for name, pattern in L1B_PATTERNS if pattern.search(text or "")]


# ------------------------------------------------------------------- the probe

MAX_ROUNDS = 8


@dataclass
class Row:
    """One task against one backend. Every field a verdict is computed from."""

    backend: str = ""
    model: str = ""
    wire: str = ""
    task: str = ""
    layer: str = ""
    tools_offered: list[str] = field(default_factory=list)
    tool_choice: str = "auto"
    rounds: int = 0
    structured_calls: int = 0
    distinct_tools: list[str] = field(default_factory=list)
    executed: int = 0
    tool_errors: int = 0
    repeat_max: int = 0
    repeat_failed_max: int = 0
    chain_depth: int = 0
    l1b_hits: list[str] = field(default_factory=list)
    called_in_round1: bool = False
    consumed: bool = False
    answer_ok: bool = False
    selection_ok: bool = True
    chain_ok: bool = True
    forced_ok: bool = True
    reply_chars: int = 0
    reasoning_chars: int = 0
    final_text: str = ""
    seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""

    @property
    def passed(self) -> bool:
        return (not self.error and self.answer_ok and self.selection_ok
                and self.chain_ok and self.forced_ok)


def _tool_choice_body(wire: str, choice: str) -> dict:
    """The wire's way of saying "you must call something".

    Two wires, two spellings, and neither is guessable from the other. An
    unknown wire gets nothing rather than a wrong key: a rejected request would
    score the task as a model failure when it is ours.
    """
    if not choice:
        return {}
    if wire == config.WIRE_ANTHROPIC:
        return {"tool_choice": {"type": "any"}}
    if wire == config.WIRE_OPENAI:
        return {"tool_choice": choice}
    return {}


def run_task(backend: config.Backend, task: Task, timeout: float = 300.0) -> Row:
    """One task, to a verdict. Never raises: a dead endpoint is a row too."""
    row = Row(backend=backend.name, model=backend.model, wire=backend.wire,
              task=task.id, layer=task.layer, tools_offered=list(task.tools),
              tool_choice=task.tool_choice or "auto")
    schemas = [TOOLS[name][1] for name in task.tools]
    started = time.time()
    try:
        client = build(backend, timeout=timeout)
    except ProviderError as exc:
        row.error = str(exc)
        return row
    # Set on the instance rather than through the backend: `tool_choice` is a
    # property of THIS task, not of the endpoint, and a backend carrying it
    # would leak the constraint into every other task in the run.
    if task.tool_choice:
        client.extra_body = {**getattr(client, "extra_body", {}),
                             **_tool_choice_body(backend.wire, task.tool_choice)}

    history: list[dict] = [{"role": "user", "content": task.prompt}]
    call_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}
    seen_tools: list[str] = []
    text = ""

    for round_no in range(1, MAX_ROUNDS + 1):
        try:
            turn = client.chat(history, tools=schemas or None, max_tokens=2048)
        except ProviderError as exc:
            row.error = str(exc)
            break
        except Exception as exc:  # a wire bug is ours, and must not read as a fail
            row.error = f"{type(exc).__name__}: {exc}"
            break

        row.rounds = round_no
        row.reasoning_chars += len(turn.thinking or "")
        usage = turn.usage or {}
        row.tokens_in += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        row.tokens_out += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if turn.text:
            text = turn.text

        if not turn.tool_calls:
            # Both channels are scanned. A thinking model puts its whole attempt
            # in `thinking` and returns an empty `text`; scoring only `text`
            # would record that as silence.
            for hit in detect_l1b(f"{turn.text or ''}\n{turn.thinking or ''}"):
                if hit not in row.l1b_hits:
                    row.l1b_hits.append(hit)
            break

        if round_no == 1:
            row.called_in_round1 = True
        row.structured_calls += len(turn.tool_calls)
        row.chain_depth += 1
        history.append({
            "role": "assistant",
            "content": turn.text or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name,
                                         "arguments": json.dumps(c.arguments)}}
                           for c in turn.tool_calls],
        })
        for call in turn.tool_calls:
            if call.name not in seen_tools:
                seen_tools.append(call.name)
            fn = TOOLS.get(call.name)
            if fn is None:
                output = "Error: no tool named '%s'" % call.name
            else:
                try:
                    output = fn[0](**(call.arguments or {}))
                    row.executed += 1
                except Exception as exc:
                    output = "Error: %s: %s" % (type(exc).__name__, exc)
            key = f"{call.name}:{json.dumps(call.arguments or {}, sort_keys=True)}"
            call_counts[key] = call_counts.get(key, 0) + 1
            if output.startswith("Error:"):
                row.tool_errors += 1
                failed_counts[key] = failed_counts.get(key, 0) + 1
            history.append({"role": "tool", "tool_call_id": call.id,
                            "name": call.name, "content": output})
    else:
        # Fell off the end of the loop still calling tools. Not an error - a
        # model that never stops is a real result, and the row says so.
        row.error = row.error or "did not settle within %d rounds" % MAX_ROUNDS

    row.seconds = round(time.time() - started, 2)
    row.distinct_tools = seen_tools
    row.final_text = text
    row.reply_chars = len(text)
    row.repeat_max = max(call_counts.values(), default=0)
    row.repeat_failed_max = max(failed_counts.values(), default=0)
    # L3 is "used the result", so it needs a call AND a later answer.
    row.consumed = bool(row.structured_calls and text.strip())
    if not row.error:
        row.answer_ok = bool(task.expect(text))
    if task.want_tool:
        row.selection_ok = (task.want_tool in seen_tools
                            and task.avoid_tool not in seen_tools)
    if task.min_chain:
        row.chain_ok = row.chain_depth >= task.min_chain and row.answer_ok
    if task.require_call_round1:
        row.forced_ok = row.called_in_round1
    return row


# ------------------------------------------------------------------- scoring

def rung(rows: list[Row]) -> str:
    """The highest rung this backend actually reached.

    Read off the recorded rows, never off a task's pass/fail alone: a model can
    answer T1 from prior knowledge without calling anything, and that is L0.
    """
    tool_rows = [r for r in rows if r.tools_offered]
    if not tool_rows or all(r.error for r in tool_rows):
        return "dead"
    if not any(r.structured_calls for r in tool_rows):
        return "L1b" if any(r.l1b_hits for r in tool_rows) else "L0"
    if not any(r.executed for r in tool_rows):
        return "L1"
    if not any(r.consumed and r.answer_ok for r in tool_rows):
        return "L2"
    chain = [r for r in rows if r.task == "T6_chain3"]
    if chain and chain[0].chain_ok:
        return "L4"
    multi = [r for r in rows if r.task == "T3_two_tools"]
    if multi and multi[0].answer_ok:
        return "L4-flat"
    return "L3"


def summarise(rows: list[Row]) -> dict:
    by_backend: dict[str, list[Row]] = {}
    for row in rows:
        by_backend.setdefault(row.backend, []).append(row)
    return {name: {"rung": rung(rs),
                   "passed": sum(1 for r in rs if r.passed),
                   "total": len(rs),
                   "failed_tasks": [r.task for r in rs if not r.passed]}
            for name, rs in by_backend.items()}


def render(rows: list[Row]) -> str:
    lines = []
    for name, stat in summarise(rows).items():
        head = f"{name:<28} {stat['rung']:<8} {stat['passed']}/{stat['total']}"
        lines.append(head)
        if stat["failed_tasks"]:
            lines.append(f"{'':<28} missed: {', '.join(stat['failed_tasks'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- entry

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="one-toolprobe", description=__doc__.splitlines()[0])
    ap.add_argument("patterns", nargs="*",
                    help="Substrings of backend name/model. Empty = every backend.")
    ap.add_argument("-t", "--task", action="append", default=[],
                    help="Restrict to a task id (T1_single, ...). Repeatable.")
    ap.add_argument("--timeout", type=float, default=300.0, help="Per call, seconds.")
    ap.add_argument("--no-local", action="store_true",
                    help="Skip llama.cpp endpoints; drop-ins only.")
    ap.add_argument("--out", default="",
                    help="JSONL rows. Default .scratch/toolprobe/<stamp>.jsonl")
    ap.add_argument("--list", action="store_true", help="Show the selection and exit.")
    args = ap.parse_args(argv)

    # Imported here rather than at module scope: bench pulls in the whole cell
    # machinery, and `one-toolprobe --help` should not pay for it.
    from .bench import select_backends

    backends = select_backends(args.patterns, include_local=not args.no_local)
    tasks = [t for t in BATTERY if not args.task or t.id in args.task]
    if not backends:
        print("no backend matched")
        return 2
    if not tasks:
        print("no task matched")
        return 2
    if args.list:
        for backend in backends:
            print(f"{backend.name:<28} {backend.display}")
        print(f"{len(tasks)} task(s): {', '.join(t.id for t in tasks)}")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or Path(".scratch/toolprobe") / f"{stamp}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    with out.open("w", encoding="utf-8") as fh:
        for backend in backends:
            for task in tasks:
                row = run_task(backend, task, timeout=args.timeout)
                rows.append(row)
                fh.write(json.dumps(row.__dict__, default=str) + "\n")
                fh.flush()
                mark = "ok " if row.passed else ("ERR" if row.error else "no ")
                print(f"{mark} {backend.name:<24} {row.task:<22} "
                      f"{row.seconds:>6.1f}s {row.error[:60]}")
    print()
    print(render(rows))
    print(f"\nrows: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scoped inference engine - tool-less, schema-checked, budgeted model calls.

This is the core behind the relay's `POST /v1/infer`. A consumer names a task
kind, a capability profile and a budget. It never names a machine or tools. The
engine picks lanes from its own configuration, walks them through
`onecli.pool`, validates the output against the caller's JSON schema, reserves
worst-case cost before any metered send, and writes one ledger row per attempt.

Fail closed:
- no config file: every call is refused (`not_configured`). The relay's existing
  routes are unaffected;
- a caller (token owner) missing from `callers`, or a kind outside its
  allowlist: refused;
- a metered lane with no audited tariff: rejected when the config is loaded;
- capability fields the engine cannot enforce: rejected, never ignored.

Config: `~/.config/one-cli/infer.json`, override `ONE_INFER_CONFIG`.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from typing import Any, Callable

from onecli.infer import ledger as infer_ledger
from onecli.budget import BudgetError, BudgetExceeded, RequestBudget
from onecli.pool import (Lane, LaneFault, NoLane, Pool, PoolExhausted,
                         RequestFault, walk)

CONFIG_ENV = "ONE_INFER_CONFIG"
DEFAULT_CONFIG = "~/.config/one-cli/infer.json"
BUDGET_DIR_ENV = "ONE_INFER_BUDGET_DIR"
DEFAULT_BUDGET_DIR = "~/.local/state/one-cli/infer-budget"

# Hard ceilings a config may lower but never raise.
MAX_OUTPUT_TOKENS = 8192
MAX_INPUT_TOKENS = 128_000
MAX_DEADLINE_MS = 120_000
MAX_ATTEMPTS = 5
IDEMPOTENCY_TTL_S = 600.0
# Tokens counted as framing per message when bounding input cost.
FRAMING_TOKENS_PER_MESSAGE = 32

LOCALITIES = ("private", "external")
COSTS = ("free", "flat", "metered")
REQUEST_LOCALITIES = ("private_only", "any", "paid_ok")
CAPABILITY_FIELDS = {"locality", "min_context", "vision", "lanes"}
BUDGET_FIELDS = {"max_input_tokens", "max_output_tokens", "deadline_ms",
                 "max_usd", "max_attempts"}
REQUEST_FIELDS = {"kind", "caller", "capability", "input", "output_schema",
                  "budget", "idempotency_key", "temperature"}
TARIFF_FIELDS = {"input_usd_per_million", "output_usd_per_million",
                 "audit_evidence"}
# Caller-supplied output_schema bounds: it is untrusted input to the validator.
MAX_SCHEMA_BYTES = 16_384
MAX_SCHEMA_DEPTH = 24
MAX_PATTERN_CHARS = 256
# A reply is checked against the schema only up to this size.
MAX_CHECKED_REPLY_CHARS = 65_536
# No regex shape check can rule out catastrophic backtracking (`(a|aa)*c`,
# `a*a*a*b`), and `re` cannot be interrupted in a thread. So a schema that uses
# patterns is checked in a child process that is killed at this deadline.
MAX_PATTERN_CHECK_SECONDS = 2.0
_CHECK_CHILD = (
    "import json, sys\n"
    "from onecli.infer import _schema_errors\n"
    "schema, value = json.load(sys.stdin)\n"
    "json.dump(_schema_errors(schema, value), sys.stdout)\n")
KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")




class InferError(Exception):
    """A refusal with an HTTP status and a stable code."""

    def __init__(self, status: int, code: str, message: str, *,
                 retryable: bool = False, attempts=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.attempts = attempts or []

    def body(self, request_id: str) -> dict:
        return {"request_id": request_id, "ok": False, "output": None,
                "attempts": [a.as_dict() for a in self.attempts],
                "error": {"code": self.code, "message": str(self), "retryable": self.retryable}}


class DetailedFault(RequestFault):
    """A request fault that carries the attempt detail pool.py records."""

    detail: dict


class SchemaFault(DetailedFault):
    """The lane answered, the answer did not satisfy the caller's schema."""


class BudgetRefusal(RequestFault):
    """This attempt would exceed a USD bound. The lane is healthy."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LaneSpec:
    name: str
    slots: int
    locality: str
    cost: str
    context: int | None = None
    vision: bool = False
    json_mode: bool = False
    kinds: tuple = ("*",)
    tariff: dict | None = None

    def serves_kind(self, kind: str) -> bool:
        return "*" in self.kinds or kind in self.kinds


@dataclass(frozen=True)
class CallerSpec:
    kinds: tuple
    max_inflight: int = 4
    usd_per_day: str | None = None


@dataclass
class EngineConfig:
    lanes: list
    callers: dict
    cooldown_s: float = 90.0
    shared_health: bool = False
    max_output_tokens: int = 2048
    max_input_tokens: int = 32_000
    max_deadline_ms: int = 60_000


def _bool(v, where):
    if not isinstance(v, bool):
        raise ValueError(f"{where} must be true or false")
    return v


def _posint(v, where, ceiling):
    if type(v) is not int or not 0 < v <= ceiling:
        raise ValueError(f"{where} must be an integer in 1..{ceiling}")
    return v


def _tariff(raw, where) -> dict:
    if not isinstance(raw, dict) or set(raw) != TARIFF_FIELDS:
        raise ValueError(f"{where}: metered lane needs an audited tariff "
                         f"with exactly {sorted(TARIFF_FIELDS)}")
    if not isinstance(raw["audit_evidence"], str) or not raw["audit_evidence"].strip():
        raise ValueError(f"{where}: tariff audit_evidence required")
    out: dict[str, Any] = {"audit_evidence": raw["audit_evidence"]}
    for k in ("input_usd_per_million", "output_usd_per_million"):
        if type(raw[k]) not in (int, float, str):
            raise ValueError(f"{where}: invalid {k}")
        rate = Decimal(str(raw[k]))
        if not rate.is_finite() or rate < 0 or rate > 10**6:
            raise ValueError(f"{where}: invalid {k}")
        out[k] = rate
    return out


def parse_config(data: dict) -> EngineConfig:
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("infer config must be an object with version 1")
    unknown = set(data) - {"version", "lanes", "callers", "limits",
                           "cooldown_s", "shared_health"}
    if unknown:
        raise ValueError(f"unknown infer config keys: {sorted(unknown)}")
    lanes = []
    raw_lanes = data.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise ValueError("infer config needs a non-empty ordered `lanes` list")
    seen = set()
    for i, raw in enumerate(raw_lanes):
        where = f"lanes[{i}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be an object")
        extra = set(raw) - {"name", "slots", "locality", "cost", "context",
                            "vision", "json", "kinds", "tariff"}
        if extra:
            raise ValueError(f"{where}: unknown keys {sorted(extra)}")
        name = raw.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"{where}: unique non-empty name required")
        seen.add(name)
        if raw.get("locality") not in LOCALITIES:
            raise ValueError(f"{where}: locality must be one of {LOCALITIES}")
        if raw.get("cost") not in COSTS:
            raise ValueError(f"{where}: cost must be one of {COSTS}")
        tariff = _tariff(raw.get("tariff"), where) if raw["cost"] == "metered" else None
        if raw["cost"] != "metered" and "tariff" in raw:
            raise ValueError(f"{where}: tariff only applies to metered lanes")
        kinds = raw.get("kinds", ["*"])
        if not isinstance(kinds, list) or not kinds or not all(isinstance(k, str) for k in kinds):
            raise ValueError(f"{where}: kinds must be a non-empty list of strings")
        context = raw.get("context")
        lanes.append(LaneSpec(
            name=name, slots=_posint(raw.get("slots"), f"{where}.slots", 64),
            locality=raw["locality"], cost=raw["cost"],
            context=None if context is None else _posint(context, f"{where}.context", 10**7),
            vision=_bool(raw.get("vision", False), f"{where}.vision"),
            json_mode=_bool(raw.get("json", False), f"{where}.json"),
            kinds=tuple(kinds), tariff=tariff))
    callers = {}
    raw_callers = data.get("callers")
    if not isinstance(raw_callers, dict):
        raise ValueError("infer config needs a `callers` object keyed by token owner")
    for owner, raw in raw_callers.items():
        where = f"callers[{owner}]"
        if not isinstance(raw, dict) or set(raw) - {"kinds", "max_inflight", "usd_per_day"}:
            raise ValueError(f"{where}: allowed keys are kinds, max_inflight, usd_per_day")
        kinds = raw.get("kinds")
        if not isinstance(kinds, list) or not kinds or not all(
                isinstance(k, str) and (k == "*" or KIND_RE.match(k)) for k in kinds):
            raise ValueError(f"{where}: kinds must be a non-empty list of kind ids")
        usd = raw.get("usd_per_day")
        if usd is not None and type(usd) not in (int, float, str):
            raise ValueError(f"{where}: usd_per_day must be a number")
        callers[owner] = CallerSpec(
            kinds=tuple(kinds),
            max_inflight=_posint(raw.get("max_inflight", 4), f"{where}.max_inflight", 256),
            usd_per_day=None if usd is None else str(usd))
    limits = data.get("limits", {})
    if not isinstance(limits, dict) or set(limits) - {
            "max_output_tokens", "max_input_tokens", "max_deadline_ms"}:
        raise ValueError("limits: allowed keys are max_output_tokens, "
                         "max_input_tokens, max_deadline_ms")
    cooldown = data.get("cooldown_s", 90)
    if type(cooldown) not in (int, float) or not 0 <= cooldown <= 86400:
        raise ValueError("cooldown_s must be 0..86400")
    return EngineConfig(
        lanes=lanes, callers=callers, cooldown_s=float(cooldown),
        shared_health=_bool(data.get("shared_health", False), "shared_health"),
        max_output_tokens=_posint(limits.get("max_output_tokens", 2048),
                                  "limits.max_output_tokens", MAX_OUTPUT_TOKENS),
        max_input_tokens=_posint(limits.get("max_input_tokens", 32_000),
                                 "limits.max_input_tokens", MAX_INPUT_TOKENS),
        max_deadline_ms=_posint(limits.get("max_deadline_ms", 60_000),
                                "limits.max_deadline_ms", MAX_DEADLINE_MS))


def config_path() -> Path:
    return Path(os.path.expanduser(os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG))


def load_config(path: Path | None = None) -> EngineConfig | None:
    """None when no config exists (engine disabled). A present but invalid
    file raises: a typo must not silently open or close lanes."""
    path = path or config_path()
    if not path.exists():
        return None
    return parse_config(json.loads(path.read_text()))


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------

@dataclass
class InferRequest:
    kind: str
    caller: str
    locality: str
    min_context: int | None
    vision: bool
    lanes: tuple | None
    system: str
    messages: list
    schema: dict | None
    max_input_tokens: int
    max_output_tokens: int
    deadline_ms: int
    max_usd: Decimal
    max_attempts: int
    temperature: float
    idempotency_key: str | None
    input_bound_tokens: int = 0


def _check_pattern(pattern: str) -> None:
    if len(pattern) > MAX_PATTERN_CHARS:
        raise _bad(f"output_schema pattern longer than {MAX_PATTERN_CHARS} chars",
                   "bad_schema")
    try:
        re.compile(pattern)
    except (re.error, RecursionError, OverflowError) as exc:
        raise _bad(f"output_schema has an invalid pattern: {exc}", "bad_schema") from exc


def _check_local_ref(root, ref: str) -> None:
    """Only `#` or `#/json/pointer` refs that resolve inside the schema."""
    if ref != "#":
        node = root
        for part in (ref[2:].split("/") if ref.startswith("#/") else [None]):
            if part is None:
                node = None
                break
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                node = None
                break
        if node is None:
            raise _bad(f"output_schema $ref {ref!r} does not resolve inside the schema",
                       "bad_schema")


def _walk_schema(node, depth: int = 0, root=None) -> None:
    root = node if root is None else root
    if depth > MAX_SCHEMA_DEPTH:
        raise _bad(f"output_schema nests deeper than {MAX_SCHEMA_DEPTH}", "bad_schema")
    if isinstance(node, dict):
        for key, val in node.items():
            if key in ("$ref", "$dynamicRef", "$recursiveRef") and isinstance(val, str):
                if not val.startswith("#"):
                    raise _bad("output_schema may only use local '#' references",
                               "bad_schema")
                _check_local_ref(root, val)
            elif key == "pattern" and isinstance(val, str):
                _check_pattern(val)
            elif key == "patternProperties" and isinstance(val, dict):
                for pat in val:
                    _check_pattern(pat)
            _walk_schema(val, depth + 1, root)
    elif isinstance(node, list):
        for item in node:
            _walk_schema(item, depth + 1, root)


def _check_output_schema(schema: dict) -> None:
    """Refuse an unsafe or invalid caller schema as the caller's error (400)."""
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    if len(json.dumps(schema)) > MAX_SCHEMA_BYTES:
        raise _bad(f"output_schema larger than {MAX_SCHEMA_BYTES} bytes", "bad_schema")
    _walk_schema(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise _bad(f"output_schema is not a valid JSON Schema: {exc.message}",
                   "bad_schema") from exc


def _bad(message: str, code: str = "bad_request") -> InferError:
    return InferError(400, code, message)


def caller_allowed(owner: str, caller: str) -> bool:
    """A body `caller` is the token owner itself or a sub-name under it."""
    return caller == owner or caller.startswith(owner + "/")


def parse_request(body, cfg: EngineConfig, owner: str) -> InferRequest:
    if not isinstance(body, dict):
        raise _bad("body must be a JSON object")
    if "tools" in body or "tool_choice" in body:
        raise _bad("/v1/infer is tool-less; tools are not accepted", "tools_refused")
    unknown = set(body) - REQUEST_FIELDS
    if unknown:
        raise _bad(f"unknown request fields: {sorted(unknown)}")
    kind = body.get("kind")
    if not isinstance(kind, str) or not KIND_RE.match(kind):
        raise _bad("kind must be a lowercase id like classify.mask")
    caller = body.get("caller")
    if not isinstance(caller, str) or not caller or len(caller) > 200:
        raise _bad("caller is required")
    if not caller_allowed(owner, caller):
        raise InferError(403, "caller_mismatch",
                         "caller must be the token owner or a name under it")

    cap = body.get("capability") or {}
    if not isinstance(cap, dict):
        raise _bad("capability must be an object")
    unsupported = set(cap) - CAPABILITY_FIELDS
    if unsupported:
        # Refuse rather than ignore: a caller asking for a quality floor this
        # engine cannot measure must not believe it got one.
        raise _bad(f"capability fields not enforced by this engine: {sorted(unsupported)}",
                   "unsupported_capability")
    locality = cap.get("locality", "private_only")
    if locality not in REQUEST_LOCALITIES:
        raise _bad(f"capability.locality must be one of {REQUEST_LOCALITIES}")
    min_context = cap.get("min_context")
    if min_context is not None:
        min_context = _checked_int(min_context, "capability.min_context", 10**7)
    vision = cap.get("vision", False)
    if not isinstance(vision, bool):
        raise _bad("capability.vision must be a boolean")
    pinned = cap.get("lanes")
    if pinned is not None:
        if not isinstance(pinned, list) or not pinned or not all(isinstance(n, str) for n in pinned):
            raise _bad("capability.lanes must be a non-empty list of lane names")
        pinned = tuple(pinned)

    inp = body.get("input")
    if not isinstance(inp, dict) or set(inp) - {"system", "messages"}:
        raise _bad("input must be {system?, messages}")
    system = inp.get("system", "")
    if not isinstance(system, str):
        raise _bad("input.system must be a string")
    messages = inp.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _bad("input.messages must be a non-empty list")
    clean = []
    for m in messages:
        if (not isinstance(m, dict) or set(m) != {"role", "content"}
                or m["role"] not in ("user", "assistant")
                or not isinstance(m["content"], str)):
            raise _bad("each message is exactly {role: user|assistant, content: string}")
        clean.append({"role": m["role"], "content": m["content"]})
    if clean[-1]["role"] != "user":
        raise _bad("the last message must be from the user")

    schema = body.get("output_schema")
    if schema is not None:
        if not isinstance(schema, dict):
            raise _bad("output_schema must be a JSON Schema object")
        _check_output_schema(schema)

    budget = body.get("budget")
    if not isinstance(budget, dict):
        raise _bad("budget is required: max_input_tokens, max_output_tokens, deadline_ms")
    if set(budget) - BUDGET_FIELDS:
        raise _bad(f"unknown budget fields: {sorted(set(budget) - BUDGET_FIELDS)}")
    for req in ("max_input_tokens", "max_output_tokens", "deadline_ms"):
        if req not in budget:
            raise _bad(f"budget.{req} is required")
    max_in = _checked_int(budget["max_input_tokens"], "budget.max_input_tokens", cfg.max_input_tokens)
    max_out = _checked_int(budget["max_output_tokens"], "budget.max_output_tokens", cfg.max_output_tokens)
    deadline_ms = _checked_int(budget["deadline_ms"], "budget.deadline_ms", cfg.max_deadline_ms)
    max_attempts = _checked_int(budget.get("max_attempts", 3), "budget.max_attempts", MAX_ATTEMPTS)
    max_usd = budget.get("max_usd", 0)
    if type(max_usd) not in (int, float, str):
        raise _bad("budget.max_usd must be a number")
    try:
        max_usd = Decimal(str(max_usd))
    except ArithmeticError as exc:
        raise _bad("budget.max_usd must be a number") from exc
    if not max_usd.is_finite() or max_usd < 0:
        raise _bad("budget.max_usd must be finite and >= 0")

    temperature = body.get("temperature", 0.0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) \
            or not 0 <= temperature <= 2:
        raise _bad("temperature must be 0..2")
    key = body.get("idempotency_key")
    if key is not None and (not isinstance(key, str) or not 0 < len(key) <= 128):
        raise _bad("idempotency_key must be a string of 1..128 characters")

    # Conservative input bound. A byte-level tokenizer emits at most one token
    # per UTF-8 byte, so bytes plus per-message framing bounds the prompt from
    # above. Refusal uses a looser 2-bytes-per-token estimate, so a caller
    # declaring 4000 tokens can send ~8KB; cost reservation always uses the
    # larger of the declared cap and the byte bound.
    total_bytes = len(system.encode()) + sum(len(m["content"].encode()) for m in clean)
    if total_bytes / 2 > max_in:
        raise InferError(413, "input_too_large",
                         f"input is ~{total_bytes // 2} tokens by estimate, "
                         f"over budget.max_input_tokens={max_in}")
    bound = max(max_in, total_bytes + FRAMING_TOKENS_PER_MESSAGE * (len(clean) + 1))

    return InferRequest(
        kind=kind, caller=caller, locality=locality, min_context=min_context,
        vision=vision, lanes=pinned, system=system, messages=clean, schema=schema,
        max_input_tokens=max_in, max_output_tokens=max_out, deadline_ms=deadline_ms,
        max_usd=max_usd, max_attempts=max_attempts, temperature=float(temperature),
        idempotency_key=key, input_bound_tokens=bound)


def _checked_int(v, where, ceiling):
    if type(v) is not int or not 0 < v <= ceiling:
        raise _bad(f"{where} must be an integer in 1..{ceiling}")
    return v


# --------------------------------------------------------------------------
# Capability filter and cost
# --------------------------------------------------------------------------

def eligible_lanes(cfg: EngineConfig, req: InferRequest) -> list:
    """Lanes that satisfy the request, in configured order. Unknown = no."""
    out = []
    for lane in cfg.lanes:
        if req.lanes is not None and lane.name not in req.lanes:
            continue
        if not lane.serves_kind(req.kind):
            continue
        if req.locality == "private_only" and (lane.locality != "private" or lane.cost == "metered"):
            continue
        if req.locality == "any" and lane.cost == "metered":
            continue
        if req.vision and not lane.vision:
            continue
        if req.min_context is not None and (lane.context is None or lane.context < req.min_context):
            continue
        if lane.cost == "metered" and req.max_usd <= 0:
            continue
        out.append(lane)
    return out


def reservation_usd(lane: LaneSpec, req: InferRequest) -> Decimal:
    """Worst-case USD for one attempt on `lane`. Zero for unmetered lanes.

    One attempt may include one schema repair call, so a schema request
    reserves two calls' worth.
    """
    if lane.cost != "metered":
        return Decimal(0)
    t = lane.tariff
    if t is None:
        raise ValueError(f"metered lane {lane.name} has no tariff")
    calls = 2 if req.schema is not None else 1
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_CEILING
        per_call = (t["input_usd_per_million"] * req.input_bound_tokens
                    + t["output_usd_per_million"] * req.max_output_tokens) / Decimal(1_000_000)
        # The repair call carries the first answer and the validator error.
        repair_extra = (t["input_usd_per_million"] * req.max_output_tokens * 2) / Decimal(1_000_000)
        return per_call * calls + (repair_extra if calls == 2 else 0)


def actual_usd(lane: LaneSpec, tokens_in: int, tokens_out: int) -> Decimal:
    if lane.cost != "metered":
        return Decimal(0)
    t = lane.tariff
    if t is None:
        raise ValueError(f"metered lane {lane.name} has no tariff")
    return (t["input_usd_per_million"] * tokens_in
            + t["output_usd_per_million"] * tokens_out) / Decimal(1_000_000)


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def parse_json_text(text: str):
    body = text.strip()
    m = _FENCE.match(body)
    if m:
        body = m.group(1)
    return json.loads(body)


def _uses_regex(node) -> bool:
    if isinstance(node, dict):
        return ("pattern" in node or "patternProperties" in node
                or any(_uses_regex(v) for v in node.values()))
    if isinstance(node, list):
        return any(_uses_regex(v) for v in node)
    return False


def _schema_errors(schema: dict, value) -> list[str]:
    from jsonschema import Draft202012Validator
    v = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in sorted(v.iter_errors(value), key=lambda e: list(e.absolute_path))][:5]


def schema_errors(schema: dict, value) -> list[str]:
    if not _uses_regex(schema):
        return _schema_errors(schema, value)
    try:
        done = subprocess.run(
            [sys.executable, "-c", _CHECK_CHILD], input=json.dumps([schema, value]),
            capture_output=True, text=True, timeout=MAX_PATTERN_CHECK_SECONDS,
            env={**os.environ, "PYTHONPATH": _PACKAGE_ROOT})
    except subprocess.TimeoutExpired:
        return [f"schema check exceeded {MAX_PATTERN_CHECK_SECONDS:g}s "
                "(a pattern is too costly for this reply)"]
    try:
        return list(json.loads(done.stdout))
    except ValueError:
        return [f"schema check failed: {done.stderr.strip()[-200:] or 'no output'}"]


_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

def default_wire_factory(name: str, timeout: float):
    """In-process provider wire for a drop-in name - no `one` subprocess."""
    from onecli import config, providers
    backend = config.resolve(name=name)
    return providers.build(backend, timeout), backend.model


@dataclass
class _Result:
    output: object
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool
    schema_result: str
    detail: dict = field(default_factory=dict)


class Engine:
    def __init__(self, cfg: EngineConfig, *,
                 wire_factory: Callable = default_wire_factory,
                 ledger_path: Path | None = None,
                 budget_dir: Path | None = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.wire_factory = wire_factory
        self.ledger_path = ledger_path
        self.budget_dir = budget_dir or Path(os.path.expanduser(
            os.environ.get(BUDGET_DIR_ENV) or DEFAULT_BUDGET_DIR))
        self.clock = clock
        self.specs = {lane.name: lane for lane in cfg.lanes}
        self.pool = Pool([(lane.name, lane.slots) for lane in cfg.lanes],
                         cooldown_s=cfg.cooldown_s,
                         shared_health=cfg.shared_health, clock=clock)
        self._lock = threading.Lock()
        self._inflight: dict[str, int] = {}
        self._idem: dict[tuple, tuple] = {}

    # -- public ----------------------------------------------------------
    def health(self) -> dict:
        return {"lanes": self.pool.snapshot(),
                "callers": sorted(self.cfg.callers),
                "shared_health": self.cfg.shared_health}

    def infer(self, owner: str, body) -> tuple[int, dict]:
        request_id = "inf_" + uuid.uuid4().hex[:16]
        try:
            caller_spec = self.cfg.callers.get(owner)
            if caller_spec is None:
                raise InferError(403, "not_configured",
                                 "this token owner is not configured for /v1/infer")
            req = parse_request(body, self.cfg, owner)
            if "*" not in caller_spec.kinds and req.kind not in caller_spec.kinds:
                raise InferError(403, "kind_not_allowed",
                                 f"{owner} may not run kind {req.kind}")
        except InferError as exc:
            self._refused(request_id, owner, body, exc)
            return exc.status, exc.body(request_id)

        if req.idempotency_key:
            cached = self._idem_get(owner, req.idempotency_key)
            if cached is not None:
                return cached

        if not self._enter(owner, caller_spec.max_inflight):
            busy = InferError(429, "caller_busy",
                              f"{owner} already has {caller_spec.max_inflight} calls in flight",
                              retryable=True)
            self._refused(request_id, owner, body, busy, req)
            return busy.status, busy.body(request_id)
        try:
            status, resp = self._run(request_id, owner, caller_spec, req)
        finally:
            self._leave(owner)
        if req.idempotency_key and status == 200:
            self._idem_put(owner, req.idempotency_key, (status, resp))
        return status, resp

    # -- admission helpers ---------------------------------------------------
    def _enter(self, owner, cap) -> bool:
        with self._lock:
            n = self._inflight.get(owner, 0)
            if n >= cap:
                return False
            self._inflight[owner] = n + 1
            return True

    def _leave(self, owner):
        with self._lock:
            self._inflight[owner] -= 1

    def _idem_get(self, owner, key):
        now = self.clock()
        with self._lock:
            for k in [k for k, (ts, _) in self._idem.items() if now - ts > IDEMPOTENCY_TTL_S]:
                self._idem.pop(k, None)
            hit = self._idem.get((owner, key))
            return hit[1] if hit else None

    def _idem_put(self, owner, key, value):
        with self._lock:
            self._idem[(owner, key)] = (self.clock(), value)

    def _daily_budget(self, owner: str, spec: CallerSpec) -> RequestBudget | None:
        if spec.usd_per_day is None:
            return None
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", owner)
        self.budget_dir.mkdir(parents=True, exist_ok=True)
        return RequestBudget(self.budget_dir / f"{safe}-{day}.sqlite",
                             spec.usd_per_day, create=True)

    # -- the call ------------------------------------------------------------
    def _run(self, request_id, owner, caller_spec, req) -> tuple[int, dict]:
        lanes = eligible_lanes(self.cfg, req)
        if not lanes:
            exc = InferError(503, "no_capable_lane",
                             "no configured lane satisfies this capability profile")
            self._refused(request_id, owner, None, exc, req)
            return exc.status, exc.body(request_id)
        allowed = {lane.name for lane in lanes}
        started = self.clock()
        deadline = started + req.deadline_ms / 1000
        state: dict[str, Any] = {"reserved": Decimal(0), "budget_refused": False,
                 "schema_failed": False, "n": 0, "lane_reserved": {}}
        try:
            daily = self._daily_budget(owner, caller_spec)
        except BudgetError as exc:
            err = InferError(503, "budget_unavailable", f"caller budget ledger: {exc}")
            self._refused(request_id, owner, None, err, req)
            return err.status, err.body(request_id)

        def call(lane: Lane, remaining: float):
            spec = self.specs[lane.name]
            cost = reservation_usd(spec, req)
            if cost > 0:
                if state["reserved"] + cost > req.max_usd:
                    state["budget_refused"] = True
                    raise BudgetRefusal(f"{lane.name}: worst case ${cost} exceeds "
                                        f"remaining request budget")
                if daily is not None:
                    try:
                        daily.reserve(f"{request_id}:{lane.name}", cost)
                    except BudgetExceeded as exc:
                        state["budget_refused"] = True
                        raise BudgetRefusal(f"{lane.name}: daily caller budget: {exc}") from exc
                state["reserved"] += cost
                state["lane_reserved"][lane.name] = cost
            return self._attempt(spec, req, remaining)

        def record(attempt):
            state["n"] += 1
            d = attempt.detail
            if d.get("schema_result") == "failed":
                state["schema_failed"] = True
            spec = self.specs[attempt.lane]
            usd = actual_usd(spec, d.get("tokens_in") or 0, d.get("tokens_out") or 0)
            infer_ledger.append(infer_ledger.make_row(
                request_id=request_id, attempt=state["n"], caller=req.caller,
                owner=owner, kind=req.kind, lane=attempt.lane, model=d.get("model"),
                outcome=attempt.outcome,
                error_class=None if attempt.outcome == "ok" else d.get("error_class", attempt.outcome),
                ms=attempt.ms, tokens_in=d.get("tokens_in"), tokens_out=d.get("tokens_out"),
                tokens_estimated=d.get("tokens_estimated"), usd=float(usd),
                reserved_usd=float(state["lane_reserved"].get(attempt.lane, 0)),
                schema_result=d.get("schema_result")), self.ledger_path)

        try:
            result, lane, attempts = walk(self.pool, call, deadline=deadline,
                                          max_attempts=req.max_attempts,
                                          allowed=allowed, on_attempt=record)
        except PoolExhausted as exc:
            if state["budget_refused"] and not any(
                    a.outcome == "lane_fault" for a in exc.attempts):
                err = InferError(429, "budget_exceeded", str(exc), attempts=exc.attempts)
            elif state["schema_failed"]:
                err = InferError(422, "schema_failed", str(exc), attempts=exc.attempts)
            elif self.clock() >= deadline:
                err = InferError(504, "deadline", f"no answer within {req.deadline_ms}ms",
                                 retryable=True, attempts=exc.attempts)
            elif isinstance(exc.last, NoLane) and not exc.attempts:
                err = InferError(503, "lanes_busy", str(exc), retryable=True)
            elif exc.attempts and all(a.outcome == "request_fault" for a in exc.attempts):
                err = InferError(422, "request_failed", str(exc), attempts=exc.attempts)
            else:
                err = InferError(502, "lanes_failed", str(exc), retryable=True,
                                 attempts=exc.attempts)
            return err.status, err.body(request_id)

        spec = self.specs[lane.name]
        usd = actual_usd(spec, result.tokens_in, result.tokens_out)
        warnings = []
        if not result.tokens_estimated and result.tokens_in > req.input_bound_tokens:
            warnings.append("provider-reported input tokens exceeded the reserved bound")
        return 200, {
            "request_id": request_id, "ok": True,
            "output": result.output,
            "validation": result.schema_result,
            "served_by": {"lane": lane.name, "model": result.model,
                          "locality": spec.locality, "cost": spec.cost},
            "attempts": [a.as_dict() for a in attempts],
            "usage": {"input_tokens": result.tokens_in,
                      "output_tokens": result.tokens_out,
                      "estimated": result.tokens_estimated,
                      "usd": float(usd), "reserved_usd": float(state["reserved"])},
            "warnings": warnings,
            "error": None,
        }

    def _attempt(self, spec: LaneSpec, req: InferRequest, remaining: float) -> _Result:
        try:
            wire, model = self.wire_factory(spec.name, max(1.0, remaining))
        except Exception as exc:  # resolve/build failure is the lane's
            raise LaneFault(f"{spec.name}: cannot build lane: {exc}") from exc
        messages = ([{"role": "system", "content": req.system}] if req.system else []) \
            + list(req.messages)
        turn = self._chat(wire, spec, req, messages)
        tin, tout, est = _usage(turn, messages)
        text = (turn.text or "").strip()
        if not text:
            empty = DetailedFault(f"{spec.name} returned empty content")
            empty.detail = _detail(model, tin, tout, est, None, "empty_reply")
            raise empty
        if req.schema is None:
            return _Result(text, text, model, tin, tout, est, "none",
                           _detail(model, tin, tout, est, "none"))
        errors = self._check(req.schema, text)
        if not errors:
            return _Result(parse_json_text(text), text, model, tin, tout, est, "passed",
                           _detail(model, tin, tout, est, "passed"))
        # One repair on the same lane, validator errors attached.
        repair = messages + [
            {"role": "assistant", "content": text[:4000]},
            {"role": "user", "content":
                "Your reply did not satisfy the required JSON schema:\n- "
                + "\n- ".join(errors)
                + "\nReply again with only the corrected JSON value."}]
        turn2 = self._chat(wire, spec, req, repair)
        t2in, t2out, est2 = _usage(turn2, repair)
        tin, tout, est = tin + t2in, tout + t2out, est or est2
        text2 = (turn2.text or "").strip()
        errors2 = self._check(req.schema, text2) if text2 else ["empty reply"]
        if errors2:
            failed = SchemaFault(f"{spec.name}: output failed schema after repair: {errors2[0]}")
            failed.detail = _detail(model, tin, tout, est, "failed", "schema_failed")
            raise failed
        return _Result(parse_json_text(text2), text2, model, tin, tout, est, "repaired",
                       _detail(model, tin, tout, est, "repaired"))

    @staticmethod
    def _chat(wire, spec, req, messages):
        kwargs = {"temperature": req.temperature, "max_tokens": req.max_output_tokens}
        if req.schema is not None and spec.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # Positional tools=None is explicit: this path never sends a tool.
        return wire.chat(messages, None, **kwargs)

    @staticmethod
    def _check(schema, text) -> list[str]:
        if len(text) > MAX_CHECKED_REPLY_CHARS:
            return [f"reply longer than {MAX_CHECKED_REPLY_CHARS} chars"]
        try:
            value = parse_json_text(text)
        except ValueError as exc:
            return [f"not JSON: {exc}"]
        return schema_errors(schema, value)

    def _refused(self, request_id, owner, body, exc: InferError, req=None):
        caller = req.caller if req else (body.get("caller") if isinstance(body, dict) else None)
        kind = req.kind if req else (body.get("kind") if isinstance(body, dict) else None)
        infer_ledger.append(infer_ledger.make_row(
            request_id=request_id, caller=str(caller)[:200] if caller else None,
            owner=owner, kind=str(kind)[:64] if kind else None,
            outcome="refused", error_class=exc.code), self.ledger_path)


def _detail(model, tin, tout, est, schema_result, error_class=None) -> dict:
    d = {"model": model, "tokens_in": tin, "tokens_out": tout,
         "tokens_estimated": est, "schema_result": schema_result}
    if error_class:
        d["error_class"] = error_class
    return d


def _usage(turn, messages) -> tuple[int, int, bool]:
    """Provider-reported usage when present, else a marked estimate."""
    u = turn.usage or {}
    tin = u.get("prompt_tokens", u.get("input_tokens"))
    tout = u.get("completion_tokens", u.get("output_tokens"))
    if type(tin) is int and type(tout) is int:
        return tin, tout, False
    chars_in = sum(len(m["content"]) for m in messages)
    return math.ceil(chars_in / 4), math.ceil(len(turn.text or "") / 4), True


_ENGINE: Engine | None = None
_ENGINE_LOCK = threading.Lock()


def engine() -> Engine | None:
    """Process-wide engine, built once from the config file. None = disabled."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            cfg = load_config()
            if cfg is None:
                return None
            _ENGINE = Engine(cfg)
        return _ENGINE

"""onecli.infer - scoped, tool-less, budgeted inference."""

import json
from decimal import Decimal

import pytest

from onecli import infer
from onecli.infer import ledger as infer_ledger
from onecli.providers.base import ProviderError, Turn

OWNER = "host:host-a"

SCHEMA = {"type": "object", "required": ["mask"],
          "properties": {"mask": {"enum": ["code", "research"]}},
          "additionalProperties": False}


def config(**over):
    data = {
        "version": 1,
        "lanes": [
            {"name": "local-a", "slots": 1, "locality": "private", "cost": "free",
             "context": 8192, "json": True},
            {"name": "local-b", "slots": 1, "locality": "private", "cost": "free",
             "context": 32768},
            {"name": "flat-x", "slots": 2, "locality": "external", "cost": "flat"},
            {"name": "paid-p", "slots": 2, "locality": "external", "cost": "metered",
             "tariff": {"input_usd_per_million": 1, "output_usd_per_million": 4,
                        "audit_evidence": "test tariff"}},
        ],
        "callers": {OWNER: {"kinds": ["classify.mask", "summarize"], "max_inflight": 2}},
    }
    data.update(over)
    return infer.parse_config(data)


class FakeWire:
    def __init__(self, script, log, name):
        self.script = script
        self.log = log
        self.name = name

    def chat(self, messages, tools=None, **kw):
        self.log.append({"lane": self.name, "messages": messages, "tools": tools, **kw})
        step = self.script[self.name].pop(0)
        if isinstance(step, Exception):
            raise step
        text, usage = step if isinstance(step, tuple) else (step, {})
        return Turn(text=text, usage=usage)


def make_engine(tmp_path, script, cfg=None):
    log = []
    eng = infer.Engine(cfg or config(),
                       wire_factory=lambda name, timeout: (FakeWire(script, log, name), f"m-{name}"),
                       ledger_path=tmp_path / "ledger.jsonl",
                       budget_dir=tmp_path / "budget")
    return eng, log


def body(**over):
    b = {"kind": "classify.mask", "caller": OWNER + "/mask_classifier",
         "input": {"system": "pick", "messages": [{"role": "user", "content": "task"}]},
         "output_schema": SCHEMA,
         "budget": {"max_input_tokens": 500, "max_output_tokens": 50, "deadline_ms": 5000}}
    b.update(over)
    return b


def rows(tmp_path):
    return list(infer_ledger.read_rows(tmp_path / "ledger.jsonl"))


# -- config ---------------------------------------------------------------

def test_metered_lane_without_tariff_is_rejected():
    with pytest.raises(ValueError, match="audited tariff"):
        infer.parse_config({"version": 1, "callers": {},
                            "lanes": [{"name": "p", "slots": 1, "locality": "external",
                                       "cost": "metered"}]})


def test_missing_config_disables_engine(tmp_path, monkeypatch):
    monkeypatch.setenv(infer.CONFIG_ENV, str(tmp_path / "absent.json"))
    assert infer.load_config() is None


# -- auth / identity ------------------------------------------------------

def test_unenrolled_owner_refused(tmp_path):
    eng, log = make_engine(tmp_path, {})
    status, resp = eng.infer("host:other", body(caller="host:other"))
    assert (status, resp["error"]["code"]) == (403, "not_configured")
    assert log == []
    assert rows(tmp_path)[0]["outcome"] == "refused"


def test_caller_spoofing_refused(tmp_path):
    eng, log = make_engine(tmp_path, {})
    status, resp = eng.infer(OWNER, body(caller="host:host-b/mask"))
    assert (status, resp["error"]["code"]) == (403, "caller_mismatch")
    # A prefix that is not a path boundary is not the owner either.
    status, _ = eng.infer(OWNER, body(caller="host:sparkle"))
    assert status == 403
    assert log == []


def test_kind_outside_allowlist_refused(tmp_path):
    eng, log = make_engine(tmp_path, {})
    status, resp = eng.infer(OWNER, body(kind="judge.rubric"))
    assert (status, resp["error"]["code"]) == (403, "kind_not_allowed")


# -- tool-less, validation --------------------------------------------------

def test_tools_are_refused(tmp_path):
    eng, log = make_engine(tmp_path, {})
    status, resp = eng.infer(OWNER, body(tools=[{"type": "function"}]))
    assert (status, resp["error"]["code"]) == (400, "tools_refused")
    assert log == []


def test_tool_role_messages_refused(tmp_path):
    eng, _ = make_engine(tmp_path, {})
    b = body(input={"messages": [{"role": "tool", "content": "x"}]})
    assert eng.infer(OWNER, b)[0] == 400


def test_unenforceable_capability_refused_not_ignored(tmp_path):
    eng, _ = make_engine(tmp_path, {})
    status, resp = eng.infer(OWNER, body(capability={"quality_floor": 0.8}))
    assert (status, resp["error"]["code"]) == (400, "unsupported_capability")


def test_invalid_schema_refused(tmp_path):
    eng, _ = make_engine(tmp_path, {})
    status, resp = eng.infer(OWNER, body(output_schema={"type": 12}))
    assert (status, resp["error"]["code"]) == (400, "bad_schema")


def test_budget_required_and_capped(tmp_path):
    eng, _ = make_engine(tmp_path, {})
    b = body()
    del b["budget"]["deadline_ms"]
    assert eng.infer(OWNER, b)[0] == 400
    b = body()
    b["budget"]["max_output_tokens"] = 100_000
    assert eng.infer(OWNER, b)[0] == 400


def test_oversized_input_refused_before_send(tmp_path):
    eng, log = make_engine(tmp_path, {})
    b = body(input={"messages": [{"role": "user", "content": "x" * 5000}]})
    status, resp = eng.infer(OWNER, b)
    assert (status, resp["error"]["code"]) == (413, "input_too_large")
    assert log == []


# -- happy path, tool-less on the wire -------------------------------------

def test_valid_output_passes_and_is_tool_less(tmp_path):
    usage = {"prompt_tokens": 20, "completion_tokens": 5}
    eng, log = make_engine(tmp_path, {"local-a": [('{"mask": "code"}', usage)]})
    status, resp = eng.infer(OWNER, body())
    assert status == 200, resp
    assert resp["output"] == {"mask": "code"}
    assert resp["validation"] == "passed"
    assert resp["served_by"]["lane"] == "local-a"
    assert resp["usage"]["input_tokens"] == 20 and not resp["usage"]["estimated"]
    assert log[0]["tools"] is None
    assert log[0]["response_format"] == {"type": "json_object"}
    assert log[0]["max_tokens"] == 50
    (row,) = rows(tmp_path)
    assert row["outcome"] == "ok" and row["caller"] == OWNER + "/mask_classifier"
    assert row["owner"] == OWNER and row["schema_result"] == "passed"
    assert "task" not in json.dumps(row)  # no prompt text in the ledger


def test_fenced_json_accepted(tmp_path):
    eng, _ = make_engine(tmp_path, {"local-a": ['```json\n{"mask": "research"}\n```']})
    status, resp = eng.infer(OWNER, body())
    assert status == 200 and resp["output"] == {"mask": "research"}


def test_text_mode_without_schema(tmp_path):
    eng, log = make_engine(tmp_path, {"local-a": ["a summary"]})
    status, resp = eng.infer(OWNER, body(kind="summarize", output_schema=None))
    assert status == 200 and resp["output"] == "a summary"
    assert "response_format" not in log[0]


# -- schema enforcement and retry classification --------------------------

def test_schema_repair_on_same_lane(tmp_path):
    eng, log = make_engine(tmp_path, {"local-a": ['{"mask": "poetry"}', '{"mask": "code"}']})
    status, resp = eng.infer(OWNER, body())
    assert status == 200 and resp["validation"] == "repaired"
    assert [c["lane"] for c in log] == ["local-a", "local-a"]
    assert "did not satisfy" in log[1]["messages"][-1]["content"]


def test_schema_failure_moves_lane_and_never_cools(tmp_path):
    eng, log = make_engine(tmp_path, {
        "local-a": ["not json", "still not json"],
        "local-b": ['{"mask": 1}', '{"wrong": true}'],
    })
    status, resp = eng.infer(OWNER, body())
    assert (status, resp["error"]["code"]) == (422, "schema_failed")
    assert [a["outcome"] for a in resp["attempts"]] == ["request_fault", "request_fault"]
    assert all(s["state"] == "ready" for s in eng.pool.snapshot())
    assert [r["schema_result"] for r in rows(tmp_path)] == ["failed", "failed"]


def test_lane_fault_cools_and_fails_over(tmp_path):
    eng, _ = make_engine(tmp_path, {
        "local-a": [ProviderError("HTTP 503 service unavailable")],
        "local-b": ['{"mask": "code"}'],
    })
    status, resp = eng.infer(OWNER, body())
    assert status == 200 and resp["served_by"]["lane"] == "local-b"
    assert [a["outcome"] for a in resp["attempts"]] == ["lane_fault", "ok"]
    states = {s["lane"]: s["state"] for s in eng.pool.snapshot()}
    assert states["local-a"] == "cooling" and states["local-b"] == "ready"


def test_empty_reply_is_request_fault(tmp_path):
    eng, _ = make_engine(tmp_path, {"local-a": [""], "local-b": [""]})
    status, resp = eng.infer(OWNER, body(kind="summarize", output_schema=None))
    assert (status, resp["error"]["code"]) == (422, "request_failed")
    assert all(s["state"] == "ready" for s in eng.pool.snapshot())


def test_all_lanes_down_is_502(tmp_path):
    eng, _ = make_engine(tmp_path, {
        "local-a": [ProviderError("HTTP 502")], "local-b": [ProviderError("timed out")]})
    status, resp = eng.infer(OWNER, body())
    assert (status, resp["error"]["code"]) == (502, "lanes_failed")
    assert resp["error"]["retryable"]


def test_max_attempts_bounds_walk(tmp_path):
    eng, log = make_engine(tmp_path, {
        "local-a": [ProviderError("HTTP 502")], "local-b": ['{"mask": "code"}']})
    b = body()
    b["budget"]["max_attempts"] = 1
    status, _ = eng.infer(OWNER, b)
    assert status == 502 and len(log) == 1


# -- capability filter / locality / paid ------------------------------------

def test_default_locality_is_private_only(tmp_path):
    eng, _ = make_engine(tmp_path, {})
    lanes = infer.eligible_lanes(eng.cfg, infer.parse_request(body(), eng.cfg, OWNER))
    assert [lane.name for lane in lanes] == ["local-a", "local-b"]


def test_min_context_and_vision_fail_closed(tmp_path):
    eng, log = make_engine(tmp_path, {"local-b": ['{"mask": "code"}']})
    status, resp = eng.infer(OWNER, body(capability={"min_context": 16000}))
    assert status == 200 and resp["served_by"]["lane"] == "local-b"
    # flat-x has no declared context: unknown never satisfies a requirement.
    status, resp = eng.infer(OWNER, body(capability={"min_context": 64000, "locality": "any"}))
    assert (status, resp["error"]["code"]) == (503, "no_capable_lane")
    status, resp = eng.infer(OWNER, body(capability={"vision": True}))
    assert (status, resp["error"]["code"]) == (503, "no_capable_lane")


def test_metered_lane_needs_paid_ok_and_max_usd(tmp_path):
    eng, log = make_engine(tmp_path, {"paid-p": ['{"mask": "code"}']})
    pin = {"lanes": ["paid-p"]}
    assert eng.infer(OWNER, body(capability={**pin, "locality": "any"}))[0] == 503
    assert eng.infer(OWNER, body(capability={**pin, "locality": "paid_ok"}))[0] == 503
    assert log == []
    # Money alone does not open a metered lane: the locality must say paid_ok.
    b = body(capability={**pin, "locality": "any"})
    b["budget"]["max_usd"] = "0.01"
    assert eng.infer(OWNER, b)[0] == 503
    assert log == []
    b = body(capability={**pin, "locality": "paid_ok"})
    b["budget"]["max_usd"] = "0.01"
    status, resp = eng.infer(OWNER, b)
    assert status == 200, resp
    assert resp["usage"]["reserved_usd"] > 0


def test_budget_refused_before_any_send(tmp_path):
    eng, log = make_engine(tmp_path, {"paid-p": ['{"mask": "code"}']})
    b = body(capability={"lanes": ["paid-p"], "locality": "paid_ok"})
    b["budget"]["max_usd"] = "0.0000001"
    status, resp = eng.infer(OWNER, b)
    assert (status, resp["error"]["code"]) == (429, "budget_exceeded")
    assert log == []
    assert all(s["state"] == "ready" for s in eng.pool.snapshot())


def test_daily_caller_budget(tmp_path):
    cfg = config(callers={OWNER: {"kinds": ["*"], "usd_per_day": "0.002"}})
    eng, log = make_engine(tmp_path, {"paid-p": ['{"mask": "code"}'] * 5}, cfg)
    b = body(capability={"lanes": ["paid-p"], "locality": "paid_ok"})
    b["budget"]["max_usd"] = "1"
    first = eng.infer(OWNER, b)
    assert first[0] == 200, first
    # Worst case per call is 0.0015 (2 calls x (500 in, 50 out) + repair);
    # the second reservation would exceed the day's 0.002.
    status, resp = eng.infer(OWNER, b)
    assert (status, resp["error"]["code"]) == (429, "budget_exceeded")
    assert len(log) == 1


def test_reservation_is_worst_case():
    cfg = config()
    req = infer.parse_request(body(), cfg, OWNER)
    paid = cfg.lanes[3]
    one_call = (Decimal(1) * req.input_bound_tokens + Decimal(4) * 50) / 1_000_000
    assert infer.reservation_usd(paid, req) > 2 * one_call
    assert infer.reservation_usd(cfg.lanes[0], req) == 0


# -- deadline, admission, idempotency ---------------------------------------

def test_deadline_is_enforced(tmp_path):
    t = [1000.0]

    class SlowWire:
        def chat(self, messages, tools=None, **kw):
            t[0] += 10
            raise ProviderError("timed out")

    eng = infer.Engine(config(), wire_factory=lambda n, to: (SlowWire(), "m"),
                       ledger_path=tmp_path / "l.jsonl", clock=lambda: t[0])
    eng.pool.clock = lambda: t[0]
    status, resp = eng.infer(OWNER, body())
    assert (status, resp["error"]["code"]) == (504, "deadline")
    assert len(resp["attempts"]) == 1


def test_wire_timeout_bounded_by_remaining_deadline(tmp_path):
    seen = []

    def factory(name, timeout):
        seen.append(timeout)
        return FakeWire({"local-a": ['{"mask": "code"}']}, [], name), "m"

    eng = infer.Engine(config(), wire_factory=factory, ledger_path=tmp_path / "l.jsonl")
    assert eng.infer(OWNER, body())[0] == 200
    assert 0 < seen[0] <= 5.0


def test_inflight_cap(tmp_path):
    eng, _ = make_engine(tmp_path, {})
    eng._inflight[OWNER] = 2
    status, resp = eng.infer(OWNER, body())
    assert (status, resp["error"]["code"]) == (429, "caller_busy")


def test_idempotency_returns_first_answer(tmp_path):
    eng, log = make_engine(tmp_path, {"local-a": ['{"mask": "code"}']})
    first = eng.infer(OWNER, body(idempotency_key="k1"))
    second = eng.infer(OWNER, body(idempotency_key="k1"))
    assert first == second and len(log) == 1

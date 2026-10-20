"""Caller-supplied output_schema is untrusted: bounded, local-only, ReDoS-safe."""

import json
import time

import pytest

from onecli import infer

OWNER = "host:host-a"


def cfg():
    return infer.parse_config({
        "version": 1,
        "lanes": [{"name": "local-a", "slots": 1, "locality": "private",
                   "cost": "free", "context": 8192, "json": True}],
        "callers": {OWNER: {"kinds": ["summarize"], "max_inflight": 2}},
    })


def request(schema):
    return {"kind": "summarize", "caller": OWNER,
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "output_schema": schema,
            "budget": {"max_input_tokens": 100, "max_output_tokens": 100,
                       "deadline_ms": 5000}}


def refused(schema):
    with pytest.raises(infer.InferError) as ei:
        infer.parse_request(request(schema), cfg(), OWNER)
    assert ei.value.status == 400
    assert ei.value.code == "bad_schema"
    return str(ei.value)


def test_oversize_schema_refused():
    refused({"type": "object", "description": "x" * (getattr(infer, "MAX_SCHEMA_BYTES", 16384) + 1)})


def test_deep_nesting_refused():
    schema = leaf = {"type": "object"}
    for _ in range(getattr(infer, "MAX_SCHEMA_DEPTH", 24) + 5):
        leaf["properties"] = {"a": {"type": "object"}}
        leaf = leaf["properties"]["a"]
    refused(schema)


def test_invalid_pattern_refused():
    refused({"type": "string", "pattern": "(unclosed"})
    refused({"type": "object", "patternProperties": {"[bad": {"type": "string"}}})


def test_overlong_pattern_refused():
    refused({"type": "string", "pattern": "a" * (getattr(infer, "MAX_PATTERN_CHARS", 256) + 1)})


@pytest.mark.parametrize("ref", ["http://example.com/s.json", "file:///etc/passwd",
                                 "other.json#/x"])
def test_remote_ref_refused(ref):
    refused({"type": "object", "properties": {"a": {"$ref": ref}}})


def test_unresolvable_local_ref_refused():
    refused({"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}})


@pytest.mark.parametrize("pattern, reply", [
    ("^(a+)+$", "a" * 40 + "!"),
    ("(a|aa)*c", "a" * 60),
    ("a*a*a*a*a*a*b", "a" * 3000),
    ("(?>(a+)+)b", "a" * 40),
])
def test_costly_pattern_check_is_cut_off(pattern, reply):
    # A shape check cannot catch every backtracking form; the deadline must.
    schema = {"type": "string", "pattern": pattern}
    t0 = time.monotonic()
    errs = infer.Engine._check(schema, json.dumps(reply))
    assert time.monotonic() - t0 < 6.0
    assert errs


def test_oversize_reply_not_validated():
    errs = infer.Engine._check({"type": "string"}, "x" * (getattr(infer, "MAX_CHECKED_REPLY_CHARS", 65536) + 1))
    assert errs and "longer" in errs[0]


def test_normal_schema_still_validates():
    schema = {"type": "object", "required": ["id", "tags"],
              "properties": {"id": {"type": "string", "pattern": "^[a-z]{1,8}-[0-9]+$"},
                             "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"}}},
              "$defs": {"tag": {"type": "string", "pattern": "^[a-z]+$"}}}
    req = infer.parse_request(request(schema), cfg(), OWNER)
    assert req.schema == schema
    assert infer.schema_errors(schema, {"id": "ab-12", "tags": ["x"]}) == []
    assert infer.schema_errors(schema, {"id": "AB", "tags": ["x"]})

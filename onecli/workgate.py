"""Versioned workflow policies shared by people, agents and CI."""

from __future__ import annotations

import argparse
import hashlib
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from onecli import secureio

BUILTINS: dict[str, Any] = {"experimental-code@1": {"name": "experimental-code", "version": 1, "events": {
    "phase_completed": [
        {"field": "meta_rounds", "op": "lte", "value": 1, "severity": "block",
         "message": "more than one round measures the measuring instrument"},
        {"field": "private_mocks", "op": "eq", "value": 0, "severity": "block",
         "message": "generated tests mock private functions of the module under test"},
        {"field": "product_delta", "op": "truthy", "severity": "warn",
         "message": "this phase produced no durable product change"}],
    "before_merge": [
        {"field": "regression_failed_before_fix", "op": "truthy", "severity": "block",
         "message": "regression test was not observed failing before the fix"},
        {"field": "ci_green", "op": "truthy", "severity": "block", "message": "CI is not green"},
        {"field": "scratch_only", "op": "falsy", "severity": "block",
         "message": "the only output is scratch work"}],
    "finish": [
        {"field": "disposition", "op": "in", "value": ["merged", "stopped"],
         "severity": "block", "message": "finish requires a merged result or explicit stop decision"},
        {"field": "evidence", "op": "truthy", "severity": "block",
         "message": "finish requires durable evidence"}]}}}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(policy: dict) -> str:
    return hashlib.sha256(_canonical(policy).encode()).hexdigest()


def load_policy(name_or_path: str) -> dict:
    if name_or_path in BUILTINS:
        policy = BUILTINS[name_or_path]
    else:
        try:
            policy = json.loads(Path(name_or_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load policy {name_or_path!r}: {exc}") from exc
    if not isinstance(policy, dict) or not isinstance(policy.get("events"), dict):
        raise ValueError("policy must be an object with an events object")
    if not policy.get("name") or not isinstance(policy.get("version"), int):
        raise ValueError("policy requires a name and integer version")
    events: dict = policy["events"]
    for rules in events.values():
        if not isinstance(rules, list):
            raise ValueError("policy event rules must be a list")
        for rule in rules:
            _check_rule(rule)
    return json.loads(_canonical(policy))


def start(path: Path, policy: dict, *, goal: str, expected: str, stop: str) -> dict:
    if not all(x.strip() for x in (goal, expected, stop)):
        raise ValueError("goal, expected output and stop condition are required")
    record = {"schema": 1, "policy": policy, "policy_sha256": _digest(policy),
              "goal": goal.strip(), "expected_output": expected.strip(),
              "stop_condition": stop.strip(),
              "started_at": datetime.now(timezone.utc).isoformat(), "checks": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        secureio.atomic_write_text(path, json.dumps(record, indent=2) + "\n", exclusive=True)
    except FileExistsError:
        raise ValueError(f"work record already exists: {path}") from None
    return record


_NEEDS_VALUE = {"eq", "lte", "in"}


def _check_rule(rule) -> None:
    if not isinstance(rule, dict):
        raise ValueError("policy rule must be an object")
    op = rule.get("op")
    if op in _NEEDS_VALUE and "value" not in rule:
        raise ValueError(f"policy rule for {rule.get('field')!r} uses {op!r} but has no value")
    if op == "lte" and (isinstance(rule["value"], bool)
                        or not isinstance(rule["value"], (int, float))):
        raise ValueError(f"policy rule for {rule.get('field')!r}: lte needs a number")
    if op == "in" and not isinstance(rule["value"], list):
        raise ValueError(f"policy rule for {rule.get('field')!r}: in needs a list")


def _passes(rule: dict, evidence: dict) -> bool:
    _check_rule(rule)
    value, op = evidence.get(rule.get("field")), rule.get("op")
    target: Any = rule.get("value")
    if op == "truthy":
        return bool(value)
    if op == "falsy":
        return not value
    if op == "eq":
        return value == target
    if op == "lte":
        return isinstance(value, (int, float)) and value <= target
    if op == "in":
        return value in target
    raise ValueError(f"unknown policy operation: {op!r}")


def evaluate(policy: dict, event: str, evidence: dict) -> dict:
    rules = policy["events"].get(event)
    if rules is None:
        raise ValueError(f"event {event!r} is not defined by this policy")
    findings = [{"severity": r["severity"], "field": r["field"], "message": r["message"]}
                for r in rules if not _passes(r, evidence)]
    verdict = "block" if any(x["severity"] == "block" for x in findings) else (
        "warn" if findings else "pass")
    return {"event": event, "verdict": verdict, "findings": findings}


@contextmanager
def _locked(path: Path):
    """Serialise read-modify-write of one record across processes."""
    fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def check(path: Path, event: str, evidence: dict) -> dict:
    with _locked(path):
        record = json.loads(path.read_text(encoding="utf-8"))
        policy = record["policy"]
        if record.get("policy_sha256") != _digest(policy):
            raise ValueError("stored policy snapshot was modified")
        result = evaluate(policy, event, evidence)
        result.update(checked_at=datetime.now(timezone.utc).isoformat(), evidence=evidence)
        record["checks"].append(result)
        secureio.atomic_write_text(path, json.dumps(record, indent=2) + "\n")
    return result


def _evidence(raw: str) -> dict:
    if raw == "-":
        raw = sys.stdin.read()
    elif raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("evidence must be a JSON object")
    return value


def _emit(value: dict, machine: bool) -> None:
    if machine:
        print(json.dumps(value, sort_keys=True))
        return
    print(value.get("verdict", "started"))
    for item in value.get("findings", []):
        print(f"{item['severity']}: {item['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="one workgate")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profiles", help="List built-in policy profiles.")
    begin = sub.add_parser("start", help="Create a policy-pinned work record.")
    begin.add_argument("--state", default=".one/workgate.json")
    begin.add_argument("--policy", default="experimental-code@1")
    begin.add_argument("--goal", required=True)
    begin.add_argument("--expected", required=True)
    begin.add_argument("--stop", required=True)
    for name in ("check", "finish"):
        command = sub.add_parser(name)
        command.add_argument("--state", default=".one/workgate.json")
        if name == "check":
            command.add_argument("--event", required=True)
        command.add_argument("--evidence", required=True, help="JSON object, @file, or - for stdin.")
    args = parser.parse_args(argv)
    try:
        if args.command == "profiles":
            value = {"profiles": sorted(BUILTINS)}
            print(json.dumps(value) if args.json else "\n".join(value["profiles"]))
            return 0
        path = Path(args.state)
        if args.command == "start":
            value = start(path, load_policy(args.policy), goal=args.goal,
                          expected=args.expected, stop=args.stop)
            _emit(
                {
                    "verdict": "started",
                    "state": str(path),
                    "policy_sha256": value["policy_sha256"],
                },
                args.json,
            )
            return 0
        event = args.event if args.command == "check" else "finish"
        value = check(path, event, _evidence(args.evidence))
        _emit(value, args.json)
        return 2 if value["verdict"] == "block" else 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"workgate: {exc}", file=sys.stderr)
        return 2


__all__ = ["BUILTINS", "load_policy", "start", "evaluate", "check", "main"]

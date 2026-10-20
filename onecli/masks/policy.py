"""Versioned, opt-in execution constraints. Mask data cannot supply code.

Completion verification is a trusted host callback, never a model assertion.
Existing runtime permissions and budgets remain authoritative.
"""
from dataclasses import dataclass


READ_TOOLS = frozenset({"read", "ls", "glob", "grep"})
DEBUG_TOOLS = READ_TOOLS | {"write", "edit", "bash", "code"}


@dataclass(frozen=True)
class MaskPolicy:
    kind: str
    version: int
    max_steps: int
    tools: frozenset[str]

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) - {"kind", "version", "max_steps", "tools"}:
            raise ValueError("mask policy must be an object with known fields")
        kind = raw.get("kind")
        if kind not in ("debug", "review"):
            raise ValueError("mask policy kind must be debug or review")
        version = raw.get("version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("unsupported mask policy version")
        steps = raw.get("max_steps", 12)
        if type(steps) is not int or not 2 <= steps <= 1000:
            raise ValueError("mask policy max_steps must be an integer in 2..1000")
        ceiling = READ_TOOLS if kind == "review" else DEBUG_TOOLS
        names = raw.get("tools", sorted(ceiling))
        if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
                or not set(names) <= ceiling):
            raise ValueError("mask policy tools exceed the task capability ceiling")
        return cls(kind, version, steps, frozenset(names))

    def as_dict(self):
        return {"kind": self.kind, "version": self.version,
                "max_steps": self.max_steps, "tools": sorted(self.tools)}

    @property
    def instruction(self):
        if self.kind == "review":
            return "Review using filesystem evidence. Tie findings to file locations. Completion requires evidence collection."
        return "Debug by reproducing, investigating, editing and verifying. Completion requires the host verifier to pass."


class PolicyRun:
    def __init__(self, policy, verifier=None):
        self.policy = policy
        self.verifier = verifier
        self.evidence = False

    def observe(self, name, output):
        if name in READ_TOOLS and output.strip() and not output.startswith(("error:", "blocked:")):
            self.evidence = True

    def refusal(self, workspace):
        if self.policy.kind == "review":
            return None if self.evidence else "Mask policy: collect filesystem evidence before completing the review."
        try:
            passed = self.verifier(workspace) is True
        except Exception:
            passed = False
        return None if passed else "Mask policy: host verification has not passed; investigate and verify before completing."

"""Tool-call guards: the seam where a call can be refused before it runs.

The agent loop's `on_event` is a notifier - it watches. A guard *decides*. That
distinction is the whole module: a guard is consulted between the model asking
for a tool and the tool running, and its verdict can stop the call.

Two kinds ship here:

`ReadOnlyGuard`   in-process, refuses any tool declaring `mutating=True`.
`SubprocessGuard` runs external programs and speaks the Claude Code PreToolUse
                  hook contract, so guards already written for that harness work
                  here unchanged:

    stdin    {"tool_name": "Bash", "tool_input": {"command": "..."}, ...}
    exit 0   allow
    exit 2   BLOCK, stderr is the reason shown to the model
    other    BLOCK: the guard could not establish permission

Timeouts and startup failures also block. An installed enforcement policy
cannot grant access merely because its process failed.

The tool-name translation matters more than it looks. Guards match on Claude's
vocabulary (`Bash`/`.tool_input.command`, `Write`/`.tool_input.file_path`), so a
one-cli `bash` call with a `command` argument is presented as `Bash`, and the
file tools' `path` is presented as `file_path`. Skip that rename and every
path-matching guard silently passes everything - the worst failure a guard has,
because it looks installed.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from onecli.shellrun import run_shell

# one-cli tool name -> the name guards were written against.
TOOL_NAME_MAP: dict[str, str] = {
    "bash": "Bash",
    # A background job is a shell command too; presented as anything else, a
    # guard or hook matching "Bash" never sees it. `code` keeps its own name:
    # a guard cannot read a program, so the program is confined instead
    # (codemode.confinement_reason) and its shell calls arrive here singly.
    "job_start": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "ls": "LS",
    "glob": "Glob",
    "grep": "Grep",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
}

# Per tool, one-cli argument -> the argument name guards read.
ARG_NAME_MAP: dict[str, dict[str, str]] = {
    "read": {"path": "file_path"},
    "write": {"path": "file_path"},
    "edit": {"path": "file_path"},
}

DEFAULT_TIMEOUT = 10.0


@dataclass
class Verdict:
    """A guard's answer. `reason` is shown to the model when denied."""

    allowed: bool = True
    reason: str = ""
    source: str = ""

    @classmethod
    def allow(cls) -> "Verdict":
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str, source: str = "") -> "Verdict":
        return cls(allowed=False, reason=reason, source=source)


class Guard(Protocol):
    def check(self, tool_name: str, arguments: dict, mutating: bool) -> Verdict:
        ...


def to_hook_payload(tool_name: str, arguments: dict, mutating: bool,
                    workspace: str = "") -> dict:
    """Translate a one-cli tool call into the Claude PreToolUse payload."""
    renames = ARG_NAME_MAP.get(tool_name, {})
    tool_input = {renames.get(k, k): v for k, v in arguments.items()}
    return {
        "tool_name": TOOL_NAME_MAP.get(tool_name, tool_name),
        "tool_input": tool_input,
        "cwd": workspace,
        # Ours, not Claude's. A guard that wants to key on "does this change
        # anything" can, without re-deriving it from the tool name.
        "onecli": {"tool": tool_name, "mutating": mutating},
    }


class ReadOnlyGuard:
    """Refuse every mutating tool.

    `--read-only` already omits these from the schema the model sees, so this
    is the second layer: the schema is what the model is told, this is what the
    loop enforces. They fail differently - a model can name a tool it was never
    offered, and a schema cannot stop a resumed session whose history still
    carries one.
    """

    name = "read-only"

    def check(self, tool_name: str, arguments: dict, mutating: bool) -> Verdict:
        if mutating:
            return Verdict.deny(
                f"{tool_name} modifies state and this session is read-only.",
                source=self.name,
            )
        return Verdict.allow()


@dataclass
class SubprocessGuard:
    """Run external guard programs; only exit zero grants permission."""

    commands: list[str]
    workspace: str = ""
    timeout: float = DEFAULT_TIMEOUT
    advisories: list[str] = field(default_factory=list)

    def check(self, tool_name: str, arguments: dict, mutating: bool) -> Verdict:
        payload = json.dumps(
            to_hook_payload(tool_name, arguments, mutating, self.workspace)
        )
        for command in self.commands:
            verdict = self._run_one(command, payload)
            if not verdict.allowed:
                return verdict
        return Verdict.allow()

    def _run_one(self, command: str, payload: str) -> Verdict:
        try:
            proc = run_shell(command, input=payload, timeout=self.timeout,
                             cwd=self.workspace or None)
        except subprocess.TimeoutExpired:
            self.advisories.append(f"{command}: timed out after {self.timeout}s")
            return Verdict.deny(self.advisories[-1], source=command)
        except OSError as exc:
            self.advisories.append(f"{command}: could not run ({exc})")
            return Verdict.deny(self.advisories[-1], source=command)

        if proc.returncode == 2:
            reason = (proc.stderr or proc.stdout or "").strip()
            return Verdict.deny(reason or f"blocked by {command}", source=command)
        if proc.returncode != 0:
            self.advisories.append(
                f"{command}: exit {proc.returncode} "
                f"{(proc.stderr or '').strip()[:200]}"
            )
            return Verdict.deny(self.advisories[-1], source=command)
        # Exit 0 can still refuse: a Claude Code guard answers with a JSON
        # envelope (`permissionDecision: deny`, `decision: block`) on stdout.
        from onecli.hooks import _context_from_stdout  # hooks imports guard
        _, refusal = _context_from_stdout(proc.stdout or "")
        if refusal is not None:
            return Verdict.deny(refusal or f"blocked by {command}", source=command)
        return Verdict.allow()


@dataclass
class GuardChain:
    """Every guard in order; the first denial wins."""

    guards: list[Guard] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.guards)

    def check(self, tool_name: str, arguments: dict, mutating: bool) -> Verdict:
        for guard in self.guards:
            verdict = guard.check(tool_name, arguments, mutating)
            if not verdict.allowed:
                return verdict
        return Verdict.allow()


def build(commands: list[str] | None = None, read_only: bool = False,
          workspace: str = "", timeout: float = DEFAULT_TIMEOUT) -> GuardChain:
    """Assemble the chain the CLI and the host both use."""
    guards: list[Guard] = []
    if read_only:
        guards.append(ReadOnlyGuard())
    if commands:
        guards.append(SubprocessGuard(list(commands), workspace=workspace,
                                      timeout=timeout))
    return GuardChain(guards)


__all__ = [
    "Verdict", "Guard", "GuardChain", "ReadOnlyGuard", "SubprocessGuard",
    "build", "to_hook_payload", "TOOL_NAME_MAP", "ARG_NAME_MAP",
]

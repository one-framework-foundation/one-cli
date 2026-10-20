"""Drivers - a subagent whose body is somebody else's CLI.

`onecli/harnesses.py` already knows how to point a dozen vendor CLIs at THIS
endpoint with THIS model and read back what they did. The arena uses that to
score them against us. A driver is the same adapter turned around: used for
work instead of for a score. A preset that names `driver: opencode` spawns
opencode - in the parent's workspace, on the parent's backend, with the
parent's key - and the parent gets back one answer, exactly as it would from an
in-process child.

What a driver child inherits, and what it cannot:

- **The backend.** The rung the parent is standing on right now, not the one it
  started on. A driver is not a second account and never falls back to the
  operator's own vendor credentials: `harnesses._env` strips every `*_API_KEY`,
  `*_BASE_URL` and `*_AUTH_TOKEN` from its environment first, so a driver that
  fails to pick up your configured endpoint fails loudly instead of quietly answering from
  somebody's personal plan.
- **The workspace.** Same directory, as cwd and as whatever `--dir`/`-C` flag
  that CLI takes.
- **The guard chain, once, at the boundary.** A vendor CLI does not call our
  tools, so there is no per-call seam to guard. What there IS is a single
  honest decision: this workspace is about to be handed to `<argv>`. That is
  presented to the guard chain as a mutating `bash` call carrying the real
  command line, so a PreToolUse guard written for Claude Code sees it and can
  refuse it. One decision, made where the decision actually is.
- **NOT the tool surface, NOT the hooks, NOT the loop guard.** We cannot
  intersect a foreign CLI's tools with ours, so we do not pretend to. The
  surface rule is therefore a refusal rather than a narrowing: a parent that
  has itself been narrowed (`--agent explorer`, no write/edit/bash) may only
  drive a CLI that takes a read-only instruction, and gets it in read-only
  mode. A parent holding the full surface may drive anything.
- **The budget, as a floor.** Only our own harness reports usage; every vendor
  CLI here reports none. A driver's spend on your configured endpoint is real, so charging
  it zero would make `--max-budget` a ceiling with a door in it. What we can
  count is the prompt in and the answer out, which is a FLOOR - the child's own
  tool loop spent more - so it is charged to the parent's meter with
  `estimated` set, the same flag the loop already raises when an endpoint
  returns no usage. A floor that says it is a floor beats a zero that says
  nothing.

A driver failing is a tool error, never a crash. Missing binary, wrong wire,
timeout, non-zero exit, unparsable output: each comes back as a sentence the
parent can read and act on, because the parent's run is not over just because
somebody else's CLI is.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from onecli import config, context as ctx, harnesses

# `one` is not a driver. Spawning our own CLI as a subprocess would buy nothing
# and lose the two things the in-process child exists for: the shared meter
# object and the shared guard chain. A preset asking for it is told so.
NOT_A_DRIVER = {
    "one": "'one' is this program - use an in-process subagent, which shares "
           "the parent's meter and guard chain instead of forking a new one",
}

# Tools a parent must hold before it may hand its workspace to a CLI whose
# surface we cannot narrow.
MUTATING_TOOLS = ("write", "edit", "bash")

# Harnesses whose adapter actually acts on `Cell.read_only`. Deliberately
# short: `_gptme_build` passes `-t read,shell` for a read-only cell, and a
# shell is not read-only, so gptme is not on this list. Being on it means the
# CLI is told to restrict itself, not that we can enforce it.
READ_ONLY_CAPABLE = ("pi", "opencode")

# A driver gets longer than an in-process child's step budget allows, because
# it is a whole agent loop of somebody else's design and we cannot count its
# steps. Wall time is the only budget that applies to all of them.
DEFAULT_TIMEOUT = 300.0


@dataclass
class DriverRun:
    """What one driver child did. Uncounted fields stay 0 - see the note."""

    driver: str
    result: harnesses.Result
    seconds: float = 0.0
    argv: str = ""
    metered: bool = False


def names() -> list[str]:
    """Every harness that could be a driver on this box, installed or not."""
    return sorted(n for n in harnesses.HARNESSES
                  if n not in NOT_A_DRIVER and n not in harnesses.SKIPPED)


def available() -> list[str]:
    """The drivers whose binary is actually on this box."""
    return [n for n in names() if harnesses.HARNESSES[n].available()]


def catalogue() -> str:
    lines = []
    for name in names():
        harness = harnesses.HARNESSES[name]
        mark = "" if harness.available() else " (not installed)"
        lines.append(f"- {name}{mark}: {harness.note or 'no note'}")
    return "\n".join(lines)


def backend_for(rung, name: str = "") -> config.Backend:
    """The Backend a harness adapter needs, read off the LIVE provider.

    Built from the provider rather than carried on the rung on purpose: a run
    started from a bare provider (no config lookup) has no Backend anywhere,
    and a driver that only worked for chain-built runs would be a driver that
    works in the tests and not in `one -m`. The provider is the one object that
    exists in both cases and knows all four facts.
    """
    provider = rung.provider
    wire = (config.WIRE_ANTHROPIC
            if type(provider).__name__.lower().startswith("anthropic")
            else config.WIRE_OPENAI)
    return config.Backend(
        name=name or getattr(rung, "name", "") or "parent",
        model=getattr(provider, "model", "") or getattr(rung, "model", ""),
        base_url=getattr(provider, "base_url", ""),
        wire=wire,
        api_key=getattr(provider, "api_key", ""),
    )


def refusal(driver: str, parent_tools: list[str] | None) -> str:
    """Why this parent may not run this driver, or "" if it may.

    The surface rule in one place. A parent with the whole surface may drive
    anything; a narrowed parent may only drive a CLI that takes a read-only
    instruction, and gets it in read-only mode. There is no third case where we
    quietly hand a write-capable CLI to a parent that cannot write.
    """
    if driver in NOT_A_DRIVER:
        return NOT_A_DRIVER[driver]
    # A known CLI that cannot be driven headless says WHY, not "no such
    # driver", so this check comes before the lookup.
    if driver in harnesses.SKIPPED:
        return f"{driver} cannot be driven headless: {harnesses.SKIPPED[driver]}"
    if driver not in harnesses.HARNESSES:
        return f"no driver named {driver!r}. Available: {', '.join(available())}"
    # Policy before environment. What this RUN may hand over is a fact about
    # the run and is the same on every host; what is installed is a fact about
    # this box. Checking installation first would answer a narrowed parent with
    # "not installed here", which reads as "install it and retry" for something
    # it would be refused anyway.
    if read_only_for(parent_tools) and driver not in READ_ONLY_CAPABLE:
        return (f"this run holds a narrowed tool surface, and {driver} has no "
                f"read-only mode we can set. Drivers that do: "
                f"{', '.join(READ_ONLY_CAPABLE)}")
    if not harnesses.HARNESSES[driver].available():
        return (f"the {driver} CLI is not installed on this host "
                f"(installed: {', '.join(available()) or 'none'})")
    return ""


def read_only_for(parent_tools: list[str] | None) -> bool:
    """Does the parent lack a mutating tool? Then so must its driver child."""
    if parent_tools is None:
        return False
    return not any(t in parent_tools for t in MUTATING_TOOLS)


def wire_refusal(driver: str, backend: config.Backend) -> str:
    """A CLI that cannot speak this endpoint's wire is a fact, not a failure."""
    harness = harnesses.HARNESSES[driver]
    if backend.wire in harness.wires:
        return ""
    return (f"{driver} cannot drive the {backend.wire} wire "
            f"(it speaks {'/'.join(harness.wires)}), and this run is on "
            f"{backend.name} which is {backend.wire}")


def guard_verdict(guard, argv: list[str], workspace: str):
    """Present the launch to the parent's guard chain as one bash call.

    A vendor CLI makes its own tool calls inside its own process, where we have
    no seam. The seam we DO have is this one: the moment the workspace is
    handed over. Presented as `bash` with the real command line so a guard
    written for Claude Code's PreToolUse contract matches it the way it matches
    any other command, and mutating=True so a read-only chain refuses it.
    """
    if not guard:
        return None
    return guard.check("bash", {"command": " ".join(argv), "cwd": workspace}, True)


def charge(meter, prompt: str, result: harnesses.Result) -> None:
    """Put a driver's spend on the parent's meter - as a floor, marked as one.

    A harness that reports usage is believed. One that does not is charged the
    prompt it was given plus the answer it returned, which is less than it
    spent (its own tool loop is invisible to us) and is flagged `estimated` so
    nothing downstream reads it as a measurement.
    """
    if meter is None:
        return
    prompt_tokens = result.cost.get("input_tokens")
    output_tokens = result.cost.get("output_tokens")
    if prompt_tokens is None and output_tokens is None:
        meter.input_tokens += ctx.estimate_tokens(prompt)
        meter.output_tokens += ctx.estimate_tokens(result.text or "")
        meter.estimated = True
        return
    meter.input_tokens += int(prompt_tokens or 0)
    meter.output_tokens += int(output_tokens or 0)
    if result.cost.get("estimated"):
        meter.estimated = True


def run(driver: str, prompt: str, *, backend: config.Backend, workspace: str,
        system: str = "", read_only: bool = False,
        timeout: float = DEFAULT_TIMEOUT, before=None) -> DriverRun:
    """Run one vendor CLI to completion. Every failure is a Result, not a raise.

    The private config directory is made and destroyed here, inside one
    function, because that is where the API key lives: every adapter writes the
    endpoint and the key into a config file for its CLI to read, and a
    key-bearing directory whose lifetime is somebody else's problem is a
    key-bearing directory that outlives the run.
    """
    harness = harnesses.HARNESSES[driver]
    cfg_dir = Path(tempfile.mkdtemp(prefix=f"one-driver-{driver}-"))
    try:
        return _run_in(harness, driver, prompt, backend=backend,
                       workspace=workspace, cfg_dir=cfg_dir, system=system,
                       read_only=read_only, timeout=timeout, before=before)
    finally:
        shutil.rmtree(cfg_dir, ignore_errors=True)


def _run_in(harness, driver: str, prompt: str, *, backend: config.Backend,
            workspace: str, cfg_dir: Path, system: str, read_only: bool,
            timeout: float, before=None) -> DriverRun:
    cell = harnesses.Cell(backend=backend, model_id="", workspace=Path(workspace),
                          prompt=prompt, system=system, read_only=read_only,
                          timeout=timeout, cfg_dir=cfg_dir)
    started = time.monotonic()
    try:
        launch = harness.build(cell)
    except OSError as exc:
        return DriverRun(driver, harnesses.Result(
            error=f"could not prepare {driver}: {exc}", stopped="crash"))
    argv = " ".join(launch.cmd)
    # The guard sees the command that will actually run, not a stand-in for it.
    # `build` is what turns a preset into an argv - config files, flags, the
    # model pin - so this is the earliest moment the real launch exists, and
    # the last moment before it happens.
    if before is not None:
        denial = before(launch.cmd)
        if denial:
            return DriverRun(driver, harnesses.Result(
                error=denial, stopped="denied"), argv=argv)
    stdin = ({"input": launch.stdin} if launch.stdin
             else {"stdin": subprocess.DEVNULL})
    try:
        proc = subprocess.run(launch.cmd, cwd=launch.cwd, env=launch.env,
                              capture_output=True, text=True, timeout=timeout,
                              **stdin)
    except subprocess.TimeoutExpired:
        return DriverRun(driver, harnesses.Result(
            error=f"{driver} exceeded {timeout:.0f}s and was killed",
            stopped="timeout"), seconds=time.monotonic() - started, argv=argv)
    except OSError as exc:
        return DriverRun(driver, harnesses.Result(
            error=f"could not start {driver}: {exc}", stopped="crash"),
            seconds=time.monotonic() - started, argv=argv)
    elapsed = time.monotonic() - started
    result = harness.parse(proc.stdout, proc.stderr)
    if not (result.text or "").strip() and not result.error and proc.returncode != 0:
        # A CLI that exits non-zero with nothing parsable has still told us
        # something; losing it to an empty answer is how a driver failure gets
        # reported to the parent as "the subagent returned nothing".
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        result.error = f"{driver} exited {proc.returncode}: {tail}"
        result.stopped = "crash"
    return DriverRun(driver, result, seconds=elapsed, argv=argv,
                     metered=bool(result.cost))


__all__ = ["DriverRun", "NOT_A_DRIVER", "MUTATING_TOOLS", "READ_ONLY_CAPABLE",
           "DEFAULT_TIMEOUT", "names", "available", "catalogue", "backend_for",
           "refusal", "read_only_for", "wire_refusal", "guard_verdict",
           "charge", "run"]

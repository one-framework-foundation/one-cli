"""Other people's agent CLIs, driven against your configured endpoints.

A harness adapter answers three questions: is the binary on this box, how is it
pointed at THIS endpoint and THIS model, and what did it do. Everything else
about a cell - the workspace, the task, the checker, the timeout - is identical
across harnesses, because that is the only thing that makes the comparison mean
anything. A harness that can only be run against its vendor's own endpoint is
not comparable and is skipped loudly rather than scored.

Every adapter gets a private config dir per cell. None of them may read the
operator's real config: a run that picks up `~/.config/opencode` or `~/.pi`
measures that machine's settings, not the harness.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config, secureio

NODE_BIN = shutil.which("node") or "node"
KILO_BIN = os.environ.get("KILO_BIN") or shutil.which("kilo") or "kilo"
# MiMo Code (@mimo-ai/cli), bin name "mimo". MIMOCODE_BIN names a specific
# install when another "mimo" command sits earlier on PATH.
MIMOCODE_BIN = os.environ.get("MIMOCODE_BIN") or shutil.which("mimo") or "mimo"

# Harnesses installed for the arena live here, never in the operator's global
# prefixes: `uv tool` output in .harnesses/bin, npm output in
# .harnesses/node/node_modules/.bin, plain binaries in .harnesses/bin.
HARNESS_PREFIX = Path(__file__).resolve().parent.parent / ".harnesses"
EXTRA_BINS = (HARNESS_PREFIX / "bin",
              HARNESS_PREFIX / "node/node_modules/.bin")


def _which(name: str) -> str:
    """Resolve a CLI: the arena prefix first, then the login shell's PATH."""
    for directory in EXTRA_BINS:
        candidate = directory / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name) or ""


@dataclass
class Cell:
    backend: config.Backend
    model_id: str
    workspace: Path
    prompt: str
    system: str = ""
    read_only: bool = False
    timeout: float = 180.0
    cfg_dir: Path = field(default_factory=Path)
    # The task's own tool-loop budget. A harness that takes a step limit gets
    # THIS number, never one baked into its adapter: a cap that differs per
    # harness measures the cap, not the harness.
    max_steps: int = 30

    @property
    def model(self) -> str:
        return self.model_id or self.backend.model

    # What a keyless endpoint gets written into a harness config. A local
    # llama.cpp or ollama needs no key and `config.Backend` leaves it "", but an
    # empty string is not the same as no key to the CLI reading it: pi's schema
    # declares `apiKey` as a string of minLength 1, so `"apiKey": ""` fails
    # validation, the whole models.json is discarded, and the run dies with
    # `Unknown provider "onelane"` before a single request is sent. Every
    # adapter here writes the key somewhere, so the substitution belongs on the
    # cell rather than in nine places.
    PLACEHOLDER_KEY = "no-key-required"

    @property
    def key(self) -> str:
        return self.backend.api_key or self.PLACEHOLDER_KEY


@dataclass
class Launch:
    cmd: list[str]
    cwd: str
    env: dict[str, str] | None = None
    stdin: str = ""


def _read_only_launch(launch: Launch, cell: Cell) -> Launch:
    """Mount the host tree read-only; only the private config remains writable.

    Reuse the bubblewrap isolation flags used by tool_sandbox. This is a
    filesystem write boundary, not an offline or confidentiality sandbox.
    Missing/unsupported bubblewrap fails at launch; never run unguarded.
    """
    workspace, cfg = cell.workspace.resolve(), cell.cfg_dir.resolve()
    if (workspace == cfg or workspace in cfg.parents or cfg in workspace.parents
            or cfg == Path("/") or cfg == Path.home().resolve()):
        raise OSError("read-only launch requires a separate private config directory")
    cmd = [shutil.which("bwrap") or "bwrap", "--unshare-all", "--unshare-user",
           "--share-net", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
           "--disable-userns", "--assert-userns-disabled", "--ro-bind", "/", "/",
           "--proc", "/proc", "--dev", "/dev", "--bind", str(cfg), str(cfg),
           "--chdir", str(workspace), "--", *launch.cmd]
    scratch = cfg / "tmp"
    scratch.mkdir(mode=0o700, exist_ok=True)
    env = dict(launch.env if launch.env is not None else os.environ)
    env.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    return Launch(cmd, cwd=launch.cwd, env=env, stdin=launch.stdin)


@dataclass
class Result:
    text: str = ""
    steps: int = 0
    tool_calls: int = 0
    error: str = ""
    stopped: str = "answered"
    # Token usage, when the harness reports any. Empty means it does not, which
    # is a different thing from a run that used no tokens.
    cost: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# declared parity (dimensions 4-8 and 12 of the arena parity protocol)
# --------------------------------------------------------------------------
# Each adapter declares, per dimension, whether the cell's value reaches the
# harness (MATCHED), the harness genuinely has no such control (UNCAPPED), or
# the control exists and we fail to pass it (ABSENT - a defect to fix, never a
# resting state). This is the *arena* parity protocol; it is unrelated to
# onecli/evals/parity.py, which is the one-cli-vs-Claude-Code feature table.
MATCHED = "MATCHED"
UNCAPPED = "UNCAPPED"
ABSENT = "ABSENT"
PARITY_STATUSES = (MATCHED, UNCAPPED, ABSENT)

# Dimension number -> short label, straight from the spec table. 9/10/11 are the
# observed dimensions and are proven by the probe, not declared here.
PARITY_DIMENSIONS: dict[int, str] = {
    4: "system prompt role",
    5: "step budget",
    6: "wall timeout",
    7: "output token budget",
    8: "thinking / reasoning level",
    12: "ambient state",
}


@dataclass(frozen=True)
class ParityDim:
    status: str
    note: str


def _parity(system, step, timeout, tokens, thinking, ambient) -> dict[int, ParityDim]:
    """(status, note) tuples for dims 4, 5, 6, 7, 8, 12 in that order."""
    return {4: ParityDim(*system), 5: ParityDim(*step), 6: ParityDim(*timeout),
            7: ParityDim(*tokens), 8: ParityDim(*thinking), 12: ParityDim(*ambient)}


@dataclass
class Harness:
    name: str
    binary: str
    wires: tuple[str, ...]
    build: Callable[[Cell], Launch]
    parse: Callable[[str, str], Result]
    note: str = ""
    parity: dict[int, ParityDim] = field(default_factory=dict)

    def available(self) -> bool:
        return bool(_which(self.binary)) if self.binary else True


def _env(cell: Cell, extra: dict[str, str], isolate_home: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    # Strip every vendor credential the child could otherwise fall back to, so a
    # cell that fails to pick up your configured endpoint fails loudly instead of quietly
    # answering from someone's personal account.
    for name in list(env):
        if name.endswith(("_API_KEY", "_AUTH_TOKEN", "_BASE_URL", "_OAUTH_TOKEN")):
            env.pop(name, None)
    # Most adapters take an explicit config path, so their HOME can be thrown
    # away. qwen-code cannot: with an empty HOME it starts, prints nothing and
    # exits 0, so it keeps the real one and reads that box's ~/.qwen settings.
    # Worth knowing when reading its rows - the credential scrub above still
    # applies, so it can only reach the endpoint we hand it.
    if isolate_home:
        env["HOME"] = str(cell.cfg_dir)
    env["PATH"] = ":".join([str(d) for d in EXTRA_BINS] + [env.get("PATH", "")])
    env.update(extra)
    return env


# --------------------------------------------------------------------------
# one - our own loop
# --------------------------------------------------------------------------

def _one_build(cell: Cell) -> Launch:
    """Our own loop, held to exactly the terms every other adapter gets.

    It takes the endpoint explicitly rather than by drop-in name, runs on an
    isolated HOME (so `~/.one-cli` and `~/.one-providers` are never shared
    with anything else running), and uses the task's own step budget rather
    than a fixed cap - the same terms as every other adapter.
    """
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "onecli",
           "--base-url", cell.backend.base_url, "--wire", cell.backend.wire,
           "--model-id", cell.model,
           "-w", str(cell.workspace), "--json", "-q",
           "--max-steps", str(cell.max_steps),
           "--timeout", str(int(cell.timeout)), "--max-tokens", "8192",
           # The arena measures task execution, not the non-interactive CLI's
           # first-response policy gate. That gate adds a model request and
           # consumes one of the task's matched steps.
           "--response-gate", "off"]
    if cell.system:
        cmd += ["--system", cell.system]
    if cell.read_only:
        cmd.append("--read-only")
    # `python -m` searches cwd before PYTHONPATH. Keep interpreter startup in
    # the trusted package root, never a task's onecli.py/package (or a shadow
    # standard-library module). The explicit -w still roots every agent tool
    # in the task workspace. This also works on Python 3.10, without -P.
    return Launch(cmd + [cell.prompt], cwd=str(root),
                  env=_env(cell, {"ONE_CLI_API_KEY": cell.backend.api_key,
                                  "PYTHONPATH": str(root),
                                  # An isolated HOME has no drop-in dir, and
                                  # discovery is bypassed anyway - but an empty
                                  # one is still the honest value to hand it.
                                  "ONE_PROVIDERS_DIR": str(cell.cfg_dir / "providers")}))


def _one_parse(stdout: str, stderr: str) -> Result:
    try:
        payload = json.loads(stdout)
    except ValueError:
        return Result(error=(stderr or stdout).strip()[-200:], stopped="crash")
    return Result(text=payload.get("text") or "", steps=payload.get("steps", 0),
                  tool_calls=payload.get("tool_calls", 0),
                  error=(payload.get("error") or "")[:200],
                  stopped=payload.get("stopped", "answered"),
                  cost=payload.get("cost") or {})


# --------------------------------------------------------------------------
# pi - earendil-works/pi coding agent
# --------------------------------------------------------------------------



def _pi_build(cell: Cell) -> Launch:
    agent_dir = cell.cfg_dir / ".pi" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    api = ("anthropic-messages" if cell.backend.wire == config.WIRE_ANTHROPIC
           else "openai-completions")
    secureio.atomic_write_text(agent_dir / "models.json", json.dumps({
        "providers": {
            "onelane": {
                "baseUrl": cell.backend.base_url,
                "api": api,
                "apiKey": cell.key,
                "models": [{"id": cell.model}],
            }
        }
    }), private=True)
    cmd = [_which("pi"), "-p", "--mode", "json", "--no-session",
           "--no-extensions", "--no-skills", "--no-context-files",
           "--provider", "onelane", "--model", cell.model]
    if cell.system:
        cmd += ["--system-prompt", cell.system]
    if cell.read_only:
        cmd += ["--tools", "read,grep,find,ls"]
    cmd.append(cell.prompt)
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"PI_CODING_AGENT_DIR": str(agent_dir),
                                  "PI_SKIP_VERSION_CHECK": "1",
                                  # A cell whose provider list depends on a
                                  # network catalog refresh is measuring the
                                  # refresh. Ours is written out in full, so
                                  # there is nothing to fetch.
                                  "PI_OFFLINE": "1"}))


def _pi_parse(stdout: str, stderr: str) -> Result:
    res = Result()
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "turn_end":
            res.steps += 1
            res.tool_calls += len(event.get("toolResults") or [])
            message = event.get("message") or {}
            # pi is the one third-party harness here that reports usage, so its
            # rows carry real token counts rather than the "unknown, not zero"
            # blank the others get. Accumulated across turns, not taken from the
            # last one, which reports only its own.
            usage = message.get("usage") or {}
            if usage:
                res.cost["input_tokens"] = (res.cost.get("input_tokens", 0)
                                            + int(usage.get("input") or 0))
                res.cost["output_tokens"] = (res.cost.get("output_tokens", 0)
                                             + int(usage.get("output") or 0))
            if message.get("errorMessage"):
                res.error = str(message["errorMessage"])[:200]
                res.stopped = "error"
        elif kind == "agent_end":
            for message in reversed(event.get("messages") or []):
                if message.get("role") != "assistant":
                    continue
                parts = [c.get("text", "") for c in message.get("content") or []
                         if c.get("type") == "text"]
                if parts:
                    res.text = "\n".join(parts)
                    break
            break
    if not res.text and not res.error:
        res.error = (stderr or stdout).strip()[-200:]
        res.stopped = "crash" if res.error else res.stopped
    return res


# --------------------------------------------------------------------------
# opencode - sst/opencode
# --------------------------------------------------------------------------

def _opencode_build(cell: Cell) -> Launch:
    anthropic = cell.backend.wire == config.WIRE_ANTHROPIC
    provider = {
        "npm": "@ai-sdk/anthropic" if anthropic else "@ai-sdk/openai-compatible",
        "name": "onelane",
        "options": {"baseURL": cell.backend.base_url, "apiKey": cell.key},
        "models": {cell.model: {"name": cell.model}},
    }
    cfg: dict = {"$schema": "https://opencode.ai/config.json",
                 "provider": {"onelane": provider}}
    if cell.read_only:
        cfg["permission"] = {"edit": "deny", "bash": "deny", "webfetch": "deny"}
    cfg_path = cell.cfg_dir / "opencode.json"
    secureio.atomic_write_text(cfg_path, json.dumps(cfg), private=True)
    prompt = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    cmd = [_which("opencode"), "run", "--auto", "--format", "json",
           "--pure", "--dir", str(cell.workspace),
           "-m", f"onelane/{cell.model}", prompt]
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"OPENCODE_CONFIG": str(cfg_path),
                                  "OPENCODE_DISABLE_AUTOUPDATE": "1"}))


def _opencode_parse(stdout: str, stderr: str) -> Result:
    res = Result()
    texts: list[str] = []

    def walk(node) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        if kind == "tool":
            res.tool_calls += 1
        if kind == "text" and isinstance(node.get("text"), str):
            texts.append(node["text"])
        if kind == "message" or node.get("role") == "assistant":
            res.steps += 1
        if node.get("error"):
            res.error = json.dumps(node["error"])[:200]
            res.stopped = "error"
        for value in node.values():
            if isinstance(value, (dict, list)):
                walk(value)

    blob = stdout.strip()
    if blob.startswith(("{", "[")):
        try:
            walk(json.loads(blob))
        except ValueError:
            for line in blob.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        walk(json.loads(line))
                    except ValueError:
                        continue
    res.text = texts[-1] if texts else ""
    if not res.text and not res.error:
        # Fall back to whatever it printed rather than lose a real answer to a
        # JSON shape we did not predict - but keep it short enough to read.
        res.text = blob[-2000:]
        if not res.text:
            res.error = stderr.strip()[-200:]
            res.stopped = "crash" if res.error else res.stopped
    return res


# --------------------------------------------------------------------------
# openclaw - openclaw 2.0, agent exec
# --------------------------------------------------------------------------

def _kilo_build(cell: Cell) -> Launch:
    anthropic = cell.backend.wire == config.WIRE_ANTHROPIC
    base_url = cell.backend.base_url.rstrip("/")
    if anthropic and not base_url.endswith("/v1"):
        # @ai-sdk/anthropic posts to "<baseURL>/messages", while an Anthropic
        # base_url here is the SDK root: without /v1 every call is a 404.
        base_url += "/v1"
    provider = {
        "npm": "@ai-sdk/anthropic" if anthropic else "@ai-sdk/openai-compatible",
        "name": "onelane",
        "options": {"baseURL": base_url, "apiKey": cell.key},
        "models": {cell.model: {"name": cell.model, "tool_call": True,
                                "limit": {"context": 128000, "output": 8192}}},
    }
    cfg: dict = {"$schema": "https://app.kilo.ai/config.json",
                 "enabled_providers": ["onelane"],
                 "provider": {"onelane": provider},
                 "agent": {"code": {"steps": cell.max_steps}}}
    if cell.system:
        cfg["agent"]["code"]["prompt"] = cell.system
    if cell.read_only:
        cfg["permission"] = {"edit": "deny", "bash": "deny", "webfetch": "deny"}
    cfg_path = cell.cfg_dir / "kilo.json"
    secureio.atomic_write_text(cfg_path, json.dumps(cfg), private=True)
    # The arena prefix can contain a broken `node` symlink. Calling the
    # interpreter directly avoids the CLI's `/usr/bin/env node` PATH lookup.
    cmd = [NODE_BIN, _which(KILO_BIN), "run", "--auto", "--pure", "--format", "json",
           "--dir", str(cell.workspace), "--agent", "code",
           "--model", f"onelane/{cell.model}", cell.prompt]
    launch = Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"KILO_CONFIG": str(cfg_path),
                                  "KILO_DISABLE_PROJECT_CONFIG": "1",
                                  "KILO_DISABLE_EXTERNAL_SKILLS": "1"}))


    return _read_only_launch(launch, cell) if cell.read_only else launch


def _kilo_parse(stdout: str, stderr: str) -> Result:
    res = Result()
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    saw_usage = False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part") or {}
        if not isinstance(part, dict):
            continue
        # The part type is the stable one: a tool call arrives as event type
        # "tool_use" wrapping part type "tool".
        kind = part.get("type") or event.get("type")
        if kind == "text":
            res.text += part.get("text", "")
        elif kind == "tool":
            res.tool_calls += 1
        elif kind in ("step_finish", "step-finish"):
            res.steps += 1
            tokens = part.get("tokens")
            if tokens:
                # "input" excludes cache hits and "output" excludes reasoning;
                # reasoning is billed as output, cache hits are kept apart as
                # in the claude adapter.
                saw_usage = True
                usage["input_tokens"] += int(tokens.get("input") or 0)
                usage["output_tokens"] += int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0)
                usage["cache_read_input_tokens"] += int((tokens.get("cache") or {}).get("read") or 0)
        elif kind == "error":
            error = event.get("error") or part.get("error") or event
            res.error = str(error.get("data", {}).get("message") or error.get("message") or error)[:200]
            res.stopped = "error"
    if saw_usage:
        res.cost = usage
    if not res.text and not res.error:
        res.error = (stderr or stdout).strip()[-200:]
        res.stopped = "crash" if res.error else res.stopped
    return res


def _openclaw_build(cell: Cell) -> Launch:
    """openclaw 2.0 driven headless through `agent exec`.

    Two things set it apart from the other node harnesses. It needs node
    >= 24.15.0, so the runtime on PATH must satisfy that. And its provider
    block wants an explicit `api` wire and `auth`: a pinned
    `models.mode="replace"` config drops in one
    `onelane` provider carrying exactly our base URL, key and wire, so nothing
    ambient leaks in. It is the only harness besides one that speaks BOTH wires.
    """
    # openclaw anchors a session's workspace root to the nearest enclosing
    # `.git`, NOT to `--cwd`, so a cell nested inside another git tree would
    # resolve the `read` tool's relative paths against that tree. `exec` honours
    # `--cwd`, so only the read path is affected. Give the cell its own `.git`
    # so root detection stops at the cell workspace. `snapshot()` ignores
    # `.git`, so this is invisible to the before/after diff.
    if not (cell.workspace / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=str(cell.workspace),
                       check=False, capture_output=True)
    anthropic = cell.backend.wire == config.WIRE_ANTHROPIC
    api = "anthropic-messages" if anthropic else "openai-completions"
    cfg: dict = {
        "models": {
            "mode": "replace",
            "providers": {
                "onelane": {
                    "baseUrl": cell.backend.base_url,
                    "apiKey": cell.key,
                    "auth": "api-key",
                    "api": api,
                    # `maxTokens` on a model entry is openclaw's output-token cap
                    # (providers/reference docs: "Max output tokens"). `agent
                    # exec` has no --max-tokens flag, so this config field is how
                    # openclaw gets the same 8192 ceiling every other harness
                    # gets (dim 7).
                    "models": [{"id": cell.model, "name": cell.model,
                                "maxTokens": 8192}],
                }
            },
        }
    }
    if cell.read_only:
        # No write, no shell: read the tree only, mirroring what a read-only
        # opencode cell gets. Two gates have to line up.
        #
        # 1) Tool policy. The base profile MUST stay `coding` - openclaw's
        #    `minimal` profile is `session_status` ONLY, so it leaves the agent with no read tool at all. Deny the mutating
        #    tools by their real names on top of `coding`: `bash` is an alias for
        #    `exec` and the patch tool is `apply_patch`, not `patch`.
        # 2) Sandbox. `agent exec` runs a NON-main session, and openclaw's
        #    default `sandbox.mode: "non-main"` sandboxes it. Inside a sandbox a
        #    SECOND gate - `tools.sandbox.tools` - applies, and its default
        #    allowlist is web+memory ONLY, so `read` (group:fs) is stripped even
        #    though tool policy allows it. The symptom is an agent that reports
        #    exactly `web_fetch, web_search, memory_search` and refuses for lack
        #    of a file-reading tool. The arena already isolates each cell in its
        #    own workspace under a private HOME, so the sandbox buys nothing here;
        #    turn it off so the read-only session runs on the host with the tool
        #    policy above as the only constraint.
        cfg["agents"] = {"defaults": {"sandbox": {"mode": "off"}}}
        # `terminal` (group:ui) is a second shell surface alongside `exec`; it is
        # offered under `coding` and must be denied too, or a read-only cell keeps
        # a way to run commands. The deny makes read-only a policy rather than an
        # accident of the environment.
        cfg["tools"] = {"profile": "coding",
                        "deny": ["write", "edit", "apply_patch",
                                 "exec", "process", "code_execution", "terminal"]}
    cfg_path = cell.cfg_dir / "openclaw.json"
    secureio.atomic_write_text(cfg_path, json.dumps(cfg), private=True)
    # A long prompt goes through a file, never the argv - shell-quoting a
    # multi-KB task is how a cell dies before the harness ever runs.
    msg = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    msg_path = cell.cfg_dir / "openclaw_message.txt"
    secureio.atomic_write_text(msg_path, msg, private=True)
    state_dir = cell.cfg_dir / "openclaw_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cmd = [_which("openclaw"), "agent", "exec",
           "--config", str(cfg_path), "--cwd", str(cell.workspace),
           "--model", f"onelane/{cell.model}", "--json",
           "--timeout", str(int(cell.timeout)),
           # dim 8 (thinking) is deliberately NOT pinned. `--thinking` exists,
           # but openclaw validates the level per model and no single level is
           # supported by every model; some accept only `off`. Pinning `off`
           # everywhere would DISABLE reasoning on openclaw
           # while `one` runs at each model's default - a new asymmetry, not a
           # removed one. So openclaw runs at the per-model default, symmetric
           # with `one`'s own UNCAPPED backend-default reasoning.
           # --isolated cannot be combined with --config; state stays isolated
           # through the private HOME plus OPENCLAW_STATE_DIR/OPENCLAW_CONFIG_PATH.
           "--message-file", str(msg_path)]
    launch = Launch(cmd, cwd=str(cell.workspace),
                    env=_env(cell, {"OPENCLAW_CONFIG_PATH": str(cfg_path),
                                    "OPENCLAW_STATE_DIR": str(state_dir)}))
    return launch


def _openclaw_parse(stdout: str, stderr: str) -> Result:
    blob = stdout.strip()
    payload = None
    start = blob.find("{")
    if start >= 0:
        try:
            payload = json.loads(blob[start:])
        except ValueError:
            for line in reversed(blob.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        payload = json.loads(line)
                        break
                    except ValueError:
                        continue
    if payload is None:
        return Result(error=(stderr or stdout).strip()[-200:], stopped="crash")
    if not payload.get("ok", True):
        err = payload.get("error")
        msg = err.get("message") if isinstance(err, dict) else err
        return Result(error=str(msg or payload.get("status") or "error")[:200],
                      stopped="error")
    usage = payload.get("usage") or {}
    cost: dict = {}
    if usage:
        cost["input_tokens"] = int(usage.get("input") or 0)
        cost["output_tokens"] = int(usage.get("output") or 0)
    if payload.get("costUsd") is not None:
        cost["cost_usd"] = payload["costUsd"]
    tool = payload.get("toolSummary") or {}
    return Result(text=payload.get("final") or "",
                  steps=int(payload.get("assistantTurns") or 0),
                  tool_calls=int(tool.get("calls") or 0),
                  cost=cost)


# --------------------------------------------------------------------------
# qwen-code and codex - vendor CLIs with an OpenAI-compatible escape hatch
# --------------------------------------------------------------------------

def _qwen_build(cell: Cell) -> Launch:
    prompt = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    cmd = [_which("qwen"), "--auth-type", "openai", "-m", cell.model, "-y",
           "-p", prompt.replace("@", r"\@")]
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"OPENAI_BASE_URL": cell.backend.base_url,
                                  "OPENAI_API_KEY": cell.key,
                                  "OPENAI_MODEL": cell.model},
                           isolate_home=False))


def _plain_parse(stdout: str, stderr: str) -> Result:
    text = stdout.strip()
    if not text:
        return Result(error=stderr.strip()[-200:], stopped="crash")
    return Result(text=text[-4000:])


# --------------------------------------------------------------------------
# kimi - Moonshot's own CLI agent
# --------------------------------------------------------------------------

def _kimi_build(cell: Cell) -> Launch:
    """`kimi --print`, one shot per cell.

    Gives the arena a kimi-cli row, so the vendor CLI and `one` can be compared
    on the same endpoint and model.

    The endpoint is handed over through the environment: `KIMI_BASE_URL` plus
    `KIMI_API_KEY`, with `KIMI_MODEL_NAME` deliberately NOT set. That last
    one is not tidiness - kimi-cli's `augment_provider_with_env_vars` lets
    `KIMI_MODEL_NAME` override `-m` unconditionally and silently, so an
    inherited value would make every `-m` here decorative and every row would
    be the same model wearing different labels.

    `--quiet` is `--print --output-format text --final-message-only`, so the
    answer is prose and steps/tool calls stay uninstrumented - the same
    limitation aider and crush carry, and it is recorded in the parity block
    rather than hidden.
    """
    prompt = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    cmd = [_which("kimi"), "--quiet", "--yolo", "--afk",
           "-m", cell.model, "-w", str(cell.workspace), prompt]
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"KIMI_BASE_URL": cell.backend.base_url,
                                  "KIMI_API_KEY": cell.key}))


def _codex_build(cell: Cell) -> Launch:
    prompt = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    base = cell.backend.base_url
    codex_home = cell.cfg_dir / "codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    cmd = [_which("codex"), "exec", "--skip-git-repo-check",
           "--dangerously-bypass-approvals-and-sandbox",
           "-C", str(cell.workspace), "-m", cell.model,
           "-c", "model_provider=onelane",
           "-c", f'model_providers.onelane={{name="onelane",'
                 f'base_url="{base}",env_key="ONELANE_KEY",'
                 f'wire_api="responses"}}',
           prompt]
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"ONELANE_KEY": cell.key,
                                  "CODEX_HOME": str(codex_home)}))


# --------------------------------------------------------------------------
# aider - Aider-AI/aider
# --------------------------------------------------------------------------

def _aider_build(cell: Cell) -> Launch:
    """aider talks to whatever OPENAI_API_BASE points at, one shot per --message.

    It has no machine-readable stream, so the answer is its prose (`_plain_parse`)
    and steps/tool calls stay uninstrumented, the way qwen-code's do.
    """
    env = {"OPENAI_API_BASE": cell.backend.base_url,
           "OPENAI_API_KEY": cell.key,
           "AIDER_ANALYTICS_DISABLE": "1", "AIDER_CHECK_UPDATE": "false"}
    if cell.backend.wire == config.WIRE_ANTHROPIC:
        env = {"ANTHROPIC_API_BASE": cell.backend.base_url,
               "ANTHROPIC_API_KEY": cell.key,
               "AIDER_ANALYTICS_DISABLE": "1", "AIDER_CHECK_UPDATE": "false"}
    prefix = "anthropic/" if cell.backend.wire == config.WIRE_ANTHROPIC else "openai/"
    cmd = [_which("aider"), "--model", prefix + cell.model, "--yes-always",
           "--no-auto-commits", "--no-git", "--no-gitignore", "--no-check-update",
           "--no-analytics", "--no-show-model-warnings", "--no-stream",
           "--no-pretty", "--no-fancy-input", "--message",
           f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt]
    return Launch(cmd, cwd=str(cell.workspace), env=_env(cell, env))


# --------------------------------------------------------------------------
# goose - block/goose
# --------------------------------------------------------------------------

def _goose_build(cell: Cell) -> Launch:
    anthropic = cell.backend.wire == config.WIRE_ANTHROPIC
    extra = {"GOOSE_PROVIDER": "anthropic" if anthropic else "openai",
             "GOOSE_MODEL": cell.model,
             "GOOSE_DISABLE_KEYRING": "1",
             "GOOSE_CLI_MIN_PRIORITY": "0.0"}
    if anthropic:
        extra |= {"ANTHROPIC_HOST": cell.backend.base_url,
                  "ANTHROPIC_API_KEY": cell.key}
    else:
        # goose splits host from path itself; OPENAI_BASE_PATH pins the route so
        # a base_url that already carries /v1 cannot become /v1/v1/chat/....
        extra |= {"OPENAI_HOST": cell.backend.base_url.rsplit("/v1", 1)[0],
                  "OPENAI_BASE_PATH": "v1/chat/completions",
                  "OPENAI_API_KEY": cell.key}
    cmd = [_which("goose"), "run", "--no-session", "-q",
           "--output-format", "json", "--max-turns", str(cell.max_steps),
           "-t", cell.prompt]
    if cell.system:
        cmd[-2:-2] = ["--system", cell.system]
    return Launch(cmd, cwd=str(cell.workspace), env=_env(cell, extra))


def _goose_parse(stdout: str, stderr: str) -> Result:
    blob = stdout.strip()
    start = blob.find("{")
    if start < 0:
        return _plain_parse(stdout, stderr)
    try:
        payload = json.loads(blob[start:])
    except ValueError:
        return _plain_parse(stdout, stderr)
    messages = payload.get("messages") or []
    res = Result(text=str(payload.get("response") or payload.get("text") or ""))
    for message in messages:
        content = message.get("content") or []
        if message.get("role") == "assistant":
            res.steps += 1
        for part in content if isinstance(content, list) else []:
            kind = part.get("type") if isinstance(part, dict) else ""
            if kind in ("toolRequest", "toolReq", "tool_request"):
                res.tool_calls += 1
            if not res.text and kind == "text" and message.get("role") == "assistant":
                res.text = str(part.get("text") or "")
    if not res.text:
        return _plain_parse(stdout, stderr)
    return res


# --------------------------------------------------------------------------
# crush - charmbracelet/crush
# --------------------------------------------------------------------------

def _crush_build(cell: Cell) -> Launch:
    anthropic = cell.backend.wire == config.WIRE_ANTHROPIC
    cfg = {
        "$schema": "https://charm.land/crush.json",
        "providers": {
            "onelane": {
                "id": "onelane",
                "name": "onelane",
                "type": "anthropic" if anthropic else "openai-compat",
                "base_url": cell.backend.base_url,
                "api_key": cell.key,
                "models": [{"id": cell.model, "name": cell.model,
                            "context_window": 128000,
                            "default_max_tokens": 8192}],
            }
        },
        # Without an explicit models map crush picks whatever provider its
        # catwalk database ranks first and silently ignores ours.
        "models": {"large": {"model": cell.model, "provider": "onelane"},
                   "small": {"model": cell.model, "provider": "onelane"}},
        "options": {"debug": False},
    }
    # CRUSH_GLOBAL_CONFIG names a DIRECTORY, not a file: crush appends
    # crush.json/crushrc to it. Pointing it at the file makes crush look for
    # <file>/crush.json and fail to load any config at all.
    cfg_home = cell.cfg_dir / "crush"
    cfg_home.mkdir(parents=True, exist_ok=True)
    secureio.atomic_write_text(cfg_home / "crush.json", json.dumps(cfg), private=True)
    prompt = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    # `run` is already non-interactive and auto-approving; --yolo is a root-only
    # flag this build rejects on the subcommand.
    # --data-dir keeps crush's own .crush/ state out of the workspace, where it
    # would otherwise show up as a dozen "lines touched" on a read-only task.
    cmd = [_which("crush"), "run", "--quiet", "--cwd", str(cell.workspace),
           "--data-dir", str(cell.cfg_dir / "crushdata"), prompt]
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"CRUSH_GLOBAL_CONFIG": str(cfg_home),
                                  "CRUSH_GLOBAL_DATA": str(cell.cfg_dir / "data"),
                                  "CRUSH_DISABLE_PROVIDER_AUTO_UPDATE": "1",
                                  "CRUSH_DISABLE_METRICS": "1",
                                  "DO_NOT_TRACK": "1"}))


# --------------------------------------------------------------------------
# gptme - gptme-ai/gptme
# --------------------------------------------------------------------------

def _gptme_build(cell: Cell) -> Launch:
    anthropic = cell.backend.wire == config.WIRE_ANTHROPIC
    extra = ({"ANTHROPIC_API_KEY": cell.key,
              "ANTHROPIC_BASE_URL": cell.backend.base_url} if anthropic else
             {"OPENAI_API_KEY": cell.key,
              "OPENAI_BASE_URL": cell.backend.base_url})
    prefix = "anthropic/" if anthropic else "openai/"
    cmd = [_which("gptme"), "-n", "--output-format", "json", "-y",
           "--no-stream", "-w", str(cell.workspace),
           "-m", prefix + cell.model]
    if cell.system:
        cmd += ["--system", cell.system]
    if cell.read_only:
        cmd += ["-t", "read,shell"]
    cmd.append(cell.prompt)
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, extra | {"GPTME_CHECK_UPDATE": "false"}))


def _gptme_parse(stdout: str, stderr: str) -> Result:
    res = Result()
    texts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        role = event.get("role")
        if role == "assistant":
            res.steps += 1
            texts.append(str(event.get("content") or ""))
        elif role == "system":
            res.tool_calls += 1
        if event.get("error"):
            res.error = str(event["error"])[:200]
            res.stopped = "error"
    res.text = texts[-1] if texts else ""
    if not res.text and not res.error:
        return _plain_parse(stdout, stderr)
    return res


# --------------------------------------------------------------------------
# mini-swe-agent - SWE-agent/mini-swe-agent
# --------------------------------------------------------------------------

def _mini_build(cell: Cell) -> Launch:
    """mini routes through litellm, so the lane is (model, api_base, api_key).

    Its trajectory file is the machine-readable output; the answer is the
    submission the agent ends on.
    """
    prefix = "anthropic/" if cell.backend.wire == config.WIRE_ANTHROPIC else "openai/"
    traj = cell.cfg_dir / "traj.json"
    cmd = [_which("mini"), "-y", "-m", prefix + cell.model,
           "-o", str(traj), "-l", "0",
           # -c REPLACES the packaged config, so the default has to be named
           # again or the agent config fails pydantic validation with
           # "system_template Field required".
           "-c", "mini.yaml",
           "-c", f"model.model_kwargs.api_base={cell.backend.base_url}",
           # The task's own step budget, not a hardcoded 20: a cap that differs
           # per harness measures the cap, not the harness (dim 5).
           "-c", f"agent.step_limit={cell.max_steps}",
           "-t", f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt]
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {("ANTHROPIC_API_KEY" if prefix == "anthropic/" else "OPENAI_API_KEY"): cell.key,
                                  "MSWEA_CONFIGURED": "true",
                                  "MSWEA_GLOBAL_CONFIG_DIR": str(cell.cfg_dir),
                                  # custom endpoints are not in litellm's price table,
                                  # and an unpriced model is a hard error
                                  "MSWEA_COST_TRACKING": "ignore_errors",
                                  "LITELLM_LOG": "ERROR"}))


def _mini_parse(stdout: str, stderr: str) -> Result:
    res = _plain_parse(stdout, stderr)
    res.text = res.text or ""
    return res


# --------------------------------------------------------------------------
# droid - factory.ai
# --------------------------------------------------------------------------

def _droid_build(cell: Cell) -> Launch:
    factory_home = cell.cfg_dir / ".factory"
    factory_home.mkdir(parents=True, exist_ok=True)
    secureio.atomic_write_text(factory_home / "settings.json", json.dumps({
        "customModels": [{
            "model": cell.model,
            "displayName": "onelane",
            "baseUrl": cell.backend.base_url,
            "apiKey": cell.key,
            "provider": ("anthropic" if cell.backend.wire == config.WIRE_ANTHROPIC
                         else "generic-chat-completion-api"),
            "maxOutputTokens": 8192,
        }]
    }), private=True)
    prompt = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    cmd = [_which("droid"), "exec", "--output-format", "json",
           "--skip-permissions-unsafe", "--cwd", str(cell.workspace),
           "-m", "custom:onelane-0", prompt]
    return Launch(cmd, cwd=str(cell.workspace), env=_env(cell, {}))


def _droid_parse(stdout: str, stderr: str) -> Result:
    res = Result()
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "assistant":
            res.steps += 1
        if kind in ("tool_use", "tool_call"):
            res.tool_calls += 1
        if kind == "result":
            res.text = str(event.get("result") or event.get("text") or "")
        if event.get("error"):
            res.error = str(event["error"])[:200]
            res.stopped = "error"
    if not res.text and not res.error:
        return _plain_parse(stdout, stderr)
    return res


# --------------------------------------------------------------------------
# hermes - NousResearch/hermes-agent
# --------------------------------------------------------------------------

def _hermes_build(cell: Cell) -> Launch:
    """hermes reads everything out of $HERMES_HOME, so the whole cell fits in a
    private one: a `providers:` entry carries the base URL and the transport
    (it speaks both wires), `.env` carries the key by the name that entry
    names, and SOUL.md is where an identity/system prompt belongs. `-z` is its
    one-shot mode - final text on stdout, no banner and no session line, which
    makes it a prose harness like aider and crush rather than an instrumented
    one."""
    home = cell.cfg_dir / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    transport = ("anthropic_messages"
                 if cell.backend.wire == config.WIRE_ANTHROPIC else "openai_chat")
    secureio.atomic_write_text(home / "config.yaml", "providers:\n"
        "  onelane:\n"
        "    name: onelane\n"
        f"    api: {cell.backend.base_url}\n"
        "    key_env: ONELANE_KEY\n"
        f"    transport: {transport}\n"
        "model:\n"
        f"  default: {cell.model}\n"
        "  provider: onelane\n"
        # Memory and the user profile are per-HOME state that would leak
        # between cells and make a run depend on the order it was run in.
        "memory:\n"
        "  memory_enabled: false\n"
        "  user_profile_enabled: false\n"
        "terminal:\n"
        "  backend: local\n"
        "  cwd: .\n", private=True)
    secureio.atomic_write_text(home / ".env", f"ONELANE_KEY={cell.key}\n", private=True)
    if cell.system:
        secureio.atomic_write_text(home / "SOUL.md", cell.system, private=True)
    # Hermes bundles writes with reads in its file toolset and exposes a
    # shell. Toolset selection alone cannot enforce a read-only cell.
    toolset = "file,terminal,search" if cell.read_only else "coding"
    cmd = [_which("hermes"), "-z", cell.prompt, "--provider", "onelane",
           "-m", cell.model, "-t", toolset, "--yolo", "--ignore-rules"]
    launch = Launch(cmd, cwd=str(cell.workspace),
                    env=_env(cell, {"HERMES_HOME": str(home),
                                    "ONELANE_KEY": cell.key,
                                    "HERMES_ACCEPT_HOOKS": "0",
                                    "DO_NOT_TRACK": "1"}))
    return _read_only_launch(launch, cell) if cell.read_only else launch



# --------------------------------------------------------------------------
# claude - Claude Code itself, pointed at a non-Anthropic endpoint
# --------------------------------------------------------------------------

def _claude_build(cell: Cell) -> Launch:
    """Claude Code's own `-p --output-format json`, one shot per cell.

    Only the Anthropic wire applies: Claude Code speaks `/v1/messages` to
    whatever `ANTHROPIC_BASE_URL` names, and has no OpenAI-wire mode. A
    read-only cell gets `--allowedTools` instead of a second permission mode,
    matching the rest of the arena's one-flag-per-restriction shape.
    """
    # stream-json (not plain json) so _claude_parse can count tool_use blocks
    # and sum usage across every assistant turn: plain json only reports the
    # final result, so tool_calls was always 0 and tokens covered only the
    # last turn.
    cmd = [_which("claude"), "-p", "--output-format", "stream-json", "--verbose",
           "--model", cell.model, "--dangerously-skip-permissions",
           "--max-turns", str(cell.max_steps),
           # MCP servers registered outside the private HOME still load and
           # added 50+ tools to every cell; the other harnesses get none.
           "--strict-mcp-config"]
    if cell.system:
        cmd += ["--system-prompt", cell.system]
    if cell.read_only:
        # --allowedTools only pre-approves a tool for the permission prompt;
        # with --dangerously-skip-permissions there is no prompt to gate, so
        # it does nothing and Write/Edit/Bash stay reachable. --tools is the
        # flag that actually shrinks the tool set. Both flags are variadic
        # ("Comma or space-separated"), so without a `--` terminator either
        # keeps consuming argv and swallows the prompt itself - claude then
        # exits 1 with "no prompt provided".
        cmd += ["--tools", "Read,Grep,Glob", "--"]
    cmd.append(cell.prompt)
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"ANTHROPIC_BASE_URL": cell.backend.base_url,
                                  "ANTHROPIC_AUTH_TOKEN": cell.key,
                                  "ANTHROPIC_MODEL": cell.model}))


def _claude_parse(stdout: str, stderr: str) -> Result:
    """Parse `--output-format stream-json`: one JSON object per line, an
    `assistant` event per content block (its `message.content` holds the
    tool_use blocks), and a closing `result` event with the final answer,
    `num_turns` and the run's usage totals.

    Token counts come from the `result` event's `usage`. The per-block
    `assistant` usage is the message_start snapshot: `output_tokens` is 0
    and one message repeats it once per block, so summing it undercounts
    output and overcounts input. It is only a fallback, deduplicated by
    message id, when the `result` event carries no usage."""
    text = stdout.strip()
    if not text:
        return Result(error=stderr.strip()[-200:], stopped="crash")
    tool_calls = 0
    turn_usage: dict[str, dict] = {}
    result_event: dict | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "assistant":
            message = event.get("message") or {}
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls += 1
            if message.get("usage"):
                key = message.get("id") or f"turn-{len(turn_usage)}"
                turn_usage[key] = message["usage"]
        elif event.get("type") == "result":
            result_event = event
    if result_event is None:
        return _plain_parse(stdout, stderr)
    res = Result(text=str(result_event.get("result") or ""),
                 steps=int(result_event.get("num_turns") or 0),
                 tool_calls=tool_calls,
                 stopped="error" if result_event.get("is_error") else "answered")
    if result_event.get("is_error"):
        res.error = str(result_event.get("result") or "")[:200]
    usages = [result_event["usage"]] if result_event.get("usage") else list(turn_usage.values())
    if usages:
        for field in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            res.cost[field] = sum(int(u.get(field) or 0) for u in usages)
    return res


# --------------------------------------------------------------------------
# mimo-code - Xiaomi's own coding agent (@mimo-ai/cli, XiaomiMiMo/MiMo-Code)
# --------------------------------------------------------------------------
# An opencode fork: same $schema config shape, same npm provider packages, same
# `run --format json --dir <ws>` one-shot mode.

def _mimocode_build(cell: Cell) -> Launch:
    # Only the OpenAI-compatible custom-provider shape is documented
    # (mimo.xiaomi.com/docs, XiaomiMiMo/MiMo-Code README); tried on the
    # Anthropic wire too and it broke: mimo-code posted to
    # "<base>/chat/completions" (the OpenAI-compatible path glued onto our
    # Anthropic base_url) instead of "<base>/v1/messages", a 404 every time
    # (AI_APICallError). Declared OPENAI-only rather than
    # scored as a loss on a wire it was never built to speak.
    provider = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "onelane",
        "options": {"baseURL": cell.backend.base_url, "apiKey": cell.key},
        "models": {cell.model: {"name": cell.model}},
    }
    cfg: dict = {"$schema": "https://mimo.xiaomi.com/mimocode/config.json",
                 "provider": {"onelane": provider}}
    if cell.read_only:
        # No dedicated read-only flag on `mimo run`; --dangerously-skip-permissions
        # is all-or-nothing (like claude's), so a read-only cell instead denies
        # every mutating permission in config and skips that flag, matching the
        # opencode fork's own permission schema.
        cfg["permission"] = {"edit": "deny", "bash": "deny", "webfetch": "deny"}
    home = cell.cfg_dir / "mimocode"
    (home / "config").mkdir(parents=True, exist_ok=True)
    secureio.atomic_write_text(home / "config" / "mimocode.jsonc", json.dumps(cfg), private=True)
    prompt = f"{cell.system}\n\n{cell.prompt}" if cell.system else cell.prompt
    # `mimo`'s shebang is `#!/usr/bin/env node`. The arena prefix can contain
    # a broken `node` symlink, so the interpreter is called directly, as for
    # kilo, and the CLI's own PATH lookup is skipped.
    cmd = [NODE_BIN, _which(MIMOCODE_BIN), "run", "--format", "json", "--dir", str(cell.workspace),
           "-m", f"onelane/{cell.model}"]
    if not cell.read_only:
        cmd.append("--dangerously-skip-permissions")
    cmd.append(prompt)
    return Launch(cmd, cwd=str(cell.workspace),
                  env=_env(cell, {"MIMOCODE_HOME": str(home)}))


def _mimocode_parse(stdout: str, stderr: str) -> Result:
    res = Result()
    input_tokens = output_tokens = 0
    saw_tokens = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        part = event.get("part") or {}
        etype = event.get("type") or part.get("type")
        if etype == "text":
            res.text += part.get("text", "")
        elif etype in ("step_finish", "step-finish"):
            # One event per agent step; sum across every step in the run
            # rather than keeping only the last, which undercounted total
            # usage (e.g. 1,275 input tokens reported for a 3-task run).
            tokens = part.get("tokens")
            if tokens:
                saw_tokens = True
                input_tokens += int(tokens.get("input") or 0)
                output_tokens += int(tokens.get("output") or 0)
        elif etype in ("error", "session.error"):
            res.error = json.dumps(part)[:200]
            res.stopped = "error"
    if saw_tokens:
        res.cost["input_tokens"] = input_tokens
        res.cost["output_tokens"] = output_tokens
    if not res.text and not res.error:
        res.error = (stderr or stdout).strip()[-200:]
        res.stopped = "crash" if res.error else res.stopped
    return res


HARNESSES: dict[str, Harness] = {
    "one": Harness("one", "", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                   _one_build, _one_parse, "this repo",
                   _parity(
                       system=(MATCHED, "sent in the system role via --system"),
                       step=(MATCHED, "--max-steps cell.max_steps"),
                       timeout=(MATCHED, "--timeout cell.timeout"),
                       tokens=(MATCHED, "--max-tokens 8192 (the one value for all)"),
                       thinking=(UNCAPPED, "no reasoning flag pinned; backend default"),
                       ambient=(MATCHED, "private HOME, empty ONE_PROVIDERS_DIR, discovery bypassed"))),
    "pi": Harness("pi", "pi", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                  _pi_build, _pi_parse, "earendil-works/pi",
                  _parity(
                      system=(MATCHED, "sent in the system role via --system-prompt"),
                      step=(UNCAPPED, "pi -p exposes no step-limit flag"),
                      timeout=(UNCAPPED, "no internal timeout flag; bounded by the arena subprocess kill"),
                      tokens=(UNCAPPED, "no output-token cap flag"),
                      thinking=(UNCAPPED, "no thinking flag"),
                      ambient=(MATCHED, "private HOME + PI_CODING_AGENT_DIR + PI_OFFLINE"))),
    "opencode": Harness("opencode", "opencode",
                        (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                        _opencode_build, _opencode_parse, "sst/opencode",
                        _parity(
                            system=(ABSENT, "no system-role flag on `run`; prepended to the user turn"),
                            step=(UNCAPPED, "no step-limit flag"),
                            timeout=(UNCAPPED, "no internal timeout; bounded by the arena subprocess kill"),
                            tokens=(UNCAPPED, "no output-token cap"),
                            thinking=(UNCAPPED, "no thinking flag"),
                            ambient=(MATCHED, "private HOME + OPENCODE_CONFIG, --pure"))),
    "kilo": Harness("kilo", KILO_BIN,
                    (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                    _kilo_build, _kilo_parse, "Kilo-Org/kilocode (OpenCode fork)",
                    _parity(
                        system=(UNCAPPED, "agent.code.prompt is in the system role, but Kilo adds its own system instructions"),
                        step=(MATCHED, "agent.code.steps = cell.max_steps"),
                        timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                        tokens=(MATCHED, "model limit.output = 8192"),
                        thinking=(UNCAPPED, "provider default"),
                        ambient=(MATCHED, "private HOME, KILO_CONFIG, project config disabled, --pure"))),
    "openclaw": Harness("openclaw", "openclaw",
                        (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                        _openclaw_build, _openclaw_parse,
                        "openclaw/openclaw 2.0 - agent exec, needs node >=24.15",
                        _parity(
                            system=(UNCAPPED, "no way to set the system role: agent exec has no --system flag, and openclaw builds its own system prompt every run with no config knob to replace or suppress it (concepts/system-prompt.md). Ours is prepended to the user turn and openclaw's stays underneath - a PERMANENT asymmetry, not a defect we can close"),
                            step=(UNCAPPED, "agent exec has no step or turn flag"),
                            timeout=(MATCHED, "--timeout cell.timeout"),
                            tokens=(MATCHED, "maxTokens 8192 on the model config entry (agent exec has no --max-tokens flag)"),
                            thinking=(UNCAPPED, "--thinking exists but no single level is accepted by every model (some accept only `off`); pinning `off` would disable reasoning on openclaw while one runs at model default, a new asymmetry. Left at per-model default, symmetric with one's backend-default"),
                            ambient=(MATCHED, "private HOME + OPENCLAW_STATE_DIR, models.mode=replace, sandbox off"))),
    "qwen": Harness("qwen", "qwen", (config.WIRE_OPENAI,),
                    _qwen_build, _plain_parse, "@qwen-code/qwen-code",
                    _parity(
                        system=(ABSENT, "no system-role flag; prepended to the user turn"),
                        step=(UNCAPPED, "no step-limit flag"),
                        timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                        tokens=(UNCAPPED, "no output-token cap"),
                        thinking=(UNCAPPED, "no thinking flag"),
                        ambient=(ABSENT, "isolate_home=False: reads the operator's ~/.qwen settings"))),
    # codex removed `wire_api = "chat"` for custom providers ("no longer
    # supported ... set wire_api = responses"), so an endpoint that does not
    # serve /v1/responses cannot drive it. It cannot then be pointed at the same
    # endpoint as the others, which is a fact about codex rather than a score: it is skipped
    # with that reason instead of being counted as a loss. The adapter is ready
    # for the day a lane speaks the Responses wire.
    "codex": Harness("codex", "codex", ("responses",),
                     _codex_build, _plain_parse,
                     "@openai/codex - custom providers need the Responses wire",
                     _parity(
                         system=(ABSENT, "no system-role flag; prepended to the user turn"),
                         step=(UNCAPPED, "no step-limit flag"),
                         timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                         tokens=(UNCAPPED, "no output-token cap"),
                         thinking=(UNCAPPED, "no thinking flag"),
                         ambient=(MATCHED, "private HOME + CODEX_HOME"))),
    "aider": Harness("aider", "aider", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                     _aider_build, _plain_parse,
                     "Aider-AI/aider - prose only, steps not instrumented",
                     _parity(
                         system=(ABSENT, "no system-role flag; prepended to the --message body"),
                         step=(UNCAPPED, "no step-limit flag; single --message shot"),
                         timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                         tokens=(UNCAPPED, "no output-token cap"),
                         thinking=(UNCAPPED, "no thinking flag"),
                         ambient=(MATCHED, "private HOME, --no-git/--no-analytics"))),
    # kimi-cli on the cell's endpoint. `--quiet` gives prose, so steps and tool
    # calls are uninstrumented here exactly as they are for aider and crush -
    # stated in the parity block, not left for a reader to discover.
    "kimi": Harness("kimi", "kimi", (config.WIRE_OPENAI,),
                    _kimi_build, _plain_parse,
                    "Moonshot kimi-cli - prose only, steps not instrumented",
                    _parity(
                        system=(ABSENT, "no system-role flag; prepended to the prompt"),
                        step=(UNCAPPED, "no step-limit flag on --print"),
                        timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                        tokens=(UNCAPPED, "no output-token cap"),
                        thinking=(UNCAPPED, "no thinking flag"),
                        ambient=(MATCHED, "private HOME, KIMI_BASE_URL + KIMI_API_KEY, "
                                           "KIMI_MODEL_NAME unset so -m is authoritative"))),
    "goose": Harness("goose", "goose", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                     _goose_build, _goose_parse, "block/goose",
                     _parity(
                         system=(MATCHED, "sent in the system role via --system"),
                         step=(MATCHED, "--max-turns cell.max_steps"),
                         timeout=(UNCAPPED, "no internal timeout; bounded by the arena subprocess kill"),
                         tokens=(UNCAPPED, "no output-token cap flag"),
                         thinking=(UNCAPPED, "no thinking flag"),
                         ambient=(MATCHED, "private HOME, --no-session, keyring disabled"))),
    "crush": Harness("crush", "crush", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                     _crush_build, _plain_parse,
                     "charmbracelet/crush - prose only, steps not instrumented",
                     _parity(
                         system=(ABSENT, "no system-role flag; prepended to the user turn"),
                         step=(UNCAPPED, "no step-limit flag"),
                         timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                         tokens=(MATCHED, "default_max_tokens 8192 in the provider model config"),
                         thinking=(UNCAPPED, "no thinking flag"),
                         ambient=(MATCHED, "private HOME + CRUSH_GLOBAL_CONFIG/DATA"))),
    "gptme": Harness("gptme", "gptme", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                     _gptme_build, _gptme_parse, "gptme-ai/gptme",
                     _parity(
                         system=(MATCHED, "sent in the system role via --system"),
                         step=(UNCAPPED, "no step-limit flag"),
                         timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                         tokens=(UNCAPPED, "no output-token cap flag"),
                         thinking=(UNCAPPED, "no thinking flag"),
                         ambient=(MATCHED, "private HOME, -n non-interactive"))),
    "mini": Harness("mini", "mini", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                    _mini_build, _mini_parse,
                    "SWE-agent/mini-swe-agent - litellm lane",
                    _parity(
                        system=(ABSENT, "no system-role flag; prepended to the -t task body"),
                        step=(MATCHED, "agent.step_limit=cell.max_steps"),
                        timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                        tokens=(UNCAPPED, "no output-token cap"),
                        thinking=(UNCAPPED, "no thinking flag"),
                        ambient=(MATCHED, "private HOME + MSWEA_GLOBAL_CONFIG_DIR"))),
    "droid": Harness("droid", "droid", (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                     _droid_build, _droid_parse,
                     "factory.ai droid - BYOK custom model, needs a Factory login",
                     _parity(
                         system=(ABSENT, "no system-role flag; prepended to the user turn"),
                         step=(UNCAPPED, "no step-limit flag"),
                         timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                         tokens=(MATCHED, "maxOutputTokens 8192 in the custom model settings"),
                         thinking=(UNCAPPED, "no thinking flag"),
                         ambient=(MATCHED, "private HOME + .factory settings"))),
    "hermes": Harness("hermes", "hermes",
                      (config.WIRE_OPENAI, config.WIRE_ANTHROPIC),
                      _hermes_build, _plain_parse,
                      "NousResearch/hermes-agent - prose only, steps not instrumented",
                      _parity(
                          system=(MATCHED, "system delivered via the SOUL.md identity file"),
                          step=(UNCAPPED, "no step-limit flag on -z one-shot"),
                          timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                          tokens=(UNCAPPED, "no output-token cap"),
                          thinking=(UNCAPPED, "no thinking flag"),
                          ambient=(MATCHED, "private HERMES_HOME, memory/profile disabled"))),
    "claude": Harness("claude", "claude", (config.WIRE_ANTHROPIC,),
                      _claude_build, _claude_parse,
                      "Claude Code itself, pointed at ANTHROPIC_BASE_URL",
                      _parity(
                          system=(MATCHED, "sent via --system-prompt, replacing the default"),
                          step=(MATCHED, "--max-turns cell.max_steps"),
                          timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                          tokens=(UNCAPPED, "no output-token cap flag"),
                          thinking=(UNCAPPED, "no thinking flag pinned; backend default"),
                          ambient=(MATCHED, "private HOME, --strict-mcp-config, --dangerously-skip-permissions"))),
    "mimo-code": Harness("mimo-code", MIMOCODE_BIN, (config.WIRE_OPENAI,),
                         _mimocode_build, _mimocode_parse,
                         "XiaomiMiMo/MiMo-Code (@mimo-ai/cli) - Xiaomi's own "
                         "coding agent, an opencode fork; resolved from "
                         "MIMOCODE_BIN or PATH; OpenAI wire only - the "
                         "Anthropic custom-provider path 404s (see _mimocode_build)",
                         _parity(
                             system=(ABSENT, "no system-role flag on `run`; prepended to the user turn"),
                             step=(UNCAPPED, "no step-limit flag"),
                             timeout=(UNCAPPED, "bounded by the arena subprocess kill"),
                             tokens=(UNCAPPED, "no output-token cap flag"),
                             thinking=(UNCAPPED, "--variant selects per-provider reasoning effort; not pinned"),
                             ambient=(MATCHED, "private MIMOCODE_HOME + --dir cell.workspace"))),
}

# Harnesses that were surveyed and cannot be driven the way this arena drives
# one. Each entry is a fact about the harness, established from its own docs or
# from running it - never a zero score. The inline reasons identify the
# provider, authentication or endpoint constraint that prevents comparison.
SKIPPED: dict[str, str] = {
    "amp": "sourcegraph/amp - `amp --help` offers login/logout only; the model "
           "lane is Amp's own server, no custom base URL exists",
    "auggie": "@augmentcode/auggie - has --print/--output-format json, but auth "
              "is an Augment OAuth session (AUGMENT_SESSION_AUTH) against "
              "Augment's endpoint; no custom base URL",
    "cursor-agent": "cursor-agent - the base-URL override is an IDE setting for "
                    "Cursor's own backend, not a CLI provider config",
    "gemini": "@google/gemini-cli - headless (-p, -o json) but the wire is "
              "Gemini/Vertex/Code Assist; it cannot speak our OpenAI or "
              "Anthropic lanes",
    "grok": "@vibe-kit/grok-cli - does not start on current Node: its ink "
            "renderer fails inside react-reconciler before any prompt runs",
    "forge": "@antinomyhq/forge - `forge -p` does not complete headless: it "
             "no longer reads API keys from environment variables and needs a "
             "tty to migrate them, so a cell cannot hand it an endpoint key",
    "octofriend": "synthetic-lab/octofriend - `octofriend --config <f> "
                  "--unchained prompt --model <m> <prompt>` produces no output "
                  "and never exits; the adapter is kept for the day its prompt "
                  "subcommand terminates",
    "plandex": "plandex - the CLI is a client for a plandex server; a cell "
               "would measure that server, not the harness",
    "swe-agent": "SWE-agent - a benchmark runner over problem statements and "
                 "container environments, not a general-purpose CLI; its "
                 "single-agent CLI lane is mini-swe-agent, which IS adapted",
}

"""one - a terminal coding agent that speaks to model endpoints directly.

No vendor CLI in the path: one-cli speaks to a model endpoint directly and runs
its own tool loop, so every model you point it at gets the same agent, the same
tools and the same transcript format.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import subprocess
import sys
import time
from typing import Any

from onecli import __version__, config, context, fallback, media, presence, providers, render, session, tui, unavailable
from onecli.serving import cli as serving_cli
from onecli import tools as toolkit
from onecli.tools import shell as shelltool
from onecli import guard
from onecli import spinner
from onecli import hooks as hooklib
from onecli import mcp as mcplib
from onecli import orc as orclib
from onecli import manager as managerlib
from onecli import permissions
from onecli import ask as asklib
from onecli import jobs as joblib
from onecli import todo as todolib
from onecli import loopguard as loops
from onecli import spill as spilllib
from onecli import drivers
from onecli import commands as cmdlib
from onecli import commandfiles
from onecli import inputline
from onecli import mentions
from onecli import events
from onecli import subagent as sublib
from onecli import senses
from onecli import update as updatelib
from onecli.senses import base as senses_base
from onecli.agent import Agent, DEFAULT_SYSTEM

READ_ONLY_TOOLS = permissions.READ_ONLY_TOOLS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="one",
        description="Terminal coding agent for any OpenAI- or Anthropic-compatible model.",
    )
    p.add_argument("prompt", nargs="*", help="The task. Omit for interactive mode.")
    p.add_argument("-p", "--print", dest="print_mode", action="store_true",
                   help="Non-interactive mode: print prompt answer and exit.")
    p.add_argument("--feedback-fd", type=int, default=None, help="Inherited private feedback socket for headless turns.")
    p.add_argument("--mask", help="Named task mask; use --masks to list, off to disable.")
    p.add_argument("--masks", action="store_true", help="List task masks without contacting a provider.")
    p.add_argument("-m", "--model", help="Backend name or model id (see --list).")
    p.add_argument("--ui", "--theme", dest="theme", default=os.environ.get("ONE_CLI_THEME", ""),
                   help="Terminal animation: braid-stream, patternflow-port, orb-pulse, scan-line, equalizer; also accepts legacy banner themes.")
    p.add_argument("--orchestrator", metavar="NAME", default="",
                   help="Run the top-level agent on NAME and send every "
                        "unpinned subagent to --model instead. 'none' (the "
                        "default) runs everything on --model.")
    p.add_argument("--orc", metavar="NAME|PATH", default="",
                   help="An orchestration config: which backend drives and "
                        "which backend each subagent preset runs on. A name is "
                        "looked for in ./.orc.json and ~/.one-cli/orc.json; a "
                        "path is read directly. Explicit flags still win.")
    p.add_argument("--orc-list", action="store_true",
                   help="Print every orchestration config this run could "
                        "resolve - name, orchestrator, worker pins and the "
                        "file it came from - then exit.")
    p.add_argument("--orc-show", metavar="NAME|PATH", default="",
                   help="Print one orchestration config in full, resolved "
                        "against the presets this run would load, then exit. "
                        "A `workers` entry naming a preset that does not exist "
                        "is flagged rather than silently ignored.")
    p.add_argument("--orc-report", action="store_true",
                   help="Summarise every run record in ~/.one-cli/runs and "
                        "exit - median wall time, steps and tokens grouped by "
                        "backend and task class, with the failure-mode "
                        "counters beside them. Runs no model.")
    from onecli.effort import LEVELS
    p.add_argument("--effort", choices=LEVELS, default="auto",
                   help="Reasoning effort via backend-declared profiles; auto uses task policy.")
    p.add_argument("--task-class", metavar="NAME", default="",
                   help="What KIND of work this run is, recorded on the run "
                        "record so runs can be grouped later. Free-form; "
                        "research, build, review, refactor and debug are "
                        "suggestions, not a list. Beats a `task_class` in the "
                        "orc config.")
    p.add_argument("--phases", metavar="PATH|JSON", default="",
                   help="Drive this run off a fixed checklist instead of one "
                        "prompt: a JSON list of phases, as a path or inline. "
                        "One implementer is re-steered phase by phase and each "
                        "phase lands on the run record with its own start, "
                        "end, verdict and cost.")
    p.add_argument("--phase-mode", choices=managerlib.MODES,
                   default=managerlib.DEFAULT_MODE,
                   help="How the checklist is worked: 'loop' re-steers ONE "
                        "long-lived implementer with its history carried "
                        "forward, 'fanout' spawns a fresh one-shot child per "
                        "phase. The two modes are measured as a paired "
                        "comparison.")
    p.add_argument("--phase-wording", choices=sorted(managerlib.WORDINGS),
                   default=managerlib.DEFAULT_WORDING,
                   help="Which sentence each phase is asked with. Recorded on "
                        "every phase, because a claim about prompt wording "
                        "that cannot be told apart afterwards is untested.")
    p.add_argument("--phase-agent", metavar="NAME", default="general",
                   help="The preset the implementer runs as. Its pin decides "
                        "the backend, exactly as a task tool call would.")
    p.add_argument("--orc-explain", action="store_true",
                   help="Print the resolved orchestration config in full - "
                        "every preset, the backend it landed on, which rung "
                        "decided it, and the file it came from.")
    p.add_argument("--model-id",
                   help="Override the model id on the resolved backend, so one "
                        "endpoint can serve several models (a provider's tiers).")
    p.add_argument("--base-url", help="Explicit endpoint, bypassing discovery.")
    p.add_argument("--wire", choices=[config.WIRE_OPENAI, config.WIRE_ANTHROPIC],
                   help="Which protocol the endpoint speaks. Default openai.")
    p.add_argument("-w", "--workspace", default=os.getcwd(),
                   help="Directory the agent works in. Default cwd.")
    p.add_argument("-s", "--max-steps", type=int, default=None,
                   help="Tool-loop budget. Default: no limit.")
    p.add_argument("--response-gate", choices=("off", "prompt", "enforce"),
                   default=None, help="First-response policy; default enforce for one CLI.")
    p.add_argument("--meditate", action="store_true",
                   help="Review the draft once using read-only evidence within the existing step budget.")
    p.add_argument("--progress-seconds", type=float, default=None)
    p.add_argument("--progress-tool-calls", type=int, default=None)
    p.add_argument("--no-progress", action="store_true", help="Disable progress checkpoints.")
    p.add_argument("--soft-turn-seconds", type=float, default=None,
                   help="Begin a final answer at the next model-step boundary after this many seconds.")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Output ceiling per model call. Default: none is sent, "
                        "so the provider allows the model's own maximum.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature. Default: none is sent, so the "
                        "provider's own default for the model applies.")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Per-request timeout in seconds.")
    p.add_argument("--read-only", action="store_true",
                   help="Withhold write, edit and bash from the model, and "
                        "refuse them at call time if it names one anyway.")
    p.add_argument("--guard", action="append", metavar="CMD", default=[],
                   help="Program consulted before every tool call. Reads the "
                        "call as JSON on stdin; exit 2 blocks it and stderr is "
                        "the reason. Uses Claude Code PreToolUse payloads; "
                        "only exit zero allows execution. "
                        "Repeatable; the first block wins.")
    p.add_argument("--guard-timeout", type=float, default=guard.DEFAULT_TIMEOUT,
                   metavar="SECONDS",
                   help=f"Per-guard time limit. Default {guard.DEFAULT_TIMEOUT}. "
                        "A guard that overruns fails closed and is reported.")
    p.add_argument("--ask", action=argparse.BooleanOptionalAction, default=None,
                   help="Let the model ask you a question when it hits a fork "
                        "it cannot resolve. Default on in interactive mode, "
                        "off otherwise - a one-shot run has nobody to ask.")
    p.add_argument("--ask-timeout", type=float, default=asklib.TIMEOUT,
                   metavar="SECONDS",
                   help=f"How long a question waits for you. Default "
                        f"{asklib.TIMEOUT:.0f}. On a timeout the model is told "
                        "to decide alone and say what it assumed.")
    p.add_argument("--repeat-limit", type=int, default=loops.REPEAT_LIMIT,
                   metavar="N",
                   help="Identical tool calls before the model is told it is "
                        f"repeating itself. Default {loops.REPEAT_LIMIT}; 0 "
                        "turns the reminder off. Advisory - it never blocks.")
    p.add_argument("--discovery-limit", type=int, default=loops.DISCOVERY_LIMIT,
                   metavar="N",
                   help="Consecutive tool rounds with no write, edit, question "
                        "or spawned work before the model must commit to a "
                        "target or ask one clarifying question. Default "
                        f"{loops.DISCOVERY_LIMIT}; 0 turns the checkpoint off.")
    p.add_argument("--tool-deadline", type=float, default=loops.DEADLINE,
                   metavar="SECONDS",
                   help="Per-call ceiling. Lowered into any tool that declares "
                        "a timeout, and reported as an advisory for one that "
                        f"sets its own clock. Default {loops.DEADLINE:g}s; 0 disables.")
    p.add_argument("--image", action="append", metavar="PATH", default=[],
                   help="Attach an image to the turn. Repeatable.")
    p.add_argument("--audio", metavar="PATH",
                   help="Attach an audio file (a voice note) to the turn.")
    p.add_argument("--video", metavar="PATH",
                   help="Attach a video to the turn. No wire we speak carries "
                        "video, so the Senses Eye perceives it as text.")
    p.add_argument("--media-route", default="auto",
                   choices=list(senses_base.ROUTES), metavar="ROUTE",
                   help="How attachments reach the model: auto (inline when "
                        "the backend has the sense, else perceive it as text), "
                        "inline, describe, transcribe, off. Default auto.")
    p.add_argument("--no-media-fallback", dest="media_fallback",
                   action="store_false", default=True,
                   help="Take the first rung of a media ladder or fail. Without "
                        "it a route climbs down to the next rung.")
    p.add_argument("--inline-audio", action="store_true",
                   help="Send raw audio to a backend that declares it, instead "
                        "of the transcript. The transcript is the default "
                        "because it is the path proven on every backend.")
    p.add_argument("--speak", nargs="?", const="", metavar="VOICE",
                   help="Synthesize the final answer through pocket-tts and "
                        f"print the file path. Default voice "
                        f"{senses.lip.DEFAULT_VOICE}.")
    p.add_argument("--autocompact", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Summarize the oldest messages away when the history "
                        "nears the model's context window. On by default.")
    p.add_argument("--compact-at", type=float, default=1.0, metavar="FRACTION",
                   help="Fraction of the context window that triggers a "
                        "compaction. Default 1.0: only when the next request "
                        "would not fit the model's window.")
    p.add_argument("--context-window", type=int, default=0, metavar="TOKENS",
                   help="Override the model's context window. Default is the "
                        "measured value for known models, else "
                        f"{context.DEFAULT_CONTEXT_WINDOW}.")
    p.add_argument("--compact-keep", type=int, default=6, metavar="N",
                   help="Messages kept verbatim after a compaction. The real "
                        "cut is walked back from here so no tool result loses "
                        "the assistant turn that called it. Default 6.")
    p.add_argument("--max-budget-usd", type=float, default=0.0, metavar="USD",
                   help="Stop the loop once the run has cost this much. Only "
                        "meaningful for a model with a known price.")
    p.add_argument("--max-input-tokens", type=int, default=None, metavar="N",
                   help="Maximum cumulative metered input across the run "
                        "(cache reads at their price ratio, default 0.1). "
                        "Default: no cumulative input ceiling; 0 disables.")
    p.add_argument("--require-priced-budget", action="store_true",
                   help="Reject missing/invalid prices or budgets before model execution. "
                        "Requires --no-fallback. This validates the existing "
                        "post-response budget, not a hard spending ceiling.")
    p.add_argument("--request-budget-policy", metavar="FILE",
                   help="Operator-audited text request tariff/bounds. Requires a durable ledger.")
    p.add_argument("--request-budget-ledger", metavar="FILE",
                   help="Shared durable request reservations; reopening never resets budget.")
    p.add_argument("--request-budget-isolated", action="store_true",
                   help="Tool-free one-shot request-budget profile; disables hooks, MCP and children.")
    p.add_argument("--request-budget-sandbox-grant", metavar="FILE",
                   help="Host-owned CodingGrant JSON for bounded offline tools; requires isolated budget mode.")
    p.add_argument("--create-request-budget-ledger", action="store_true",
                   help="Explicitly initialize request budget storage for a new work item.")
    p.add_argument("--fallback-model", dest="fallback_model", action="append",
                   metavar="NAMES", default=None,
                   help="The chain a turn walks down when a backend is "
                        "overloaded, rate-limited or unreachable. Comma "
                        "separated and repeatable; `auto` (the default) ranks "
                        "what discovery already found, `off` pins the run to "
                        "one backend. A model declining the task is not a "
                        "failure and never spends a rung.")
    p.add_argument("--on-unavailable-model", dest="on_unavailable_model",
                   choices=unavailable.MODES, default=None,
                   help="When the model named with -m is offline or fails: "
                        "`ask` (default; a terminal offers online alternatives, "
                        "otherwise as `error`), `error` (exit 2 with the "
                        "alternatives), or `fallback` (switch to the first online "
                        f"alternative and say so). Host default: {unavailable.MODE_ENV}.")
    p.add_argument("--fallback-allow-paid", dest="fallback_allow_paid",
                   action="store_true", default=False,
                   help="Let `--on-unavailable-model fallback` switch to a paid "
                        f"lane. Host default: {unavailable.ALLOW_PAID_ENV}=1.")
    p.add_argument("--no-fallback", dest="fallback_model",
                   action="store_const", const=["off"],
                   help="Stay on one backend. What a benchmark wants: a cell "
                        "that answers on a spare is measuring the wrong model.")
    p.add_argument("--permission-mode", choices=list(permissions.MODES),
                   help="How careful this run is. `default` runs every tool; "
                        "`acceptEdits` is the same surface (one-cli never "
                        "stops to ask); `plan` withholds every mutating tool "
                        "and asks for a plan instead of edits; "
                        "`bypassPermissions` skips the allow/deny lists but "
                        "still runs --guard; `readOnly` is --read-only.")
    p.add_argument("--dangerously-skip-permissions", action="store_true",
                   help="Skip permission prompts; alias for --permission-mode bypassPermissions.")
    p.add_argument("--allowed-tools", "--allowedTools", dest="allowed_tools",
                   action="append", metavar="PATTERN", default=[],
                   help="Tool patterns the model may call, Claude Code syntax: "
                        "Read, 'Bash(git *)', 'Write(src/**)'. Repeatable and "
                        "comma-separated. A non-empty list is a whitelist.")
    p.add_argument("--disallowed-tools", "--disallowedTools",
                   dest="disallowed_tools", action="append",
                   metavar="PATTERN", default=[],
                   help="Tool patterns that are refused. Beats --allowed-tools "
                        "on a call that matches both.")
    p.add_argument("--add-dir", action="append", metavar="DIR", default=[],
                   help="An extra directory the file tools may reach. "
                        "Repeatable. Passing any of these turns the workspace "
                        "plus these directories into the only readable roots.")
    p.add_argument("--no-project-context", action="store_true",
                   help="Do not discover and append CLAUDE.md files.")
    p.add_argument("--done-check", default="", metavar="CMD",
                   help="Shell command, run in the workspace, that must exit 0 "
                        "before a final reply counts as done. A failure is fed "
                        "back to the model (3 times, reset by a tool call); "
                        "after that the run ends as unverified, exit 1.")
    p.add_argument("--hooks", action="append", metavar="PATH", default=[],
                   help="Extra Claude Code shaped settings.json to read hooks "
                        "from. User settings are also read. Named files are "
                        "trusted executable configuration. Repeatable.")
    p.add_argument("--trust-project-config", action="store_true",
                   help="Allow this workspace's .claude/settings*.json hooks "
                        "and .mcp.json servers to execute on the host for this "
                        "run. Only use after inspecting and trusting the project.")
    p.add_argument("--mcp-config", action="append", metavar="PATH|JSON",
                   default=[],
                   help="MCP servers to connect, as a Claude Code shaped "
                        "settings file or an inline JSON object. Repeatable. "
                        "Named configs may execute commands. Project .mcp.json "
                        "requires --trust-project-config; --strict-mcp-config "
                        "still excludes it.")
    p.add_argument("--strict-mcp-config", action="store_true",
                   help="Use only the servers named by --mcp-config, ignoring "
                        "the project's .mcp.json.")
    p.add_argument("--agents", action="append", metavar="PATH", default=[],
                   help="A JSON file of named subagent presets, each with a "
                        "description, an optional system prompt, an optional "
                        "tool list and an optional max_steps. Repeatable. The "
                        "built-in 'general' and 'explorer' are always present.")
    p.add_argument("--agent", metavar="NAME",
                   help="Run this session AS a named preset: its system prompt "
                        "and its tool surface. The preset can only narrow what "
                        "the run already had.")
    p.add_argument("--list-agents", action="store_true",
                   help="Print the subagents this run would offer (built-ins "
                        "plus every --agents file) and the vendor CLIs a "
                        "preset can name as its `driver`, then exit.")
    p.add_argument("--max-depth", type=int, default=sublib.DEFAULT_MAX_DEPTH,
                   metavar="N",
                   help=f"How deep subagents may nest. Default "
                        f"{sublib.DEFAULT_MAX_DEPTH}; 0 withholds the task "
                        f"tool entirely.")
    p.add_argument("--no-hooks", action="store_true",
                   help="Run no hooks at all, whatever the settings files say. "
                        "--guard is unaffected.")
    p.add_argument("--system", help="Replace the system prompt.")
    p.add_argument("--system-file", metavar="PATH",
                   help="Replace the system prompt with the contents of a "
                        "file. For a parent process with a large prompt to "
                        "inject: an argv string is capped at 128KB by the "
                        "kernel, a file is not. Refused alongside --system.")
    p.add_argument("-r", "--resume", nargs="?", const="@last",
                   help="Continue a session. Bare -r resumes the newest.")
    p.add_argument("--json", action="store_true",
                   help="Emit one JSON object on stdout instead of prose.")
    p.add_argument("--stream", dest="stream", action="store_true", default=None,
                   help="Print the answer as it arrives. Default on when stdout "
                        "is a terminal and neither --json nor --quiet is set. "
                        "With --json, stream transport progress on stderr while "
                        "keeping one final JSON object on stdout.")
    p.add_argument("--no-stream", dest="stream", action="store_false",
                   help="Wait for the whole turn before printing anything.")
    p.add_argument("--text-events", action="store_true",
                   help="With --json --stream, include public answer text deltas "
                        "on stderr for a streaming UI. Never includes reasoning.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Suppress the step trace; print only the answer.")
    p.add_argument("--plain", action="store_true",
                   help="No colour, no box-drawing. Same as NO_COLOR=1. "
                        "Already the default when stdout is not a terminal.")
    p.add_argument("-c", "--chat", action="store_true",
                   help="Chat instead of work: a question and an answer, with "
                        "a tool loop underneath it rather than a run around "
                        "it. Read tools run freely; bash asks before running. "
                        "`one-chat` is the same session with a tenth of the "
                        "startup, and takes --tools.")
    p.add_argument("--list", action="store_true", help="List reachable backends.")
    p.add_argument("--session-search", default="", help="Filter saved sessions by name or opening prompt.")
    p.add_argument("--sessions-here", action="store_true", help="List sessions in the selected workspace.")
    p.add_argument("--sessions", action="store_true", help="List saved sessions.")
    p.add_argument("--version", action="version", version=f"one-cli {__version__}")
    # Resolved in main() from --image/--audio/--video. Defaulted here so a
    # parsed Namespace always carries the field, whoever built it.
    # Resolved in main() by the Senses router alongside the attachments.
    p.set_defaults(attachments=[], media_preamble="", media_language="")
    return p


def cmd_list(args=None) -> int:
    found = config.discover()
    if not found:
        print(render.red("no backends found"), file=sys.stderr)
        return 1
    state = presence.survey(found)
    if args is not None and getattr(args, "json", False):
        # --list --json: the same backends, with the MODEL ID the text listing
        # does not carry. A caller that needs to know which model a backend
        # actually runs can ask here instead of parsing provider files itself.
        #
        # A NEW output rather than a fourth column: the text listing is parsed
        # in the wild as "url is the last field", and appending a column would
        # break every one of those readers at once, on hosts that update on
        # their own schedule. api_key is never included.
        print(json.dumps([{"name": b.name, "model": b.model, "wire": b.wire,
                           "base_url": b.base_url, "source": b.source,
                           "status": state[b.name],
                           "reason": presence.reason(b, state[b.name])}
                          for b in found], indent=2))
        return 0
    print(render.kicker("backends"), file=sys.stderr)
    for b in found:
        # stdout stays the machine-readable columns it always was; only the
        # frame around it is painted, so `one --list | awk` keeps working.
        # The live status rides on stderr in front of the row for the same
        # reason: a person sees it, a pipe does not get a fourth field.
        print(_status_tag(state[b.name]), end="", file=sys.stderr, flush=True)
        print(render.ink(f"{b.name:44}") + render.dim(f"{b.wire:10}")
              + render.faint(b.base_url), flush=True)
    print(render.footer([_plural(len(found), "backend")]), file=sys.stderr)
    return 0


def _status_tag(state: str) -> str:
    tag = f"{state:14}"
    if state in (presence.OFFLINE, presence.MODEL_MISSING):
        return render.red(f"{state:14}")
    if state == presence.ONLINE:
        return render.ink(tag)
    return render.faint(tag)


def cmd_sessions(args) -> int:
    """`--sessions`: what is on disk, and what each one was ABOUT.

    The byte count used to be the only thing distinguishing two rows, which
    answers a question nobody asks - you come back the next morning knowing
    what you were doing, not how large it was. `one-chat`'s own `/sessions`
    had already worked this out and shows the opening line; the two listings
    now read the same walk out of `session.opening_line` instead of one of
    them being useful and the other not.
    """
    rows = session.search(getattr(args, "session_search", ""),
                          args.workspace if getattr(args, "sessions_here", False) else None)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(render.dim("no sessions yet"), file=sys.stderr)
        return 0
    print(render.kicker("sessions"), file=sys.stderr)
    for row in rows:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["modified_epoch"]))
        print(render.ink(row["session"]) + "  " + render.dim(stamp) + "  "
              + render.faint(render.one_line(row.get("title") or row["opening_line"], 56)))
    print(render.footer([_plural(len(rows), "session")]), file=sys.stderr)
    return 0


def _exit_for(stopped: str) -> int:
    """One run outcome, one exit code, whatever the output format is.

    answered is 0, a provider that would not answer is 2, and everything else
    that stopped short - empty, budget, budget_usd - is 1.
    """
    return {"answered": 0, "error": 2}.get(stopped, 1)


def _check_workspace(path: str) -> None:
    """Refuse a workspace we cannot work in, before anything is paid for.

    Same argument as parsing the permission patterns up here: a mistyped path
    should cost nothing. Without this check the model would answer from an
    empty tree, every tool call would come back "no such path", and the run
    would exit 0 as if it had worked.
    """
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(full):
        raise ValueError(f"workspace does not exist: {full}")
    if not os.path.isdir(full):
        raise ValueError(f"workspace is not a directory: {full}")
    if not os.access(full, os.R_OK | os.X_OK):
        raise ValueError(f"workspace is not readable: {full}")


def _want_stream(args) -> bool:
    """--json and --quiet each promise one clean thing on stdout.

    This controls terminal printing only. With --json --stream, run_once
    separately enables wire streaming into the stderr progress emitter.
    """
    if args.json or args.quiet:
        return False
    if args.stream is None:
        from onecli import terminal
        return terminal.ACTIVE is not None or sys.stdout.isatty()
    return args.stream


def _run_emitter(args, state: dict):
    """The live NDJSON sink for a run, or None when nobody would read it.

    Always on under --json and never otherwise. Under --json the terminal trace
    is suppressed, so stderr carries no run output at all and the events have
    the channel to themselves; without --json the trace already IS the live
    signal and a second machine copy on the same stream would only corrupt it.
    """
    sinks = []
    if getattr(args, "json", False):
        if getattr(args, "text_events", False):
            sinks.append(events.emitter(
                sys.stderr, step=lambda: state.get("step", 0),
                include_text=True))
        else:
            sinks.append(events.emitter(
                sys.stderr, step=lambda: state.get("step", 0)))
    # A parent process that exports BG_SIGNALS reads progress from that file;
    # feed it the run's own plan (onecli.events.signals_sink).
    signals_path = os.environ.get("BG_SIGNALS", "").strip()
    if signals_path:
        sinks.append(events.signals_sink(signals_path))
    if not sinks:
        return None
    if len(sinks) == 1:
        return sinks[0]

    def emit(kind: str, data: dict) -> None:
        for sink in sinks:
            sink(kind, data)
    return emit


def _tracer(quiet: bool, state: dict | None = None, model: str = "",
            emit=None) -> Any:
    """Build the `on_event` callback for a run.

    `emit` is an optional second sink (onecli.events.emitter) fed EVERY event
    before the terminal trace decides what to draw. Under --json the terminal
    trace is fully quiet, so that sink is the only live signal a parent process
    gets while the turn runs.
    """
    state = state if state is not None else {"step": 0}
    state.setdefault("step", 0)
    live: dict[str, Any] = {"spin": None}

    def halt() -> None:
        """Clear any running rail before anything is printed on that line."""
        if live["spin"] is not None:
            live["spin"].stop()
            live["spin"] = None

    def close_line() -> None:
        """End a half-written streamed line before anything else prints.

        Streamed text goes to stdout a chunk at a time and stops wherever the
        model stopped - usually mid-sentence, with no trailing newline.
        Everything else a run prints (tool lines, notices, a permission
        question) goes to stderr, and both land on the same terminal line.
        Without this the next tool line is appended to the end of the model's
        sentence.
        """
        if state.get("midline"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            state["midline"] = False

    def settle() -> None:
        """Give the terminal back: no rail running, no unterminated line."""
        halt()
        close_line()

    def run_rail(motif, label: str) -> None:
        halt()
        from onecli import terminal
        if quiet or terminal.ACTIVE is not None:
            return
        live["spin"] = spinner.Spinner(motif, label).start()

    def trace(kind: str, data: dict) -> None:
        from onecli import terminal
        if terminal.ACTIVE is not None:
            terminal.ACTIVE.activity(kind, data)
        if emit is not None:
            emit(kind, data)
        if kind == "waiting":
            state["step"] = data["step"]
            # Final-answer suppression belongs to this response, not an earlier
            # tool-bearing response that happened to stream successfully.
            state["streamed"] = False
            # Waiting on the model. The rail is fed by tokens, so it moves only
            # while the wire is actually delivering - and freezes, greyed, with
            # a waiting time when it is not. It has to start BEFORE the request,
            # not after it: `turn` arrives with the answer already in hand, and
            # a rail started there erases the answer it was meant to cover.
            run_rail(spinner.ARRIVE, model)
            return
        if kind in ("acknowledgment", "progress_update"):
            settle()
            if not quiet:
                print(render.untrusted(data["text"]), file=sys.stderr, flush=True)
            return
        if kind == "turn":
            state["step"] = data["step"].index
            # The wait is over, so the rail that described it goes away. Its
            # own frames begin with a carriage return and an erase, so anything
            # printed while one is live is wiped on the next frame.
            halt()
            return
        if kind == "text_delta":
            # Only the answer goes to stdout; a scratchpad printed at a user is
            # the bug Turn.thinking exists to prevent.
            if state.get("stream") and data["kind"] == "text":
                # The answer arriving IS the progress indicator; a rail beside
                # it would be a second, worse one.
                halt()
                # Sanitised per chunk, same as the non-streamed print. A
                # control sequence split across two SSE frames survives this,
                # which is why the ESC byte itself is stripped even when the
                # rest of its sequence has not arrived yet.
                sys.stdout.write(render.untrusted(data["chunk"]))
                sys.stdout.flush()
                state["streamed"] = True
                state["midline"] = not data["chunk"].endswith("\n")
            elif live["spin"] is not None:
                live["spin"].feed(len(data.get("chunk", "")))
            return
        if quiet:
            return
        settle()
        if kind == "tool_start":
            call = data["call"]
            # The diamond is filled when the call can change something. That is
            # the one fact worth reading at a glance while a run scrolls.
            if terminal.ACTIVE is not None:
                terminal.ACTIVE.tool_start(call.name, call.arguments,
                    data.get('mutating', toolkit.is_mutating(call.name)))
            else:
                print(render.tool_line(state["step"], call.name, call.arguments,
                                   mutating=data.get("mutating",
                                                     toolkit.is_mutating(call.name))),
                  file=sys.stderr)
            # A subprocess we cannot see inside. Its clock is wall time, not
            # data, so the motif is ungated and its readout prints a duration -
            # never a token count it does not have.
            run_rail(spinner.WORK, call.name)
        elif kind == "tool_end":
            output = str(data.get("output") or "")
            if terminal.ACTIVE is not None:
                terminal.ACTIVE.tool_result(output)
                return
            from onecli.toolview import plain_result
            print(plain_result(output, tui.get_terminal_width()), file=sys.stderr)
        elif kind == "todos":
            # A list that changed is drawn whole - it is the one thing on
            # screen that says where the run is rather than what it just did.
            # A list merely put back into context has not changed, and redrawing
            # it says "look" about nothing new.
            if data.get("reason", "changed") == "changed":
                print(render.plan(data["items"]), file=sys.stderr)
            else:
                print(render.note(f"plan {data['reason']} - {data['summary']}",
                                  token="faint"), file=sys.stderr)
        elif kind == "todo_notice":
            print(render.note(f"plan unchanged - {data['summary']}",
                              token="faint"), file=sys.stderr)
        elif kind == "job_notice":
            print(render.note(data["note"], token="faint"), file=sys.stderr)
        elif kind == "loop_advisory":
            print(render.yellow(render.untrusted(data["note"])), file=sys.stderr)
        elif kind == "compact":
            if data.get("ok"):
                print(render.note(
                    f"compacted {data['messages_before']} messages to "
                    f"{data['messages_after']}, "
                    f"{data['before_tokens']} -> {data['after_tokens']} tokens",
                    token="faint"), file=sys.stderr)
            else:
                print(render.note(
                    f"compaction skipped: {data.get('error', '')}"),
                    file=sys.stderr)
        elif kind == "fallback":
            print(render.note(
                f"{data['from']} failed, retrying on {data['to']}: "
                f"{data['reason'][:160]}"), file=sys.stderr)
    # The asker prints a question over whatever the run left on screen, and the
    # end of a run has to close the answer's last line. Both want `settle`;
    # `halt` alone survives for the paths that only need the rail gone.
    handle: Any = trace  # function attributes are how callers reach the rail
    handle.halt = halt
    handle.settle = settle
    return handle


def _safe_build_provider(b, timeout, args):
    try:
        provider = providers.build(
            b, timeout, effort=getattr(args, "effort", "auto"),
            task_class=getattr(args, "task_class", "") or (_orc(args).task_class if _orc(args) else "")
        )
    except TypeError:
        provider = providers.build(b, timeout)
    return providers.bind_session(provider, getattr(args, "_session_id", None))


def _priced_budget(args, backend):
    price = config.price_for(backend.model, backend.name)
    if getattr(args, "require_priced_budget", False):
        import math
        budget = args.max_budget_usd
        if (type(budget) not in (int, float) or not math.isfinite(budget)
                or budget <= 0 or not isinstance(price, (tuple, list))
                or len(price) != 2 or any(type(x) not in (int, float)
                    or not math.isfinite(x) or x < 0 for x in price)):
            raise ValueError("constrained handoff requires a finite priced budget")
        if fallback.parse_spec(args.fallback_model) != []:
            raise ValueError("--require-priced-budget requires --no-fallback")
    return price


def _budget_sandbox(args):
    path = getattr(args, "request_budget_sandbox_grant", None)
    if path is None:
        return None
    if not getattr(args, "request_budget_isolated", False):
        raise ValueError('sandbox grant requires isolated request budget')
    from onecli.sandbox.tools import CodingGrant, SandboxTools
    try:
        grant = json.loads(pathlib.Path(path).read_text())
        if not isinstance(grant, dict):
            raise ValueError('host sandbox grant must be an object')
        if 'artifacts' in grant:
            if not isinstance(grant['artifacts'], list):
                raise ValueError('host sandbox artifacts must be a list')
            grant['artifacts'] = tuple(grant['artifacts'])
        if 'allowed_masks' in grant:
            if not isinstance(grant['allowed_masks'], list):
                raise ValueError('host sandbox allowed_masks must be a list')
            grant['allowed_masks'] = tuple(grant['allowed_masks'])
        return SandboxTools(CodingGrant(**grant))
    except TypeError as exc:
        raise ValueError('invalid host sandbox grant fields') from exc


def _context_settings(args, backend) -> dict:
    """Everything the agent needs to bound a run, resolved once.

    The price notice is printed here rather than at the ceiling check: a run
    that is silently unbillable should say so before it starts, not after it
    fails to stop.
    """
    price = _priced_budget(args, backend)
    # The server's own n_ctx beats the model's architectural window: a local
    # llama.cpp slot is routinely far smaller than the model can address, and
    # an agent that believes the larger number never compacts and simply runs
    # the endpoint into the ground mid-loop. An explicit flag still wins.
    window = args.context_window or (config.context_window_for(backend.model, backend.name)
        if getattr(args, "request_budget_isolated", False) else config.resolve_context_window(
        backend.model, backend.name, backend.base_url,
        getattr(backend, "context_window", None)))
    if args.max_budget_usd and price is None and not args.json:
        print(render.yellow(
            f"no known price for {backend.model} - counting tokens only, "
            "--max-budget-usd cannot stop this run"), file=sys.stderr)

    chain = fallback.build(
        backend,
        unavailable.chain_spec(
            args, fallback.parse_spec(args.fallback_model)
            if args.fallback_model is not None else fallback.spec_from_env()),
        build_provider=lambda b, timeout: _safe_build_provider(b, timeout, args),
        timeout=args.timeout,
        # An explicit --base-url is a statement about where this run goes.
        # Ranking the drop-in dir underneath it would answer somewhere the
        # operator never named; a chain they DID name still applies.
        allow_auto=backend.source != "flag",
        # A model the operator named only falls onto a paid lane if they said
        # so; an unnamed run keeps the chain it always had.
        allow_paid=not args.model or unavailable.allow_paid(args),
    )
    if not args.json:
        for rung in chain.rungs:
            decision = getattr(rung.provider, "effort_decision", None) or {}
            if decision.get("selected"):
                print(render.note(f"effort {rung.name}: {decision}"), file=sys.stderr)
    if chain.warnings and not args.json:
        for note in chain.warnings:
            print(render.yellow(f"fallback chain: {note}"), file=sys.stderr)

    return {
        "autocompact": args.autocompact,
        "context_window": window,
        "compact_at": args.compact_at,
        "keep_last": args.compact_keep,
        "max_input_tokens": args.max_input_tokens,
        "max_budget_usd": args.max_budget_usd,
        "price": price,
        "model_name": backend.model,
        "chain": chain,
        "request_admission": getattr(args, "_request_admission", None),
        "request_budget_isolated": getattr(args, "request_budget_isolated", False),
        "sandbox_tools": getattr(args, "_budget_sandbox", None),
    }


def _mode(args) -> str:
    """The one permission mode this run uses. `--read-only` folds in here."""
    mode = getattr(args, "permission_mode", None)
    if getattr(args, "dangerously_skip_permissions", False):
        mode = "bypassPermissions"
    return permissions.normalise_mode(mode, args.read_only)


def _guards(args, mode: str | None = None) -> guard.GuardChain | None:
    """The chain both entry paths install.

    Read-only appears here as well as in `tool_names` on purpose: the schema is
    what the model is offered, the guard is what the loop enforces, and only
    the second one survives a resumed history that still carries a write call.

    The permission guards sit next to the guard-chain guards rather than in
    front of them, so a refusal from either arrives at the model the same way -
    as the tool result, with the reason.
    """
    if getattr(args, "request_budget_isolated", False):
        return None
    mode = _mode(args) if mode is None else mode
    return permissions.chain(
        os.path.abspath(args.workspace), mode, commands=args.guard,
        timeout=args.guard_timeout, allowed=args.allowed_tools,
        disallowed=args.disallowed_tools, add_dirs=args.add_dir)


def _progressive_context(args) -> bool:
    """Whether an installed extension selects the boot context per request."""
    selected = False
    return selected


def _system(args, mode: str) -> str:
    """The system prompt: base, then the mode's instructions, then CLAUDE.md."""
    from onecli import run_limits
    # A bounded research run supplies its own reviewed contract through
    # --system-file. Do not append project context from every ancestor and
    # the user's home directory to it.
    bounded_host = bool(getattr(args, 'system_file', None) and
                          run_limits.load(os.path.abspath(args.workspace)))
    return permissions.compose_system(
        args.system or DEFAULT_SYSTEM,
        workspace=os.path.abspath(args.workspace),
        mode=mode,
        project=not args.no_project_context and not bounded_host
                and not getattr(args, "request_budget_isolated", False),
        trust_project=getattr(args, "trust_project_config", False),
    )


def _hooks(args, sid: str) -> hooklib.HookSet | None:
    """The event hooks for this session.

    Separate from `_guards` on purpose: `--guard` is a flat list of PreToolUse
    programs an operator names on the command line, and this is the harness's
    own config file read from disk. Both may be active; the guard runs first.
    """
    built = hooklib.from_args(args, sid)
    for problem in (built.errors if built else ()):
        print(render.yellow(f"hook configuration error: {problem}"), file=sys.stderr)
    return built


def _submit_prompt(hooks: hooklib.HookSet | None, prompt: str) -> tuple[str | None, str]:
    """Run UserPromptSubmit (hooklib.submit_prompt), its advisories on stderr."""
    return hooklib.submit_prompt(
        hooks, prompt, lambda note: print(render.yellow(f"hook: {note}"), file=sys.stderr))


def collect_attachments(args, backend) -> list[media.Attachment]:
    """Read every attached file once, then send it through the Senses router.

    Nothing is ever downgraded silently. The router either leaves the bytes
    inline for a backend whose model has that sense, or perceives them into
    text through a named ladder, or refuses naming every rung it tried. The
    refusal in `media.py` is the last rung, not the first answer: a modality a
    wire cannot carry is a reason to perceive it another way, not to bounce it.

    The perceptions land on `args.media_preamble` and ride the turn as text.
    """
    out: list[media.Attachment] = []
    for path in args.image:
        out.append(media.load(path, expect=media.IMAGE))
    if args.audio:
        out.append(media.load(args.audio, expect=media.AUDIO))
    if args.video:
        out.append(media.load(args.video, expect=media.VIDEO))
    routed = senses.route(out, backend,
                            mode=getattr(args, "media_route", "auto"),
                            fallback=getattr(args, "media_fallback", True),
                            allow_inline_audio=getattr(args, "inline_audio", False))
    args.media_preamble = routed.preamble()
    args.media_language = routed.language
    return routed.inline


def with_perceptions(args, prompt: str) -> str:
    """Prepend what the Senses perceived, once, to the turn they came with."""
    preamble = getattr(args, "media_preamble", "")
    if not preamble:
        return prompt
    args.media_preamble = ""
    return f"{preamble}\n\n{prompt}" if prompt else preamble


def speak_answer(args, text: str, sid: str) -> None:
    """Senses Lip. A dead TTS is printed, never swallowed - the whole reason
    this is not `tts_client.synthesize_voice`, which returns None either way."""
    if getattr(args, "speak", None) is None or not text:
        return
    voice = senses.tongue.voice_for(getattr(args, "media_language", ""),
                                      default=args.speak or "")
    spoken = senses.lip.speak(text, senses.lip.output_path(sid), voice=voice)
    senses.logbook.record(spoken, member="lip", path=spoken.text)
    if spoken.ok:
        print(render.dim(f"spoke: {spoken.text}"), file=sys.stderr)
    else:
        print(render.red(f"--speak failed: {spoken.ladder()}"), file=sys.stderr)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _spend(cost: dict | None) -> str:
    """What the run cost, or what it used when nobody knows the price.

    A local model has no price and reporting `$0.00` for it would be a lie of
    the most reassuring kind, so an unpriced run reports tokens instead.
    """
    if not cost:
        return ""
    if cost.get("usd") is not None:
        return f"${cost['usd']:.4f}"
    total = (cost.get("input_tokens") or 0) + (cost.get("output_tokens") or 0)
    if not total:
        return ""
    processed = cost.get('input_tokens') or 0
    output = cost.get('output_tokens') or 0
    context = cost.get('context_tokens') or 0
    cached = cost.get('cached_input_tokens') or 0
    bits = [f"{processed:,} input processed"]
    if context:
        bits.append(f"{context:,} current context")
    if cached:
        bits.append(f"{cached:,} cached")
    bits.append(f"{output:,} out")
    return " · ".join(bits)


def _facts(args, backend, chain, mode: str) -> list[str]:
    """The four things worth knowing before a run starts, in fixed order.

    Fixed order is the point: the eye learns where the model name lives and
    stops reading the rest. A fact that does not apply is dropped, never
    replaced by "none" - an empty slot says the same thing more quietly.
    """
    rows = [render.ink(backend.model) + render.faint(f"  {backend.name} · {backend.wire}"),
            render.dim(render.short_path(os.path.abspath(args.workspace)))]
    if chain is not None and getattr(chain, "spares", None):
        rows.append(render.faint("fallback " + " → ".join(
            r.name for r in chain.rungs)))
    if mode != "default":
        rows.append(render.paint("warm", mode))
    return rows


def _run_header(args, backend, chain, mode: str) -> str:
    return render.header(render.brand(__version__),
                         _facts(args, backend, chain, mode))


def _banner(args, backend, chain, mode: str, sid: str) -> str:
    from onecli import tui
    theme = (getattr(args, "theme", None) or os.environ.get("ONE_CLI_THEME", "") or tui.DEFAULT_THEME).strip().lower()
    return tui.render_banner(theme, __version__, backend.name, backend.model,
                              os.path.abspath(args.workspace), sid, mode)


def _mcp(args):
    """Connect the configured MCP servers and add their tools to the registry.

    A server that will not start is reported and skipped: one broken entry in a
    config must not cost the operator the whole run, and a silent skip would
    leave them wondering why a tool is missing.
    """
    if getattr(args, "request_budget_isolated", False):
        return None
    try:
        specs = mcplib.load_configs(args.mcp_config,
                                    workspace=os.path.abspath(args.workspace),
                                    strict=args.strict_mcp_config,
                                    trust_project=getattr(args, "trust_project_config", False))
    except (mcplib.MCPError, ValueError) as exc:
        print(render.red(f"mcp: {exc}"), file=sys.stderr)
        return None
    if not specs:
        return None

    client = mcplib.MCPClient(specs)
    client.start()
    added = client.register(toolkit.REGISTRY, toolkit.Tool)
    for name, why in client.failures:
        print(render.yellow(f"mcp: server {name} unavailable - {why}"),
              file=sys.stderr)
    if added and not args.quiet and not args.json and getattr(args, "verbose", False):
        print(render.dim(f"mcp: {len(added)} tools from "
                         f"{len(client.servers)} server(s)"), file=sys.stderr)
    return client


def _asker(args, interactive_run: bool, trace=None):
    """An asker, or None - and None is what withholds the tool.

    Default on interactively, off otherwise: a one-shot run has nobody at the
    terminal, and a question there is a hang. `--ask` forces it on for a piped
    session that does have someone watching; `--no-ask` forces it off.
    """
    want = args.ask if args.ask is not None else (interactive_run
                                                  and asklib.available())
    if not want:
        return None
    return asklib.Asker(timeout=max(0.0, args.ask_timeout),
                        before=getattr(trace, "settle", None))


def _close_jobs(store) -> int:
    for survivor in store.close():
        print(render.yellow(f"job stopped at exit: {render.untrusted(survivor)}"), file=sys.stderr)
    for survivor in shelltool.reap():
        print(render.yellow(f"background process stopped at exit: {render.untrusted(survivor)}"),
              file=sys.stderr)
    return 0


def cmd_list_agents(args) -> int:
    """What `task` can address in this run, and what a preset may be driven by.

    A preset file is the only place a driver is named, and a driver only works
    if that CLI is installed here - two facts that live in different places and
    are both invisible until a spawn fails. One command answers both.
    """
    presets = _presets(args)
    spawner = sublib.Spawner(parent=None, presets=presets)
    print(render.bold("agents"))
    print(spawner.catalogue())
    unavailable = [p.driver for p in presets.values()
                   if p.driver and p.driver not in drivers.available()]
    for name in sorted(set(unavailable)):
        print(render.yellow(f"  warning: {drivers.refusal(name, None)}"))
    print()
    print(render.bold("drivers a preset can name"))
    print(drivers.catalogue())
    return 0


def cmd_orc_list(args) -> int:
    """`--orc-list`: every layout on the SEARCH paths, one line each."""
    orcs, errors = orclib.catalogue()
    for err in errors:
        print(render.yellow(f"orc: {err}"), file=sys.stderr)
    if args.json:
        print(json.dumps([{"name": o.name, "orchestrator": o.orchestrator,
                           "workers": o.workers, "task_class": o.task_class,
                           "source": o.path} for o in orcs], indent=2))
        return 0
    print(orclib.summarise(orcs))
    return 0


def cmd_orc_show(args) -> int:
    """`--orc-show NAME|PATH`: one layout, resolved against the real presets."""
    orc, errors = orclib.find(args.orc_show)
    for err in errors:
        print(render.yellow(f"orc: {err}"), file=sys.stderr)
    if orc is None:
        return 1
    print(orclib.describe(orc, _presets(args)))
    return 0


def cmd_orc_report(args) -> int:
    """`--orc-report`: what the run records add up to. Runs no model.

    The third read verb on this surface and deliberately the same shape as the
    first two - a flag dispatched out of `main`, like `--list`, `--sessions`,
    `--list-agents`, `--orc-list` and `--orc-show`. The only verb dispatcher in
    `main` belongs to `onecli/serving/cli.py` (`models`, `tune`, `serve`,
    `setup`, `doctor`), which is a different family; adding a second one here
    would give this surface two answers to "how do I read something back".

    An unreadable record is named on stderr and counted into the header, so a
    corrupt file is visible in the summary rather than only in a warning that
    scrolls past.
    """
    records, errors = orclib.records()
    for err in errors:
        print(render.yellow(f"orc: {err}"), file=sys.stderr)
    if args.json:
        groups, nodeless = orclib.group(records)
        print(json.dumps({
            "records": len(records),
            "unreadable": len(errors),
            "nodeless": nodeless,
            "groups": [{
                "backend": g.backend,
                "task_class": g.task_class,
                "nodes": g.nodes,
                "runs": len(g.runs),
                "models": sorted(g.models),
                # The medians the table prints, as numbers rather than as the
                # strings a person reads. A caller that wanted p50 wall time
                # had to parse a column of "1.2s" out of a box-drawn table.
                "median": {k: orclib.median(v)
                           for k, v in sorted(g.readings.items())},
                "samples": {k: len(v) for k, v in sorted(g.readings.items())},
                "stopped": dict(sorted(g.stopped.items())),
                "retry_reasons": dict(sorted(g.reasons.items())),
            } for g in groups],
        }, indent=2))
        return 0
    print(orclib.report(records, unreadable=len(errors)))
    return 0


def _presets(args) -> dict[str, sublib.Preset]:
    """The subagent presets for this run. A broken file is named, not fatal."""
    presets, errors = sublib.load_presets(args.agents)
    for err in errors:
        print(render.yellow(f"agents: {err}"), file=sys.stderr)
    return presets


def _as_preset(args, presets: dict[str, sublib.Preset]) -> sublib.Preset | None:
    """The preset this whole run adopts, from `--agent`. None when unset.

    An unknown name is fatal rather than ignored: a run asked to BE the
    reviewer agent and silently given the general surface has quietly done a
    different job from the one requested.
    """
    if not args.agent:
        return None
    preset = presets.get(args.agent)
    if preset is None:
        raise SystemExit(render.red(
            f"no agent named {args.agent!r}. Available: {', '.join(sorted(presets))}"))
    return preset


def _orc(args) -> "orclib.Orc | None":
    """This run's orchestration config, or None. A bad one is named, not fatal.

    Same contract as `_presets`: an operator who mistyped a path gets a warning
    and a run on the flags they gave, rather than an exit code and no work.
    Cached on `args` because both entry paths and the launch echo ask for it,
    and reading the file three times could report the same error three times.
    """
    if getattr(args, "_orc_cached", "unset") != "unset":
        return args._orc_cached
    orc = None
    if getattr(args, "orc", ""):
        orc, errors = orclib.find(args.orc)
        for err in errors:
            print(render.yellow(f"orc: {err}"), file=sys.stderr)
    args._orc_cached = orc
    return orc


def _orchestrator_name(args) -> tuple[str, str]:
    """Who drives, and which rung said so.

    `--orchestrator` is an explicit flag and beats the file. `none` is an
    explicit refusal and beats it too - an operator turning orchestration off
    for one invocation must not have it handed back by a config in the repo.
    """
    flag = (getattr(args, "orchestrator", "") or "").strip()
    if flag:
        return ("" if flag.lower() == "none" else flag), orclib.FLAG
    orc = _orc(args)
    if orc is not None and orc.orchestrator:
        return orc.orchestrator, orclib.FILE
    return "", orclib.DEFAULT


def _presets_without_a_backend(args) -> list[str]:
    """Presets this run could spawn that no rung can put on a backend.

    The precedence chain is not rewritten here - `Spawner.pin_for` is the one
    place it is written down, so the run's own Spawner is built and asked. A
    run with no spawner (`--max-depth 0`) can spawn nothing at all, so nothing
    is missing a backend and the list is empty.
    """
    presets = _presets(args)
    spawner = _spawner(args, presets)
    if spawner is None:
        return []
    return sorted(n for n, p in presets.items() if not spawner.pin_for(p))


def _worker_name(args) -> str:
    """The backend unpinned children run on, or "" for the parent's own.

    Only an orchestrator run has one - whether the orchestrator came from the
    flag or from the config file. `--orchestrator` swaps the roles of the two
    names the operator already gave: the top-level agent moves to the
    orchestrator backend and `--model` - the one they picked - becomes what
    the workers run on. Without `--model` there is no second name, so there is
    nothing to hand out and children inherit as before.
    """
    name, _ = _orchestrator_name(args)
    if not name:
        return ""
    return (args.model or "").strip()


def _response_gate(args, interactive: bool = False):
    if getattr(args, "response_gate", None):
        return args.response_gate
    orc = _orc(args)
    if orc and orc.response_gate:
        return orc.response_gate
    return "off" if interactive else "enforce"


def _mask_options(args):
    from onecli.masks import options
    if getattr(args, 'mask', None) not in (None, 'off') and getattr(args, 'request_budget_isolated', False):
        raise ValueError('mask selection is not supported in isolated request-budget mode')
    return options(getattr(args, 'mask', None))


def _progress_policy(args):
    from onecli import progress_policy as progress
    spec = _orc(args)
    from onecli.masks import select
    mask = select(getattr(args, 'mask', None))
    result = progress.resolve((mask.loop_policy or {}).get('progress_policy') if mask else None)
    result = progress.resolve(spec.progress_policy if spec else None, base=result)
    overrides = {}
    for flag, key in (("progress_seconds", "seconds"), ("progress_tool_calls", "tool_calls")):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "no_progress", False):
        overrides["enabled"] = False
    return progress.resolve(overrides, base=result)


def _apply_orc_budgets(args) -> None:
    """`max_depth` and `max_steps` from the config, where the flag was left alone.

    "Left alone" is measured against the parser's own default rather than a
    sentinel, because both flags have a meaningful default that the help text
    prints and scripts rely on. The cost is that typing the default value
    explicitly does not beat the file - which is a distinction nobody can see
    from the outside, and not worth a second representation of "unset".
    """
    orc = _orc(args)
    if orc is None:
        return
    defaults = build_parser().parse_args([])
    if orc.max_depth and args.max_depth == defaults.max_depth:
        args.max_depth = orc.max_depth
    if orc.max_steps and args.max_steps == defaults.max_steps:
        args.max_steps = orc.max_steps


def _resolved_orc(args, presets: dict[str, sublib.Preset]) -> "orclib.Resolved":
    """The whole tree after every rung has spoken, for the echo and the record."""
    name, source = _orchestrator_name(args)
    orc = _orc(args)
    return orclib.resolve(presets, orc, name or (args.model or ""),
                          source if name else orclib.DEFAULT,
                          _worker_name(args), args.max_depth, args.max_steps,
                          task_class=getattr(args, "task_class", ""),
                          response_gate=_response_gate(args), progress_policy=_progress_policy(args))


def _echo_orc(args, presets: dict[str, sublib.Preset]) -> None:
    """Print the resolved config before the first request goes out.

    Only when there is something to say: an orchestration config in play, or an
    operator who asked. Every plain `one "fix this"` printing a four-line tree
    about a config it does not have is noise, and noise is how a launch echo
    stops being read.
    """
    if args.quiet or args.json:
        return
    if not (args.orc_explain or _orc(args) is not None
            or (args.orchestrator or "").strip().lower() not in ("", "none")):
        return
    print(orclib.echo(_resolved_orc(args, presets), explain=args.orc_explain),
          file=sys.stderr)


def _run_phases(args, agent, presets, prompt, checklist, out):
    """Drive the run off the checklist and leave the ledger where the record
    writer will find it.

    The ledger is put on `out` rather than returned beside the Result because
    the record is written from the `finally` block that EVERY termination
    passes through - answered, budget, interrupt, crash. A run killed halfway
    down a checklist has a ledger worth having, and it is the one a stall is
    read out of.
    """
    preset = presets[args.phase_agent]

    def announce(phase) -> None:
        if args.quiet or args.json:
            return
        print(render.dim(f"  phase {phase.index}/{len(checklist)} "
                         f"{phase.name}: {phase.verdict} "
                         f"({phase.seconds:.1f}s, {phase.steps} steps, "
                         f"on {phase.backend})"), file=sys.stderr)

    ledger, result = managerlib.run(
        agent.spawner, preset, prompt, checklist,
        mode=args.phase_mode, wording=args.phase_wording, on_phase=announce)
    out["phases"] = ledger
    return result


def _spawner(args, presets: dict[str, sublib.Preset]) -> sublib.Spawner | None:
    """The run's authority to spawn children, or None when it has none.

    `parent` is filled in by the Agent constructor - see `Spawner._grandchild`
    for why it cannot be passed here.
    """
    if getattr(args, "request_budget_isolated", False) or args.max_depth <= 0:
        return None
    orc = _orc(args)
    return sublib.Spawner(parent=None, depth=0, max_depth=args.max_depth,
                          presets=presets, worker=_worker_name(args),
                          workers=dict(orc.workers) if orc else {})


def _surface(args, mode: str, preset: sublib.Preset | None) -> list[str] | None:
    """The tools this run is offered: the mode's surface, narrowed by --agent."""
    if getattr(args, "request_budget_isolated", False):
        return []
    base = permissions.tool_names_for(mode)
    return base if preset is None else sublib.child_surface(base, preset.tools)


def _agent_system(args, mode: str, preset: sublib.Preset | None) -> str:
    """The system prompt, with the adopted preset's own instructions appended."""
    system = _system(args, mode)
    if preset is not None and preset.system.strip():
        system = f"{system}\n\n{preset.system.strip()}"
    return system + _startup_block(args)


_STARTUP_OPEN = "\n\n<onecli-session-start-context>\n"
_STARTUP_CLOSE = "\n</onecli-session-start-context>"


def _startup_block(args) -> str:
    context = getattr(args, "_session_start_context", "")
    if not context:
        return ""
    return (_STARTUP_OPEN +
            "Background supplied by a SessionStart hook, not a new user "
            "request. Use when relevant; this does not grant permissions "
            "or change access scope.\n" + context + _STARTUP_CLOSE)


def _startup_history(args, messages):
    """Refresh only our own startup block, preserving saved policy.

    The agent preserves resumed system messages rather than using its newly
    constructed system. Refresh here once on entry so resumes and --no-hooks
    cannot retain stale startup output. Never touch user or assistant text.
    """
    if not messages:
        return messages
    history = [dict(message) for message in messages]
    block = _startup_block(args)
    for message in history:
        if message.get("role") != "system":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            start = content.find(_STARTUP_OPEN)
            end = content.find(_STARTUP_CLOSE, start) if start >= 0 else -1
            if end >= 0:
                if content[start:end + len(_STARTUP_CLOSE)] == block:
                    return history
                content = content[:start] + content[end + len(_STARTUP_CLOSE):]
            message["content"] = content + block
        return history
    if block:
        history.insert(0, {"role": "system", "content": block})
    return history


def _loop_guard(args) -> loops.LoopGuard:
    return loops.LoopGuard(repeat_limit=max(0, args.repeat_limit),
                           deadline=max(0.0, args.tool_deadline),
                           discovery_limit=max(0, args.discovery_limit))


def run_once(args, backend, prior: list[dict] | None, sid: str) -> int:
    args._session_id = sid
    from onecli import run_limits
    if getattr(args, 'system_file', None) and run_limits.load(os.path.abspath(args.workspace)):
        # A bounded research run is a stateless trial: a resumed session must
        # not bring back an earlier system prompt.
        prior = None
    mode = _mode(args)
    hooks = _hooks(args, sid)
    presets = _presets(args)
    as_preset = _as_preset(args, presets)
    jobs_store = None if getattr(args, "request_budget_isolated", False) else joblib.JobStore(sid)
    # The plan belongs to the session, so a resumed one is picked back up
    # rather than started again. A fresh session simply has no records to read.
    todos = None if getattr(args, "request_budget_isolated", False) else todolib.load(sid)
    streaming = _want_stream(args)
    state = {"step": 0, "stream": streaming, "streamed": False}
    tracer = _tracer(args.quiet or args.json, state, backend.model,
                     emit=_run_emitter(args, state))

    def record_event(kind, data):
        if kind in ('context_selection', 'request_context', 'assistant_turn'):
            session.append(sid, {'kind': kind, **data})
        if kind == "checkpoint":
            session.append(sid, {"kind": "messages", "messages": data["messages"]})
            if not state.get("session_announced"):
                tracer("session", {"id": sid})
                state["session_announced"] = True
        tracer(kind, data)

    from onecli.turn_feedback import TurnFeedback
    feedback_fd = getattr(args, "feedback_fd", None)
    feedback = TurnFeedback(feedback_fd) if feedback_fd is not None else None
    turn_guard = _guards(args, mode)
    agent = Agent(
        session_id=sid,
        turn_feedback=feedback,
        **_mask_options(args),
        workspace=os.path.abspath(args.workspace),
        system=_agent_system(args, mode, as_preset),
        progressive_context=_progressive_context(args),
        max_steps=args.max_steps,
        soft_turn_seconds=args.soft_turn_seconds,
        response_gate=_response_gate(args),
        meditate=getattr(args, "meditate", False),
        progress_policy=_progress_policy(args),
        tool_names=_surface(args, mode, as_preset),
        on_event=record_event,
        # JSON controls stdout formatting, not the provider transport. Real
        # arrivals go to the throttled stderr emitter; stdout stays one object.
        stream=streaming or (args.json and args.stream is True),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        guard=turn_guard,
        hooks=hooks,
        trust_project=getattr(args, "trust_project_config", False),
        done_check=args.done_check,
        spill=spilllib.SpillStore(sid),
        loop_guard=_loop_guard(args),
        jobs=jobs_store,
        asker=_asker(args, False, tracer),
        spawner=_spawner(args, presets),
        todos=todos,
        **_context_settings(args, backend),
    )

    prompt = with_perceptions(args, " ".join(args.prompt))
    # `@path` is the operator's shorthand in both modes. It was wired into the
    # REPL only, so `one "explain @src/parser.py"` from a shell sent the
    # literal string and the model had to go read the file - the one shape a
    # script or a parent process is most likely to type.
    prompt, mentioned = ((prompt, []) if getattr(args, "request_budget_isolated", False)
                         else mentions.expand(prompt, os.path.abspath(args.workspace),
                                              path_guards=permissions.path_guards(turn_guard)))
    if mentioned and not (args.quiet or args.json):
        print(render.dim("  sent with the question: " + ", ".join(mentioned)),
              file=sys.stderr)
    submitted, refusal = _submit_prompt(hooks, prompt)
    if submitted is None:
        print(render.red(f"prompt blocked: {refusal}"), file=sys.stderr)
        return 2
    prompt = submitted
    if not (args.quiet or args.json):
        print(_run_header(args, backend, agent.chain, mode), file=sys.stderr)
        _echo_orc(args, presets)

    # A bad checklist costs a sentence and exit 2, BEFORE the clock starts and
    # before a single token is spent. `--phases` names a file the operator
    # typed; getting it wrong is the common case, not the exceptional one.
    checklist: list[managerlib.Item] = []
    if args.phases:
        checklist, why = managerlib.load(args.phases)
        if why:
            print(render.red(f"--phases: {why}"), file=sys.stderr)
            return 2
        if agent.spawner is None:
            print(render.red("--phases needs subagents; --max-depth is 0"),
                  file=sys.stderr)
            return 2
        if args.phase_agent not in presets:
            print(render.red(f"--phase-agent: no preset named "
                             f"{args.phase_agent!r}. Available: "
                             f"{', '.join(sorted(presets))}"), file=sys.stderr)
            return 2
    started = time.monotonic()
    # What the run finished holding, for the single record write below. A
    # dict rather than locals because an interrupt leaves the names in the
    # body unbound and the record still has to be written.
    out: dict = {"result": None, "elapsed": 0.0, "status": "", "phases": []}

    def _body() -> int:
        try:
            if checklist:
                result = _run_phases(args, agent, presets, prompt, checklist,
                                     out)
            else:
                result = agent.run(prompt, prior, attachments=args.attachments)
        finally:
            # The rail is stopped by the NEXT event, so a turn that ends
            # without one - no streamed text, no tool call - leaves a thread
            # redrawing over whatever is printed after it. The run ending is
            # that missing event.
            tracer.halt()
        out["result"] = result
        out["elapsed"] = elapsed = time.monotonic() - started
        out["status"] = str(result.stopped or "")
        session.append(sid, {"kind": "run", "backend": backend.name,
                             "answered_on": agent.chain.current.name,
                             "model": backend.model, "prompt": prompt,
                             "caller": _caller(),
                             # What the run actually spent. Without it the record
                             # can say a run happened but not what it cost, so
                             # "sum the damage" is unanswerable after the fact.
                             "cost": result.cost,
                             "request_usage": [dict(step.turn.usage) for step in result.steps],
                             "request_effort": [dict(step.turn.request_effort) for step in result.steps],
                             "request_finish_reasons": [step.turn.finish_reason for step in result.steps],
                             "bounded_http_attempts": agent.wire_requests if agent.run_limits else None,
                             "stopped": result.stopped,
                             "seconds": elapsed, "mask": getattr(result, "mask", None)})
        session.append(sid, {"kind": "messages", "messages": result.messages})

        if args.json:
            print(json.dumps({
                "session": sid,
                "permission_mode": mode,
                "backend": backend.name,
                "model": backend.model,
                # Which rung actually answered. Equal to `backend` unless the chain
                # moved; a caller attributing a result to a model reads this one.
                "answered_on": agent.chain.current.name,
                "wire": backend.wire,
                "text": result.text,
                "steps_text": result.steps_text,
                "answer_advisories": result.answer_advisories,
                "stopped": result.stopped,
                "error": result.error,
                "steps": len(result.steps),
                "tool_calls": result.tool_calls,
                "mask": getattr(result, "mask", None),
                "seconds": round(elapsed, 2),
                "cost": result.cost,
                "context_tokens": result.context_tokens,
                "context_window": result.context_window,
                "compactions": result.compactions,
                "fallbacks": result.fallbacks,
                "resamples": result.resamples,
                # Every retry the wire made, and the class it gave for each. The
                # resamples above are the malformed-tool-call slice of this.
                "retries": result.retries,
                "retry_reasons": result.retry_reasons,
                # Every loop-guard note the model was shown. A run that answered
                # after three of these answered while stuck, which is not the same
                # thing as answering.
                "advisories": [{"tool": t, "note": n} for t, n in result.advisories],
                # The plan as it stood at the end. A caller reading this can tell a
                # finished list from an abandoned one; the exit code cannot.
                "todos": result.todos,
                "request_effort": [dict(step.turn.request_effort) for step in result.steps],
                "fallback_chain": agent.chain.as_dict(),
            }, indent=2))
            # The same mapping the text path below uses, so the exit code does
            # not depend on the output format.
            return _exit_for(result.stopped)

        if result.stopped == "context":
            print(render.red(f"context blocked: {render.untrusted(result.error)}"), file=sys.stderr)
        if result.stopped == "error":
            print(render.red(f"provider error: {render.untrusted(result.error)}"), file=sys.stderr)
            if args.model and agent.chain.cursor == 0:
                print(render.red(f"{backend.name}: {render.untrusted(unavailable.reason_of(result.error))}"),
                      file=sys.stderr)
                print(unavailable.listing([] if backend.source == "flag" else unavailable.suggestions(backend)), file=sys.stderr)
            return _exit_for(result.stopped)
        if result.stopped == "empty":
            print(render.yellow(render.untrusted(result.error)), file=sys.stderr)
            return _exit_for(result.stopped)

        if result.text:
            # A streamed answer is already on stdout; reprinting it doubles it. All
            # that is owed is the newline the stream never wrote.
            if state["streamed"]:
                tracer.settle()
            else:
                print(render.untrusted(result.text))
        speak_answer(args, result.text, sid)
        if mode == "plan":
            print(render.dim("plan mode - no changes were made."), file=sys.stderr)
        if result.stopped in ("budget", "budget_input", "budget_usd", "truncated",
                              "unverified"):
            # `truncated` prints the answer above and the caveat here. The text
            # was recovered from completed steps, not spoken as a final reply,
            # and a reader who is not told that reads it as one.
            print(render.yellow(f"\n{render.untrusted(result.error)}"), file=sys.stderr)

        # A run that answers with items still open has not finished its own plan.
        # Printing the list once at the end is the difference between noticing that
        # and taking the last paragraph's word for it.
        if not args.quiet and any(t["state"] != "done" for t in result.todos):
            print(render.plan(result.todos, title="plan left"), file=sys.stderr)

        if not args.quiet:
            # The chain's current rung, not the backend we started on: after a
            # failover the answer came from somewhere else and the footer that
            # still named the primary would be reporting the wrong model.
            answered_on = agent.chain.current.name
            display = (backend.display if answered_on == backend.name
                       else f"{answered_on} (from {backend.name})")
            bits = [display, _plural(len(result.steps), "step"),
                    _plural(result.tool_calls, "tool"), f"{elapsed:.1f}s",
                    _spend(result.cost), f"session {sid}"]
            print(render.footer(bits), file=sys.stderr)
        return _exit_for(result.stopped)

    try:
        code = _body()
    except KeyboardInterrupt:
        print(render.yellow("\ninterrupted"), file=sys.stderr)
        out["status"] = "interrupt"
        code = 130
    except BaseException:
        # An interrupt is not the only way out that skips the linear path.
        # An unexpected exception is still a run that spent real tokens, so
        # it gets a record too and then goes on up unchanged - swallowing it
        # here would hide a bug behind an exit code.
        out["status"] = "crash"
        code = 1
        raise
    finally:
        # Nothing outlives the CLI that started it. A run that exits leaving
        # a background command behind has handed the user a process they did
        # not start and cannot see.
        for survivor in jobs_store.close() if jobs_store is not None else ():
            print(render.yellow(f"job stopped at exit: {render.untrusted(survivor)}"), file=sys.stderr)
        for survivor in shelltool.reap():
            print(render.yellow(f"background process stopped at exit: {render.untrusted(survivor)}"),
                  file=sys.stderr)
        # One JSON per run beside the sessions and the jobs, written from the
        # ONE exit point every termination passes through: answered, budget,
        # error, interrupt, or an exception nobody predicted. There is no
        # second call site to drift from this one, and `exit_code` is the
        # value actually returned rather than an independent guess at it.
        # Never fatal - see `orc.write`.
        spent = out["elapsed"] or (time.monotonic() - started)
        orclib.write(orclib.record_for(
            sid, _resolved_orc(args, presets), out["result"],
            agent.spawner.spawns if agent.spawner else [],
            seconds=spent,
            exit_code=code,
            exit_status=out["status"],
            started_at=time.time() - spent,
            chain_name=agent.chain.current.name,
            chain_model=agent.chain.current.model,
            phases=out["phases"]))
    return code


def interactive(args, backend, prior: list[dict] | None, sid: str,
                client=None) -> int:
    from onecli import terminal
    with terminal.session(f"one  |  {backend.model}  |  {render.short_path(args.workspace)}", theme=args.theme) as surface:
        if surface is not None:
            surface.restore_draft(sid)
        return _interactive_loop(args, backend, prior, sid, client)


def _interactive_loop(args, backend, prior, sid, client=None):
    args._session_id = sid
    mode = _mode(args)
    messages = prior
    settings = _context_settings(args, backend)
    chain = settings["chain"]
    from onecli import terminal
    if terminal.ACTIVE is None:
        print(_banner(args, backend, chain, mode, sid))
    hooks = _hooks(args, sid)
    presets = _presets(args)
    as_preset = _as_preset(args, presets)
    streaming = _want_stream(args)
    # One store for the whole interactive session, so spill files are numbered
    # across turns instead of restarting at 001 on every line.
    spill_store = spilllib.SpillStore(sid)
    # Jobs belong to the session, not the turn: a suite started on one line is
    # still running - and still readable - on the next.
    jobs_store = None if getattr(args, "request_budget_isolated", False) else joblib.JobStore(sid)
    # One state object for the whole session, because a slash command answering
    # "what has this cost" needs a meter that outlives the turn - and every
    # Agent below is built fresh per turn on purpose.
    files, notes = commandfiles.load(args.workspace,
                                     reserved=set(cmdlib.COMMANDS),
                                     trust_project=getattr(args, "trust_project_config", False))
    for note in notes:
        print(render.yellow(f"command file: {note}"), file=sys.stderr)
    repl = cmdlib.Repl(
        args=args, backend=backend, chain=chain, mode=mode, sid=sid,
        messages=messages, jobs=jobs_store, hooks=hooks, presets=presets,
        surface=_surface(args, mode, as_preset), settings=settings,
        mcp_client=client, files=files,
    )
    if files and not args.quiet:
        print(render.dim(f"{_plural(len(files), 'command')} from "
                         ".claude/commands"), file=sys.stderr)
    inputline.install(cmdlib.names_for(repl), os.path.abspath(args.workspace))
    # The way out is said once, on the first Ctrl-C that does not take it.
    told_how_to_leave = False
    last_checkpoint_note = ""
    # One list for the session, not one per line: a plan written on the first
    # turn is what the fourth turn is still working through.
    todos = None if getattr(args, "request_budget_isolated", False) else todolib.load(sid)
    while True:
        if repl.rebuild:
            # A `/model` or `/mode` change lands here rather than inside the
            # command: rebuilding the chain, the surface and the system prompt
            # is the loop's job, and doing it at the top of the next turn means
            # a half-applied switch cannot exist.
            backend = repl.backend
            mode = repl.mode
            settings = _context_settings(args, backend)
            chain = settings["chain"]
            repl.chain, repl.settings = chain, settings
            repl.surface = _surface(args, mode, as_preset)
            repl.rebuild = False
        try:
            line = inputline.read(_prompt(repl),
                                  render.faint("  … ")).strip()
        except EOFError:
            print()
            repl.tasks.close()
            repl.checkpoints.close()
            return _close_jobs(jobs_store)
        except KeyboardInterrupt:
            # Ctrl-C at the prompt throws away the line being typed. It is not
            # the way out: the banner says ctrl-d, and a session that dies to
            # the same key that cancels a turn takes its history and its jobs
            # with it one keystroke after an interrupt that was survivable.
            # Reaching for it twice - once for the tool call, once out of habit
            # - is the normal way to use Ctrl-C, not a request to quit.
            print()
            if not told_how_to_leave:
                print(render.dim("ctrl-d or /exit to leave"), file=sys.stderr)
                told_how_to_leave = True
            continue
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            # A bare word is what a newcomer types to leave; it is not a prompt.
            line = "/exit"
        from onecli.model_picker import handle_picker
        menu = handle_picker(repl, line)
        if menu is not None:
            print(menu, file=sys.stderr)
            continue
        if line.startswith("!"):
            # A shell line is the operator's, not the model's: it runs here,
            # under their own permissions, without spending a turn. What it
            # printed is offered to the next turn as context, because the
            # reason to run it was almost always to show the model something.
            print(_shell(args, line[1:].strip(), repl), file=sys.stderr)
            continue
        if cmdlib.is_command(line):
            # A slash command never reaches the model, so it never runs a
            # UserPromptSubmit hook and never lands in the transcript as a
            # turn. It is the operator talking to the session, not to the
            # agent. A command FILE is the exception - it is a saved prompt,
            # and it costs a turn like any other.
            try:
                out = cmdlib.dispatch(line, repl)
            except Exception as exc:                        # noqa: BLE001
                # One bad command must not take the session down with it: the
                # history in this process is the expensive thing on the table.
                out = cmdlib.Outcome(
                    text=render.red(f"{line.split()[0]} failed: {exc}"))
            if out.text:
                print(out.text, file=sys.stderr)
            if repl.leave:
                repl.tasks.close()
                repl.checkpoints.close()
                return _close_jobs(jobs_store)
            messages = repl.messages
            if not out.prompt:
                continue
            line = out.prompt
        for lost in mentions.missing(line, os.path.abspath(args.workspace)):
            print(render.yellow(f"no such file: {lost} (sent as typed)"), file=sys.stderr)
        # Built before the expansion so a file the Read rules deny is left in
        # the sentence as typed, the same answer the read tool would give.
        turn_guard = _guards(args, mode)
        line, sent = mentions.expand(line, os.path.abspath(args.workspace),
                                     path_guards=permissions.path_guards(turn_guard))
        if sent and not args.quiet:
            print(render.dim("  sent with the question: " + ", ".join(sent)),
                  file=sys.stderr)
        if repl.shell_context:
            line = repl.take_shell_context() + line
        prompt, refusal = _submit_prompt(hooks, line)
        if prompt is None:
            print(render.red(f"prompt blocked: {refusal}"), file=sys.stderr)
            continue
        args.prompt = [prompt]
        messages = _startup_history(args, messages)
        istate = {"step": 0, "stream": streaming, "streamed": False}
        itracer = _tracer(args.quiet, istate, backend.model)
        def context_tracer(kind, data):
            if kind in ('context_selection', 'request_context', 'assistant_turn'):
                session.append(sid, {'kind': kind, **data})
            # Mid-turn, as run_once does: a long turn that is killed or
            # crashes otherwise leaves nothing of itself on disk.
            if kind == "checkpoint":
                session.append(sid, {"kind": "messages", "messages": data["messages"]})
            itracer(kind, data)
        agent = Agent(
            session_id=sid,
            **_mask_options(args),
            workspace=os.path.abspath(args.workspace),
            system=_agent_system(args, mode, as_preset),
            progressive_context=_progressive_context(args),
            max_steps=args.max_steps,
            soft_turn_seconds=args.soft_turn_seconds,
            response_gate=_response_gate(args, interactive=True),
            meditate=getattr(args, "meditate", False),
            progress_policy=_progress_policy(args),
            tool_names=repl.surface,
            on_event=context_tracer,
            stream=streaming,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            guard=turn_guard,
            hooks=hooks,
            trust_project=getattr(args, "trust_project_config", False),
            done_check=getattr(args, "done_check", ""),
            spill=spill_store,
            jobs=jobs_store,
            asker=_asker(args, True, itracer),
            todos=todos,
            # A fresh spawner per turn, so one turn's children are not counted
            # against the next one's cap.
            spawner=_spawner(args, presets),
            # A fresh guard per turn: the repeat count belongs to the task the
            # model is working on, not to the session it lives in.
            loop_guard=_loop_guard(args),
            **settings,
        )
        # Attachments belong to the turn they were passed with, so they ride the
        # first interactive turn only and are not re-sent on every later line.
        # Every turn gets a checkpoint record, so /undo maps to exactly this turn.
        # A surface without a mutating tool cannot edit the workspace, so it
        # takes no snapshot (a non-git one copies up to MAX_BYTES).
        note = repl.checkpoints.begin_turn(
            prompt, messages or [],
            edits=repl.surface is None or any(toolkit.is_mutating(n) for n in repl.surface))
        if note and note != last_checkpoint_note:
            print(render.yellow(f"no checkpoint for this turn: {note}"), file=sys.stderr)
        last_checkpoint_note = note or ""
        try:
            if terminal.ACTIVE is not None:
                terminal.ACTIVE.allow_steer = True
            result = agent.run(with_perceptions(args, prompt), messages,
                               attachments=args.attachments)
        except KeyboardInterrupt:
            # Ctrl-C during a turn belongs to the turn. The session, its
            # history and its jobs outlive it, and the prompt comes back - a
            # long tool call is the most common reason to reach for it, and
            # having to restart the whole session to escape one is why people
            # stop using a REPL.
            itracer.halt()
            messages = agent.interrupted_messages
            repl.messages = messages
            session.append(sid, {"kind": "messages", "messages": messages})
            print(render.yellow("\ninterrupted - conversation retained"),
                  file=sys.stderr)
            args.attachments = []
            continue
        finally:
            if terminal.ACTIVE is not None:
                terminal.ACTIVE.allow_steer = False
            # Without this the rail outlives the turn and erases the prompt on
            # every frame, so the next line cannot be typed at all.
            itracer.halt()
        args.attachments = []
        if result.stopped == "context":
            print(render.red(f"context blocked: {render.untrusted(result.error)}"), file=sys.stderr)
        if result.stopped == "error":
            print(render.red(f"provider error: {render.untrusted(result.error)}"), file=sys.stderr)
        if result.stopped == "unverified":
            print(render.yellow(render.untrusted(result.error)), file=sys.stderr)
        # Kept on an error too: the turn's tool calls already changed files,
        # and dropping them from the history left the model resuming from
        # before its own work (a 429 mid-turn).
        messages = result.messages
        repl.messages = messages
        repl.record(result)
        session.append(sid, {"kind": "mask_run", "mask": getattr(result, "mask", {}),
                             "stopped": result.stopped, "error": result.error})
        session.append(sid, {"kind": "messages", "messages": messages})
        if result.text:
            if istate["streamed"]:
                itracer.settle()
            else:
                print(render.untrusted(result.text))
            speak_answer(args, result.text, sid)
        if not args.quiet:
            answered_on = agent.chain.current.name
            bits = [answered_on, _plural(len(result.steps), "step"),
                    _plural(result.tool_calls, "tool"), _spend(result.cost)]
            print(render.footer(bits), file=sys.stderr)


def _prompt(repl) -> str:
    """The prompt, with the session's own status line above it.

    One string rather than a print plus a prompt: readline measures what it is
    given, and a status line printed separately would be scrolled away by the
    first redraw of the line under it.
    """
    line = cmdlib.status(repl)
    return ("\n" + line if line else "") + render.prompt_mark()


# A shell line is the operator's own command, run in the workspace with their
# own permissions - no guard, no permission mode, no tool call. Those gates
# exist to bound what the MODEL may do; applying them to a human at their own
# terminal would be theatre. The output is not cut: the operator asked for it.
SHELL_TIMEOUT = 120.0


def _shell(args, command: str, repl) -> str:
    if not command:
        return render.dim("!<command> runs a shell command in the workspace")
    workspace = os.path.abspath(args.workspace)
    try:
        done = subprocess.run(command, shell=True, cwd=workspace,
                              capture_output=True, text=True,
                              errors="replace", timeout=SHELL_TIMEOUT)
        output = (done.stdout or "") + (done.stderr or "")
        code = done.returncode
    except subprocess.TimeoutExpired:
        output, code = f"timed out after {SHELL_TIMEOUT:.0f}s", 124
    except OSError as exc:
        output, code = str(exc), 127
    repl.remember_shell(command, output, code)
    body = output.rstrip() if output.strip() else render.dim("(no output)")
    tail = "" if code == 0 else render.yellow(f"\nexit {code}")
    return body + tail


def _resolve_system_file(args) -> None:
    """Fold --system-file into args.system.

    A parent process driving one-cli (a harness, a scheduler, another tool) composes a
    system prompt far larger than the kernel's 128KB per-argument ceiling, so it
    needs a path rather than a string. Both flags at once is an operator
    mistake, not a precedence question, so it is refused instead of ranked.
    """
    if not args.system_file:
        return
    if args.system:
        raise ValueError("--system and --system-file are exclusive")
    args.system = pathlib.Path(args.system_file).read_text(encoding="utf-8")


# A Claude Code hook writes its verdict to stdout as one JSON object per line.
# A child that inherits that fd and reads it sees a stack of those objects, and
# nothing in them is a task. Recognised by the two keys the envelope always
# carries - a real prompt does not begin with either.
_HOOK_KEYS = ('"continue"', "'continue'", '"suppressOutput"', '"hookSpecificOutput"',
              '"hookEventName"', '"permissionDecision"')


def _is_hook_envelope(line: str) -> bool:
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return False
    if line == "{}":
        return True
    try:
        obj = json.loads(line)
    except ValueError:
        return False
    if not isinstance(obj, dict):
        return False
    return bool({"continue", "suppressOutput", "hookSpecificOutput",
                 "systemMessage", "decision"} & set(obj))


def _strip_hook_envelope(text: str) -> tuple[str, int]:
    """Drop leading hook-envelope lines from a piped body.

    Stripping rather than refusing outright, because the observed shape is a
    stack of envelopes followed by a REAL task: the caller's prompt is in there,
    it just arrived behind somebody else's stdout. Refusing the whole body would
    throw the task away along with the junk. When nothing survives, there was no
    task, and the caller is told so instead of a backend being billed for it.
    """
    lines = text.splitlines()
    cut = 0
    while cut < len(lines) and (not lines[cut].strip() or _is_hook_envelope(lines[cut])):
        cut += 1
    if cut == 0:
        return text, 0
    return "\n".join(lines[cut:]).strip(), cut


def _prompt_from_stdin(args) -> None:
    """Take the prompt from stdin when none was given on argv.

    Interactive mode is still what an empty argv means at a terminal. It is only
    a PIPED stdin that turns "no prompt" into "the prompt is coming down the
    pipe" - which is how a parent process sends a turn too long for argv.

    An argv prompt wins outright and stdin is never read at all: a parent that
    spawns us with a task on argv and forgets to close its stdin must not have
    that fd turned into a second, contradictory prompt.
    """
    if args.prompt or sys.stdin.isatty():
        return
    piped = sys.stdin.read().strip()
    if not piped:
        return
    piped, dropped = _strip_hook_envelope(piped)
    if dropped:
        print(render.yellow(
            f"dropped {dropped} hook-envelope line(s) from stdin - a parent "
            "spawned us without closing its stdin"), file=sys.stderr)
    if not piped:
        raise ValueError(
            "stdin carried only Claude Code hook output, which is not a task. "
            "Pass the prompt on argv, or spawn with stdin=DEVNULL.")
    args.prompt = [piped]


def _caller() -> str:
    """Who spawned us, for the run record.

    An inherited-fd bug is invisible in a record that names only the backend:
    the backend rotation is what it looks like from the outside, and that is an
    inference. The parent's own argv makes it a query. ONE_CLI_CALLER wins where
    a caller names itself; /proc is the fallback and simply misses off Linux.
    """
    named = os.environ.get("ONE_CLI_CALLER", "").strip()
    if named:
        return named[:200]
    try:
        raw = pathlib.Path(f"/proc/{os.getppid()}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(raw.decode("utf-8", "replace").split("\0")).strip()[:200]


def _first_run():
    """No backend configured: guide a person, or tell a script what to do."""
    from onecli import onboard
    if not onboard.interactive():
        print(onboard.NO_BACKEND_HELP, file=sys.stderr)
        return None
    return onboard.run()


def main(argv: list[str] | None = None) -> int:
    from onecli.execution_evidence import initialize
    initialize()
    raw = list(sys.argv[1:] if argv is None else argv)
    budget_isolated = any(x.startswith("--") and "--request-budget-isolated".startswith(x) for x in raw)
    if budget_isolated and raw and raw[0] in serving_cli.VERBS:
        print("--request-budget-isolated forbids alternate commands", file=sys.stderr)
        return 2
    # `one update` is a command only with nothing but flags after it, so a
    # prompt such as `one update the readme` still reaches the model. The bare
    # word also needs a terminal on stdin: `echo x | one update` is the prompt
    # "update" (an argv prompt, so the pipe is not read), not the installer.
    if raw and raw[0] == "update" and all(a.startswith("-") for a in raw[1:]) \
            and (raw[1:] or sys.stdin.isatty()):
        return updatelib.main(raw[1:])
    # `one recover scan|review|prepare` is a command; any other second word is
    # still a prompt (`one recover the deleted file`).
    if raw and raw[0] == "recover" and len(raw) > 1 and raw[1] in ("scan", "review", "prepare"):
        if budget_isolated:
            print("--request-budget-isolated forbids alternate commands", file=sys.stderr)
            return 2
        from onecli import recover
        return recover.main(raw[1:])
    # The serving verbs sit in front of the flat prompt CLI. `one` takes a
    # prompt as its positional, so a first word that names a verb is a command
    # and anything else is still a prompt; `one -- models ...` escapes a
    # prompt that genuinely starts with one of these words.
    if raw and raw[0] in serving_cli.VERBS:
        code = serving_cli.dispatch(raw)
        if code is not None:
            return code
    if raw and raw[0] == "--" and len(raw) > 1 and raw[1] in serving_cli.VERBS:
        raw = raw[1:]
        argv = raw

    args = build_parser().parse_args(argv)
    # Resolve the workspace once: the no-follow file tools refuse a symlink
    # anywhere in a path, so a symlinked -w (macOS /var/folders too) would
    # pass the guard and then fail every read, write and search below it.
    args.workspace = os.path.realpath(args.workspace)
    try:
        if args.masks:
            from onecli.masks import listing
            print(listing(args.mask))
            return 0
        _mask_options(args)  # Fail before provider discovery or execution.
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.request_budget_isolated and (args.list or args.sessions or args.list_agents
            or args.orc_list or args.orc_show or args.orc_report):
        print("--request-budget-isolated forbids listing commands", file=sys.stderr)
        return 2

    if args.plain:
        # Set the variable rather than a module flag: a hook, a guard or a
        # child `one` inherits the environment, so --plain reaches them too.
        os.environ["ONE_CLI_PLAIN"] = "1"
    render.refresh()

    if args.list:
        return cmd_list(args)
    if args.sessions or args.sessions_here or args.session_search:
        return cmd_sessions(args)
    if args.list_agents:
        return cmd_list_agents(args)
    if args.orc_list:
        return cmd_orc_list(args)
    if args.orc_show:
        return cmd_orc_show(args)
    if args.orc_report:
        return cmd_orc_report(args)

    try:
        _resolve_system_file(args)
        _prompt_from_stdin(args)
        _check_workspace(args.workspace)
    except (ValueError, OSError) as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2

    if args.request_budget_sandbox_grant and not args.request_budget_isolated:
        print(render.red("sandbox grant requires isolated request budget"), file=sys.stderr)
        return 2

    if args.request_budget_isolated:
        if not (args.require_priced_budget and args.request_budget_policy
                and args.request_budget_ledger):
            print(render.red("--request-budget-isolated requires priced request budget policy and ledger"), file=sys.stderr)
            return 2
        if (args.chat or args.resume or not args.prompt or args.hooks or args.mcp_config
                or args.guard or args.phases or args.image or args.audio
                or args.video or args.speak is not None):
            print(render.red("--request-budget-isolated forbids interactive/chat, hooks, MCP, guards, phases and media"), file=sys.stderr)
            return 2

    if args.feedback_fd is not None:
        if (args.request_budget_policy or args.request_budget_ledger
                or args.create_request_budget_ledger or args.request_budget_isolated):
            print(render.red("--feedback-fd cannot be combined with request budget flags: "
                             "the receipt would precede budget admission"), file=sys.stderr)
            return 2
        try:
            if not stat.S_ISSOCK(os.fstat(args.feedback_fd).st_mode):
                raise ValueError("not a socket")
        except (OSError, ValueError) as exc:
            print(render.red(f"--feedback-fd {args.feedback_fd} is not a usable socket: {exc}"),
                  file=sys.stderr)
            return 2

    if not args.json:
        updatelib.startup()

    # Handed over after the system prompt and a piped body are resolved. The
    # chat builds the same permission chain from these args (permissions.chain):
    # -w, modes, allow/deny rules, --guard and --add-dir all hold there too.
    if args.chat:
        unsupported = [flag for flag, value in (
            ("--max-budget-usd", args.max_budget_usd != 0),
            ("--max-steps", args.max_steps is not None),
            ("--max-input-tokens", args.max_input_tokens is not None),
        ) if value]
        if unsupported:
            print(render.red(f"{', '.join(unsupported)} unsupported in --chat"),
                  file=sys.stderr)
            return 2
        if (args.require_priced_budget or args.request_budget_policy
                or args.request_budget_ledger or args.create_request_budget_ledger):
            print(render.red("--require-priced-budget is unsupported in --chat"),
                  file=sys.stderr)
            return 2
        from onecli import chat as chatlib
        return chatlib.from_cli(args)

    try:
        _mode(args)
        # Parse the patterns before the first request goes out. A typo in a
        # rule should cost nothing; finding it at the first tool call means the
        # model has already been paid for.
        permissions.parse_rules(args.allowed_tools)
        permissions.parse_rules(args.disallowed_tools)
    except ValueError as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2

    # --orchestrator moves the top-level agent onto its own backend; the
    # picked --model stays as the workers' (see `_worker_name`). Resolved here
    # so the window, price and fallback chain downstream are all the
    # orchestrator's - the children resolve their own in the Spawner.
    # The orc file is read FIRST so a config can name the orchestrator, and
    # the flag is applied on top so an explicit one still wins - see
    # `_orchestrator_name` for the rung order.
    _apply_orc_budgets(args)
    orchestrator, source = _orchestrator_name(args)
    driving = args.model
    if orchestrator:
        # An orchestrator splits the tree, so the children need somewhere to
        # go. The old guard tested `--model` and refused an ORC FILE that named
        # an orchestrator - demanding a flag nobody typed, whose value the
        # file's `workers` map would then have overridden for every preset.
        #
        # The precise condition is the one that actually breaks a run: an
        # orchestrator is set, there is no blanket worker default, and some
        # preset this run could spawn has no pin from ANY rung. A file whose
        # map covers every preset needs no `--model` at all.
        stranded = _presets_without_a_backend(args)
        if stranded:
            if source == orclib.FLAG:
                # The flag case is unchanged: with only a name there is
                # genuinely nothing to hand the children, and `--model` is
                # exactly the missing thing to name.
                print(render.red("--orchestrator needs --model: it names who "
                                 "drives, and --model names who does the work"),
                      file=sys.stderr)
            else:
                print(render.red(
                    f"orchestrator {orchestrator!r} leaves nothing for these "
                    f"agents to run on: {', '.join(stranded)}. Give each a "
                    f"backend in the config's `workers` map, or a default for "
                    f"all of them with --model"), file=sys.stderr)
            return 2
        driving = orchestrator

    try:
        backend = config.resolve(name=driving, model=args.model_id,
                                 base_url=args.base_url, wire=args.wire,
                                 **({"include_local": False} if args.request_budget_isolated else {}))
    except config.NoBackend as exc:
        first = None if args.request_budget_isolated else _first_run()
        if first is None:
            if args.request_budget_isolated:
                print(render.red(str(exc)), file=sys.stderr)
            return 2
        backend = first
    except RuntimeError as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2
    if args.model and (refused := presence.refusal(backend)):
        backend = unavailable.resolve(args, backend, refused)
        if backend is None:
            return 2

    try:
        if args.require_priced_budget:
            _priced_budget(args, backend)
        if args.request_budget_policy or args.request_budget_ledger or args.create_request_budget_ledger:
            if not (args.require_priced_budget and args.request_budget_policy
                    and args.request_budget_ledger):
                raise ValueError('request budget requires policy, ledger and --require-priced-budget')
            if args.image or args.audio or args.video or args.speak is not None:
                raise ValueError('request budget forbids auxiliary media providers')
            from onecli.request_admission import RequestAdmission
            args._request_admission = RequestAdmission.load(
                args.request_budget_policy, args.request_budget_ledger, args.max_budget_usd,
                create=args.create_request_budget_ledger)
            # Validate the actual configured transport before attachment/hooks.
            args._request_admission.bind(_safe_build_provider(backend, args.timeout, args))
            args._budget_sandbox = _budget_sandbox(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2

    try:
        args.attachments = collect_attachments(args, backend)
    except media.MediaError as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2

    prior: list[dict] | None = None
    sid = session.new_id()
    if args.resume:
        want = session.latest_id() if args.resume == "@last" else args.resume
        if not want:
            print(render.red("no session to resume"), file=sys.stderr)
            return 2
        try:
            prior = session.load_messages(want)
            if args.mask is None:
                args.mask = session.metadata(want).get('mask')
                _mask_options(args)
            if args.request_budget_policy and any(
                    media.blocks_of(m.get("content")) for m in prior):
                raise media.MediaError('request budget forbids resumed media providers')
            prior = senses.route_history(
                prior, backend, mode=args.media_route,
                fallback=args.media_fallback,
                allow_inline_audio=args.inline_audio)
        except (FileNotFoundError, ValueError) as exc:
            print(render.red(str(exc)), file=sys.stderr)
            return 2
        sid = want
        # A resume keeps the conversation and NOT the backend - the model is
        # resolved from this command line, so `-r` with no `-m` continues
        # yesterday's chat on whatever the default happens to be today. The
        # answer that comes back is then indistinguishable from a resume that
        # lost the history, and on a host whose drop-in dir holds paid
        # endpoints it is also a bill. Not changed, said: an explicit -m is a
        # legitimate reason to move a conversation, and silence was the bug.
        was = session.backend_of(want)
        if was and was != backend.name:
            print(render.yellow(
                f"resuming {want} on {backend.name}; it last ran on {was}"),
                file=sys.stderr)

    try:
        session.append(sid, {"kind": "metadata", "workspace": str(pathlib.Path(args.workspace).resolve()), "mask": args.mask or "off"})
    except OSError as exc:
        print(render.red(f"cannot save session {session.path_for(sid)}: {exc}"), file=sys.stderr)
        return 1

    # SessionStart / SessionEnd bracket the whole run, so they are fired here
    # rather than inside either entry path - one session, one pair, whichever
    # way the agent was invoked.
    boot = _hooks(args, sid)
    # Rebuilt for this invocation, including resume. Keep it out of saved
    # user messages so repeated resumes cannot accumulate old hook output.
    args._session_start_context = ""
    if boot:
        started = boot.session_start("resume" if args.resume else "startup")
        if started.blocked:
            print(render.red(f"session blocked: {started.reason}"), file=sys.stderr)
            boot.session_end("blocked")
            return 2
        args._session_start_context = started.context_text
        for note in started.advisories:
            if getattr(args, "verbose", False):
                print(render.yellow(f"hook: {note}"), file=sys.stderr)
        if started.context_text and not args.json and getattr(args, "verbose", False):
            print(render.dim(started.context_text), file=sys.stderr)

    prior = _startup_history(args, prior)

    client = _mcp(args)
    try:
        if not args.prompt:
            code = interactive(args, backend, prior, sid, client=client)
        else:
            code = run_once(args, backend, prior, sid)
    except providers.ProviderError as exc:
        print(render.red(str(exc)), file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        # Both entry paths handle their own interrupt, so reaching this one
        # means it landed somewhere else - setup, a hook, the MCP handshake.
        # A stack trace is not a message to a person either way.
        print(render.yellow("\ninterrupted"), file=sys.stderr)
        code = 130
    finally:
        # Servers are subprocesses we started; they go down with us however the
        # run ended, including on a traceback.
        if client is not None:
            client.stop()

    if boot:
        boot.session_end("exit")
    return code

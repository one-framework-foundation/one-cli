"""one chat - the fast lane.

`one` is a coding agent: every turn is a run, with a header, a
permission mode and a spine of tool calls scrolling past. That is the
right shape for work and the wrong shape for a conversation.

So this is the other entry: same backends, same transcript, chat-shaped. A turn
is a question and an answer - and when the answer depends on something the model
cannot know, it reads the file, greps the tree or runs a command underneath
the conversation and keeps talking. The loop is not a mode you switch into; it
runs behind the chat, one dim line per call, and the answer streams as prose.
What that costs is bounded on purpose: a short surface (`read`, `ls`, `glob`,
`grep`, `bash`, and web tools when an extension provides them), and nothing that
can change a file until you ask for it with `/tools all`.

Nothing is imported until it is needed: the agent, the tool registry and the
media path load on the turn that first uses them, never at startup - a test
asserts that, because it is the kind of claim a later import quietly breaks. The
whole conversation stays in one canonical history that any backend can be handed
mid-sentence: switching model does not restart the conversation, because the
history is ours and not the vendor's.

Everything a chat needs is a slash command, and every slash command is in one
table: it is what `/help` prints, what TAB completes, and what the tests drive.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import queue
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from onecli import __version__, config, presence, providers, render, session, tui, unavailable
from onecli.context import CostMeter, last_exchange_start, usage_tokens
from onecli.providers.base import ProviderError

HISTORY_FILE = pathlib.Path.home() / ".one-cli" / "chat_history"
HISTORY_MAX = 5000

DEFAULT_SYSTEM = (
    "You are one, a terminal chat assistant.\n"
    "Answer the question that was asked, in as few words as it takes. No "
    "preamble, no restating the question, no closing summary. Plain text: the "
    "reader is in a terminal, so headings and decoration cost more than they "
    "give. Code goes in a fenced block. Never use emoji.\n"
    "If you do not know something, say so instead of guessing."
)


# --- what the chat can reach for -------------------------------------------
#
# `auto` is the default surface: everything needed to answer a question about
# this machine. Four of its tools only look; `bash` can do anything, so it is
# the one that stops and asks first - a question about free disk space is a
# command, and a chat that cannot run one is answering from imagination.
# `all` is the agent's own surface - edits, background jobs - and is a thing you
# ask for. `off` is the pure fast lane, where no schema is sent at all.
#
# The rule the gate turns on: a call that can change something needs a person
# to say yes. With nobody at the keyboard there is no yes to be had, so `auto`
# refuses and `all` proceeds - because `all` was typed at the command line and
# is itself the answer, while `auto` is the default nobody chose.

TOOL_MODES = ("off", "auto", "all")
DEFAULT_TOOL_MODE = "auto"
CHAT_TOOLS = ["read", "ls", "glob", "grep", "bash", "web_search", "web_fetch"]

TOOL_SYSTEM = """\
You can look things up before answering. Tools on this turn: {names}.
Working directory: {cwd}

Reach for one only when the answer depends on something you cannot know from \
this conversation - a file, the state of this machine, something on the web. \
Anything you already know, answer straight away; a question about how something \
works does not need a tool call first.

For substantial work, give a short initial update and report meaningful findings, \
blockers or changes of approach. The reader can see routine tool calls; do not \
narrate each one, repeat summaries, or ask permission to read something. Finish \
with one concise answer covering the outcome and relevant verification.

A command that could change something is shown to the person before it runs, \
and they may refuse it. Prefer the command that only reads."""


def tool_surface(mode: str) -> list[str] | None:
    """The tool names offered on a turn. None means the whole registry.

    Resolved against what is actually registered, not against a wish list: the
    web tools only exist when an extension registers them, and a schema for a
    tool the registry cannot run is a promise the loop then breaks.
    """
    if mode == "all":
        return None
    from onecli import tools as toolkit  # lazy: the fast lane never pays for it

    return [n for n in CHAT_TOOLS if n in toolkit.REGISTRY]


def chat_surface(chat) -> list[str] | None:
    """The tool mode's surface, narrowed by the permission mode.

    `--read-only` and `--permission-mode plan` withhold what changes things
    here exactly as they do for `one`; the ReadOnlyGuard in the chain is the
    second layer, for a call the model names without being offered it.
    """
    if chat.tools == "off":
        return []
    names = tool_surface(chat.tools)
    if chat.mode not in ("readOnly", "plan"):
        return names
    from onecli import permissions
    allowed = permissions.tool_names_for(chat.mode) or []
    return allowed if names is None else [n for n in names if n in allowed]


def strip_tools(messages: list[dict]) -> list[dict]:
    """The conversation with the tool plumbing folded away.

    A turn that declares no tools cannot carry tool calls: hand an Anthropic
    endpoint a `tool_use` block with no `tools` array to match it and the whole
    request is rejected, so `/tools off` after a stretch of tool use would break
    the conversation rather than quieten it. What the model said survives; the
    calls and their results do not.
    """
    kept = []
    for msg in messages:
        if msg.get("role") == "tool":
            continue
        if msg.get("tool_calls"):
            content = msg.get("content")
            if not content:
                continue
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
        kept.append(msg)
    return kept


class InputLane:
    """Non-blocking background input lane that keeps the typing lane open at all times.

    Monitors stdin while the agent and tools execute so typed user messages
    are queued without blocking the runtime or corrupting the stream.
    """

    def __init__(self, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self.queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
            self._thread = None

    @contextlib.contextmanager
    def paused(self):
        """Hand stdin to a direct reader, then take it back.

        An approval prompt reads the same descriptor; with the lane still
        polling, the two race and the answer can land in the queue instead.
        """
        running = self._thread is not None
        if running:
            self.stop()
        try:
            yield
        finally:
            if running:
                self.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if hasattr(select, "select") and hasattr(self.stream, "fileno"):
                    try:
                        r, _, _ = select.select([self.stream], [], [], 0.1)
                        if not r:
                            continue
                    except (ValueError, OSError):
                        break
                line = self.stream.readline()
                if not line:
                    break
                text = line.strip()
                if text:
                    self.queue.put(text)
            except Exception:
                break

    def pending(self) -> list[str]:
        items = []
        while not self.queue.empty():
            try:
                items.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return items


# --- one conversation ------------------------------------------------------


class Chat:
    """The conversation, its backend, and the dials that shape a turn.

    The history is canonical (`role`/`content`), which is what makes `/model`
    cheap: the wire converts on the way out, so the same messages can be sent
    to an OpenAI endpoint and then to an Anthropic one with nothing rewritten.
    """

    def __init__(self, backend, *, system: str = "", temperature: float | None = None,
                 max_tokens: int | None = None, timeout: float = 120.0, stream: bool = True,
                 thinking: bool = False, sid: str = "", out=None, err=None,
                 tools: str = DEFAULT_TOOL_MODE, mask: str | None = None,
                 workspace: str = "", guard=None, mode: str = "default",
                 trust_project: bool = False, task_options: dict | None = None,
                 hooks=None):
        self.mask = mask
        # The session's hooks (hooks.from_args), the same set `one` builds:
        # UserPromptSubmit runs on every turn, and the tool loop gets the rest.
        self.hooks = hooks
        # The permission stack `one` builds (permissions.chain) - `-w`, the
        # permission mode, allow/deny rules, --guard and --add-dir. Empty
        # workspace means the current directory at turn time; a None guard
        # means the default stack for it, which still confines file tools.
        self.workspace = os.path.realpath(workspace) if workspace else workspace
        self.guard = guard
        self.mode = mode
        self.trust_project = trust_project
        self.task_options = task_options or {"permission_mode": mode}
        self.backend = backend
        self.system = system or DEFAULT_SYSTEM
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stream = stream
        self.thinking = thinking
        self.tools = tools if tools in TOOL_MODES else DEFAULT_TOOL_MODE
        # `all` can change things, so the first thing that would change
        # something asks. Answering "a" stands the question down for the rest
        # of the session; a session with nobody at the keyboard never asks at
        # all, because a one-shot that blocks on a question nobody can see is
        # worse than either answer.
        self.allow_all = False
        self.ask_line = None
        self.sid = sid or session.new_id()
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr

        self.messages: list[dict] = []
        self.staged: list[str] = []          # text queued by /file and !cmd
        self.attachments: list = []          # media queued by /img
        self.turns: list[dict] = []          # per-turn stats, newest last
        self.last_answer = ""
        self.meter = self._meter(backend)
        # Tokens are comparable across models and dollars are not, so the
        # running total keeps the two apart: every turn's tokens, and the money
        # only from the turns whose model had a price.
        self.total_in = 0
        self.total_out = 0
        self.total_usd = 0.0
        self.total_tools = 0
        self.priced = True
        self._jobs = None
        self._wire = None
        self._names: list[config.Backend] = []
        self.input_lane = InputLane(stream=sys.stdin)
        self.input_queue: list[str] = []
        from onecli.tasks import TaskManager
        self.tasks = TaskManager(
            execute=True,
            **self.task_options,
            workspace=self.workspace or os.getcwd(),
            default_model=getattr(backend, "name", "") or "",
        )
        from onecli.checkpoints import CheckpointManager
        self.checkpoints = CheckpointManager(workspace=self.workspace or os.getcwd(), sid=self.sid)
        self._checkpoint_note = ""

    # -- backend ------------------------------------------------------------

    @staticmethod
    def _meter(backend) -> CostMeter:
        price = config.price_for(backend.model, backend.name)
        return CostMeter(model=backend.model,
                         price_in=(price or (0.0, 0.0))[0],
                         price_out=(price or (0.0, 0.0))[1],
                         price_known=price is not None)

    @property
    def wire(self):
        if self._wire is None:
            self._wire = providers.build(self.backend, self.timeout)
        return providers.bind_session(self._wire, self.sid)

    def use(self, backend) -> None:
        """Point the conversation at another backend, keeping every message.

        A fresh meter, because a dollar figure that spans two price tables is a
        number nobody can check. The tokens already spent are reported by the
        turn log, which is per-turn and keeps its own model name.
        """
        self.backend = backend
        self._wire = None
        self.meter = self._meter(backend)

    def backends(self, refresh: bool = False) -> list[config.Backend]:
        """Discovered backends, cached: discovery probes the local llama-server
        and a TAB completion must not pay for that on every keypress."""
        if refresh or not self._names:
            self._names = config.discover()
        return self._names

    # -- background jobs ----------------------------------------------------

    def jobs(self):
        """The background-job store, made on the turn that first needs one.

        A long command does not have to hold the conversation still: in `all`
        the model can start it, keep answering, and be told when it lands. The
        store is per-session and nothing outlives the session that started it.
        """
        if self._jobs is None:
            from onecli import jobs as joblib  # lazy: only `all` reaches this

            self._jobs = joblib.JobStore(self.sid)
        return self._jobs

    def close(self) -> list[str]:
        """Stop anything still running. Returns what had to be killed."""
        survivors: list[str] = []
        if self._jobs is not None:
            survivors.extend(self._jobs.close())
            self._jobs = None
        if self.tasks is not None:
            survivors.extend(self.tasks.close())
        from onecli.tools import shell as shelltool  # lazy, like the job store
        survivors.extend(shelltool.reap())
        self.checkpoints.close()
        return survivors

    # -- history ------------------------------------------------------------

    def history(self, *, tools: bool = False) -> list[dict]:
        """The conversation as this turn sends it.

        With tools, the transcript goes out whole and the system prompt says
        what may be reached for: the calls and their results are what the next
        step reasons from. Without, the plumbing is folded away, because a turn
        that declares no tools may not carry any.
        """
        if not tools:
            sys_content = self.system
            if self.tasks:
                task_prompt = self.tasks.context_prompt()
                if task_prompt:
                    sys_content += "\n\n" + task_prompt
            return ([{"role": "system", "content": sys_content}]
                    + strip_tools(self.messages))
        names = chat_surface(self)
        listed = ", ".join(names) if names else "every tool one has"
        system = self.system + "\n\n" + TOOL_SYSTEM.format(
            names=listed, cwd=self.workspace or os.getcwd())
        if self.tasks:
            task_prompt = self.tasks.context_prompt()
            if task_prompt:
                system += "\n\n" + task_prompt
        return [{"role": "system", "content": system}] + self.messages

    def reset(self, keep_system: bool = True) -> None:
        self.messages = []
        self.turns = []
        self.last_answer = ""
        self.staged = []
        self.attachments = []
        self.sid = session.new_id()
        self.meter = self._meter(self.backend)
        self.total_in = self.total_out = 0
        self.total_usd = 0.0
        self.total_tools = 0
        self.allow_all = False
        self.priced = True
        from onecli.tasks import TaskManager
        self.tasks.close()
        self.tasks = TaskManager(
            execute=True,
            **self.task_options,
            workspace=self.workspace or os.getcwd(),
            default_model=getattr(self.backend, "name", "") or "",
        )
        self.checkpoints.reset_turns()
        if not keep_system:
            self.system = DEFAULT_SYSTEM

    def last_user(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                return " ".join(b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text")
        return ""

    def undo(self) -> bool:
        """Drop the last exchange, and its turn's checkpoint record unrestored.
        Returns False when there is nothing to drop."""
        if not self.drop_exchange():
            return False
        self.checkpoints.forget_turn()
        return True

    def drop_exchange(self) -> bool:
        """Drop the last turn from the conversation only: from where it began
        when that is recorded (a turn can hold several user-role messages),
        else from its last request."""
        cut = self.checkpoints.turn_start(self.messages)
        if cut is None:
            cut = last_exchange_start(self.messages)
        if cut is None:
            return False
        del self.messages[cut:]
        if self.turns:
            self.turns.pop()
        self.last_answer = ""
        return True

    def compose(self, text: str) -> str | list:
        """Fold anything staged by /file, /img or !cmd into this turn.

        Staged context rides the turn it was staged for and is then gone: a
        file pasted into one question must not silently re-attach itself to
        every later one.
        """
        body, attachments = self.compose_parts(text)
        if not attachments:
            return body
        from onecli import media  # lazy: a text chat never pays for it

        return media.user_content(body, attachments)

    def compose_parts(self, text: str) -> tuple[str, list]:
        """The same fold, left in two pieces.

        The tool loop builds its own user message from a prompt and a list of
        attachments, so handing it a pre-folded content list would mean taking
        the images back out again.
        """
        body = text
        if self.staged:
            body = "\n\n".join(self.staged + [text]) if text else "\n\n".join(self.staged)
        self.staged = []
        attachments = self.attachments
        self.attachments = []
        return body, attachments

    # -- the turn -----------------------------------------------------------

    def ask(self, text: str, echo: bool = True) -> dict:
        """One turn. Streams to `self.out` when echoing, and always returns the
        stats the footer and `/cost` are built from."""
        content = self.compose(text)
        self.messages.append({"role": "user", "content": content})
        history = self.history()

        state = {"ttft": 0.0, "printed": False}
        started = time.monotonic()

        def on_delta(kind: str, chunk: str) -> None:
            if not state["ttft"]:
                state["ttft"] = time.monotonic() - started
            if kind == "thinking" and not self.thinking:
                return
            if not echo:
                return
            # Model text is untrusted terminal input (render.untrusted).
            chunk = render.untrusted(chunk)
            self.out.write(render.faint(chunk) if kind == "thinking" else chunk)
            self.out.flush()
            state["printed"] = True

        try:
            turn = self.wire.chat(history, tools=None,
                                  temperature=self.temperature,
                                  max_tokens=self.max_tokens,
                                  on_delta=on_delta if (self.stream and echo) else None)
        except KeyboardInterrupt:
            # Keep the unfinished request, including staged attachments, across
            # the next turn and session resume. No completed answer is implied.
            self.messages.append({"role": "user", "content":
                "The previous turn was interrupted. Its request remains unfinished unless my next message cancels or replaces it. Consider my newest message together with that request."})
            session.append(self.sid, {"kind": "messages", "messages": self.messages})
            raise
        except ProviderError as exc:
            self.messages.pop()
            return {"error": str(exc), "seconds": time.monotonic() - started}

        elapsed = time.monotonic() - started
        text_out = turn.text or ""
        if state["printed"]:
            self.out.write("\n")
            self.out.flush()
        elif echo and text_out:
            self.out.write(render.untrusted(text_out) + "\n")
            self.out.flush()

        self.messages.append({"role": "assistant", "content": text_out})
        self.last_answer = text_out
        before = (self.meter.input_tokens, self.meter.output_tokens, self.meter.usd)
        self.meter.add_turn(history, turn.usage, text_out)
        self.total_in += self.meter.input_tokens - before[0]
        self.total_out += self.meter.output_tokens - before[1]
        self.total_usd += self.meter.usd - before[2]
        self.priced = self.priced and self.meter.price_known
        stat = {
            "model": self.backend.name,
            "seconds": elapsed,
            "ttft": state["ttft"],
            "input_tokens": self.meter.input_tokens - before[0],
            "output_tokens": self.meter.output_tokens - before[1],
            "thinking": turn.thinking or "",
            "text": text_out,
            "usd": self.meter.usd,
        }
        self.turns.append(stat)
        session.append(self.sid, {"kind": "messages", "messages": self.messages})
        return stat

    # -- footer -------------------------------------------------------------

    def footer(self, stat: dict) -> str:
        bits = [self.backend.display]
        if stat.get("ttft"):
            bits.append(f"{stat['ttft']:.2f}s to first token")
        bits.append(f"{stat['seconds']:.1f}s")
        out_tokens = stat.get("output_tokens") or 0
        if out_tokens and stat["seconds"] > 0:
            bits.append(f"{out_tokens / stat['seconds']:.0f} tok/s")
        bits.append(f"{_thousands(stat.get('input_tokens') or 0)} in · "
                    f"{_thousands(out_tokens)} out")
        calls = stat.get("tool_calls") or 0
        if calls:
            bits.append(f"{calls} tool call" + ("s" if calls != 1 else ""))
        if self.total_usd:
            bits.append(f"${self.total_usd:.4f}")
        return render.footer(bits)


def _thousands(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


# --- the command table -----------------------------------------------------
#
# One table, three consumers: /help, TAB completion, and the dispatcher. A
# command that is not in here does not exist, which is the point - there is no
# second place to look for what a chat can do.


class Command:
    def __init__(self, name, usage, blurb, fn, aliases):
        self.name, self.usage, self.blurb, self.fn, self.aliases = (
            name, usage, blurb, fn, aliases)


COMMANDS: dict[str, Command] = {}
ALIASES: dict[str, str] = {}


def command(name: str, usage: str, blurb: str, *aliases: str):
    def deco(fn):
        COMMANDS[name] = Command(name, usage, blurb, fn, aliases)
        for alias in aliases:
            ALIASES[alias] = name
        return fn
    return deco


def lookup(name: str) -> Command | None:
    name = name.lstrip("/")
    if name in COMMANDS:
        return COMMANDS[name]
    if name in ALIASES:
        return COMMANDS[ALIASES[name]]
    return None


def dispatch(chat: Chat, line: str) -> bool:
    """Run one slash command. Returns False when the session should end."""
    head, _, rest = line.strip().partition(" ")
    cmd = lookup(head)
    if cmd is None:
        near = _nearest(head.lstrip("/"))
        hint = f" - did you mean /{near}?" if near else ""
        _say(chat, render.red(f"no command {head}{hint}"))
        return True
    try:
        result = cmd.fn(chat, rest.strip())
    except Exception as exc:
        _say(chat, render.red(f"{head} failed: {render.untrusted(str(exc))}"))
        return True
    return True if result is None else bool(result)


def _nearest(name: str) -> str:
    """The command a typo most likely meant, or "". Prefix match first, because
    a chat's typos are truncations far more often than they are transpositions."""
    if not name:
        return ""
    everything = list(COMMANDS) + list(ALIASES)
    for candidate in everything:
        if candidate.startswith(name) or name.startswith(candidate):
            return candidate
    return ""


def _say(chat: Chat, text: str) -> None:
    print(text, file=chat.err)


# -- what a chat can do -----------------------------------------------------


@command("help", "/help", "This list.", "h", "?")
def _help(chat: Chat, arg: str):
    _say(chat, render.kicker("commands"))
    for name in sorted(COMMANDS):
        cmd = COMMANDS[name]
        # The aliases too. They were registered and never printed, so /quit,
        # /q, /h, /? and /ls all worked and none of them were findable - a
        # persona reported /quit as an undocumented command after using it.
        line = "  " + render.ink(cmd.usage.ljust(26)) + render.dim(cmd.blurb)
        if cmd.aliases:
            line += render.faint("  also " + " ".join("/" + a
                                                      for a in cmd.aliases))
        _say(chat, line)
    _say(chat, render.dim("  a line ending in \\ continues on the next; \"\"\" "
                          "opens a paste block; !cmd runs a shell command."))


@command("exit", "/exit", "End the session.", "quit", "q")
def _exit(chat: Chat, arg: str):
    return False


@command("mask", "/mask [name|off]", "Inspect or switch the task mask.")
def _mask(chat: Chat, arg: str):
    from onecli.masks import listing, select
    try:
        if not arg:
            _say(chat, listing(chat.mask))
            return
        select(arg)
        chat.mask = arg
        session.append(chat.sid, {'kind': 'metadata', 'mask': arg})
        _say(chat, f'Mask: {arg}. Applies to the next turn; conversation retained.')
    except ValueError as exc:
        _say(chat, render.red(str(exc)))


@command("models", "/models [refresh]", "Every backend one-cli can reach.", "ls")
def _models(chat: Chat, arg: str):
    found = chat.backends(refresh=arg.strip() == "refresh")
    if not found:
        _say(chat, render.red("no backends found"))
        return
    _say(chat, render.kicker("backends"))
    for b in found:
        live = b.name == chat.backend.name
        mark = render.accent(render.glyph("◆")) if live else " "
        _say(chat, f"  {mark} " + render.ink(f"{b.name:38}")
             + render.dim(f"{b.wire:10}") + render.faint(b.base_url))
    _say(chat, render.footer([f"{len(found)} backends", chat.backend.display]))


@command("model", "/model [name] [model-id]",
         "Switch backend, keeping the conversation.", "m")
def _model(chat: Chat, arg: str):
    if not arg:
        from onecli.model_picker import open_picker
        _say(chat, open_picker(chat, chat.backends(refresh=True)))
        return
    # A second word is a model id on that same endpoint: one gateway serves a
    # vendor's whole tier list, so naming the tier must not need a new drop-in.
    name, _, model_id = arg.partition(" ")
    try:
        backend = config.resolve(name=name.strip(), model=model_id.strip() or None)
    except RuntimeError as exc:
        _say(chat, render.red(str(exc)))
        return
    refused = presence.refusal(backend)
    if refused:
        _say(chat, render.red(refused))
        return
    chat.use(backend)
    _say(chat, render.dim(f"→ {backend.display}  ({len(chat.messages)} messages carried over)"))


@command("new", "/new", "Forget the conversation, keep the settings.", "clear", "reset")
def _new(chat: Chat, arg: str):
    chat.reset()
    _say(chat, render.dim(f"new conversation - session {chat.sid}"))


@command("system", "/system [text|@file]", "Show or replace the system prompt.", "sys")
def _system(chat: Chat, arg: str):
    if not arg:
        _say(chat, render.dim(render.untrusted(chat.system)))
        return
    if arg.startswith("@"):
        path = pathlib.Path(os.path.expanduser(arg[1:]))
        try:
            chat.system = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            _say(chat, render.red(f"cannot read {path}: {exc}"))
            return
    else:
        chat.system = arg
    _say(chat, render.dim(f"system prompt set ({len(chat.system)} chars)"))


@command("temp", "/temp <0.0-2.0>", "Sampling temperature.", "temperature")
def _temp(chat: Chat, arg: str):
    if not arg:
        _say(chat, render.dim(f"temperature {chat.temperature}"))
        return
    try:
        chat.temperature = float(arg)
    except ValueError:
        _say(chat, render.red(f"not a number: {arg}"))
        return
    _say(chat, render.dim(f"temperature {chat.temperature}"))


@command("max", "/max <n>", "Max tokens in a reply.", "max-tokens")
def _max(chat: Chat, arg: str):
    if not arg:
        _say(chat, render.dim(f"max tokens {chat.max_tokens}"))
        return
    try:
        chat.max_tokens = int(arg)
    except ValueError:
        _say(chat, render.red(f"not a number: {arg}"))
        return
    _say(chat, render.dim(f"max tokens {chat.max_tokens}"))


@command("stream", "/stream [on|off]", "Print the answer as it arrives.")
def _stream(chat: Chat, arg: str):
    chat.stream = _toggle(arg, chat.stream)
    _say(chat, render.dim(f"stream {'on' if chat.stream else 'off'}"))


@command("think", "/think [on|off]", "Show a thinking model's scratchpad.")
def _think(chat: Chat, arg: str):
    chat.thinking = _toggle(arg, chat.thinking)
    _say(chat, render.dim(f"thinking {'shown' if chat.thinking else 'hidden'}"))


def _toggle(arg: str, current: bool) -> bool:
    arg = arg.strip().lower()
    if arg in ("on", "yes", "1", "true"):
        return True
    if arg in ("off", "no", "0", "false"):
        return False
    return not current


@command("retry", "/retry [model]", "Ask the last question again, optionally elsewhere.", "r")
def _retry(chat: Chat, arg: str):
    prompt = chat.last_user()
    if not prompt:
        _say(chat, render.red("nothing to retry"))
        return
    chat.undo()
    if arg:
        _model(chat, arg)
    stat = turn(chat, prompt)
    _report(chat, stat)


@command("undo", "/undo [id]", "Revert last turn workspace edits and drop its question.", "u")
def _undo(chat: Chat, arg: str):
    if not chat.messages:
        _say(chat, render.red("nothing to undo"))
        return
    cp_id = int(arg.strip()) if arg.strip().isdigit() else None
    # The workspace and the conversation move together, one turn each: a turn
    # without a checkpoint refuses rather than rolling back an older turn too.
    res = chat.checkpoints.undo_turn(chat.messages, cp_id)
    if not res.success:
        _say(chat, render.red(f"cannot undo: {render.untrusted(str(res.error))}"))
        return
    cut = len(chat.messages) if res.cut is None else res.cut
    msg_dropped = len(chat.messages) - cut
    del chat.messages[cut:]
    del chat.turns[max(0, len(chat.turns) - res.turns):]
    chat.last_answer = ""
    if res.checkpoint_id:
        details = []
        if res.reverted_files:
            details.append(f"{len(res.reverted_files)} file(s) restored")
        if res.removed_files:
            details.append(f"{len(res.removed_files)} untracked file(s) removed")
        detail_str = f" ({', '.join(details)})" if details else ""
        _say(chat, render.paint("brand", f"reverted workspace to checkpoint #{res.checkpoint_id}{detail_str}"))
    if msg_dropped:
        _say(chat, render.dim(f"dropped - {len(chat.messages)} messages left"))


@command("diff", "/diff [id]", "Show workspace diff against turn checkpoint.", "d")
def _diff(chat: Chat, arg: str):
    if not hasattr(chat, "checkpoints") or not chat.checkpoints:
        _say(chat, render.dim("no checkpoints available"))
        return
    cp_id = int(arg.strip()) if arg.strip().isdigit() else None
    diff_text = chat.checkpoints.get_diff(cp_id)
    _say(chat, render.untrusted(diff_text))


@command("checkpoints", "/checkpoints", "List turn checkpoints in this session.")
def _checkpoints(chat: Chat, arg: str):
    if not hasattr(chat, "checkpoints") or not chat.checkpoints:
        _say(chat, render.dim("no checkpoints available"))
        return
    cps = chat.checkpoints.list_checkpoints()
    if not cps:
        _say(chat, render.dim("no checkpoints recorded in this session"))
        return
    _say(chat, render.kicker(f"checkpoints · {len(cps)}"))
    for cp in cps:
        stat = chat.checkpoints.get_diff_stat(cp.id)
        stat_str = f"  {stat}" if stat else ""
        _say(chat, f"  #{cp.id} " + render.ink(render.untrusted(cp.prompt)) + render.dim(render.untrusted(stat_str)))


@command("edit", "/edit", "Reopen the last question in $EDITOR and resend it.", "e")
def _edit(chat: Chat, arg: str):
    prompt = chat.last_user()
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as fh:
        fh.write(prompt)
        path = fh.name
    try:
        subprocess.call([editor, path])
        edited = pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        _say(chat, render.red(f"cannot run {editor}: {exc}"))
        return
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if not edited:
        _say(chat, render.dim("empty - nothing sent"))
        return
    if prompt:
        chat.undo()
    stat = turn(chat, edited)
    _report(chat, stat)


@command("copy", "/copy", "Put the last answer on the clipboard.", "y")
def _copy(chat: Chat, arg: str):
    if not chat.last_answer:
        _say(chat, render.red("no answer yet"))
        return
    from onecli import terminal
    text = chat.last_answer
    # The terminal's own clipboard (OSC 52) works across remote logins and
    # tmux; a local tool is only the fallback when there is no live session.
    if terminal.ACTIVE is not None and terminal.ACTIVE.clipboard(text):
        _say(chat, render.dim(f"copied {len(text)} chars via the terminal"))
        return
    for tool in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["pbcopy"]):
        if shutil.which(tool[0]):
            try:
                subprocess.run(tool, input=text.encode(), check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                _say(chat, render.red(f"{tool[0]}: {exc}"))
                return
            _say(chat, render.dim(f"copied {len(text)} chars"))
            return
    _say(chat, render.red("no clipboard tool (wl-copy, xclip, pbcopy)"))


@command("save", "/save [path]", "Write the conversation as markdown.")
def _save(chat: Chat, arg: str):
    path = pathlib.Path(os.path.expanduser(arg)) if arg else pathlib.Path(
        f"chat-{chat.sid}.md")
    from onecli import secureio
    try:
        # A transcript carries everything said and every tool output: owner-only.
        secureio.atomic_write_text(path, transcript(chat), private=True)
    except OSError as exc:
        _say(chat, render.red(f"cannot write {path}: {exc}"))
        return
    _say(chat, render.dim(f"saved {path}"))


def transcript(chat: Chat) -> str:
    """The conversation as markdown - what /save writes and a test asserts on."""
    lines = [f"# one chat {chat.sid}", "",
             f"- backend: {chat.backend.display}",
             f"- wire: {chat.backend.wire}", ""]
    for msg in strip_tools(chat.messages):
        content = msg.get("content")
        if not isinstance(content, str):
            content = " ".join(b.get("text", "") for b in content or []
                               if isinstance(b, dict) and b.get("type") == "text")
        lines.append("## " + ("you" if msg.get("role") == "user" else "one"))
        lines.append("")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


@command("sessions", "/sessions", "Saved sessions, newest first.")
def _sessions(chat: Chat, arg: str):
    rows = session.listing()
    if not rows:
        _say(chat, render.dim("no sessions yet"))
        return
    _say(chat, render.kicker("sessions"))
    for sid, mtime, _size in rows:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        # The first question, not the byte count: an id and a size cannot tell
        # you which of yesterday's twelve sessions is the one you want back.
        _say(chat, "  " + render.ink(sid) + "  " + render.dim(stamp) + "  "
             + render.faint(render.one_line(render.untrusted(session.opening_line(sid)), 56)))


@command("resume", "/resume [id|@last]", "Continue a saved conversation.")
def _resume(chat: Chat, arg: str):
    want = arg.strip() or "@last"
    sid = session.latest_id() if want == "@last" else want
    if not sid:
        _say(chat, render.red("no session to resume"))
        return
    try:
        messages = session.load_messages(sid)
        from onecli.masks import select
        restored = chat.mask if chat.mask is not None else session.metadata(sid).get('mask')
        select(restored)
    except (FileNotFoundError, ValueError) as exc:
        _say(chat, render.red(str(exc)))
        return
    chat.mask = restored
    # The stored history carries the system message of the run that wrote it;
    # ours lives in a field, so it is lifted out rather than duplicated.
    chat.messages = [m for m in messages if m.get("role") != "system"]
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            chat.system = m["content"]
            break
    chat.sid = sid
    chat.checkpoints.reset_turns()
    _say(chat, render.dim(f"resumed {sid} - {len(chat.messages)} messages"))


@command("cost", "/cost", "Tokens and dollars for this conversation.", "usage")
def _cost(chat: Chat, arg: str):
    """The whole conversation, not the current backend.

    A `/model` switch resets the price table, not the conversation, so the
    totals here span every model that answered - and the money is flagged as
    partial when one of them had no price, rather than quietly under-reporting.
    """
    turns = len(chat.turns)
    answered = {t.get("model") for t in chat.turns}
    bits = [f"{_thousands(chat.total_in)} in", f"{_thousands(chat.total_out)} out",
            f"{turns} turn" + ("" if turns == 1 else "s")]
    if chat.total_usd:
        bits.append(f"${chat.total_usd:.4f}" + ("" if chat.priced else " (partial)"))
    elif len(answered) > 1:
        bits.append("no price for these models")
    else:
        bits.append("no price for this model")
    if chat.meter.estimated:
        bits.append("some turns estimated")
    _say(chat, render.footer(bits))


@command("history", "/history", "The conversation so far.", "hist")
def _history(chat: Chat, arg: str):
    if not chat.messages:
        _say(chat, render.dim("nothing yet"))
        return
    for msg in chat.messages:
        who = "you" if msg.get("role") == "user" else "one"
        content = msg.get("content")
        if not isinstance(content, str):
            content = " ".join(b.get("text", "") for b in content or []
                               if isinstance(b, dict) and b.get("type") == "text")
        _say(chat, render.dim(f"{who}: ") + render.one_line(render.untrusted(content), 88))


@command("file", "/file <path>", "Stage a text file into the next question.", "f")
def _file(chat: Chat, arg: str):
    if not arg:
        _say(chat, render.red("usage: /file <path>"))
        return
    path = pathlib.Path(os.path.expanduser(arg))
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _say(chat, render.red(f"cannot read {path}: {exc}"))
        return
    chat.staged.append(f"--- {path.name} ---\n{body}\n--- end {path.name} ---")
    _say(chat, render.dim(f"staged {path.name} ({len(body)} chars)"))


@command("img", "/img <path>", "Stage an image into the next question.", "image")
def _img(chat: Chat, arg: str):
    if not arg:
        _say(chat, render.red("usage: /img <path>"))
        return
    from onecli import media

    try:
        chat.attachments.append(media.load(os.path.expanduser(arg), expect="image"))
    except media.MediaError as exc:
        _say(chat, render.red(str(exc)))
        return
    _say(chat, render.dim(f"staged {len(chat.attachments)} image(s) - "
                          f"the next question carries them"))


@command("compare", "/compare <a,b,..> [question]",
         "Ask several backends the same thing, at once.", "cmp", "vs")
def _compare(chat: Chat, arg: str):
    names, _, prompt = arg.partition(" ")
    wanted = [n for n in names.split(",") if n.strip()]
    prompt = prompt.strip() or chat.last_user()
    if not wanted or not prompt:
        _say(chat, render.red("usage: /compare model-a,model-b [question]"))
        return
    for row in fan_out(chat, wanted, prompt):
        _say(chat, render.rule())
        _say(chat, render.kicker(row["name"]) + "  "
             + render.faint(row.get("label", "")))
        if row.get("error"):
            _say(chat, render.red(render.untrusted(row["error"])))
            continue
        print(render.untrusted(row["text"]), file=chat.err)
        _say(chat, render.dim(f"  {row['seconds']:.1f}s · "
                              f"{_thousands(row['output_tokens'])} out"))


def fan_out(chat: Chat, names: list[str], prompt: str) -> list[dict]:
    """The same question to several backends, in parallel, in the order asked.

    Not streamed and not added to the history: a comparison is a side question
    about the models, so answering it must not change what the conversation
    remembers.
    """
    import concurrent.futures as futures  # lazy: only a comparison needs threads

    submitted, refusal = _submit(chat, prompt)
    if submitted is None:
        return [{"name": name, "error": f"prompt blocked: {refusal}", "seconds": 0.0}
                for name in names]
    prompt = submitted
    history = chat.history() + [{"role": "user", "content": prompt}]

    def one(name: str) -> dict:
        started = time.monotonic()
        try:
            backend = config.resolve(name=name)
            wire = providers.bind_session(providers.build(backend, chat.timeout), chat.sid)
            turn = wire.chat(history, tools=None, temperature=chat.temperature,
                             max_tokens=chat.max_tokens)
        except (RuntimeError, ProviderError) as exc:
            return {"name": name, "error": str(exc),
                    "seconds": time.monotonic() - started}
        _, out_tokens = usage_tokens(turn.usage or {})
        # The short name is the header - a kicker letterspaces what it is
        # given, and a full display label comes out as a wall of gapped
        # capitals. The label rides alongside it instead.
        return {"name": backend.name, "label": backend.display,
                "text": turn.text or "", "seconds": time.monotonic() - started,
                "output_tokens": out_tokens}

    with futures.ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
        return list(pool.map(one, names))


@command("tools", "/tools [off|auto|all]",
         "What the chat may reach for while it answers.")
def _tools(chat: Chat, arg: str):
    want = arg.strip().lower()
    if want in ("on", "read", "chat"):
        want = "auto"       # what the first cut called it
    if not want:
        _say(chat, render.dim(f"tools {chat.tools}") + "  "
             + render.faint(_surface_line(chat)))
        return
    if want not in TOOL_MODES:
        _say(chat, render.red("tools takes " + ", ".join(TOOL_MODES)))
        return
    chat.tools = want
    if want == "off":
        _say(chat, render.dim("tools off - straight to the model, "
                              "nothing is read"))
    elif want == "auto":
        _say(chat, render.dim("tools auto - reads and searches while it "
                              "answers; bash asks before it runs"))
    else:
        # The one state worth a colour: everything the agent can do, in a
        # window whose other turns look like a conversation.
        _say(chat, render.yellow("tools all - the chat can now write files, "
                                 "run commands and start background jobs"))


def _surface_line(chat: Chat) -> str:
    if chat.tools == "off":
        return "no tools on the wire"
    names = tool_surface(chat.tools)
    return ", ".join(names) if names else "every tool one has"


@command("tasks", "/tasks", "List active and finished detached offload tasks.")
def _tasks(chat: Chat, arg: str):
    if not chat.tasks:
        _say(chat, render.dim("no background tasks active"))
        return
    chat.tasks.poll()
    if not chat.tasks.list_all():
        _say(chat, render.dim("no background tasks active"))
        return
    _say(chat, render.kicker(f"detached tasks · {len(chat.tasks.list_all())}"))
    for t in chat.tasks.list_all():
        card = chat.tasks.render_card(t.task_id)
        _say(chat, card)


@command("task", "/task [spawn <title> | advance <id> | done <id> | cancel <id> | log <id> | show <id>]",
         "Manage detached offload tasks with procedural ASCII cards.")
def _task(chat: Chat, arg: str):
    if not arg:
        return _tasks(chat, "")
    if chat.tasks:
        chat.tasks.poll()
    parts = arg.split(None, 1)
    subcmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "spawn":
        from onecli.tasks import parse_spawn_args
        prompt, runner, model, workspace = parse_spawn_args(rest)
        title = prompt.splitlines()[0][:80] if prompt else "Detached Background Task"
        t = chat.tasks.spawn(title=title, prompt=prompt, runner=runner, model=model, workspace=workspace)
        _say(chat, render.paint("brand", f"detached task #{t.task_id} spawned"))
        _say(chat, chat.tasks.render_card(t.task_id))
    elif subcmd == "advance":
        tid = rest or (chat.tasks.list_active()[0].task_id if chat.tasks.list_active() else "")
        found = chat.tasks.get(tid)
        if not found:
            _say(chat, render.red(f"task #{tid} not found"))
            return
        found.advance_phase()
        _say(chat, chat.tasks.render_card(tid))
    elif subcmd == "done":
        tid = rest or (chat.tasks.list_active()[0].task_id if chat.tasks.list_active() else "")
        found = chat.tasks.get(tid)
        if not found:
            _say(chat, render.red(f"task #{tid} not found"))
            return
        found.complete()
        _say(chat, chat.tasks.render_card(tid))
    elif subcmd == "cancel":
        tid = rest or (chat.tasks.list_active()[0].task_id if chat.tasks.list_active() else "")
        found = chat.tasks.get(tid)
        if not found:
            _say(chat, render.red(f"task #{tid} not found"))
            return
        found.cancel()
        _say(chat, render.yellow(f"task #{tid} cancelled"))
        _say(chat, chat.tasks.render_card(tid))
    elif subcmd in ("log", "output"):
        tid = rest or (chat.tasks.list_all()[-1].task_id if chat.tasks.list_all() else "")
        found = chat.tasks.get(tid)
        if not found:
            _say(chat, render.red(f"task #{tid} not found"))
            return
        _say(chat, render.untrusted(found.output()))
    elif subcmd in ("show", "card"):
        tid = rest or (chat.tasks.list_all()[-1].task_id if chat.tasks.list_all() else "")
        found = chat.tasks.get(tid)
        if not found:
            _say(chat, render.red(f"task #{tid} not found"))
            return
        _say(chat, chat.tasks.render_card(tid))
    else:
        from onecli.tasks import parse_spawn_args
        prompt, runner, model, workspace = parse_spawn_args(arg)
        title = prompt.splitlines()[0][:80] if prompt else "Detached Background Task"
        t = chat.tasks.spawn(title=title, prompt=prompt, runner=runner, model=model, workspace=workspace)
        _say(chat, render.paint("brand", f"detached task #{t.task_id} spawned"))
        _say(chat, chat.tasks.render_card(t.task_id))


@command("queue", "/queue [clear]", "Show or clear pending message and task queues.")
def _queue(chat: Chat, arg: str):
    if (arg or "").strip().lower() == "clear":
        chat.input_queue.clear()
        _say(chat, render.paint("brand", "message queue cleared"))
        return
    lines = []
    lines.append(render.kicker("queues"))
    if chat.input_queue:
        lines.append(render.paint("accent", f"  Message Queue ({len(chat.input_queue)} pending):"))
        for i, q_msg in enumerate(chat.input_queue, 1):
            lines.append(f"    {i}. {render.ink(render.untrusted(q_msg[:70]))}")
    else:
        lines.append(render.dim("  Message Queue: empty (typing lane open)"))

    if chat.tasks:
        active = chat.tasks.list_active()
        if active:
            lines.append(render.paint("live", f"  Detached Task Queue ({len(active)} active):"))
            for t in active:
                lines.append(f"    #{t.task_id} [{t.runner.upper()}] {render.untrusted(t.phase_name)} ({t.elapsed:.1f}s): {render.untrusted(t.title[:50])}")
        else:
            lines.append(render.dim("  Detached Task Queue: idle (0 active)"))

    if chat._jobs is not None:
        chat._jobs.poll()
        running_jobs = [j for j in chat._jobs.jobs.values() if j.running]
        if running_jobs:
            lines.append(render.paint("live", f"  Background Jobs ({len(running_jobs)} running):"))
            for j in running_jobs:
                lines.append(f"    {j.id} ({j.seconds:.0f}s): {render.untrusted(j.command[:50])}")
        else:
            lines.append(render.dim("  Background Jobs: idle (0 running)"))
    _say(chat, "\n".join(lines))


@command("bg", "/bg <command>", "Run a shell command in the background.")
def _bg(chat: Chat, arg: str):
    if not arg.strip():
        _say(chat, render.yellow("usage: /bg <command>"))
        return
    store = chat.jobs()
    workspace = getattr(chat, "workspace", "") or os.getcwd()
    try:
        job = store.start(arg.strip(), workspace)
    except OSError as exc:
        _say(chat, render.red(f"error: could not start job: {render.untrusted(str(exc))}"))
        return
    _say(chat, f"job {job.id} started: {render.untrusted(job.command)}\n"
               f"running in background. inspect with /logs {job.id} or /ps, cancel with /kill {job.id}")


@command("ps", "/ps [id]", "List background jobs or view job output.", "jobs")
def _ps(chat: Chat, arg: str):
    store = chat.jobs()
    rest = arg.strip()
    workspace = getattr(chat, "workspace", "") or os.getcwd()
    from onecli import jobs as joblib
    if rest.startswith("cancel "):
        _say(chat, render.untrusted(joblib.job_cancel(workspace, rest.split(None, 1)[1].strip(), jobs=store)))
    elif rest:
        _say(chat, render.untrusted(joblib.job_output(workspace, rest, jobs=store)))
    else:
        _say(chat, render.untrusted(joblib.job_list(workspace, jobs=store)))


@command("kill", "/kill <id>", "Cancel a background job.", "stop")
def _kill(chat: Chat, arg: str):
    if not arg.strip():
        _say(chat, render.yellow("usage: /kill <id>"))
        return
    store = chat.jobs()
    workspace = getattr(chat, "workspace", "") or os.getcwd()
    from onecli import jobs as joblib
    _say(chat, render.untrusted(joblib.job_cancel(workspace, arg.strip(), jobs=store)))


@command("logs", "/logs <id> [offset] [limit]", "View output of a background job.", "log")
def _logs(chat: Chat, arg: str):
    store = chat.jobs()
    workspace = getattr(chat, "workspace", "") or os.getcwd()
    from onecli import jobs as joblib
    parts = arg.strip().split()
    if not parts:
        if store.jobs:
            job_id = list(store.jobs.keys())[-1]
            parts = [job_id]
        else:
            _say(chat, render.yellow("usage: /logs <id> [offset] [limit]"))
            return
    job_id = parts[0]
    offset = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else joblib.DEFAULT_LINES
    _say(chat, render.untrusted(joblib.job_output(workspace, job_id, offset=offset, limit=limit, jobs=store)))


# --- the loop underneath the conversation ----------------------------------


def _tracer(chat: Chat, echo: bool = True) -> Any:
    """Agent events, rendered as a chat rather than as a run.

    The answer streams to stdout and everything else goes to stderr: a piped
    `one-chat "..."` gets the answer and nothing else, exactly as it does when
    no tool was touched. What a call did is one dim line - a chat that prints a
    run's spine at somebody has stopped being a chat.
    """
    from onecli import spinner  # lazy: only a tool turn needs the rail

    state: dict[str, Any] = {"streamed": False, "step_streamed": False, "midline": False, "tools": 0, "spin": None}

    def halt() -> None:
        if state["spin"] is not None:
            state["spin"].stop()
            state["spin"] = None

    def close_line() -> None:
        if state.get("midline"):
            chat.out.write("\n")
            chat.out.flush()
            state["midline"] = False

    def settle() -> None:
        halt()
        close_line()

    def rail(motif, label: str) -> None:
        settle()
        if echo:
            state["spin"] = spinner.Spinner(motif, label, stream=chat.err).start()

    def trace(kind: str, data: dict) -> Any:
        if kind == "turn":
            state["step_streamed"] = False
            rail(spinner.ARRIVE, chat.backend.model)
            return
        if kind == "text_delta":
            if not echo:
                return
            if data["kind"] == "thinking":
                if chat.thinking:
                    settle()
                    chat.out.write(render.faint(render.untrusted(data["chunk"])))
                    chat.out.flush()
                    state["midline"] = not data["chunk"].endswith("\n")
                elif state["spin"] is not None:
                    state["spin"].feed(len(data.get("chunk", "")))
                return
            # The answer arriving is the progress indicator; a rail beside it
            # would be a second, worse one.
            halt()
            chat.out.write(render.untrusted(data["chunk"]))
            chat.out.flush()
            state["streamed"] = True
            state["step_streamed"] = True
            state["midline"] = not data["chunk"].endswith("\n")
            return
        if kind == "tool_start":
            state["tools"] += 1
            call = data["call"]
            # The approval does not depend on echo: without a visible prompt
            # (`one --chat --json`) a mutating call still goes through it.
            if not echo:
                refusal = _confirm(chat, call, attended=False) \
                    if data.get("mutating") else None
                return {"deny": refusal[1]} if refusal else None
            settle()
            _say(chat, render.node(render.untrusted(call.name),
                                   render.untrusted(_detail(call.arguments)),
                                   mutating=bool(data.get("mutating"))))
            if data.get("mutating"):
                refusal = _confirm(chat, call)
                if refusal:
                    state["denied"] = True
                    _say(chat, render.dim("    not run - " + refusal[0]))
                    # A refusal is a tool result, not an exception: a model told
                    # why it was refused can take another route, one handed an
                    # error can only stop.
                    return {"deny": refusal[1]}
            # A subprocess we cannot see inside: its clock is wall time, not
            # data, so the motif is ungated and prints a duration.
            rail(spinner.WORK, render.untrusted(call.name))
            return
        if not echo:
            return
        halt()
        if kind == "tool_end":
            # A refusal has already had its line. The loop reports it twice -
            # once as a denial, once as the tool result the model gets - and a
            # chat that prints both has said the same no three times.
            if state.pop("denied", False):
                return
            output = str(data.get("output") or "")
            from onecli.toolview import plain_result
            _say(chat, plain_result(output, tui.get_terminal_width()))
        elif kind == "tool_denied":
            if state.get("denied"):
                return
            # The name is the model's and the reason a hook's or a guard's:
            # both are untrusted terminal input.
            _say(chat, render.red(f"    {render.untrusted(data['call'].name)} blocked: "
                                  f"{render.untrusted(str(data['verdict'].reason))}"))
        elif kind == "task_spawned":
            settle()
            if chat.tasks:
                _say(chat, render.paint("brand", f"detached task #{data['task_id']} spawned"))
                _say(chat, chat.tasks.render_card(data["task_id"]))
        elif kind == "fallback":
            _say(chat, render.note(render.untrusted(
                f"{data['from']} failed, retrying on {data['to']}")))
        elif kind == "compact":
            _say(chat, render.note("compacted the conversation to fit the "
                                   "context window", token="faint"))

    handle: Any = trace  # function attributes are how callers reach the rail
    handle.halt = halt
    handle.settle = settle
    handle.state = state
    return handle


# What the person sees, and what the model is told, for each way a call can be
# refused. Two audiences, one decision - kept together so they cannot drift.
DECLINED = ("you said no",
            "the person declined this call - do not repeat it; say what you "
            "would have done, or take a route that changes nothing")
UNATTENDED = ("nothing can change here, and nobody is at the keyboard to allow "
              "it (--tools all)",
              "nothing that changes something may run in this session, and "
              "there is nobody to ask - answer without it, or say plainly what "
              "you would have run")


def _confirm(chat: Chat, call, attended: bool = True) -> tuple[str, str] | None:
    """Ask before a call that can change something. None means run it.

    Only ever asked when there is somebody at the keyboard - a question printed
    at a pipe is a hang, not a safeguard. Unattended, the mode decides: `--tools
    all` said what it wanted at the command line, and `auto` did not. A turn
    that does not echo (`--json`) is unattended: its stdout belongs to the
    caller's parser, so there is nowhere to put the question.
    """
    if chat.allow_all:
        return None
    reader = chat.ask_line
    if reader is None:
        if not attended or not sys.stdin.isatty():
            return None if chat.tools == "all" else UNATTENDED
        reader = input
    detail = _detail(call.arguments)
    # Preserve every command line, but never let terminal controls hide text.
    visible = "".join(c if c == "\n" or c.isprintable() else ascii(c)[1:-1]
                      for c in detail)
    question = render.yellow(
        f"    run {render.untrusted(call.name)}\n{visible}\n? [y/N/a] ")
    lane = getattr(chat, "input_lane", None)
    try:
        with lane.paused() if lane is not None else contextlib.nullcontext():
            answer = (reader(question) or "").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(file=chat.err)
        return DECLINED
    if answer in ("a", "all", "always"):
        chat.allow_all = True
        return None
    return None if answer in ("y", "yes") else DECLINED


def _detail(arguments: dict) -> str:
    """The one field worth reading on a tool line."""
    for key in ("path", "command", "pattern", "query", "url", "prompt", "source"):
        value = arguments.get(key)
        if value:
            return str(value)
    # An approval must show what it approves: an MCP tool with its own
    # argument names would otherwise ask "run x" over a blank line.
    return json.dumps(arguments, ensure_ascii=False, default=str) if arguments else ""


# --- input -----------------------------------------------------------------


PASTE_MARK = '"""'


class Reader:
    """Line input that understands continuations and paste blocks.

    A chat gets pasted into far more than it gets typed into, so a bare
    `input()` is wrong twice: a multi-line paste arrives as several unrelated
    questions, and a long typed thought has nowhere to go. Both are one state
    machine, kept out of the REPL so a test can drive it without a terminal.
    """

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.pasting = False

    @property
    def waiting(self) -> bool:
        return self.pasting or bool(self.buffer)

    def feed(self, line: str) -> str | None:
        """Take one raw line. Returns the finished prompt, or None if more is
        expected."""
        if self.pasting:
            if line.strip() == PASTE_MARK:
                self.pasting = False
                body = "\n".join(self.buffer)
                self.buffer = []
                return body
            self.buffer.append(line)
            return None
        if line.strip() == PASTE_MARK:
            self.pasting = True
            return None
        if line.endswith("\\"):
            self.buffer.append(line[:-1])
            return None
        if self.buffer:
            self.buffer.append(line)
            body = "\n".join(self.buffer)
            self.buffer = []
            return body
        return line


def shell(chat: Chat, line: str) -> None:
    """`!cmd` - run it, show it, stage it for the next question.

    Staging rather than only printing, because the reason to run a command mid
    chat is almost always to ask about its output.
    """
    cmd = line[1:].strip()
    if not cmd:
        return
    try:
        # errors="replace": a command printing non-UTF-8 bytes is output to
        # show, not a reason to end the session.
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              errors="replace", timeout=120, cwd=chat.workspace or os.getcwd())
    except (OSError, subprocess.SubprocessError) as exc:
        _say(chat, render.red(f"{cmd}: {exc}"))
        return
    body = (proc.stdout or "") + (proc.stderr or "")
    print(render.untrusted(body.rstrip()), file=chat.err)
    chat.staged.append(f"--- $ {cmd} ---\n{body}\n--- end ---")
    _say(chat, render.dim(f"staged {len(body)} chars of output"))


# --- the loop --------------------------------------------------------------


_ANSI = re.compile(r"(\033\[[0-9;]*m)")


def _rl_prompt(text: str) -> str:
    """Mark the colour escapes invisible to readline.

    Without \001/\002 readline counts the escape bytes as columns, and a line
    long enough to wrap starts overwriting the prompt. It is the single most
    common bug in a coloured REPL.
    """
    return _ANSI.sub("\001\\1\002", text)


def _setup_readline(chat: Chat):
    try:
        import readline
    except ImportError:
        return None
    try:
        from onecli import secureio
        with secureio.directory(HISTORY_FILE.parent, create=True, private=True):
            pass
        if HISTORY_FILE.exists():
            readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass
    readline.set_history_length(HISTORY_MAX)
    readline.parse_and_bind("tab: complete")
    readline.set_completer_delims(" \t\n")
    readline.set_completer(completer(chat, readline))
    return readline


def completer(chat: Chat, readline):
    """TAB completes what the table says exists, and what discovery found."""
    def complete(text: str, state: int):
        buffer = readline.get_line_buffer().lstrip()
        head = buffer.split(" ")[0]
        looked = lookup(head)
        if " " in buffer and looked and looked.name in ("model", "compare"):
            options = [b.name for b in chat.backends()]
        elif buffer.startswith("/"):
            options = ["/" + n for n in sorted(list(COMMANDS) + list(ALIASES))]
        else:
            options = []
        matches = [o for o in options if o.startswith(text)]
        return matches[state] if state < len(matches) else None
    return complete


def banner(chat: Chat) -> str:
    return "\n".join([
        render.brand(__version__) + render.faint("  chat"),
        render.header(render.dim(chat.backend.display), [
            f"{chat.backend.wire} · {chat.backend.base_url}",
            f"session {chat.sid}",
            f"tools {chat.tools} · {_surface_line(chat)}",
            "/help for commands, /exit to leave",
        ]),
    ])


def _failure(stat: dict) -> str:
    """The line a failed turn is reported with."""
    kind = "prompt blocked" if stat.get("blocked") else "provider error"
    # Provider and hook text: untrusted terminal input.
    return f"{kind}: {render.untrusted(str(stat['error']))}"


def _report(chat: Chat, stat: dict) -> None:
    if stat.get("error"):
        _say(chat, render.red(_failure(stat)))
        return
    _say(chat, chat.footer(stat))


def repl(chat: Chat, quiet: bool = False) -> int:
    if not quiet:
        print(banner(chat), file=chat.err)
    rl = _setup_readline(chat)
    reader = Reader()
    while True:
        if chat.tasks:
            for notice in chat.tasks.notices():
                _say(chat, render.paint("brand", render.untrusted(notice)))
        if chat._jobs is not None:
            for notice in chat._jobs.notices():
                _say(chat, render.paint("brand", render.untrusted(notice)))

        if chat.input_queue:
            prompt = chat.input_queue.pop(0)
            _say(chat, render.paint("brand", f"↳ [queued #{len(chat.input_queue) + 1} processing]: ") + render.ink(render.untrusted(prompt)))
        else:
            mark = render.prompt_mark() if not reader.waiting else "  "
            active_bg = 0
            if chat.tasks:
                active_bg += len(chat.tasks.list_active())
            if chat._jobs is not None:
                chat._jobs.poll()
                active_bg += sum(1 for j in chat._jobs.jobs.values() if j.running)
            if active_bg:
                mark = render.paint("live", f"[{active_bg} bg] ") + mark
            try:
                line = input(_rl_prompt(mark))
            except EOFError:
                print(file=chat.err)
                break
            except KeyboardInterrupt:
                # Ctrl-C at the prompt clears what is half-typed rather than ending
                # the session - the session ends with Ctrl-D or /exit, deliberately.
                print(file=chat.err)
                reader = Reader()
                continue

            fed = reader.feed(line)
            if fed is None:
                continue
            prompt = fed.strip()
            if not prompt:
                continue
        if prompt.lower() in ("exit", "quit"):
            prompt = "/exit"

        from onecli.model_picker import handle_picker
        menu = handle_picker(chat, prompt)
        if menu is not None:
            _say(chat, menu)
            continue

        try:
            if prompt.startswith("!"):
                shell(chat, prompt)
                continue
            if prompt.startswith("/"):
                if not dispatch(chat, prompt):
                    break
                continue
        except KeyboardInterrupt:
            # /retry, /edit, /compare and a !command run turns and processes
            # too; Ctrl-C stops the one, never the session.
            print(file=chat.err)
            _say(chat, render.yellow("interrupted"))
            continue

        chat.input_lane.start()
        try:
            stat = turn(chat, prompt)
        except KeyboardInterrupt:
            print(file=chat.err)
            _say(chat, render.yellow("interrupted"))
            continue
        finally:
            chat.input_lane.stop()
            pending = chat.input_lane.pending()
            if pending:
                chat.input_queue.extend(pending)
                _say(chat, render.paint("brand", f"  ↳ queued {len(pending)} message(s) typed during execution (see /queue)"))
        _report(chat, stat)

    for survivor in chat.close():
        _say(chat, render.yellow(f"job stopped at exit: {render.untrusted(survivor)}"))
    if rl is not None:
        try:
            from onecli import secureio
            secureio.write_history(rl, HISTORY_FILE, limit=HISTORY_MAX)
        except OSError:
            pass
    return 0


def turn(chat: Chat, prompt: str, echo: bool = True) -> dict:
    """One question, answered.

    `off` is one request and nothing else. Every other mode goes through the
    loop, which is not a different lane: most turns never call a tool and come
    back on the same single request, and the ones that do are still a question
    with an answer - the looking happens underneath.
    """
    submitted, refusal = _submit(chat, prompt)
    if submitted is None:
        return {"error": refusal, "blocked": True, "stopped": "blocked", "seconds": 0.0}
    prompt = submitted
    plain = chat.tools == "off" and chat.mask in (None, "off")
    # Every turn gets a checkpoint record, so /undo maps to exactly this turn.
    # A plain turn cannot edit the workspace and needs no snapshot.
    start = len(chat.messages)
    note = chat.checkpoints.begin_turn(prompt, chat.messages, edits=not plain)
    if note and note != chat._checkpoint_note:
        _say(chat, render.yellow(f"no checkpoint for this turn: {render.untrusted(note)}"))
    chat._checkpoint_note = note or ""
    try:
        return chat.ask(prompt, echo=echo) if plain else agent_turn(chat, prompt, echo=echo)
    finally:
        if len(chat.messages) <= start:
            chat.checkpoints.forget_turn()  # the turn left no exchange behind


def agent_turn(chat: Chat, prompt: str, echo: bool = True) -> dict:
    """The tool loop, run underneath the conversation.

    Imported here and nowhere else: the agent pulls in the tool registry, the
    permission model and the hook system, and a chat that never reaches for a
    tool must not pay for any of it at startup.

    In `auto` the boundary is the surface itself - four read tools, the web and
    bash, which asks before it runs (`_confirm`). `all` is the
    agent's own surface and is entered deliberately, by typing it.
    """
    from onecli import permissions
    from onecli.agent import Agent
    from onecli.masks import options

    started = time.monotonic()
    workspace = os.path.abspath(chat.workspace or os.getcwd())
    guard = chat.guard if chat.guard is not None else permissions.chain(
        workspace, chat.mode)
    text, attachments = chat.compose_parts(prompt)
    trace = _tracer(chat, echo)
    session.append(chat.sid, {"kind": "metadata", "mask": chat.mask or "off"})
    agent = Agent(
        **options(chat.mask),
        provider=chat.wire,
        workspace=workspace,
        guard=guard,
        trust_project=chat.trust_project,
        hooks=chat.hooks,
        model_name=chat.backend.name,
        tool_names=chat_surface(chat),
        max_tokens=chat.max_tokens,
        temperature=chat.temperature,
        stream=bool(chat.stream and echo),
        on_event=trace,
        price=config.price_for(chat.backend.model, chat.backend.name),
        context_window=config.resolve_context_window(chat.backend.model,
                                                     chat.backend.name,
                                                     chat.backend.base_url),
        jobs=chat.jobs() if chat.tools == "all" else None,
        tasks=chat.tasks,
    )
    try:
        result = agent.run(text, chat.history(tools=chat.tools != "off"),
                           attachments=attachments)
    except KeyboardInterrupt:
        chat.messages = [m for m in agent.interrupted_messages
                         if m.get("role") != "system"]
        session.append(chat.sid, {"kind": "messages", "messages": chat.messages})
        raise
    finally:
        trace.settle()

    if result.stopped == "error":
        # Kept, with the error recorded: the turn's tool calls may already have
        # changed files, and dropping them left the model resuming from before
        # its own work while /undo, finding no exchange, rolled back the turn
        # before. The agent stops only at a request, so every call is answered.
        # A turn whose first request failed did nothing and is not kept.
        if result.steps:
            chat.messages = [m for m in result.messages if m.get("role") != "system"]
        session.append(chat.sid, {"kind": "mask_run", "mask": result.mask,
                                  "stopped": result.stopped, "error": result.error})
        session.append(chat.sid, {"kind": "messages", "messages": chat.messages})
        return {"error": result.error, "seconds": time.monotonic() - started}

    # The agent returns the whole history including the system message; ours
    # lives in a field, so it is dropped rather than stored twice.
    chat.messages = [m for m in result.messages if m.get("role") != "system"]
    chat.last_answer = result.text or ""
    if echo and result.text and not trace.state.get("step_streamed"):
        chat.out.write(render.untrusted(result.text) + "\n")
        chat.out.flush()
    if echo and result.error and not result.text:
        _say(chat, render.yellow(render.untrusted(result.error)))
    elif echo and result.stopped in ("budget", "budget_input", "budget_usd", "truncated") and result.error:
        _say(chat, render.yellow(render.untrusted(result.error)))
    session.append(chat.sid, {"kind": "messages", "messages": chat.messages})
    # The agent reports its spend as the meter's dict, not a float. Folded into
    # the same running totals as a plain turn, so /cost stays true across a
    # session that answered some turns by looking and some by knowing.
    session.append(chat.sid, {"kind": "mask_run", "mask": result.mask, "stopped": result.stopped})
    spend = result.cost if isinstance(result.cost, dict) else {}
    in_tokens = int(spend.get("input_tokens") or 0)
    out_tokens = int(spend.get("output_tokens") or 0)
    usd = float(spend.get("usd") or 0.0)
    chat.total_in += in_tokens
    chat.total_out += out_tokens
    chat.total_usd += usd
    chat.total_tools += result.tool_calls
    chat.priced = chat.priced and bool(spend.get("price_known"))
    stat = {"model": chat.backend.name, "seconds": time.monotonic() - started,
            "ttft": 0.0, "input_tokens": in_tokens, "output_tokens": out_tokens,
            "text": result.text or "", "usd": usd,
            "tool_calls": result.tool_calls, "steps": len(result.steps)}
    # A loop that ran out of budget answered with whatever it had. Saying so is
    # the difference between a short answer and a truncated one.
    if result.stopped not in ("answered", ""):
        stat["stopped"] = result.stopped
        if echo:
            _say(chat, render.yellow(render.untrusted(result.error or
                                     f"stopped: {result.stopped}")))
    chat.turns.append(stat)
    return stat


# --- entry -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="one-chat",
        description="Fast chat against any configured model. A light tool "
                    "loop (read and search freely, bash after approval; "
                    "--tools off disables it), no vendor CLI, one transcript.")
    p.add_argument("prompt", nargs="*", help="Ask one question and exit. "
                                             "Omit for a session.")
    p.add_argument("--mask", help="Named task mask, or off.")
    p.add_argument("--masks", action="store_true", help="List task masks without contacting a provider.")
    p.add_argument("-m", "--model", help="Backend name or model id (see --list).")
    p.add_argument("--model-id", help="Override the model id on that backend.")
    p.add_argument("--base-url", help="Explicit endpoint, bypassing discovery.")
    p.add_argument("--wire", choices=[config.WIRE_OPENAI, config.WIRE_ANTHROPIC])
    p.add_argument("--system", help="System prompt.")
    p.add_argument("--system-file", help="System prompt, from a file.")
    p.add_argument("-t", "--temperature", type=float, default=None,
                   help="Default: none is sent; the provider's default applies.")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout. Default 120 - a chat that hangs "
                        "for ten minutes is not a chat.")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tools", choices=list(TOOL_MODES), default=DEFAULT_TOOL_MODE,
                   help="What the chat may reach for: off (nothing), auto "
                        "(read and search freely, bash after approval; the "
                        "default), all (the agent's own surface - edits, "
                        "shell, jobs).")
    p.add_argument("--think", action="store_true",
                   help="Print a thinking model's scratchpad too.")
    p.add_argument("--resume", nargs="?", const="@last", metavar="ID")
    p.add_argument("--sessions", action="store_true",
                   help="List saved sessions, newest first, with what each "
                        "one opened with.")
    p.add_argument("--compare", metavar="A,B",
                   help="Ask these backends the same question, in parallel.")
    p.add_argument("--json", action="store_true", help="One-shot, machine-readable.")
    p.add_argument("-q", "--quiet", action="store_true", help="Answer only.")
    p.add_argument("--plain", action="store_true", help="No colour, no glyphs.")
    p.add_argument("--list", action="store_true", help="List reachable backends.")
    p.add_argument("--version", action="version", version=f"one-cli {__version__}")
    return p


def _system_prompt(args) -> str:
    if args.system_file:
        if args.system:
            raise ValueError("--system and --system-file are exclusive")
        return pathlib.Path(os.path.expanduser(args.system_file)).read_text(
            encoding="utf-8")
    return args.system or ""


def _stdin_prompt(args) -> None:
    """A piped body is the question, so `echo q | one-chat` works.

    Only when stdin is not a terminal and no prompt was typed: an interactive
    session must not read its own keyboard here.
    """
    if args.prompt or sys.stdin.isatty():
        return
    body = sys.stdin.read().strip()
    if body:
        args.prompt = [body]


def _submit(chat: Chat, prompt: str) -> tuple[str | None, str]:
    """UserPromptSubmit for one chat prompt (hooks.submit_prompt)."""
    if not chat.hooks:
        return prompt, ""
    from onecli import hooks as hooklib
    chat.hooks.session_id = chat.sid  # /new and /resume change the session
    return hooklib.submit_prompt(
        chat.hooks, prompt, lambda note: _say(chat, render.yellow(f"hook: {render.untrusted(note)}")))


def _session_hooks(chat: Chat, args):
    """The hooks `one` would run for these flags (hooks.from_args). A broken
    configuration is reported here and still built: its PreToolUse spec
    denies every tool, so the chat fails closed exactly as the run does."""
    from onecli import hooks as hooklib
    built = hooklib.from_args(args, chat.sid)
    for problem in (built.errors if built else ()):
        _say(chat, render.yellow(f"hook configuration error: {render.untrusted(str(problem))}"))
    return built


def from_cli(args) -> int:
    """`one --chat` - the same session, entered through the agent's parser.

    The flags overlap almost exactly, so the chat is reachable without learning
    a second command; the separate `one-chat` script exists for startup cost,
    not for a different feature set.

    The permission flags build the same guard chain `one` builds
    (permissions.chain): `-w`, `--read-only`, `--permission-mode`, the allow
    and deny lists, `--guard` and `--add-dir`. The chat used to drop them all,
    so `one --chat --read-only` was a session that could write.
    """
    from onecli import permissions
    from onecli.tasks import session_options
    mode = getattr(args, "permission_mode", None)
    if getattr(args, "dangerously_skip_permissions", False):
        mode = "bypassPermissions"
    workspace = os.path.realpath(getattr(args, "workspace", None) or os.getcwd())
    try:
        mode = permissions.normalise_mode(mode, bool(getattr(args, "read_only", False)))
        guard = permissions.chain(
            workspace, mode, commands=getattr(args, "guard", None),
            timeout=getattr(args, "guard_timeout", None),
            allowed=getattr(args, "allowed_tools", None),
            disallowed=getattr(args, "disallowed_tools", None),
            add_dirs=getattr(args, "add_dir", None))
    except ValueError as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2
    try:
        backend = config.resolve(name=args.model, model=getattr(args, "model_id", None),
                                 base_url=args.base_url, wire=args.wire)
    except RuntimeError as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2
    if args.model and (refused := presence.refusal(backend)):
        backend = unavailable.resolve(args, backend, refused)
        if backend is None:
            return 2
    stream = args.stream if args.stream is not None else sys.stdout.isatty()
    chat = Chat(backend, system=getattr(args, "system", "") or "",
                temperature=args.temperature, max_tokens=args.max_tokens,
                timeout=min(getattr(args, "timeout", 120.0) or 120.0, 600.0),
                stream=bool(stream),
                tools=getattr(args, "tools", DEFAULT_TOOL_MODE) or DEFAULT_TOOL_MODE,
                mask=getattr(args, "mask", None),
                workspace=workspace, guard=guard, mode=mode,
                trust_project=bool(getattr(args, "trust_project_config", False)),
                task_options=session_options(args, mode))
    if getattr(args, "resume", None):
        _resume(chat, args.resume)
    chat.hooks = _session_hooks(chat, args)
    prompt = " ".join(args.prompt).strip() if args.prompt else ""
    if prompt:
        # --json is a promise about stdout, so the answer must not also be
        # streamed onto it: a caller piping into a parser would get the text
        # twice, once loose and once inside the object.
        want_json = bool(getattr(args, "json", False))
        if want_json:
            chat.stream = False
        try:
            stat = turn(chat, prompt, echo=not want_json)
        except KeyboardInterrupt:
            print(render.yellow("\ninterrupted"), file=sys.stderr)
            return 130
        finally:
            chat.close()
        if want_json:
            # `one -c --json` prints JSON, never prose: a caller that asked for
            # JSON needs something a parser accepts.
            print(json.dumps({
                "session": chat.sid,
                "backend": chat.backend.name,
                "model": chat.backend.model,
                "wire": chat.backend.wire,
                "text": stat.get("text", ""),
                "stopped": stat.get("stopped",
                                    "error" if stat.get("error") else "answered"),
                "error": stat.get("error", ""),
                "steps": stat.get("steps", 0),
                "tool_calls": stat.get("tool_calls", 0),
                "seconds": round(float(stat.get("seconds") or 0.0), 3),
                "input_tokens": stat.get("input_tokens", 0),
                "output_tokens": stat.get("output_tokens", 0),
                "usd": stat.get("usd", 0.0),
            }, indent=2))
            return 2 if stat.get("error") else 0
        if stat.get("error"):
            print(render.red(_failure(stat)), file=sys.stderr)
            return 2
        if not args.quiet:
            print(chat.footer(stat), file=sys.stderr)
        return 0
    return repl(chat, quiet=args.quiet)


def cmd_sessions(args) -> int:
    """The listing `one --sessions` prints, from this entry point too."""
    rows = session.listing()
    if getattr(args, "json", False):
        print(json.dumps([{
            "session": sid,
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt)),
            "modified_epoch": mt,
            "bytes": size,
            "opening_line": session.opening_line(sid),
        } for sid, mt, size in rows], indent=2))
        return 0
    if not rows:
        print(render.dim("no sessions yet"), file=sys.stderr)
        return 0
    print(render.kicker("sessions"), file=sys.stderr)
    for sid, mtime, _size in rows:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        print(render.ink(sid) + "  " + render.dim(stamp) + "  "
              + render.faint(render.one_line(render.untrusted(session.opening_line(sid)), 56)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from onecli.masks import listing, select
        if args.masks:
            print(listing(args.mask))
            return 0
        select(args.mask)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.plain:
        os.environ["ONE_CLI_PLAIN"] = "1"
    render.refresh()

    if args.list:
        found = config.discover()
        if not found:
            print(render.red("no backends found"), file=sys.stderr)
            return 1
        for b in found:
            print(render.ink(f"{b.name:44}") + render.dim(f"{b.wire:10}")
                  + render.faint(b.base_url))
        return 0

    if args.sessions:
        # `one` has had `--sessions` all along and `one-chat` had only the
        # in-session `/sessions`, so the way to find a chat to resume was to
        # start a chat first. The listing is the same one - the two entry
        # points are the same sessions on the same disk, and a flag that
        # exists on one of them and not the other is a thing to memorise.
        return cmd_sessions(args)

    try:
        system = _system_prompt(args)
    except (ValueError, OSError) as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2

    _stdin_prompt(args)

    try:
        backend = config.resolve(name=args.model, model=args.model_id,
                                 base_url=args.base_url, wire=args.wire)
    except RuntimeError as exc:
        print(render.red(str(exc)), file=sys.stderr)
        return 2
    if args.model and (refused := presence.refusal(backend)):
        backend = unavailable.resolve(args, backend, refused)
        if backend is None:
            return 2

    stream = args.stream if args.stream is not None else sys.stdout.isatty()
    chat = Chat(backend, system=system, temperature=args.temperature,
                max_tokens=args.max_tokens, timeout=args.timeout,
                stream=stream and not args.json, thinking=args.think,
                tools=args.tools, mask=args.mask)
    if args.resume:
        _resume(chat, args.resume)
    chat.hooks = _session_hooks(chat, args)

    prompt = " ".join(args.prompt).strip()

    if args.compare:
        if not prompt:
            print(render.red("--compare needs a question"), file=sys.stderr)
            return 2
        try:
            rows = fan_out(chat, [n for n in args.compare.split(",") if n.strip()], prompt)
        except KeyboardInterrupt:
            print(render.yellow("\ninterrupted"), file=sys.stderr)
            return 130
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        for row in rows:
            print(render.kicker(row["name"]), file=sys.stderr)
            print(render.untrusted(row.get("error") or row["text"]))
        return 0

    if prompt:
        try:
            stat = turn(chat, prompt, echo=not args.json)
        except KeyboardInterrupt:
            print(render.yellow("\ninterrupted"), file=sys.stderr)
            return 130
        finally:
            chat.close()
        if args.json:
            # `model` is the model id and `backend` is the endpoint we asked;
            # the turn stat's own `model` field is the backend name, so it is
            # dropped here rather than silently overwriting the id.
            print(json.dumps({"session": chat.sid, "backend": backend.name,
                              "model": backend.model, "wire": backend.wire,
                              **{k: v for k, v in stat.items()
                                 if k not in ("thinking", "model")}},
                             indent=2, default=str))
            return 0 if not stat.get("error") else 1
        if stat.get("error"):
            print(render.red(_failure(stat)), file=sys.stderr)
            return 2
        if not args.quiet:
            print(chat.footer(stat), file=sys.stderr)
        return 0

    if not sys.stdin.isatty():
        print(render.red("no question on stdin and no terminal to ask on"),
              file=sys.stderr)
        return 2
    return repl(chat, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())

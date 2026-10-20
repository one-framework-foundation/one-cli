"""Slash commands for the interactive session.

A REPL whose only two commands are `/exit` and `/quit` makes every question
about the session - what is this costing, how full is the context, which model
am I on, what is still running in the background - a reason to kill the session
and start a new one with different flags. The session is the expensive thing:
it holds the history, the jobs and the spill files. So the answers live here,
where they cost a line instead of a restart.

The rules this module keeps to:

1. **A command reports or changes state. It never runs the model.** The one
   exception is `/compact`, which has to ask the model to summarize - and it is
   the operation the model is *for*, not a turn the user typed.
2. **Nothing here prints.** Every command returns text and the loop prints it,
   so the same dispatch is testable without a terminal and the loop keeps one
   place where output goes to stderr.
3. **A change of model or mode does not rebuild the world here.** It sets a
   flag on the state; the loop rebuilds on the next turn. Commands that reach
   into `_context_settings` would need half of `cli.py` imported, which is how
   a circular import starts.
"""

from __future__ import annotations

import difflib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from onecli import config
from onecli import context as ctx
from onecli import jobs as joblib
from onecli import permissions
from onecli import presence
from onecli import render
from onecli import session


@dataclass
class Repl:
    """Everything a command may read or change, and nothing else.

    The turn counters live here rather than on the Agent because a fresh Agent
    is built for every interactive turn - a per-turn meter can answer "what did
    that line cost", never "what has this session cost", which is the only
    version of the question anyone asks.
    """

    args: Any = None
    backend: Any = None
    chain: Any = None
    mode: str = "default"
    sid: str = ""
    messages: list[dict] | None = None
    jobs: Any = None
    hooks: Any = None
    presets: dict = field(default_factory=dict)
    surface: list[str] | None = None
    mcp_client: Any = None
    files: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    priced: bool = False
    context_tokens: int = 0
    tasks: Any = None
    checkpoints: Any = None

    def __post_init__(self):
        workspace = getattr(self.args, "workspace", "") or os.getcwd()
        if self.tasks is None:
            from onecli.tasks import TaskManager, session_options
            model = getattr(self.args, "model", "") or (getattr(self.backend, "name", "") if self.backend else "") or ""
            self.tasks = TaskManager(
                execute=True,
                workspace=workspace,
                default_model=model,
                **session_options(self.args, self.mode),
            )
        if self.checkpoints is None:
            from onecli.checkpoints import CheckpointManager
            self.checkpoints = CheckpointManager(workspace=workspace, sid=self.sid)

    # What `!command` lines have produced since the last turn. Held rather
    # than sent immediately: the reason to run one is almost always to show the
    # model something, and a turn that never comes should not have paid for it.
    shell_context: list[str] = field(default_factory=list)

    # Set by a command, read by the loop.
    leave: bool = False
    rebuild: bool = False

    def remember_shell(self, command: str, output: str, code: int) -> None:
        self.shell_context.append(
            f"$ {command}\n{output.rstrip()}"
            + ("" if code == 0 else f"\n[exit {code}]"))

    def take_shell_context(self) -> str:
        """The shell block for the next turn, consumed on the way out."""
        if not self.shell_context:
            return ""
        blocks = "\n\n".join(self.shell_context)
        self.shell_context = []
        return ("<shell>\nCommands the user ran in the terminal before this "
                f"message, with their output:\n\n{blocks}\n</shell>\n\n")

    @property
    def window(self) -> int:
        return int(self.settings.get("context_window") or 0)

    def record(self, result) -> None:
        """Fold one turn's accounting into the session's."""
        self.turns += 1
        # Read with getattr: a Result is the shape the loop hands over, and a
        # meter that raises on a field it did not expect would take the whole
        # session down over an accounting line.
        cost = getattr(result, "cost", None) or {}
        self.input_tokens += cost.get("input_tokens") or 0
        self.output_tokens += cost.get("output_tokens") or 0
        if cost.get("usd") is not None:
            self.usd += cost["usd"]
            self.priced = True
        self.context_tokens = (getattr(result, "context_tokens", 0)
                               or self.context_tokens)


@dataclass
class Command:
    name: str
    summary: str
    run: Callable[[Repl, str], str]
    usage: str = ""
    aliases: tuple[str, ...] = ()


@dataclass
class Outcome:
    """What the loop does next with a line that started with a slash.

    Exactly one of these is ever set. `text` is printed and the prompt comes
    straight back; `prompt` is sent to the model as if it had been typed.
    """

    text: str = ""
    prompt: str = ""


def _rows(pairs: Sequence[tuple[str, object]]) -> str:
    """Two columns, the left one dim - armature and figure, per render.py."""
    if not pairs:
        return ""
    width = max(len(k) for k, _ in pairs)
    return "\n".join(render.dim(f"  {k:<{width}}  ") + render.ink(str(v))
                     for k, v in pairs)


# --- the commands ----------------------------------------------------------


def _help(repl: Repl, rest: str) -> str:
    lines = [render.kicker("commands")]
    for cmd in ORDER:
        left = "/" + cmd.name + (f" {cmd.usage}" if cmd.usage else "")
        line = render.accent(f"  {left:<22}") + render.dim(cmd.summary)
        # Same gap as the chat's own /help: an alias that is registered and
        # never printed is a command nobody can find.
        if cmd.aliases:
            line += render.faint("  also " + " ".join("/" + a
                                                      for a in cmd.aliases))
        lines.append(line)
    if repl.files:
        lines.append("")
        lines.append(render.kicker("commands in this workspace"))
        for name, cmd in sorted(repl.files.items()):
            left = "/" + name + (f" {cmd.argument_hint}"
                                 if cmd.argument_hint else "")
            lines.append(render.accent(f"  {left:<22}")
                         + render.dim(cmd.description or _one_line(cmd.body, 48)))
    lines.append("")
    lines.append(render.faint("  @path sends the file with the question · "
                              "!cmd runs a shell command"))
    lines.append(render.faint("  ctrl-c drops the line or the turn · "
                              "ctrl-d leaves · tab completes"))
    return "\n".join(lines)


def _exit(repl: Repl, rest: str) -> str:
    repl.leave = True
    return ""


def _clear(repl: Repl, rest: str) -> str:
    """Forget the conversation, keep the session.

    The transcript file is appended to, never rewritten, so the cleared history
    is still on disk and a later `--resume` of this id still replays the whole
    thing. That is deliberate: `/clear` is about what the model is carrying,
    not about destroying a record.
    """
    had = len(repl.messages or [])
    repl.messages = None
    repl.context_tokens = 0
    repl.checkpoints.reset_turns()
    session.append(repl.sid, {"kind": "cleared", "dropped": had})
    if not had:
        return render.dim("nothing to clear")
    return render.dim(f"cleared {had} messages - the transcript on disk is intact")


def _rewind(repl: Repl, rest: str) -> str:
    """Drop the last exchange - the question and everything it produced.

    The turn that went wrong is usually the last one, and the alternatives are
    both bad: leave it in the history and every later turn reasons from it, or
    /clear and lose the twenty good turns in front of it. Repeatable, so three
    /rewinds walk back three exchanges.
    """
    messages = list(repl.messages or [])
    # Where the last turn began when recorded: one turn can hold several
    # user-role messages (interrupt notes, harness nudges).
    cut = repl.checkpoints.turn_start(messages)
    if cut is None:
        cut = ctx.last_exchange_start(messages)
    if cut is None:
        return render.dim("nothing to rewind")
    dropped = len(messages) - cut
    repl.messages = messages[:cut]
    session.append(repl.sid, {"kind": "messages", "messages": repl.messages})
    repl.context_tokens = ctx.history_tokens(repl.messages)
    repl.checkpoints.forget_turn()  # the exchange is gone; its files stay as they are
    return render.dim(f"rewound one exchange - {dropped} messages dropped, "
                      f"{cut} kept")


def _undo(repl: Repl, rest: str) -> str:
    """Revert the last turn's workspace edits and drop its exchange."""
    messages = list(repl.messages or [])
    if ctx.last_exchange_start(messages) is None:
        return render.dim("nothing to undo")
    cp_id = int(rest.strip()) if rest.strip().isdigit() else None
    # The workspace and the conversation move together, one turn each: a turn
    # without a checkpoint refuses rather than rolling back an older turn too.
    res = repl.checkpoints.undo_turn(messages, cp_id)
    if not res.success:
        return render.red(f"cannot undo: {res.error}")

    cut = len(messages) if res.cut is None else res.cut
    msg_count = len(messages) - cut
    repl.messages = messages[:cut]
    session.append(repl.sid, {"kind": "messages", "messages": repl.messages})
    repl.context_tokens = ctx.history_tokens(repl.messages)

    parts = []
    if res.checkpoint_id:
        details = []
        if res.reverted_files:
            details.append(f"{len(res.reverted_files)} file(s) restored")
        if res.removed_files:
            details.append(f"{len(res.removed_files)} untracked file(s) removed")
        detail_str = f" ({', '.join(details)})" if details else ""
        parts.append(render.paint("brand", f"reverted workspace to checkpoint #{res.checkpoint_id}{detail_str}"))

    if msg_count:
        parts.append(render.dim(f"dropped {msg_count} message(s) from conversation"))
    return "\n".join(parts)


def _diff(repl: Repl, rest: str) -> str:
    """Show workspace diff against turn checkpoint."""
    if repl.checkpoints is None:
        return render.dim("no checkpoints recorded")
    cp_id = int(rest.strip()) if rest.strip().isdigit() else None
    diff_text = repl.checkpoints.get_diff(cp_id)
    if not diff_text or diff_text.startswith("workspace clean"):
        return render.dim("workspace clean - no changes since checkpoint")
    # File contents the model wrote; a diff line is as untrusted as the answer.
    return render.untrusted(diff_text)


def _checkpoints(repl: Repl, rest: str) -> str:
    """List turn checkpoints in this session."""
    if repl.checkpoints is None:
        return render.dim("no checkpoints recorded")
    cps = repl.checkpoints.list_checkpoints()
    if not cps:
        return render.dim("no checkpoints recorded in this session")
    lines = [render.kicker(f"checkpoints · {len(cps)}")]
    for cp in cps:
        stat = repl.checkpoints.get_diff_stat(cp.id)
        stat_str = f"  {stat}" if stat else ""
        lines.append(f"  #{cp.id} {render.ink(cp.prompt)}" + render.dim(stat_str))
    return "\n".join(lines)


def _resume(repl: Repl, rest: str) -> str:
    """Adopt another session's history without leaving this one.

    The transcript keeps being written under THIS id, so resuming somebody
    else's conversation never rewrites the one it came from.
    """
    if not rest or rest.startswith("? ") or rest == "?" or rest == "@workspace":
        workspace = getattr(repl.args, "workspace", None) if rest == "@workspace" else None
        rows = session.search(rest[2:].strip() if rest.startswith("?") else "", workspace, 10)
        if not rows:
            return render.dim("no matching sessions")
        lines = [render.kicker("sessions")]
        for row in rows:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["modified_epoch"]))
            title = row.get("title") or row["opening_line"]
            lines.append(render.ink(row["session"]) + "  " + render.dim(stamp)
                         + "  " + render.one_line(title, 56))
        lines.append(render.faint("/resume <id> adopts history · /resume ? words searches · /resume @workspace"))
        return "\n".join(lines)
    want = session.latest_id() if rest == "@last" else rest
    if not want:
        return render.red("no session to resume")
    try:
        loaded = session.load_messages(want)
    except (FileNotFoundError, ValueError) as exc:
        return render.red(str(exc))
    if not loaded:
        return render.yellow(f"session {want} has no messages")
    repl.messages = loaded
    repl.context_tokens = ctx.history_tokens(loaded)
    repl.checkpoints.reset_turns()
    session.append(repl.sid, {"kind": "messages", "messages": loaded})
    return render.dim(f"adopted {len(loaded)} messages from {want} - "
                      f"this session is still {repl.sid}")


def _rename(repl: Repl, rest: str) -> str:
    try:
        session.rename(repl.sid, rest)
    except (OSError, ValueError) as exc:
        return render.red(str(exc))
    return render.dim(f"session named {rest.strip()}")


def _fork(repl: Repl, rest: str) -> str:
    session.append(repl.sid, {"kind": "messages", "messages": repl.messages or []})
    target = session.fork(repl.sid, rest)
    return render.dim(f"saved fork {target}; continue with one --resume {target}")


def _model(repl: Repl, rest: str) -> str:
    if not rest:
        from onecli.model_picker import open_picker
        return open_picker(repl, config.discover())
    try:
        picked = config.resolve(name=rest)
    except RuntimeError as exc:
        return render.red(str(exc))
    refused = presence.refusal(picked)
    if refused:
        return render.red(refused)
    if picked.name == repl.backend.name and picked.model == repl.backend.model:
        return render.dim(f"already on {picked.name}")
    repl.backend = picked
    repl.rebuild = True
    return render.dim(f"switched to {picked.name} ({picked.model})"
                      " - the conversation carries over")


def _mask(repl: Repl, rest: str) -> str:
    from onecli.masks import listing, select
    try:
        if not rest:
            return listing(getattr(repl.args, 'mask', None))
        if getattr(repl.args, 'request_budget_isolated', False):
            raise ValueError('mask selection is not supported in isolated request-budget mode')
        select(rest)
        repl.args.mask = rest
        session.append(repl.sid, {'kind': 'metadata', 'mask': rest})
        return f'Mask: {rest}. Applies to the next turn; conversation retained.'
    except ValueError as exc:
        return render.red(str(exc))


def _models(repl: Repl, rest: str) -> str:
    found = config.discover()
    if not found:
        return render.red("no backends found")
    here = repl.backend.name if repl.backend else ""
    lines = [render.kicker("backends")]
    for b in found:
        mark = render.accent(" *") if b.name == here else "  "
        lines.append(mark + render.ink(f"{b.name:<40}") + render.faint(b.wire))
    return "\n".join(lines)


def _mode(repl: Repl, rest: str) -> str:
    if not rest:
        return _rows([("mode", repl.mode),
                      ("choices", ", ".join(permissions.MODES))])
    if rest not in permissions.MODES:
        near = difflib.get_close_matches(rest, permissions.MODES, n=1)
        hint = f" - did you mean {near[0]}?" if near else ""
        return render.red(f"unknown mode {rest}{hint}")
    if rest == repl.mode:
        return render.dim(f"already in {rest}")
    repl.mode = rest
    repl.tasks.permission_mode = rest
    repl.rebuild = True
    return render.dim(f"permission mode is now {rest}")


def _tools(repl: Repl, rest: str) -> str:
    from onecli import tools as toolkit

    names = repl.surface if repl.surface is not None else list(toolkit.REGISTRY)
    if not names:
        return render.yellow("no tools on this run's surface")
    width = render.width() - 4
    line, out = "", []
    for name in sorted(names):
        if line and len(line) + len(name) + 2 > width:
            out.append("  " + line)
            line = ""
        line += (", " if line else "") + name
    if line:
        out.append("  " + line)
    head = render.kicker(f"tools · {len(names)} on the surface · {repl.mode}")
    return "\n".join([head] + [render.ink(row) for row in out])


def _queue(repl: Repl, rest: str) -> str:
    from onecli import terminal
    pending = terminal.ACTIVE.pending_messages() if terminal.ACTIVE is not None else []
    return 'Queued messages:\n' + ('\n'.join(f'{i}. {text}' for i, text in enumerate(pending, 1)) or 'empty')


def _context(repl: Repl, rest: str) -> str:
    used = repl.context_tokens or ctx.history_tokens(repl.messages or [])
    window = repl.window
    pairs = [("history", f"{len(repl.messages or [])} messages"),
             ("tokens", f"{used:,}")]
    if window:
        share = used / window
        pairs.append(("window", f"{window:,}  ({share:.0%} full)"))
        if repl.settings.get("autocompact"):
            at = repl.settings.get("compact_at", 1.0)
            pairs.append(("compacts at", f"{int(window * at):,} tokens"))
        else:
            pairs.append(("autocompact", "off"))
    return _rows(pairs)


def _compact(repl: Repl, rest: str) -> str:
    """Summarize the old half of the history away, now rather than at the
    threshold. The only command that spends money, so it says what it saved."""
    messages = repl.messages or []
    keep = int(repl.settings.get("keep_last", 6))
    if not ctx.has_compactable(messages, keep):
        return render.dim("nothing old enough to compact")
    provider = repl.chain.current.provider
    max_tokens = getattr(repl.args, "max_tokens", None)

    def summarize(head: list[dict]) -> str:
        turn = provider.chat(
            [{"role": "system", "content": ctx.SUMMARY_INSTRUCTION},
             {"role": "user", "content": _render_for_summary(head)}],
            None, temperature=getattr(repl.args, "temperature", None),
            max_tokens=max_tokens)
        return turn.text or ""

    new, report = ctx.compact(messages, summarize, keep_last=keep)
    if not report.ok:
        return render.red(f"compaction failed: {render.untrusted(report.error)}")
    saved = report.before_tokens - report.after_tokens
    if saved <= 0:
        # A summary no smaller than what it replaces is a loss, not a
        # compaction: keep the real messages.
        return render.dim(f"nothing to compact - a summary would not be smaller "
                          f"({report.before_tokens:,} -> {report.after_tokens:,} tokens)")
    repl.messages = new
    repl.context_tokens = report.after_tokens
    session.append(repl.sid, {"kind": "messages", "messages": new})
    return render.dim(
        f"compacted {report.summarized} messages into one summary - "
        f"{report.before_tokens:,} -> {report.after_tokens:,} tokens "
        f"({saved:,} freed)")


def _cost(repl: Repl, rest: str) -> str:
    total = repl.input_tokens + repl.output_tokens
    pairs = [("turns", repl.turns),
             ("input", f"{repl.input_tokens:,} tokens"),
             ("output", f"{repl.output_tokens:,} tokens"),
             ("total", f"{total:,} tokens")]
    if repl.priced:
        pairs.append(("spend", f"${repl.usd:.4f}"))
    else:
        # An unpriced model reporting $0.00 is the most reassuring possible
        # lie. Say which model has no price instead.
        pairs.append(("spend", f"no price known for {repl.backend.model}"))
    pairs.append(("elapsed", _elapsed(time.time() - repl.started)))
    return _rows(pairs)


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _bg(repl: Repl, rest: str) -> str:
    if repl.jobs is None:
        return render.dim("background jobs are off for this run")
    cmd = rest.strip()
    if not cmd:
        return "usage: /bg <command>"
    workspace = os.path.abspath(getattr(repl.args, "workspace", "") or os.getcwd())
    try:
        job = repl.jobs.start(cmd, workspace)
    except OSError as exc:
        return f"error: could not start job: {exc}"
    return (f"job {job.id} started: {job.command}\n"
            f"running in background. inspect with /logs {job.id} or /ps, cancel with /kill {job.id}")


def _ps(repl: Repl, rest: str) -> str:
    if repl.jobs is None:
        return render.dim("background jobs are off for this run")
    workspace = os.path.abspath(getattr(repl.args, "workspace", "") or os.getcwd())
    # Job commands and output are what the model chose to run and what it printed.
    if rest.startswith("cancel "):
        return render.untrusted(joblib.job_cancel(workspace, rest.split(None, 1)[1].strip(),
                                                  jobs=repl.jobs))
    if rest:
        return render.untrusted(joblib.job_output(workspace, rest.strip(), jobs=repl.jobs))
    return render.untrusted(joblib.job_list(workspace, jobs=repl.jobs))


_jobs = _ps


def _kill(repl: Repl, rest: str) -> str:
    if repl.jobs is None:
        return render.dim("background jobs are off for this run")
    job_id = rest.strip()
    if not job_id:
        return "usage: /kill <id>"
    workspace = os.path.abspath(getattr(repl.args, "workspace", "") or os.getcwd())
    return joblib.job_cancel(workspace, job_id, jobs=repl.jobs)


def _logs(repl: Repl, rest: str) -> str:
    if repl.jobs is None:
        return render.dim("background jobs are off for this run")
    workspace = os.path.abspath(getattr(repl.args, "workspace", "") or os.getcwd())
    parts = rest.strip().split()
    if not parts:
        if getattr(repl.jobs, "jobs", None):
            job_id = list(repl.jobs.jobs.keys())[-1]
            parts = [job_id]
        else:
            return "usage: /logs <id> [offset] [limit]"
    job_id = parts[0]
    offset = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else joblib.DEFAULT_LINES
    # A job's output is whatever the model's command printed.
    return render.untrusted(
        joblib.job_output(workspace, job_id, offset=offset, limit=limit, jobs=repl.jobs))


def _session(repl: Repl, rest: str) -> str:
    path = session.path_for(repl.sid)
    return _rows([("id", repl.sid),
                  ("transcript", render.short_path(str(path))),
                  ("turns", repl.turns),
                  ("messages", len(repl.messages or [])),
                  ("resume", f"one --resume {repl.sid}")])


def _cwd(repl: Repl, rest: str) -> str:
    """The workspace, and what it is allowed to reach.

    Read-only on purpose: the workspace is the boundary every guard, hook and
    permission rule was built against at start-up. Moving it mid-session would
    leave those pointed at the old tree, which is a hole, not a feature.
    """
    workspace = os.path.abspath(repl.args.workspace)
    pairs = [("workspace", render.short_path(workspace))]
    extra = getattr(repl.args, "add_dir", None) or []
    for i, d in enumerate(extra):
        pairs.append(("also writable" if i == 0 else "", render.short_path(d)))
    return _rows(pairs)


def _hooks(repl: Repl, rest: str) -> str:
    hooks = repl.hooks
    specs = list(getattr(hooks, "specs", []) or [])
    if not specs:
        return render.dim("no hooks configured for this workspace")
    by_event: dict[str, list] = {}
    for spec in specs:
        by_event.setdefault(spec.event, []).append(spec)
    lines = [render.kicker(f"hooks · {len(specs)}")]
    for event in sorted(by_event):
        lines.append(render.accent(f"  {event}"))
        for spec in by_event[event]:
            matcher = f"[{spec.matcher}] " if getattr(spec, "matcher", "") else ""
            lines.append(render.dim("    " + matcher)
                         + render.ink(_one_line(spec.command)))
    return "\n".join(lines)


def _one_line(text: str, cap: int = 78) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= cap else flat[:cap - 1] + "…"


def _mcp(repl: Repl, rest: str) -> str:
    client = repl.mcp_client
    if client is None:
        return render.dim("no mcp servers on this run")
    lines = [render.kicker(f"mcp · {len(client.servers)} server(s)")]
    for server in client.servers:
        lines.append(render.ink(f"  {server.spec.name}")
                     + render.faint(f"  {len(server.tools)} tools"))
    for name, why in client.failures:
        lines.append(render.yellow(f"  {name}  unavailable - {render.untrusted(why)}"))
    return "\n".join(lines)


def _agents(repl: Repl, rest: str) -> str:
    if not repl.presets:
        return render.dim("no subagent presets loaded (--agents)")
    lines = [render.kicker(f"agents · {len(repl.presets)}")]
    for name, preset in sorted(repl.presets.items()):
        driver = getattr(preset, "driver", "") or "one"
        lines.append(render.ink(f"  {name:<20}") + render.faint(driver))
    return "\n".join(lines)


def status(repl: Repl) -> str:
    """The one line above the prompt: where this session stands, right now.

    Only facts that change. The model is there because `/model` can move it
    mid-session and a footer two screens up is not an answer to "which model am
    I talking to". The meter is there because the expensive failure is
    discovering the window was full after the compaction fired.
    """
    if not repl.turns:
        return ""
    bits = [render.faint(repl.backend.model if repl.backend else "")]
    window, used = repl.window, repl.context_tokens
    if window and used:
        share = used / window
        bits.append(render.meter(share, 10)
                    + render.faint(f" {share:.0%}"))
    total = repl.input_tokens + repl.output_tokens
    if repl.priced:
        bits.append(render.faint(f"${repl.usd:.4f}"))
    elif total:
        bits.append(render.faint(f"{total:,} tok"))
    return "  " + render.faint(" · ").join(b for b in bits if b)


def _tasks(repl: Repl, rest: str) -> str:
    if repl.tasks is None:
        return render.dim("no background tasks active")
    repl.tasks.poll()
    if not repl.tasks.list_all():
        return render.dim("no background tasks active")
    cards = [render.kicker(f"detached tasks · {len(repl.tasks.list_all())}")]
    for t in repl.tasks.list_all():
        cards.append(repl.tasks.render_card(t.task_id))
    return "\n".join(cards)


def _task(repl: Repl, rest: str) -> str:
    if not rest:
        return _tasks(repl, "")
    if repl.tasks:
        repl.tasks.poll()
    parts = rest.split(None, 1)
    subcmd = parts[0].lower()
    tail = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "spawn":
        from onecli.tasks import parse_spawn_args
        prompt, runner, model, workspace = parse_spawn_args(tail)
        title = prompt.splitlines()[0][:80] if prompt else "Detached Background Task"
        t = repl.tasks.spawn(title=title, prompt=prompt, runner=runner, model=model, workspace=workspace)
        return render.paint("brand", f"detached task #{t.task_id} spawned") + "\n" + repl.tasks.render_card(t.task_id)
    elif subcmd == "advance":
        tid = tail or (repl.tasks.list_active()[0].task_id if repl.tasks.list_active() else "")
        t = repl.tasks.get(tid)
        if not t:
            return render.red(f"task #{tid} not found")
        t.advance_phase()
        return repl.tasks.render_card(tid)
    elif subcmd == "done":
        tid = tail or (repl.tasks.list_active()[0].task_id if repl.tasks.list_active() else "")
        t = repl.tasks.get(tid)
        if not t:
            return render.red(f"task #{tid} not found")
        t.complete()
        return repl.tasks.render_card(tid)
    elif subcmd == "cancel":
        tid = tail or (repl.tasks.list_active()[0].task_id if repl.tasks.list_active() else "")
        t = repl.tasks.get(tid)
        if not t:
            return render.red(f"task #{tid} not found")
        t.cancel()
        return render.yellow(f"task #{tid} cancelled") + "\n" + repl.tasks.render_card(tid)
    elif subcmd in ("log", "output"):
        tid = tail or (repl.tasks.list_all()[-1].task_id if repl.tasks.list_all() else "")
        t = repl.tasks.get(tid)
        if not t:
            return render.red(f"task #{tid} not found")
        # A worker's output is model text.
        return render.untrusted(t.output())
    elif subcmd in ("show", "card"):
        tid = tail or (repl.tasks.list_all()[-1].task_id if repl.tasks.list_all() else "")
        t = repl.tasks.get(tid)
        if not t:
            return render.red(f"task #{tid} not found")
        return repl.tasks.render_card(tid)
    else:
        t = repl.tasks.spawn(title=rest, prompt=rest)
        return render.paint("brand", f"detached task #{t.task_id} spawned") + "\n" + repl.tasks.render_card(t.task_id)


def _render_for_summary(messages: list[dict]) -> str:
    from onecli.agent import _render_for_summary as impl

    return impl(messages)


# --- registry --------------------------------------------------------------

ORDER: list[Command] = [
    Command("help", "this list", _help, aliases=("?",)),
    Command("model", "pick a provider and model, or switch by name", _model,
            usage="[name]"),
    Command("mask", "inspect or switch task mask", _mask, usage="[name|off]"),
    Command("models", "every backend this host can reach", _models),
    Command("mode", "show or change the permission mode", _mode, usage="[name]"),
    Command("tools", "what the model is allowed to call", _tools),
    Command("tasks", "list detached offload tasks", _tasks),
    Command("task", "spawn or update detached task cards", _task, usage="[spawn|advance|done] [name/id]"),
    Command("bg", "run a command in the background", _bg, usage="<command>"),
    Command("ps", "list background jobs", _ps),
    Command("kill", "cancel a background job", _kill, usage="<id>", aliases=("stop",)),
    Command("logs", "read background job output", _logs, usage="<id> [offset] [limit]", aliases=("log",)),
    Command("queue", "show pending messages; Enter twice while busy steers", _queue),
    Command("context", "how full the context window is", _context),
    Command("compact", "summarize the old half of the history away", _compact),
    Command("cost", "tokens and spend for this session", _cost),
    Command("jobs", "background jobs (`/jobs <id>`, `/jobs cancel <id>`)",
            _jobs, usage="[id]"),
    Command("hooks", "the hooks this workspace runs", _hooks),
    Command("mcp", "connected mcp servers", _mcp),
    Command("agents", "subagent presets `task` can address", _agents),
    Command("rewind", "drop the last exchange, keep everything before it",
            _rewind),
    Command("undo", "revert last turn workspace edits and drop exchange",
            _undo, usage="[id]", aliases=("u",)),
    Command("diff", "show git diff since turn checkpoint",
            _diff, usage="[id]", aliases=("d",)),
    Command("checkpoints", "list turn checkpoints",
            _checkpoints),
    Command("resume", "list past sessions, or adopt one's history here",
            _resume, usage="[id]"),
    Command("rename", "name the current session", _rename, usage="<name>"),
    Command("fork", "save an independent copy of this conversation", _fork, usage="[name]"),
    Command("session", "id, transcript path, how to resume", _session),
    Command("cwd", "the workspace and what else is writable", _cwd),
    Command("clear", "forget the conversation, keep the session", _clear),
    Command("exit", "leave", _exit, aliases=("quit",)),
]

COMMANDS: dict[str, Command] = {}
for _cmd in ORDER:
    COMMANDS[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        COMMANDS[_alias] = _cmd

NAMES: list[str] = ["/" + c.name for c in ORDER]


def names_for(repl: Repl | None = None) -> dict[str, str]:
    """Every name tab should offer, with the one-line summary the terminal
    shows beside it - built-ins plus this workspace's files."""
    names = {"/" + c.name: c.summary for c in ORDER}
    files = repl.files if repl is not None and repl.files else {}
    for name in sorted(files):
        names["/" + name] = files[name].description or "command file"
    return names


def is_command(line: str) -> bool:
    """A line the REPL handles itself rather than sending to the model.

    A bare `/` is not one, and neither is `/path/to/thing` - a leading slash is
    how absolute paths start, and a user who types one is talking to the model
    about a file. The test is a single word of command characters; `:` is in
    the set because a command file in a subdirectory is named `git:commit`.
    """
    if not line.startswith("/") or line == "/":
        return False
    head = line[1:].split(None, 1)[0]
    return bool(head) and all(ch.isalnum() or ch in "-_?:" for ch in head)


def dispatch(line: str, repl: Repl) -> Outcome:
    """Run one slash command.

    A built-in answers with text the loop prints. A command file answers with a
    prompt the loop sends to the model - which is the whole difference between
    the two: a built-in costs nothing, a command file costs a turn.
    """
    head, _, rest = line[1:].partition(" ")
    key, rest = head.lower(), rest.strip()
    cmd = COMMANDS.get(key)
    if cmd is not None:
        return Outcome(text=cmd.run(repl, rest))
    onfile = repl.files.get(key) if repl.files else None
    if onfile is not None:
        return Outcome(prompt=onfile.render(rest))
    pool = list(COMMANDS) + list(repl.files or {})
    near = difflib.get_close_matches(key, pool, n=1)
    hint = f" - did you mean /{near[0]}?" if near else " - /help lists them"
    return Outcome(text=render.red(f"unknown command /{head}{hint}"))


__all__ = ["Repl", "Command", "Outcome", "COMMANDS", "NAMES", "ORDER",
           "names_for", "is_command", "dispatch", "status"]

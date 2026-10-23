"""The agent loop.

Model -> tool calls -> results -> model, until the model answers without asking
for a tool or the step budget runs out. Canonical history is OpenAI-shaped;
each wire converts on the way out. That choice keeps exactly one history format
in the loop, which is what makes `--resume` across two different backends work.

Between the model asking for a tool and the tool running there are two gates:
the guard chain (`onecli.guard`) and the PreToolUse hooks (`onecli.hooks`),
plus whatever `on_event("tool_start", ...)` returns. A denial never raises - it comes back to the model as the tool result,
because a model that is told *why* it was refused can take another route, and
one that gets an exception can only stop.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from onecli import config
from onecli import context as ctx
from onecli import response_gate as response_policy
from onecli import fallback
from onecli import lanehealth
from onecli import loopguard as loops
from onecli import media
from onecli import run_limits
from onecli import spill as spilllib
from onecli import permissions
from onecli.evals.run_analysis import audit_run
from onecli import tools as toolkit
from onecli.guard import GuardChain, Verdict
from onecli.hooks import HookSet
from onecli.providers import ProviderError, Turn
from onecli.providers.base import ToolCall

if TYPE_CHECKING:
    from onecli.masks.policy import PolicyRun
from onecli.providers.base import is_transient
from onecli.shellrun import run_shell

DEFAULT_SYSTEM = """You are one, a terminal coding agent.

You work by calling tools. Read before you edit; never guess a file's contents.
For substantial work, give a short initial update, then report only meaningful
findings, blockers or changes of approach. Routine tool calls are already visible;
do not narrate each call or repeat intermediate summaries. Keep working until the
authorized task is complete. End with one concise answer: the outcome, what was
verified, and any remaining blocker. Use short paragraphs, bullets when useful,
and backticks for paths and commands. Match the user's requested level of detail;
no preamble or restating the request. If a tool returns an error, read it and try a
different approach rather than repeating the same call. When the request names
something you cannot pin down after a short search (a project, file or person
that matches several candidates or none), ask one concise question listing the
candidates instead of searching on. Once a lookup has resolved what the user
means (an exact path or a named component) and you know the route, stop
reading: act, or ask one confirmation question. Do not read source code to
describe something you are about to change.

Search the task's known files or smallest relevant project/docs directory first.
Use grep/glob with a path and file filter, not shell recursion or home-directory
listings. Search tracked sources; omit archives, backups and other worktrees.
Expand one scope at a time only after a narrow search misses, explaining why.
Stop discovery when the owning code or requested evidence is found.

Workspace: {workspace}
""".strip()

# Headroom for a summarization call. See Agent._summarize: compaction runs
# exactly when the window is full, so the call that is supposed to rescue the
# run is the one most likely to be starved by the run's own --max-tokens.
SUMMARY_MIN_TOKENS = 1024
SUMMARY_OVERHEAD_TOKENS = 512        # the instruction and the wire's own shape
SUMMARY_INPUT_SAFETY = 0.5           # CHARS_PER_TOKEN is a floor, not a measure
SUMMARY_MIN_INPUT_CHARS = 2000


def _tighter(limit: int | None, cap: int | None) -> int | None:
    """The stricter of two limits, where None means no limit."""
    if cap is None:
        return limit
    return cap if limit is None else min(limit, cap)


def _merge_reasons(rungs) -> dict:
    """One retry-reason map for the whole chain. Only classes that happened."""
    out: dict[str, int] = {}
    for r in rungs:
        for reason, n in (getattr(r.provider, "retry_reasons", None) or {}).items():
            out[reason] = out.get(reason, 0) + n
    return out


@dataclass
class Step:
    index: int
    turn: Turn
    tool_results: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0
    denials: list[tuple[str, str]] = field(default_factory=list)
    advisories: list[tuple[str, str]] = field(default_factory=list)



# How often a failing done-check refuses the model's final reply before the run
# ends as `unverified`. A tool call resets the count: the model is working on
# it, not repeating a claim. Same shape as Cline's empty-turn nudge cap.
DONE_CHECK_RETRIES = 3
DONE_CHECK_TIMEOUT = 300
DONE_CHECK_OUTPUT = 2000
DONE_CHECK_NUDGE = (
    "Not done yet: the task's done-check `{command}` failed.\n{output}\n"
    "Your reply is not a completion until this check passes. Fix what it "
    "reports with tools, then answer.")

@dataclass
class Result:
    text: str
    steps: list[Step]
    messages: list[dict]
    # answered | truncated | unverified | budget | budget_usd | empty | error.
    # `unverified` is a run whose declared done-check still failed when the
    # model stopped for the last allowed time: the reply is not a completion.
    # `truncated` is a run that did real work and then could not produce a
    # final reply: `text` is recovered from the completed steps rather than
    # spoken by the model. `empty` is now only a run with nothing to recover.
    stopped: str = "answered"
    error: str = ""
    compactions: list[dict] = field(default_factory=list)
    fallbacks: list[dict] = field(default_factory=list)
    # Calls the endpoint answered only on a second sample of the same request,
    # because it could not parse its own model's reply the first time. A run
    # that needed three of these answered, but not cleanly.
    resamples: int = 0
    # Every request the wire retried rather than failed, and the class the wire
    # itself gave. `resamples` is the malformed-tool-call slice of this, kept
    # because the JSON output already names it.
    retries: int = 0
    retry_reasons: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    context_tokens: int = 0
    context_window: int = 0
    # The plan as it stood when the run ended. A run that answered with three
    # of eight items still open is a different outcome from one that finished
    # its list, and the exit code cannot say which.
    todos: list[dict] = field(default_factory=list)
    mask_policy: dict = field(default_factory=dict)
    mask: dict = field(default_factory=dict)

    @property
    def steps_text(self) -> list[dict]:
        """Nonempty assistant text in chronological order, including tool turns."""
        return [{"step": step.index, "text": step.turn.text}
                for step in self.steps if step.turn.text]

    @property
    def answer_advisories(self) -> list[str]:
        """Diagnostics, separate from loop-guard interventions."""
        if any(step.turn.tool_calls and step.turn.text.strip()
               and step.turn.text.strip() != self.text.strip()
               for step in self.steps):
            return ["Earlier text accompanied a tool call and differs from the final answer; inspect steps_text."]
        return []

    @property
    def tool_calls(self) -> int:
        return sum(len(s.turn.tool_calls) for s in self.steps)

    @property
    def loop_stops(self) -> int:
        """How many times the loop guard had to speak. It already counts them
        as advisories; this is that count, not a second tally beside it."""
        return len(self.advisories)

    @property
    def denied(self) -> list[tuple[str, str]]:
        """(tool, reason) for every call a guard refused."""
        return [d for s in self.steps for d in s.denials]

    @property
    def advisories(self) -> list[tuple[str, str]]:
        """(tool, note) for every loop-guard advisory the model was shown."""
        return [a for s in self.steps for a in s.advisories]


class Agent:
    """The tool loop. Pass EITHER `provider` (one backend) or `chain` (a
    ladder) - never both, so there is one answer to "which backend is live"."""

    def __init__(self, provider=None, workspace: str = "", system: str | None = None,
                 max_steps: int | None = None, tool_names: list[str] | None = None,
                 on_event: Callable[[str, dict], object] | None = None,
                 max_tokens: int | None = None, temperature: float | None = None,
                 guard: GuardChain | None = None, stream: bool = False,
                 hooks: HookSet | None = None,
                 autocompact: bool = True,
                 context_window: int = ctx.DEFAULT_CONTEXT_WINDOW,
                 compact_at: float = 1.0,
                 keep_last: int = 6,
                 max_input_tokens: int | None = None,
                 max_budget_usd: float = 0.0,
                 price: tuple[float, float] | None = None,
                 model_name: str = "",
                 cache_read_weight: float | None = None,
                 chain=None,
                 spill=None,
                 loop_guard=None,
                 jobs=None,
                 asker=None,
                 spawner=None,
                 todos=None,
                 tasks=None,
                 response_gate="off", progress_policy=None, meditate=False,
                 request_admission=None, request_budget_isolated=False,
                 sandbox_tools=None, progressive_context=False,
                 mask_policy=None, mask_verifier=None, loop_policy=None,
                 mask_name="", mask_preamble="", context_contract=None, turn_feedback=None,
                 done_check: str = "", trust_project: bool = False,
                 session_id: str | None = None):
        from onecli import execution_evidence
        self.execution_evidence = execution_evidence.initialize() if session_id else None
        if self.execution_evidence:
            self.execution_evidence.session_id = session_id
        self.session_id = session_id
        self._save_execution_checkpoint: Callable[..., None] | None = None
        from onecli.masks.policy import MaskPolicy
        # A shell command that must exit 0 before a final reply counts as done.
        # The model's own "done" is a claim; this is the evidence (a run can
        # say "done" with the deliverable never copied).
        self.done_check = done_check
        # --trust-project-config: whether the workspace's own skills may load.
        self.trust_project = trust_project
        self.done_refusals = 0
        from onecli.masks import loop as mask_loop
        from onecli.context_contract import parse as parse_context
        self.turn_feedback = turn_feedback
        self.context_contract = parse_context(context_contract) if context_contract is not None else None
        self.context_observations: list[dict] = []
        self.mask_name, self.mask_preamble = mask_name, mask_preamble
        self.loop_policy = mask_loop.parse(loop_policy) if loop_policy is not None else {}
        if self.loop_policy:
            max_steps = _tighter(max_steps, self.loop_policy.get('max_steps'))
        # None: an empty turn is always answered with a recovery nudge. A mask
        # may opt in to a cap; there is none by default.
        self.empty_retries = self.loop_policy.get('empty_retries')
        # Tools marked explicit run only when the caller names them.
        if tool_names is None:
            tool_names = [n for n, t in toolkit.REGISTRY.items() if not t.explicit]
        self.mask_policy = None if mask_policy is None else MaskPolicy.parse(mask_policy)
        if self.mask_policy and request_budget_isolated:
            raise ValueError("mask policies do not yet support isolated request budgets")
        if self.mask_policy and sandbox_tools is not None:
            raise ValueError("mask policies do not yet support sandbox tool adapters")
        if self.mask_policy and self.mask_policy.kind == "debug" and not callable(mask_verifier):
            raise ValueError("debug mask policy requires a trusted host verifier")
        self.mask_verifier = mask_verifier
        self._mask_run: PolicyRun | None = None
        if self.mask_policy:
            max_steps = _tighter(max_steps, self.mask_policy.max_steps)
            surface = toolkit.REGISTRY if tool_names is None else tool_names
            tool_names = [n for n in surface if n in self.mask_policy.tools]
        self.progressive_context = progressive_context
        self.request_budget_isolated = bool(request_budget_isolated)
        if sandbox_tools is not None:
            from onecli.sandbox.tools import SandboxTools
            if type(sandbox_tools) is not SandboxTools or not self.request_budget_isolated:
                raise ValueError('sandbox tools require host adapter and isolated request budget')
        self.sandbox_tools = sandbox_tools
        if self.request_budget_isolated:
            from onecli.request_admission import RequestAdmission
            if type(request_admission) is not RequestAdmission:
                raise ValueError("isolated request budget requires RequestAdmission")
            if hooks or guard is not None or spawner is not None:
                raise ValueError("isolated request budget forbids hooks, guards and spawners")
            tool_names = []
            jobs = asker = tasks = todos = None
            autocompact = False
        self.request_admission = request_admission
        self.tasks = tasks
        # The chain owns which backend is live. A run with no chain is a chain
        # of one, so the loop below has a single path rather than two.
        if chain is None:
            if provider is None:
                raise ValueError("Agent needs a provider or a chain")
            chain = fallback.Chain(
                rungs=[fallback.Rung(name=model_name or "primary",
                                     provider=provider, model=model_name)],
                source="off",
            )
        from onecli import progress_policy as progress
        self.progress_policy = progress.resolve(progress_policy, base=progress.resolve(
            self.loop_policy.get('progress_policy'), base={**progress.DEFAULT, "enabled": False}))
        self.meditate = bool(meditate)
        self._meditating = False
        self.response_gate = response_policy.mode(response_gate)
        self.chain = chain
        # Real path, once: the no-follow file tools refuse symlinked paths.
        self.workspace = workspace = os.path.realpath(workspace) if workspace else workspace
        self.run_limits = run_limits.load(workspace)
        if sandbox_tools is not None and self.run_limits:
            raise ValueError('sandbox tools cannot override workspace research limits')
        self.request_bytes_sent = 0
        self.wire_requests = 0
        self.wire_bytes = 0
        if self.run_limits:
            max_steps = _tighter(max_steps, self.run_limits['max_steps'])
            max_tokens = _tighter(max_tokens, self.run_limits['max_output_tokens'])
            autocompact = False  # no hidden summarization calls under this budget
            surface = list(toolkit.REGISTRY) if tool_names is None else tool_names
            tool_names = [n for n in surface if n == 'web_fetch']
            for rung in self.chain.rungs:
                if hasattr(rung.provider, 'request_gate'):
                    previous_gate = rung.provider.request_gate
                    def limited_gate(body, previous=previous_gate):
                        self._admit_wire_request(body)
                        if previous is not None:
                            previous(body)
                    # Keep shared admission hooks when a child loads limits.
                    setattr(limited_gate, '_request_admission',
                            getattr(previous_gate, '_request_admission', None))
                    rung.provider.request_gate = limited_gate
                    rung.provider.timeout = min(rung.provider.timeout, 45)
        if request_admission is not None:
            if len(self.chain.rungs) != 1:
                raise ValueError('request budget requires a single audited route')
            request_admission.bind(self.chain.current.provider)
        # `.replace`, not `.format`: the system prompt now carries discovered
        # CLAUDE.md text, and a stray brace in someone's project notes must not
        # be read as a format field.
        self.system = (system or DEFAULT_SYSTEM).replace("{workspace}", workspace)
        # A tool may carry a note about itself (an untrusted-content warning,
        # say). A run whose surface holds it gets the note, whatever prompt the
        # caller passed; a run without the tool is not told about it.
        surface_names = tool_names if tool_names is not None else toolkit.REGISTRY
        notes = dict.fromkeys(toolkit.REGISTRY[n].system_note for n in surface_names
                              if n in toolkit.REGISTRY and toolkit.REGISTRY[n].system_note)
        for note in notes:
            self.system = self.system + "\n\n" + note
        self.max_steps = max_steps
        self.jobs = jobs
        self.asker = asker
        self.todos = todos
        # The spawner is handed in without a parent and adopts this agent here.
        # It cannot be built by the caller with `parent=self` because self does
        # not exist yet, and building it afterwards would mean widening the
        # tool surface after the surface was decided.
        self.spawner = spawner
        if spawner is not None:
            spawner.parent = self
        # A run with nowhere to put a background job is not offered one, and a
        # run with nobody to ask is not offered a question. A tool whose only
        # possible answer is "not available in this run" costs a step to
        # discover and teaches the model nothing.
        withheld: set[str] = set()
        if jobs is None:
            withheld.update(toolkit.JOB_TOOL_NAMES)
        if asker is None:
            withheld.update(toolkit.ASK_TOOL_NAMES)
        if spawner is None:
            withheld.update(toolkit.TASK_TOOL_NAMES)
        if todos is None:
            withheld.update(toolkit.TODO_TOOL_NAMES)
        if withheld:
            surface = list(toolkit.REGISTRY) if tool_names is None else tool_names
            tool_names = [n for n in surface if n not in withheld]
        self.tool_names = tool_names
        # A run that can spawn is told WHICH agents exist. The `task` schema can
        # only say "a name"; the names themselves are per-run, coming from
        # --agents files, so they belong in the prompt beside the tool. Without
        # this a preset the operator wrote is a tool the model cannot address -
        # it defaults to `general` and the definition may as well not exist.
        if spawner is not None and (tool_names is None
                                    or any(n in tool_names
                                           for n in toolkit.TASK_TOOL_NAMES)):
            self.system = (self.system + "\n\nSubagents you can hand a job to "
                           "with `task`:\n" + spawner.catalogue())
        self.on_event = on_event or (lambda kind, data: None)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.guard = guard
        self.autocompact = autocompact
        self.tracker = ctx.ContextTracker(window=context_window,
                                          compact_at=compact_at,
                                          keep_last=keep_last,
                                          reply_tokens=max_tokens or 0,
                                          counter=self._served_count)
        self.max_budget_usd = max_budget_usd
        # Context size bounds one request. Stateless tool loops resend that
        # request on every step, so they also need a cumulative input bound.
        # Zero explicitly disables it. The bound meters cost-relevant input
        # (CostMeter.metered_input_tokens): cache reads count at their price
        # relative to fresh input, so re-sending a cached prefix is cheap and
        # new context is not.
        self.max_input_tokens = max_input_tokens or 0
        self.meter = ctx.CostMeter(
            model=model_name,
            price_in=price[0] if price else 0.0,
            price_out=price[1] if price else 0.0,
            price_known=price is not None,
            cache_read_weight=(config.cache_read_weight(model_name)
                               if cache_read_weight is None else cache_read_weight),
        )
        self.compactions: list[ctx.Compaction] = []
        self._current_request: dict | None = None
        self.stream = stream
        self.hooks = hooks
        self.spill = spill
        if spill is not None:
            permissions.add_read_root(self.guard, spill.root)
        # Advisory by default and cheap either way: an off switch is
        # `LoopGuard(repeat_limit=0, deadline=0)`, not a None nobody remembers
        # to check at three callsites.
        self.loop = loop_guard if loop_guard is not None else loops.LoopGuard()
        if self.loop_policy:
            from dataclasses import replace
            self.loop = replace(self.loop,
                repeat_limit=mask_loop.ceiling(self.loop.repeat_limit, self.loop_policy.get('repeat_limit', self.loop.repeat_limit)),
                deadline=mask_loop.ceiling(self.loop.deadline, self.loop_policy.get('tool_deadline', self.loop.deadline)),
                discovery_limit=mask_loop.ceiling(self.loop.discovery_limit, self.loop_policy.get('discovery_limit', self.loop.discovery_limit)))

    @property
    def provider(self):
        """The rung the run currently stands on - compaction included, since a
        summarization call to a dead endpoint is the same failure twice."""
        return self.chain.current.provider

    @audit_run
    def run(self, prompt: str, messages: list[dict] | None = None,
            attachments: list[media.Attachment] | None = None) -> Result:
        if messages:
            recovered = []
            for index, message in enumerate(messages):
                recovered.append(message)
                if message.get('role') != 'assistant' or not message.get('tool_calls'):
                    continue
                following = messages[index + 1:]
                boundary = next((i for i, m in enumerate(following)
                                 if m.get('role') != 'tool'), len(following))
                answered = {m.get('tool_call_id') for m in following[:boundary]}
                for call in message['tool_calls']:
                    if call['id'] not in answered:
                        recovered.append({'role': 'tool', 'tool_call_id': call['id'],
                                          'name': call['function']['name'],
                                          'content': 'Interrupted before a result was recorded. Outcome unknown; inspect state before retrying.'})
            messages = recovered
        self.interrupted_messages = list(messages or [])
        self._active_history = self.interrupted_messages
        try:
            return self._run(prompt, messages, attachments)
        except KeyboardInterrupt:
            history = list(self._active_history)
            # A cancelled tool may have performed side effects. Never call it
            # successful or claim it did not run; close the wire pair explicitly.
            for index, message in enumerate(list(history)):
                following = history[index + 1:]
                boundary = next((i for i, m in enumerate(following) if m.get('role') == 'assistant'), len(following))
                answered = {m.get('tool_call_id') for m in following[:boundary] if m.get('role') == 'tool'}
                for call in message.get('tool_calls', []):
                    if call['id'] not in answered:
                        history.append({'role': 'tool', 'tool_call_id': call['id'],
                                        'name': call['function']['name'],
                                        'content': 'Interrupted before a result was recorded. Outcome unknown; inspect state before retrying.'})
            history.append({'role': 'user', 'content':
                'The previous turn was interrupted. Its request remains unfinished unless my next message cancels or replaces it. Consider my newest message together with that request.'})
            self.interrupted_messages = history
            raise
        finally:
            if self.turn_feedback is not None:
                self.turn_feedback.close()

    def _run(self, prompt: str, messages: list[dict] | None = None,
             attachments: list[media.Attachment] | None = None) -> Result:
        self.context_observations = []
        if self.mask_policy:
            from onecli.masks.policy import PolicyRun
            self._mask_run = PolicyRun(self.mask_policy, self.mask_verifier)
            self.on_event("mask_policy", self.mask_policy.as_dict())
        history: list[dict] = list(messages) if messages else [
            {"role": "system", "content": self.system}
        ]
        self._active_history = history
        # A plan read back off disk is state the model has never seen in this
        # session, so it is introduced as a user turn before the prompt.
        if self.todos is not None and getattr(self.todos, "restored", False):
            history.append({"role": "user",
                            "content": self.todos.block("carried over from the "
                                                        "resumed session")})
            self.todos.restored = False
            self.on_event("todos", self._todo_event("resumed"))
        self._current_request = {
            "role": "user",
            "content": media.user_content(prompt, attachments or []),
        }
        history.append(self._current_request)
        # Named subjects with no source ("the Shifting Gardens") are looked up
        # before they are drawn; nothing in the words reads as a reference.
        lookup = loops.lookup_note(prompt if isinstance(prompt, str) else "",
                                   set(self.tool_names or ()))
        if lookup:
            history.append({"role": "user", "content": lookup})

        from onecli import progress_policy as progress
        cadence = progress.Cadence(self.progress_policy, time.monotonic())
        # Once per conversation: every REPL turn is a new run() over the same
        # history, and appending it each time stacked one identical copy per
        # "Continue" into the context.
        if self.progress_policy["enabled"] and not any(
                m.get("content") == progress.PROMPT for m in history):
            history.append({"role": "user", "content": progress.PROMPT})
        assessing = self.response_gate == "enforce"
        assessment_attempts = 0
        if self.response_gate == "prompt":
            history.append({"role": "user", "content": response_policy.PROMPT})
        schemas = [] if self.request_budget_isolated else toolkit.schemas(self.tool_names)
        if self.sandbox_tools is not None:
            schemas = self.sandbox_tools.schemas()
        from onecli.tool_discovery import Discovery
        discovery = Discovery(schemas, self.tracker.window)
        if self.sandbox_tools is not None or self.request_budget_isolated:
            discovery.enabled = False
        steps: list[Step] = []
        # Retry one consecutive empty response. Productive turns reset this
        # allowance; a recovered stall must not make a later stall terminal.
        # All retries still consume the existing step and cost budgets.
        self._meditating = False
        meditate_used = False
        empty_streak = 0
        empty_retry_limit = self.empty_retries
        if empty_retry_limit is None and self.session_id:
            empty_retry_limit = 1
        boosted_max_tokens = 0
        recovery_prompt = None
        cutoff_retries = 0
        overflow_retries = 0
        from onecli import session as session_store
        if self.session_id:
            try:
                saved = session_store.load_records(self.session_id, 'execution_checkpoint')
            except FileNotFoundError:
                saved = []
            if saved and saved[-1].get('status') != 'answered':
                retry_state = saved[-1].get('retry_state', {})
                empty_streak = retry_state.get('empty_streak', 0)
                cutoff_retries = retry_state.get('cutoff_retries', 0)
                overflow_retries = retry_state.get('overflow_retries', 0)

        def save_execution_checkpoint(pending=(), status='running'):
            if not self.session_id:
                return
            pending_actions = [{'call_id': c.id, 'name': c.name, 'input': c.arguments}
                               for c in pending]
            retry_state = {'empty_streak': empty_streak, 'cutoff_retries': cutoff_retries,
                           'overflow_retries': overflow_retries}
            session_store.append(self.session_id, {'kind': 'messages', 'messages': history}, durable=True)
            record = {'kind': 'execution_checkpoint',
                      'pending_actions': pending_actions, 'retry_state': retry_state,
                      'status': status}
            session_store.append(self.session_id, record, durable=True)
            if self.execution_evidence:
                self.execution_evidence.emit({
                    'type': 'executor_checkpoint', 'session_id': self.session_id,
                    'pending_actions': pending_actions, 'retry_state': retry_state,
                    'status': status})
        self._save_execution_checkpoint = save_execution_checkpoint
        # Low-progress checkpoint (loopguard): due after a run of discovery
        # rounds, sent as one request with tools withheld.
        self.loop.discovery_rounds = 0
        self.loop.resolved = self.loop.commitment = ""
        checkpoint_due = False
        checkpoint_final = False
        checkpoint_ignored = 0

        from onecli.context_contract import Assembly, ContextError
        assembly = Assembly(self.context_contract, self.workspace, self._authorize_context) if self.context_contract else None

        # Largest step-to-step growth of the request estimate seen so far. The
        # final-answer reservation below must cover the NEXT request, which is
        # this one plus whatever the tool results add.
        ceiling_growth = 0
        prev_estimate = 0
        for index in itertools.count(1):
            if self.max_steps is not None and index > self.max_steps:
                break
            if self.turn_feedback is not None:
                self.turn_feedback.consume(history, acknowledge=False)
            # Compaction must budget the same fitted system prompt sent on the wire.
            fitted_history, _ = ctx.fit_system(
                history, self.tracker.window, self.tracker.ratio)
            if (self.autocompact and self.tracker.should_compact(fitted_history)
                    and ctx.has_compactable(history, self.tracker.keep_last)):
                history = self._compact(history, index)

            if assembly:
                try:
                    observation = assembly.reconcile(history)
                    self.context_observations.append({'kind': 'assembly', **observation})
                    self.on_event('context_contract', observation)
                except ContextError as exc:
                    self.on_event('context_blocked', {'reason': str(exc)})
                    return self._result('', steps, history, stopped='context', error=str(exc))

            # A job that finished while the model was thinking about something
            # else is announced once, at the top of the next turn. Without this
            # the model must poll - and a model that must poll either forgets
            # to, or polls every turn and has re-invented waiting.
            for note in self._job_notices():
                history.append({"role": "user", "content": note})
            for note in self._todo_notices():
                history.append({"role": "user", "content": note})
            for note in self._task_notices():
                history.append({"role": "user", "content": note})

            # Reserve the last existing step for delivery, not more research.
            # No extra call or budget increase; single-step tool callers keep
            # their existing contract. A provider ignoring the empty tool
            # list still cannot execute a tool on this step.
            final_step = self._last_step(index) and (self.max_steps or 0) > 1
            input_ceiling_final = False
            updating = not assessing and not final_step and cadence.due(time.monotonic())
            # Keep the tool declaration stable on the reserved answer step.
            # Execution is refused below if the model ignores the no-tools
            # instruction; deleting schemas here needlessly invalidates the
            # provider's cached prefix on the largest request of the run.
            # Assessment is a JSON decision, never a tool turn. Do not offer
            # executable tools; still reject rogue tool calls below.
            request_schemas = [] if assessing else discovery.schemas()
            if self._meditating:
                from onecli.meditate import REVIEW_TOOLS
                request_schemas = [s for s in request_schemas
                                   if s["function"]["name"] in REVIEW_TOOLS]

            self._active_history = history
            request_history = (response_policy.assessment_history(
                                   history, terminal=self._last_step(index))
                               if assessing else history)
            request_history, elided = ctx.fit_system(
                request_history, self.tracker.window, self.tracker.ratio)
            if elided:
                self.on_event("system_elided", {"characters": elided,
                                                "window": self.tracker.window})
            # Selected mask instructions are request-local, so switching or
            # disabling a mask never leaves stale persona instructions behind.
            mask_text = self.mask_preamble
            if assembly:
                mask_text += "\n" + assembly.instruction()
            if self.mask_policy:
                mask_text += "\n" + self.mask_policy.instruction
            if mask_text:
                request_history = list(request_history)
                pos = next((i for i, m in enumerate(request_history) if m.get('role') != 'system'), len(request_history))
                request_history.insert(pos, {'role': 'system', 'content':
                    f"Active mask: {self.mask_name or (self.mask_policy.kind if self.mask_policy else 'custom')}\n" + mask_text})

            # Reserve an affordable final-answer request. Compaction limits the
            # CURRENT context; this limits the sum of full prompts processed by
            # a stateless provider across the whole tool loop.
            # The local estimator cannot see provider-side tokenization and
            # some wires account tool schemas separately.  Once a request has
            # completed, its reported context is the safer lower bound for the
            # next (normally larger) request.
            estimated_request = max(
                self.tracker.estimate(request_history),
                self.meter.context_tokens,
            )
            # The per-request window bound is NOT this ceiling and does not
            # read the metered figure: fit_system, autocompact (tracker, on
            # full prompt tokens) and the overflow compact-and-retry below
            # bound each request against the window at full size. A cache
            # discount makes a re-send cheap; it never makes it fit.
            if self.max_input_tokens and not assessing:
                spent = self.meter.metered_input_tokens
                next_cost = self.meter.metered(estimated_request)
                if spent + next_cost > self.max_input_tokens:
                    return self._result(
                        _salvage(steps), steps, history,
                        stopped="budget_input",
                        error=(f"Cumulative input ceiling reached before step {index} "
                               f"({spent:,.0f} metered of {self.meter.input_tokens:,} "
                               f"processed; next request about {estimated_request:,}; "
                               f"ceiling {self.max_input_tokens:,})"),
                    )
                if prev_estimate:
                    ceiling_growth = max(ceiling_growth,
                                         estimated_request - prev_estimate)
                prev_estimate = estimated_request
                # Reserve this request plus a final answer at least as large as
                # the next one will be. Reserving only 2x the current estimate
                # is not enough when the request is growing: the next step no
                # longer fits and the run stops with no answer at all.
                # Growth is new, uncached context, so it counts at full weight.
                if (not final_step and spent + 2 * next_cost + ceiling_growth
                        > self.max_input_tokens):
                    final_step = True
                    input_ceiling_final = True
            # Controls belong to this request, never to a checkpoint or the
            # next tool turn. In particular, a failed HTTP request must not
            # leave a durable instruction forbidding tools on resume.
            if assessing and assessment_attempts:
                request_history = request_history + [{"role": "user", "content":
                    "The assessment was invalid; no tools executed. " + response_policy.ASSESS}]
            if final_step and not assessing:
                request_history = request_history + [{"role": "user", "content": (
                    "This is the final response step within the current budget. "
                    "Use the evidence already collected to answer the user's "
                    "request now. Do not call tools. Report useful findings, "
                    "verified artifacts, and what remains incomplete. Do not "
                    "claim unfinished work is done or promise background work "
                    "that was not started.")}]

            checkpointing = checkpoint_due and not final_step and not assessing
            if checkpointing:
                offered = {s["function"]["name"] for s in request_schemas}
                prompt, checkpoint_final = loops.checkpoint_prompt(
                    self.loop, can_act=bool(offered & loops.ACTING_TOOLS))
                request_history = request_history + [{"role": "user", "content": prompt}]
                self.on_event("discovery_checkpoint", {
                    "step": index, "rounds": self.loop.discovery_rounds,
                    **({"resolved": self.loop.resolved} if self.loop.resolved else {}),
                    **({"final": True} if checkpoint_final else {})})

            if recovery_prompt and not final_step:
                request_history = request_history + [{"role": "user", "content": recovery_prompt}]
                recovery_prompt = None
            if updating:
                request_history = request_history + [{"role": "user", "content": cadence.reminder(time.monotonic())}]
                # A cadence reminder is part of the normal task request. It
                # neither removes tools nor forces an extra public response.
                cadence.defer(time.monotonic())
            if self.run_limits:
                size = run_limits.request_bytes(request_history, request_schemas)
                limits = self.run_limits
                if (size > limits['max_request_bytes'] or
                        self.request_bytes_sent + size > limits['max_total_request_bytes'] or
                        self.meter.input_tokens + size > limits['max_input_tokens']):
                    return self._result('', steps, history, stopped='budget',
                                        error='Workspace research input limit reached before request')

            # Announced before the request, because this is the only moment a
            # host can know a wait has STARTED. `turn` below reports one that
            # has already ended.
            # Persist only complete exchanges: every assistant tool call in
            # this snapshot has its result, so a resumed provider accepts it.
            save_execution_checkpoint()
            self.on_event("checkpoint", {"messages": history})
            self.on_event("waiting", {"step": index})
            if self.turn_feedback is not None:
                self.turn_feedback.acknowledge()
            started = time.monotonic()
            try:
                turn = self._chat(request_history, request_schemas,
                                  max_tokens=boosted_max_tokens or None,
                                  public_stream=not (assessing or self.meditate),
                                  json_output=assessing)
            except ProviderError as exc:
                overflow = ctx.parse_overflow(str(exc))
                keep_last = self.tracker.keep_last if overflow_retries == 0 else 0
                if (overflow and overflow_retries < 2
                        and ctx.has_compactable(history, keep_last)):
                    # The server named the real size and the real window.
                    # First preserve the recent exchange. If that still does
                    # not fit, summarize it too while pinning the open request.
                    overflow_retries += 1
                    self.tracker.overflowed(request_history, *overflow)
                    self.on_event("context_overflow", {
                        "n_prompt_tokens": overflow[0], "n_ctx": overflow[1]})
                    history = self._compact(history, index, keep_last=keep_last)
                    continue
                if overflow:
                    return self._result("", steps, history, stopped="error",
                                        error=(f"the conversation ({overflow[0]} tokens) "
                                               f"does not fit this model's {overflow[1]}-token "
                                               "context even after compacting"))
                return self._result("", steps, history,
                                    stopped="error", error=str(exc))

            overflow_retries = 0
            self.tracker.observe(request_history, turn.usage)
            self.meter.add_turn(request_history, turn.usage, turn.text)

            step = Step(index=index, turn=turn, seconds=time.monotonic() - started)
            steps.append(step)
            self.on_event("turn", {"step": step})
            # Persisted per call so a bad turn can be attributed to its cause:
            # finish_reason says whether the output cap cut it, usage says how
            # much of it was reasoning, and the reasoning itself is otherwise
            # lost (the history keeps only the provider echo, when any).
            self.on_event("assistant_turn", {
                "index": len(history), "step": index,
                "finish_reason": turn.finish_reason, "usage": turn.usage,
                "reasoning": turn.thinking, "seconds": round(step.seconds, 3)})

            if assessing:
                assessment_attempts += 1
                try:
                    if turn.tool_calls:
                        raise ValueError("tools requested before assessment")
                    decision, message = response_policy.parse(turn.text)
                except (ValueError, TypeError):
                    self.on_event("response_gate", {"decision": "invalid",
                                  "attempt": assessment_attempts})
                    if assessment_attempts >= 2 or self._over_budget():
                        return self._result("", steps, history, stopped="error",
                            error="First-response assessment invalid; no tools executed")
                    continue
                # Store only public text in history. The structured decision is
                # control data, never streamed as an answer or tool arguments.
                turn.text = message
                history.append({"role": "assistant", "content": message})
                self.on_event("response_gate", {"decision": decision,
                              "attempt": assessment_attempts})
                if decision != "proceed":
                    # Stop hooks must not override a request for clarification
                    # or a refusal and accidentally unlock tools.
                    return self._result(message, steps, history,
                                        stopped="assessment" if self.mask_policy else "answered")
                self.on_event("acknowledgment", {"text": message})
                cadence.sent(time.monotonic())
                if self._over_budget():
                    return self._result(message, steps, history, stopped="budget_usd",
                                        error="Budget reached during first-response assessment")
                if self._last_step(index):
                    # The gate spent the run's last model call. The message is
                    # public text, but the work it acknowledges has no turn left
                    # to run in, and the loop's generic "stopped after N steps
                    # without a final answer" hides the cause, so this stop
                    # names it.
                    return self._result(
                        message, steps, history, stopped="budget",
                        error=("the first-response gate used the run's last model "
                               f"call ({index} of {self.max_steps}); the "
                               "acknowledged work could not start"))
                assessing = False
                history.append({"role": "user", "content":
                    ctx.ACK_CONTINUATION})
                continue

            if checkpointing:
                if (turn.tool_calls and not checkpoint_final
                        and loops.checkpoint_commits(
                            turn.text or "",
                            [(c.name, c.arguments) for c in turn.tool_calls])):
                    # The reply decided and acted in one turn: that is the
                    # commit the checkpoint asked for, so its calls run.
                    checkpoint_due = False
                    checkpoint_ignored = 0
                    loops.committed(self.loop, turn.text or "; ".join(
                        f"{c.name} {json.dumps(c.arguments)[:160]}" for c in turn.tool_calls))
                    self.on_event("discovery_checkpoint", {
                        "step": index, "decision": "commit", "acted": True})
                elif turn.tool_calls:
                    # Same refusal as the reserved final step: a tool call on
                    # a tools-withheld step is never executed.
                    checkpoint_ignored += 1
                    self.on_event("discovery_checkpoint", {
                        "step": index, "decision": "ignored",
                        "attempt": checkpoint_ignored})
                    if checkpoint_ignored >= 2:
                        return self._result(
                            _salvage(steps), steps, history, stopped="budget",
                            error=(f"{self.loop.discovery_rounds} tool rounds without "
                                   "progress, and the model kept calling tools "
                                   "instead of committing to a target or asking; "
                                   "no further tools executed"))
                    continue
                elif (not checkpoint_final
                        and (turn.text or "").lstrip().startswith(loops.COMMIT_PREFIX)):
                    checkpoint_due = False
                    checkpoint_ignored = 0
                    loops.committed(self.loop, turn.text or "")
                    self.on_event("discovery_checkpoint", {
                        "step": index, "decision": "commit"})
                    history.append({"role": "assistant", "content": turn.text})
                    continue
                elif (turn.text or "").strip():
                    self.on_event("discovery_checkpoint", {
                        "step": index, "decision": "ask"})
                    history.append({"role": "assistant", "content": turn.text})
                    return self._result(turn.text, steps, history,
                                        stopped="answered")

            if turn.tool_calls or (turn.text or "").strip():
                empty_streak = 0
            if turn.tool_calls:
                self.done_refusals = 0

            if (turn.text or "").strip():
                cadence.sent(time.monotonic())
                self.on_event("assistant_text", {"text": turn.text})
            if not turn.tool_calls:
                # A turn with neither text nor a tool call is a silent stall.
                # Naming it beats printing nothing and exiting 0, which reads
                # as a successful run that answered the question.
                if not (turn.text or "").strip():
                    # An empty turn has two causes and both are recoverable. finish_reason=max_tokens means the ceiling was
                    # spent before any visible content - a reasoning model
                    # thinking past its budget - so retrying at the same
                    # ceiling would reproduce it exactly; raise it instead.
                    # Any other finish_reason is a model that simply said
                    # nothing, and asking for the answer usually gets it.
                    #
                    # This matters most at the END of a run: by step N the
                    # tools have already done the work, and failing here
                    # discards all of it and reports "returned nothing" for a
                    # task that in fact completed.
                    if empty_retry_limit is None or empty_streak < empty_retry_limit:
                        empty_streak += 1
                        if turn.finish_reason in ("max_tokens", "length"):
                            boosted_max_tokens = self._boosted_max_tokens()
                            nudge = ("Your previous turn hit the output limit "
                                     "before producing any visible text. The "
                                     "limit has been raised. Keep any reasoning short. "
                                     "Continue unfinished work with tools; answer "
                                     "only when complete or genuinely blocked.")
                        else:
                            nudge = ("Your previous turn was empty. Continue the "
                                     "authorized task from the evidence already collected. "
                                     "Use tools if work remains; give the final answer "
                                     "only when complete or genuinely blocked, including "
                                     "any required output lines.")
                        self.on_event("empty_retry", {
                            "step": index,
                            "finish_reason": turn.finish_reason or "none",
                            "attempt": empty_streak,
                            "max_tokens": boosted_max_tokens or self.max_tokens,
                        })
                        recovery_prompt = nudge
                        continue
                    # The retry did not help either. Everything the run
                    # already DID is still in `steps`, and reporting "returned
                    # nothing" for a run that read nine files and wrote three
                    # is a false statement about the work, not just a terse
                    # one. Hand back the last thing the model said - or, if it
                    # never said anything, what the tools produced - and call
                    # the outcome `truncated`, which is what it is.
                    salvaged = _salvage(steps)
                    why = (f"model returned an empty turn at step {index} "
                           f"(finish_reason={turn.finish_reason or 'none'})"
                           + (f" and again after {empty_streak} {'retry' if empty_streak == 1 else 'retries'}" if empty_streak
                              else "; recovery disabled by mask"))
                    if salvaged:
                        return self._result(
                            salvaged, steps, history, stopped="truncated",
                            error=(why + "; the answer below is recovered from "
                                   f"{len(steps)} completed step(s), not a "
                                   "final reply"),
                        )
                    return self._result("", steps, history, stopped="empty",
                                        error=why)
                history = history + [{"role": "assistant", "content": turn.text, **turn.echo}]
                # Text that ends at the output ceiling is a turn cut off
                # mid-output, usually a large write whose tool call was lost
                # to the limit. Ending the run on it reports unfinished work
                # as answered.
                if turn.finish_reason in ("max_tokens", "length"):
                    if cutoff_retries < 2:
                        cutoff_retries += 1
                        boosted_max_tokens = self._boosted_max_tokens()
                        self.on_event("cutoff_retry", {
                            "step": index, "max_tokens": boosted_max_tokens})
                        recovery_prompt = (
                            "Your previous turn was cut off at the output limit, "
                            "so any tool call in it was lost. The limit has been "
                            "raised. Continue the task with tools, and write large "
                            "files in several smaller writes or edits.")
                        continue
                    return self._result(
                        _salvage(steps) or turn.text, steps, history,
                        stopped="truncated",
                        error=(f"turn {index} still hit the output limit after "
                               f"{cutoff_retries} retries; the run did not finish"))
                if self.meditate and not meditate_used:
                    meditate_used = True
                    if (self.max_steps is None or self.max_steps - index >= 2) and not self._over_budget():
                        from onecli.meditate import REVIEW
                        self._meditating = True
                        self.on_event("meditate", {"status": "reviewing", "step": index})
                        history.append({"role": "user", "content": REVIEW})
                        continue
                    self.on_event("meditate", {"status": "skipped_budget", "step": index})
                if self._meditating:
                    self.on_event("meditate", {"status": "reviewed", "step": index})
                    self._meditating = False
                failed = self._done_check_failure()
                if failed is not None:
                    if self.done_refusals < DONE_CHECK_RETRIES:
                        self.done_refusals += 1
                        self.on_event("done_check_failed", {
                            "step": index, "attempt": self.done_refusals,
                            "output": failed[-400:]})
                        history.append({"role": "user", "content": DONE_CHECK_NUDGE.format(
                            command=self.done_check, output=failed)})
                        continue
                    return self._result(
                        turn.text, steps, history, stopped="unverified",
                        error=(f"done-check `{self.done_check}` still failed after "
                               f"{self.done_refusals} nudges; the run did not finish: "
                               f"{failed[-400:]}"))
                refusal = self._stop_refusal()
                if refusal is not None:
                    # A Stop hook said the work is not finished. Feed the reason
                    # back as a user turn so the model can act on it - the loop
                    # continuing with no explanation would just repeat itself.
                    history.append({"role": "user", "content": refusal})
                    continue
                if (not self._last_step(index) and self.turn_feedback is not None
                        and self.turn_feedback.consume(history, acknowledge=False)):
                    continue
                if input_ceiling_final:
                    # The ceiling, not the model, ended this run: tools were
                    # withheld and the reply is whatever fit. Reporting it as
                    # `answered` let the host deliver "I need one more
                    # tool-use turn" as a finished job.
                    return self._result(
                        turn.text, steps, history, stopped="budget_input",
                        error=("final step forced by the cumulative input ceiling; "
                               "tools were withheld and the work may be incomplete "
                               f"({self.meter.metered_input_tokens:,.0f} metered of "
                               f"{self.meter.input_tokens:,} processed; ceiling "
                               f"{self.max_input_tokens:,})"))
                return self._result(turn.text, steps, history)

            if self.request_budget_isolated and self.sandbox_tools is None:
                return self._result("", steps, history, stopped="restricted",
                                    error="isolated request budget forbids all tool execution")

            if final_step:
                return self._result(
                    "", steps, history,
                    stopped="budget_input" if input_ceiling_final else "budget",
                    error=("model requested more tools on the final-response step "
                           "reserved by the cumulative input ceiling; no further "
                           "tools executed" if input_ceiling_final else
                           "model requested more tools on the reserved "
                           "final-response step; no further tools executed"))

            history.append(self._assistant_message(turn))

            round_results: list[tuple[str, dict, str]] = []
            for call_index, call in enumerate(turn.tool_calls):
                save_execution_checkpoint(turn.tool_calls[call_index:])
                if self.run_limits and (call.name != 'web_fetch' or
                        call.arguments.get('url') not in self.run_limits['urls']):
                    return self._result('', steps, history, stopped='restricted',
                                        error='Workspace research tool or URL limit: stopped without execution')
                discovering = discovery.enabled and call.name == "tool_search"
                verdict = self._gate(call, discovery=discovering)
                call_started = time.monotonic()
                if verdict is None:
                    if discovering:
                        output = discovery.search(call.arguments.get("query"))
                    elif self.sandbox_tools is not None:
                        output = self.sandbox_tools.execute(call.name, call.arguments)
                    else:
                        from onecli.execution_evidence import call_scope
                        with call_scope(call.id, call.name, self.execution_evidence):
                            output = toolkit.run(call.name,
                                         self.loop.clamp(call.name, call.arguments),
                                         self.workspace, spill=self.spill,
                                         jobs=self.jobs, asker=self.asker,
                                         spawner=self.spawner, todos=self.todos,
                                         path_guards=permissions.path_guards(self.guard),
                                         policy=self._external_policy(),
                                         trust_project=self.trust_project,
                                         call_tool=self._code_caller(step),
                                         room=self._result_room(history))
                else:
                    output = f"blocked: {verdict.reason}"
                    step.denials.append((call.name, verdict.reason))
                    self.on_event("tool_denied",
                                  {"call": call, "verdict": verdict})
                output = self._post_tool_use(call, output, verdict is None)
                # A refused call counts as a repeat too. A model asking three
                # times for a tool it is not allowed to have is the clearest
                # loop there is, and the one a blocked-call-only path misses.
                output = self._loop_note(call, output, step,
                                         time.monotonic() - call_started)
                # The last bound before the result joins the history, whatever
                # produced it: sandbox tools, tool discovery and PostToolUse
                # feedback never pass through toolkit.run's spill. A sandboxed
                # model cannot read a host spill path, so it gets truncation.
                output = spilllib.apply(output, call.name,
                                        None if self.sandbox_tools is not None
                                        else self.spill,
                                        self._result_room(history))
                # Reap between tool calls, not only between turns: a job's
                # end time is otherwise measured from whenever we next looked,
                # which reported a six-second sleep as thirteen seconds.
                if self.jobs is not None:
                    self.jobs.poll()
                cadence.completed(output, denied=verdict is not None)
                step.tool_results.append((call.name, output))
                round_results.append((call.name, call.arguments, output))
                self.on_event("tool_end", {"call": call, "output": output,
                                           "denied": verdict is not None})
                # Asked of the list rather than switched on the tool name: the
                # loop should not have to know which tool writes the plan.
                if self.todos is not None and self.todos.take_change():
                    self.on_event("todos", self._todo_event())
                history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": output,
                })
                save_execution_checkpoint(turn.tool_calls[call_index + 1:])
                if self.run_limits and (verdict is not None or
                        'verdict=BLOCK' in output or 'fetch refused' in output or
                        output.startswith(('blocked:', 'error:'))):
                    return self._result('', steps, history, stopped='budget',
                                        error='Research source access blocked; no retry or shell workaround')

            if loops.discovery_round(self.loop, round_results):
                checkpoint_due = True

            # Checked once the step is fully accounted for, so a run that
            # answers on the turn that crosses the ceiling still delivers the
            # answer. A ceiling stops the loop and says so; it never quietly
            # truncates the history to fit under itself.
            if self._over_budget():
                return self._result(
                    turn.text or "", steps, history, stopped="budget_usd",
                    error=(f"stopped at ${self.meter.usd:.4f}, over the "
                           f"${self.max_budget_usd:.4f} budget, after "
                           f"{index} steps"),
                )

        return self._result(
            steps[-1].turn.text if steps else "",
            steps, history, stopped="budget",
            error=f"stopped after {self.max_steps} steps without a final answer",
        )

    def _result(self, text: str, steps: list[Step], history: list[dict],
                stopped: str = "answered", error: str = "") -> Result:
        if self._save_execution_checkpoint is not None:
            self._save_execution_checkpoint(status=stopped)
        if not text.strip() and stopped in ("budget", "budget_input", "budget_usd"):
            text = next((step.turn.text for step in reversed(steps)
                         if step.turn.text.strip()), text)
        if self.run_limits and stopped != 'answered':
            # Preserve a resumable transcript if a limit stops a tool batch.
            history = list(history)
            for i in range(len(history) - 1, -1, -1):
                if history[i].get('role') != 'assistant':
                    continue
                answered = {m.get('tool_call_id') for m in history[i + 1:]}
                for call in history[i].get('tool_calls', []):
                    if call['id'] not in answered:
                        history.append({'role': 'tool', 'tool_call_id': call['id'],
                                        'name': call['function']['name'],
                                        'content': 'Not executed: workspace research limit stopped this run.'})
                break
        return Result(
            text=text, steps=steps, messages=history,
            stopped=stopped, error=error,
            compactions=[c.as_dict() for c in self.compactions],
            fallbacks=list(self.chain.history),
            resamples=sum(getattr(r.provider, "output_retries", 0)
                          for r in self.chain.rungs),
            retries=sum(getattr(r.provider, "output_retries", 0)
                        + getattr(r.provider, "transient_retries", 0)
                        for r in self.chain.rungs),
            retry_reasons=_merge_reasons(self.chain.rungs),
            cost=self.meter.as_dict(),
            context_tokens=self.tracker.tokens,
            context_window=self.tracker.window,
            todos=self.todos.as_dicts() if self.todos is not None else [],
            mask_policy=self.mask_policy.as_dict() if self.mask_policy else {},
            mask=({'name': self.mask_name, 'loop_policy': self.loop_policy,
                   'context_contract': self.context_contract,
                   'context_observations': self.context_observations,
                   'resolved': {'max_steps': self.max_steps, 'repeat_limit': self.loop.repeat_limit,
                                'tool_deadline': self.loop.deadline, 'empty_retries': self.empty_retries,
                                'progress_policy': self.progress_policy}}
                  if self.mask_name or self.loop_policy or self.context_contract else {}),
        )

    def _authorize_context(self, path):
        from onecli.context_contract import ContextError
        if self.request_budget_isolated or self.sandbox_tools is not None:
            raise ContextError('automatic context retrieval unavailable for sandbox adapters')
        if 'read' not in self.tool_names:
            raise ContextError('context retrieval requires host read capability')
        verdict = self._gate(ToolCall(id='context-read', name='read', arguments={'path': path}))
        if verdict is not None and not verdict.allowed:
            raise ContextError('context retrieval denied by host')

    def _result_room(self, history: list[dict]) -> int:
        """Characters one tool result may take: half of what is left of the
        context window, so the run can still act on it. This scales with the
        model's window; there is no fixed result size."""
        free = self.tracker.threshold - self.tracker.estimate(history)
        return max(spilllib.PREVIEW, free * ctx.CHARS_PER_TOKEN // 2)

    def _last_step(self, index: int) -> bool:
        return self.max_steps is not None and index == self.max_steps

    def _over_budget(self) -> bool:
        # A model with no price in the table has usd == 0.0, so an unpriced run
        # can never be stopped here. That is the honest behaviour: the CLI says
        # once that it is counting tokens only.
        return bool(self.max_budget_usd) and self.meter.usd >= self.max_budget_usd

    def _chat(self, history: list[dict], schemas,
              max_tokens: int | None = None, public_stream: bool = True,
              json_output: bool = False) -> Turn:
        """One model call, walking down the fallback chain as rungs fail.

        Only a transient provider failure moves the turn (429, 5xx, timeout,
        unreachable). A model that declines the task comes back as an ordinary
        Turn and never lands here, so no refusal can spend a rung.

        A demotion is one-way for the life of the chain: the next turn starts
        on the backend that just worked, not back at the endpoint that is still
        rate-limited. Falling off the bottom re-raises the last failure, so a
        run whose whole chain is down fails loudly rather than hanging.
        """
        # The kwarg is only sent when it is wanted, so a provider that
        # predates the streaming contract stays callable.
        extra = {"on_delta": self._delta} if self.stream and public_stream else {}
        ceiling = max_tokens or self.max_tokens
        if self.run_limits:
            ceiling = _tighter(ceiling, self.run_limits['max_output_tokens'])
        while True:
            try:
                if self.run_limits:
                    self.request_bytes_sent += run_limits.request_bytes(history, schemas)
                chat = (getattr(self.provider, "chat_json", self.provider.chat)
                        if json_output else self.provider.chat)
                turn = chat(
                    history, schemas,
                    temperature=self.temperature, max_tokens=ceiling,
                    **extra,
                )
                from onecli.providers.context_metrics import reported_cache
                observation = {
                    'provider': self.chain.current.name,
                    **getattr(self.provider, 'context_metrics', {'measurement': 'unavailable'}),
                    'reported_cache': reported_cache(turn.usage or {}),
                }
                self.context_observations.append({'kind': 'wire', **observation})
                self.on_event('wire_context', observation)
                # This lane answered: drop any strikes it had accumulated, so a
                # recovered endpoint is not held out by an old backoff window.
                lanehealth.mark_up(self.chain.current.name)
                return turn
            except ProviderError as exc:
                # Record the refusal BEFORE deciding what to do with this turn.
                # The window is what stops the NEXT run - and every other
                # process sharing the state file - from walking into the same
                # dead endpoint; whether THIS turn can fall somewhere else is a
                # separate question, answered on the two lines below.
                # Only an ENDPOINT failure opens a window: a 400 on a malformed
                # request is ours, and sidelining a healthy lane for it would
                # turn one bad payload into an outage.
                if is_transient(exc):
                    lanehealth.mark_down(self.chain.current.name, str(exc)[:200])
                if self.run_limits:
                    raise  # a bounded pilot must not silently spend on another lane
                if not is_transient(exc) or not self.chain.has_next():
                    raise
                note = self.chain.advance(str(exc)[:400])
                self.on_event("fallback", note)

    def _admit_wire_request(self, body):
        """Count actual HTTP attempts, including retries hidden inside the wire."""
        size = len(json.dumps(body).encode())
        limits = self.run_limits
        if (self.wire_requests >= limits['max_steps'] or
                size > limits['max_request_bytes'] or
                self.wire_bytes + size > limits['max_total_request_bytes'] or
                self.meter.input_tokens + size > limits['max_input_tokens'] or
                body.get('max_tokens', body.get('max_completion_tokens', 0)) > limits['max_output_tokens']):
            raise ProviderError('Workspace research wire budget reached before HTTP request')
        self.wire_requests += 1
        self.wire_bytes += size

    def _compact(self, history: list[dict], step: int,
                 keep_last: int | None = None) -> list[dict]:
        """Summarize the oldest messages away and report what it cost.

        A compaction that fails must still SHRINK. Returning the history
        untouched is right for a transport blip - losing the middle of a
        conversation to one bad socket is worse than sitting near the window -
        and fatal for the failure that actually happens on a full window: the
        summarization call is starved by the same ceiling that is overflowing,
        comes back empty, and the loop retries an identical starved call every
        step while the history never shrinks. The run then dies on
        finish_reason=length with every completed step discarded.

        So the two failures are handled differently, which is what
        `Compaction.failure` is for. "transport" earns one retry and then the
        deterministic fallback; "starved" goes straight to the fallback,
        because a second identical call has nothing new to be starved by.
        "empty_head" means there was nothing to compact and shrinking cannot
        help either, so the history is genuinely returned untouched.
        """
        keep_last = self.tracker.keep_last if keep_last is None else keep_last
        new_history, report = ctx.compact(
            history, self._summarize, keep_last=keep_last, step=step)
        if report.failure == "transport":
            new_history, report = ctx.compact(
                history, self._summarize, keep_last=keep_last,
                step=step)
        if report.failure in ("transport", "starved"):
            new_history, report = ctx.shrink(
                history, keep_last=keep_last, step=step,
                reason=report.error)
        if (report.ok and self._current_request is not None
                and self._current_request not in new_history):
            # Policies and continuation nudges also use the user role. They
            # cannot replace the actual request when the transcript shrinks.
            new_history.append(dict(self._current_request))
            report.after_tokens = ctx.history_tokens(new_history)
            report.messages_after = len(new_history)
        self.compactions.append(report)
        self.on_event("compact", report.as_dict())
        if self.hooks:
            for note in self.hooks.pre_compact().advisories:
                self.on_event("hook_advisory", {"event": "PreCompact", "note": note})
        if report.ok:
            # The plan is state, and compaction only summarizes transcript. The
            # tool result that carried the list has just become one clause of a
            # summary, so the list is restated whole - losing it is losing the
            # run's spine while keeping its footnotes.
            if self.todos is not None and self.todos.items:
                new_history = new_history + [
                    {"role": "user",
                     "content": self.todos.block("still open after compaction")}]
                self.on_event("todos", self._todo_event("compaction"))
            self.tracker.observe(ctx.fit_system(
                new_history, self.tracker.window, self.tracker.ratio)[0])
            # The next completed turn decides whether this compaction actually
            # helped. Without this the tracker cannot tell a compaction that
            # worked from one that left the run exactly as over-budget as it
            # started, and a run in the second case compacts on every step for
            # the rest of its life.
            self.tracker.note_compaction()
        return new_history

    def _summarize(self, head: list[dict]) -> str:
        """Ask the model to summarize `head`, as flat text.

        The transcript is flattened into one user message rather than replayed
        as history: a tool result sent on its own would be an orphan on the
        summarization call itself, which is the very failure compaction exists
        to avoid.

        Both ends of this call get their own headroom, because compaction runs
        exactly when there is none. The INPUT is capped so the summarize
        request cannot itself exceed the window it is trying to get back under
        - an uncapped render of the head is by definition about as large as the
        history that just overflowed. The OUTPUT floor matters for the same
        reason: at a small --max-tokens the summary is cut off mid-sentence or
        never starts, the wire returns no text, and compaction reports
        "summarizer returned nothing" for a call that was never given room to
        answer.
        """
        turn = self.provider.chat(
            [
                {"role": "system", "content": ctx.SUMMARY_INSTRUCTION},
                {"role": "user", "content": _render_for_summary(
                    head, max_chars=self._summary_input_chars())},
            ],
            None, temperature=self.temperature, max_tokens=self._summary_max_tokens(),
        )
        return turn.text or ""

    def _served_count(self, messages: list[dict]) -> int | None:
        """The backend's own token count of `messages`, when it can give one."""
        base_url = getattr(self.provider, "base_url", "") or ""
        return config.served_token_count(base_url, messages) if base_url else None

    def _summary_max_tokens(self) -> int | None:
        """Output ceiling for a summarization call: none unless the run set
        one, and then never below SUMMARY_MIN_TOKENS."""
        if self.max_tokens is None:
            return None
        return max(self.max_tokens, SUMMARY_MIN_TOKENS)

    def _boosted_max_tokens(self) -> int:
        """A raised ceiling for a turn cut off at the output limit. With no
        ceiling configured the provider's own default cut it, so the retry
        sends nothing and the provider decides again."""
        return self.max_tokens * 4 if self.max_tokens else 0

    def _summary_input_chars(self) -> int:
        """Character cap on the head handed to the summarizer.

        Derived from the window rather than fixed: what is left after the
        summary's own output room and the instruction, halved, because
        CHARS_PER_TOKEN is a floor and a history estimated at N can bill 1.4N.
        """
        room = (self.tracker.window - (self._summary_max_tokens() or SUMMARY_MIN_TOKENS)
                - SUMMARY_OVERHEAD_TOKENS)
        return max(SUMMARY_MIN_INPUT_CHARS,
                   int(room * ctx.CHARS_PER_TOKEN * SUMMARY_INPUT_SAFETY
                       / self.tracker.ratio))

    def _delta(self, kind: str, chunk: str) -> None:
        """Relay a wire chunk as an event, so an embedding host sees progress.

        The loop itself never acts on a delta - the Turn the wire returns is
        still the only thing it reads - which is what keeps a streamed run and
        a non-streamed run the same run.
        """
        self.on_event("text_delta", {"kind": kind, "chunk": chunk})

    def _gate(self, call, discovery=False) -> Verdict | None:
        """Decide whether `call` may run. None means run it.

        Two independent deciders, both able to refuse. `on_event` goes first so
        an embedding host sees the call before any subprocess is spawned; it
        opts in to deciding by returning a Verdict or a {"deny": reason} dict,
        and anything else (including the usual None) means it is only watching.
        """
        if not isinstance(call.arguments, dict):
            return Verdict.deny("tool arguments must be a JSON object", source="arguments")
        surface_tool = toolkit.REGISTRY.get(call.name)
        if (surface_tool is not None and surface_tool.explicit
                and call.name not in self.tool_names):
            return Verdict.deny("this tool requires an explicitly selected tool surface", source="tool-surface")
        if self.request_budget_isolated:
            if self.sandbox_tools is None or not self.sandbox_tools.permits(call.name, call.arguments):
                return Verdict.deny("tool outside isolated sandbox grant", source="request-budget")
        elif not discovery and call.name not in self.tool_names:
            return Verdict.deny("tool outside effective tool surface (mask and host capability intersection)", source="tool-surface")
        if self._meditating:
            from onecli.meditate import REVIEW_TOOLS
            if call.name not in REVIEW_TOOLS:
                return Verdict.deny("meditate review permits only read-only evidence tools",
                                    source="meditate")
        mutating = False if discovery else toolkit.is_mutating(call.name)
        observed = self.on_event("tool_start", {"call": call, "mutating": mutating})
        verdict = _as_verdict(observed)
        if verdict is not None and not verdict.allowed:
            return verdict

        if self.guard:
            verdict = self.guard.check(call.name, call.arguments, mutating)
            if not verdict.allowed:
                return verdict

        if self.hooks:
            outcome = self.hooks.pre_tool_use(call.name, call.arguments, mutating)
            self._report(outcome)
            if outcome.blocked:
                return Verdict.deny(outcome.reason, source="PreToolUse")
        return None

    def _post_tool_use(self, call, output: str, ran: bool) -> str:
        """Let the PostToolUse hooks add to what the model sees.

        A blocked call never ran, so there is no response to hand a PostToolUse
        hook - firing it there would report a tool result that does not exist.
        """
        if ran and self._mask_run:
            self._mask_run.observe(call.name, output)
        if not (self.hooks and ran):
            return output
        outcome = self.hooks.post_tool_use(
            call.name, call.arguments, {"output": output},
            toolkit.is_mutating(call.name))
        self._report(outcome)
        if outcome.feedback_text:
            return f"{output}\n\n[PostToolUse] {outcome.feedback_text}"
        return output

    def _external_policy(self) -> list:
        """The deciders that see a tool's name and arguments but cannot see
        inside a program: every --guard program, and the hooks. Code mode
        confines its program when any are present (codemode.confinement_reason).
        """
        def leaves(guard) -> list:
            children = getattr(guard, "guards", None)
            if children is None:
                return [guard]
            return [leaf for child in children for leaf in leaves(child)]
        policy = [g for g in leaves(self.guard) if getattr(g, "commands", None)] \
            if self.guard is not None else []
        if self.hooks:
            policy.append(self.hooks)
        return policy

    def _code_caller(self, step: Step):
        """The binding a code-mode program calls, gated like any tool call.

        This is the whole security argument for code mode: the program does not
        touch the filesystem, it asks the agent to, and the ask goes through the
        same guards, hooks and on_event decider as a call the model made
        directly. Skip this and a --read-only run can write through a Python
        program, and every external guard is a decoration.
        """
        def call_tool(name: str, arguments: dict) -> tuple[str, bool]:
            if self.request_budget_isolated:
                return 'blocked: isolated request budget forbids code-mode dispatch', True
            if name == "code":
                return ("blocked: a code-mode program may not start another "
                        "one - do the work in this program"), True
            if name in toolkit.TASK_TOOL_NAMES:
                # A program that spawns agents is a scheduler, and only what it
                # prints comes back - the parent would lose every child's step
                # count, cost and stop reason. Fan out from the loop, where each
                # child is one visible tool call, not from inside one.
                return ("blocked: spawn subagents from the loop, not from a "
                        "code-mode program"), True
            call = ToolCall(id=f"code-{len(step.tool_results)}", name=name,
                            arguments=arguments)
            verdict = self._gate(call)
            if verdict is not None:
                step.denials.append((name, verdict.reason))
                self.on_event("tool_denied", {"call": call, "verdict": verdict})
                return f"blocked: {verdict.reason}", True
            output = toolkit.run(name, self.loop.clamp(name, arguments),
                                 self.workspace, spill=self.spill,
                                 path_guards=permissions.path_guards(self.guard),
                                 trust_project=self.trust_project,
                                 jobs=self.jobs, asker=self.asker)
            output = self._post_tool_use(call, output, True)
            self.on_event("tool_end", {"call": call, "output": output,
                                           "denied": verdict is not None})
            return output, False
        return call_tool

    def _job_notices(self) -> list[str]:
        if self.jobs is None:
            return []
        notes = self.jobs.notices()
        for note in notes:
            self.on_event("job_notice", {"note": note})
        return notes

    def _todo_notices(self) -> list[str]:
        if self.todos is None:
            return []
        notes = self.todos.notices()
        for note in notes:
            self.on_event("todo_notice", {"note": note,
                                          "summary": self.todos.summary()})
        return notes

    def _task_notices(self) -> list[str]:
        if getattr(self, "tasks", None) is None:
            return []
        notes = self.tasks.notices()
        for note in notes:
            self.on_event("task_notice", {"note": note})
        return notes

    def _todo_event(self, reason: str = "changed") -> dict:
        """`reason` is what the host draws on: a list that CHANGED is worth the
        whole block, and one merely restated into context is worth a line."""
        return {"items": self.todos.as_dicts(),
                "rendered": self.todos.render(),
                "summary": self.todos.summary(),
                "reason": reason}

    def _loop_note(self, call, output: str, step: Step, seconds: float) -> str:
        """Tell the model when it is repeating itself, or overrunning.

        The note is appended to the tool result rather than raised or logged,
        because the transcript is the only channel the model actually reads.
        """
        notes = self.loop.observe(call.name, call.arguments, output, seconds)
        for note in notes:
            step.advisories.append((call.name, note))
            self.on_event("loop_advisory", {"call": call, "note": note})
        return self.loop.annotate(output, notes)

    def _done_check_failure(self) -> str | None:
        """The done-check's output tail when it fails, None when it passes."""
        if not self.done_check:
            return None
        try:
            proc = run_shell(self.done_check, cwd=self.workspace or None,
                             timeout=DONE_CHECK_TIMEOUT)
        except subprocess.TimeoutExpired:
            return f"timed out after {DONE_CHECK_TIMEOUT}s"
        if proc.returncode == 0:
            return None
        tail = (proc.stdout + proc.stderr).strip()[-DONE_CHECK_OUTPUT:]
        return f"exit {proc.returncode}" + (f"\n{tail}" if tail else "")

    def _stop_refusal(self) -> str | None:
        """The Stop hooks' reason for keeping the loop alive, or None."""
        if self._mask_run:
            reason = self._mask_run.refusal(self.workspace)
            if reason:
                self.on_event("mask_completion_denied", {"reason": reason})
                return reason
        if not self.hooks:
            return None
        outcome = self.hooks.stop()
        self._report(outcome)
        return outcome.reason if outcome.blocked else None

    def _report(self, outcome) -> None:
        for note in outcome.advisories:
            self.on_event("hook_advisory", {"event": outcome.event, "note": note})

    @staticmethod
    def _assistant_message(turn: Turn) -> dict:
        return {
            "role": "assistant",
            "content": turn.text or "",
            "tool_calls": [Agent._tool_call_message(c) for c in turn.tool_calls],
            **turn.echo,
        }

    @staticmethod
    def _tool_call_message(call) -> dict:
        """One tool call as the provider must see it echoed back.

        `call.extra` is opaque: the loop never reads it, never branches on the
        provider that produced it, and adds no key at all when it is empty, so
        every wire that sends nothing extra gets the identical request it got
        before.
        """
        message = {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }
        message.update(call.extra)
        return message


def _as_verdict(value: object) -> Verdict | None:
    """Read a denial out of whatever `on_event` returned, or None."""
    if isinstance(value, Verdict):
        return value
    if isinstance(value, dict) and value.get("deny"):
        return Verdict.deny(str(value["deny"]), source="on_event")
    return None


# How much of a tool result the salvage path quotes, per step and in total.
SALVAGE_CHARS_PER_STEP = 600
SALVAGE_MAX_CHARS = 4000


def _salvage(steps: list) -> str:
    """The best answer recoverable from a run that never produced a final one.

    Preference order is the model's own words first - the last non-empty turn
    text is an actual answer, just not the last one - and only then the tool
    output, which is evidence rather than an answer and is labelled as such.
    """
    for step in reversed(steps):
        text = (getattr(step.turn, "text", "") or "").strip()
        if text:
            return text
    lines: list[str] = []
    used = 0
    for step in reversed(steps):
        for name, output in reversed(step.tool_results or []):
            body = (output or "").strip()
            if not body:
                continue
            if len(body) > SALVAGE_CHARS_PER_STEP:
                body = body[:SALVAGE_CHARS_PER_STEP] + " ...[truncated]"
            line = f"- {name}: {body}"
            if used + len(line) > SALVAGE_MAX_CHARS:
                used = -1
                break
            lines.append(line)
            used += len(line)
        if used < 0:
            break
    if not lines:
        return ""
    return ("The run ended without a final reply. What its tools produced, "
            "most recent first:\n" + "\n".join(lines))


def _render_for_summary(messages: list[dict], max_chars: int = 0) -> str:
    """History as plain text a model can read without any tool-call structure.

    `max_chars` caps the result from the END backwards: the newest messages are
    the ones a continuing agent needs, and an uncapped render of an overflowing
    history is exactly as unsendable as the history itself.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        if content.strip():
            lines.append(f"[{role}] {content.strip()}")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            lines.append(f"[{role} calls {fn.get('name', '?')}] {fn.get('arguments', '')}")
    if max_chars > 0:
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            if used + len(line) + 1 > max_chars:
                kept.append("[...earlier messages omitted to fit the summarizer]")
                break
            kept.append(line)
            used += len(line) + 1
        lines = list(reversed(kept))
    return "\n".join(lines)

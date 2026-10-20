"""Loop guards - naming a run that has stopped making progress, while it runs.

Three failures live here, and they are the same failure seen from two sides: a
run that is spending without moving.

**Repetition.** A model that gets an answer it does not like often asks the same
question again, byte for byte. The tell is there in the transcript long before
the run ends - the same call, three times, with the same result. The reminder is
advisory on purpose: the model is not wrong to re-read a file it just wrote, and
a guard that blocks a legitimate second look costs more than the loop it
prevents. What it cannot be allowed to do is repeat *without noticing*, so the
notice is delivered where the model must read it - appended to the tool result
it just received.

**Duration.** A per-call deadline is policy, not a suggestion: a single tool
call must not be able to outlive the run it belongs to. Where a tool declares a
`timeout` parameter the ceiling is *enforced* by lowering that argument before
the call runs; where it does not - an MCP tool with its own clock, say - it can
only be *reported*, so an overrun comes back as an advisory rather than a lie
about what was enforced.

**Discovery without a decision.** Neither tell above catches a model that
searches with *different* arguments every round and never chooses: every call
distinct, no artifact, no question, while the transcript grows. After
`discovery_limit` consecutive rounds with no progress tool (a write, an edit, a
question to the user, a spawned task or job), the loop sends one control step
with tools withheld: commit to a named candidate with a reason, or end the turn
with one clarifying question. Shell and code calls count as discovery here,
because the tool cannot tell a `grep` from a build; the cost of a false positive
is one short commit line, after which the count starts over.

**The same refusal, whatever the arguments.** The repeat count above is keyed
on the arguments, so a model that rephrases a refused call never trips it.
`refusal_note` keys on the result instead: the second time one tool returns the
same `error:` or `blocked:` line, naming no path, for different arguments, the
note says the refusal does not depend on them and repeats the refusal's own
advice. Advisory, like the rest: the call already ran and was already refused.

**Decided, so act.** A checkpoint reply that carries a commit and an acting
call in the same turn must have its call executed, not ignored. After a commit,
further searching spends the window instead of resetting it; other shell work
keeps counting as the action, because a long build is mostly shell.

Reminders and deadlines live in one module for the same reason spill does:
there is no plugin seam here to justify splitting them.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field

from onecli import offload_policy

# Identical calls tolerated before the model is told. Two is a re-read; three is
# a pattern.
REPEAT_LIMIT = 3
# Per-call ceiling in seconds. 0 disables both the clamp and the report.
DEADLINE = offload_policy.TOOL_DEADLINE
# Consecutive tool rounds without a progress tool before the checkpoint. 0
# disables it.
DISCOVERY_LIMIT = 6
# Tools whose call is itself a decision or an artifact. Everything else counts
# as discovery for the checkpoint.
PROGRESS_TOOLS = frozenset({"write", "edit", "ask_user", "task", "job_start"})
# Tools that neither count nor reset: reading a named file and polling a job
# are directed steps, not a search for what the request means.
NEUTRAL_TOOLS = frozenset({"read", "todo_write", "job_list", "job_output"})
# A resolving lookup at or above this confidence resolves what the user means.
RESOLVED_CONFIDENCE = 0.6
# Discovery rounds allowed once the target is resolved, and after a commit.
RESOLVED_LIMIT = 1
POST_COMMIT_LIMIT = 3
# Checking rounds allowed once a call has produced its files.
POST_OUTPUT_LIMIT = 2
# Tools that act on a decision. A surface without any of them (plan mode) can
# only read further, so a resolved checkpoint there takes the answer.
ACTING_TOOLS = frozenset({"write", "edit", "bash", "job_start", "task"})
# A result line that refuses the call rather than reporting on the work.
_REFUSAL = re.compile(r"^(?:error|blocked): ")
_CONFIDENCE = re.compile(r"^Confidence ([01](?:\.\d+)?) ", re.M)
_HEADING = re.compile(r"^## (.+)$", re.M)
# Shell commands that only look: a round of these after a decision is still
# reading. Anything not on this list (a build, a test) is acting.
_READ_COMMANDS = frozenset({"ls", "cat", "head", "tail", "sed", "grep", "rg", "find",
                            "wc", "stat", "file", "tree", "pwd", "less",
                            "du", "which", "type", "jq", "realpath", "readlink"})
# Calls that re-learn a tool whose syntax the instructions already give: a
# shell line matching RELEARN_COMMANDS, a read of a path matching
# RELEARN_SOURCES. An integration adds its own tools' patterns.
RELEARN_COMMANDS: list[re.Pattern] = [re.compile(r"--help\b")]
RELEARN_SOURCES: list[re.Pattern] = []
# A command that only describes or measures files already made.
INSPECT_COMMANDS: list[re.Pattern] = [re.compile(r"/tmp/")]
# Commands that look like a relearn pattern but make deliverables.
MAKE_COMMANDS: list[re.Pattern] = []
# Hooks naming the owner of a hand-rolled shell route: (name, arguments) -> note.
ROUTE_HINTS: list = []
_MEDIA_PATH = re.compile(r"(/[^\s'\"]+\.(?:png|jpe?g|webp|gif|mp4|webm|wav|mp3|flac|pdf))\b",
                         re.I)
# Words that start a sentence or an instruction, never a subject.
_NOT_NAMES = frozenset({"I", "A", "An", "The", "Please", "Draw", "Make", "Put", "Render",
                        "Create", "Generate", "Give", "Show", "Send", "Save", "Use",
                        "No", "Do", "Don't", "Then", "And", "Also", "Just", "One",
                        "Two", "Three", "Can", "Could", "Would", "Write", "Find"})
# A request to depict something: only then are its names subjects to look up.
_MEDIA_INTENT = re.compile(r"\b(?:draw|paint|picture|pic|image|pixel|render|illustrat\w*|"
                           r"art|poster|card|video|clip|song|music|portrait|scene)s?\b", re.I)
_NAME = re.compile(r"\b[A-Z][\w']*(?:\s+(?:of|the|and|de|van|in|on)?\s*[A-Z][\w']*)*")
# The checkpoint reply that resumes tool work. Anything else ends the turn.
COMMIT_PREFIX = "COMMIT:"
CHECKPOINT = (
    "[loop-guard checkpoint] You have spent {rounds} tool rounds searching "
    "without writing anything, asking anything or settling on a target. Tools "
    "are withheld for this one reply. Choose exactly one:\n"
    "1. If the evidence already identifies what the user means, reply with one "
    "line starting `" + COMMIT_PREFIX + "` naming the chosen candidate (path, "
    "project or plan) and the reason. Tools return on the next step; do the "
    "work, do not search further for alternatives.\n"
    "2. Otherwise end the turn with ONE concise clarifying question that lists "
    "the top candidates you found (at most four), so the user can pick. Do not "
    "start the reply with `" + COMMIT_PREFIX + "`.\n"
    "If a configured lookup tool could settle which candidate the user means, "
    "that is the one check worth naming as your committed next step.")
RESOLVED_CHECKPOINT = (
    "[loop-guard checkpoint] The target is already resolved: {resolved}. "
    "Tools are withheld for this one reply. Do not read further to describe or "
    "double-check it. Choose exactly one:\n"
    "1. Reply with one line starting `" + COMMIT_PREFIX + "` naming the action "
    "you will take now (the recipe, command or edit). Tools return for that "
    "action only.\n"
    "2. Otherwise end the turn with your answer, or ONE confirmation question "
    "naming the target and route, so the user can say go.")
FINAL_CHECKPOINT = (
    "[loop-guard checkpoint] {reason}, and {rounds} tool rounds have passed "
    "without writing anything or asking anything. Tools stay withheld. End the "
    "turn now with your answer: the result, the plan, or ONE confirmation "
    "question naming the target and route. Do not search further.")


def fingerprint(name: str, arguments: dict) -> str:
    """One call's identity: the tool plus its arguments, order-independent."""
    try:
        body = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        body = repr(sorted(arguments.items())) if arguments else ""
    return f"{name}:{hashlib.sha1(body.encode()).hexdigest()[:16]}"


@dataclass
class LoopGuard:
    """Per-run state. One instance per agent; never shared across runs, because
    a repeat count that survives a run would fire on the first call of the next
    one."""

    repeat_limit: int = REPEAT_LIMIT
    deadline: float = DEADLINE
    counts: dict[str, int] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    unchanged_counts: dict[str, int] = field(default_factory=dict)
    advisories: list[tuple[str, str]] = field(default_factory=list)

    resource_failures: dict[str, int] = field(default_factory=dict)
    discovery_limit: int = DISCOVERY_LIMIT
    discovery_rounds: int = 0
    # What a tool result resolved the target to, and the commit line, if any.
    # The decisions on them are module functions so a reload reaches a live run.
    resolved: str = ""
    commitment: str = ""
    # The files an acting call reported, once it has; checking them is then capped.
    outputs: str = ""

    # -- before the call ---------------------------------------------------

    def clamp(self, name: str, arguments: dict) -> dict:
        """Return the arguments to actually run with.

        A tool that declares an integer `timeout` gets ours when it asked for
        more than policy allows, or asked for nothing. The argument is only ever
        lowered: a model that asks for 5 seconds gets 5, because a shorter
        deadline than policy is the model's business.
        """
        if self.deadline <= 0 or not _accepts_timeout(name):
            return arguments
        ceiling = max(1, int(self.deadline))
        raw_timeout = arguments.get("timeout")
        asked: float | None
        try:
            asked = float(raw_timeout) if raw_timeout is not None else None
        except (TypeError, ValueError):
            asked = None
        if asked is not None and 0 < asked <= ceiling:
            from onecli import tools as toolkit
            spec = toolkit.REGISTRY[name].parameters["properties"]["timeout"]
            timeout = max(1, int(asked)) if spec["type"] == "integer" else asked
            return {**arguments, "timeout": timeout}
        return {**arguments, "timeout": ceiling}

    # -- after it ----------------------------------------------------------

    def observe(self, name: str, arguments: dict, output: str,
                seconds: float = 0.0) -> list[str]:
        """Record the call and return the advisories it earned, if any."""
        notes: list[str] = []
        # Structured diagnostics only: free-form error text is not a receipt.
        if name == "bash" and output.startswith("exit ") and not output.startswith("exit 0\n"):
            seen = set()
            for line in output.splitlines():
                if not line.startswith("LOOK_DIAGNOSTIC "):
                    continue
                try:
                    record = json.loads(line[len("LOOK_DIAGNOSTIC "):])
                    if not isinstance(record, dict) or record.get("kind") not in {
                        "model_load_failed", "model_missing", "credentials_missing", "credentials_rejected"
                    }:
                        continue
                    values = [record.get(k) for k in ("endpoint", "model", "kind")]
                    if not all(isinstance(v, str) and 0 < len(v) < 512 for v in values):
                        continue
                    resource = json.dumps(values)
                except (ValueError, TypeError):
                    continue
                if resource in seen:
                    continue
                seen.add(resource)
                count = self.resource_failures.get(resource, 0) + 1
                self.resource_failures[resource] = count
                if count >= 2:
                    notes.append("[loop-guard] same resource failure across calls: "
                                 + resource + ". Changing the command did not repair this resource. "
                                 "Inspect or repair the cause before retrying; use another route only "
                                 "when already authorized. Advisory only; no call was blocked.")
        for hint in ROUTE_HINTS:
            routed = hint(name, arguments)
            if routed:
                notes.append(routed)
                break
        key = fingerprint(name, arguments)
        refused = refusal_note(self, name, key, output)
        if refused:
            notes.append(refused)
        self.counts[key] = self.counts.get(key, 0) + 1
        # Recorded on every call, not only on a reported one: the comparison
        # that makes "nothing changed" true needs the previous result, which
        # arrived before the count reached the limit.
        digest = hashlib.sha1((output or "").encode()).hexdigest()[:16]
        same = self.digests.get(key) == digest
        self.digests[key] = digest
        count = self.unchanged_counts.get(key, 0) + 1 if same else 1
        self.unchanged_counts[key] = count

        if self.repeat_limit > 0 and count >= self.repeat_limit:
            notes.append(_repeat_note(name, count, same, self.repeat_limit))

        # Only for a tool we could NOT clamp. A clamped tool that hits its
        # ceiling has already returned its own honest "timed out after 3s", and
        # adding "nothing cut it short" beside it is a lie the model then
        # repeats, for instance as a claim that the call was "denied at
        # submission", which never happened. The margin absorbs the difference
        # between the deadline we set and the wall clock we measured.
        if (self.deadline > 0 and not _accepts_timeout(name)
                and seconds > self.deadline * 1.1 + 0.5):
            notes.append(
                f"[loop-guard] {name} took {seconds:.0f}s, over the "
                f"{self.deadline:.0f}s per-call deadline. This tool keeps its "
                f"own clock, so nothing cut it short - split the work into "
                f"smaller calls, or ask for less in one call.")

        for note in notes:
            self.advisories.append((name, note))
        return notes

    def annotate(self, output: str, notes: list[str]) -> str:
        """Put the advisories where the model cannot miss them."""
        if not notes:
            return output
        return "\n\n".join([output, *notes])


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def _repeat_note(name: str, count: int, identical: bool,
                 repeat_limit: int = REPEAT_LIMIT) -> str:
    same = (" The result is byte-identical to the last one."
            if identical else "")
    tail = ("You are repeating yourself. Stop repeating these arguments; "
            "use the result above and take the next authorized action.")
    if count >= 2 * repeat_limit:
        tail = ("This is a loop. Stop this repeated call, not the task. "
                "Take a different concrete step toward the authorized result; "
                "report a blocker only if the actual tool result establishes one.")
    return (f"[loop-guard] {_ordinal(count)} identical call to "
            f"{name}.{same} Advisory only: this repetition warning did not "
            "block the call or revoke authorization. The tool result above "
            "determines success or refusal. This warning does not prevent "
            f"response text or host markers. {tail}")


def refusal_note(guard: LoopGuard, name: str, key: str, output: str) -> str:
    """A note when `name` returns the same refusal for different arguments.

    Keyed on the refusal's exact first line. A refusal that names a path
    (`no such file: /w/a`, `old_string not found in /w/a.py`) is about that
    target, so changing the arguments can fix it: it never counts. State
    lives on the guard's instance dict so a guard built before a reload keeps
    working."""
    line = (output or "").lstrip().split("\n", 1)[0]
    if not _REFUSAL.match(line) or "/" in line:
        return ""
    seen = vars(guard).setdefault("refusals", {})
    callers = seen.setdefault(f"{name}:{line}", set())
    callers.add(key)
    if len(callers) < 2:
        return ""
    return (f"[loop-guard] {name} has now refused {len(callers)} different calls "
            "with the same reason. The refusal does not depend on the arguments, "
            "so rewording the call will not get past it. Follow the refusal's own "
            f"advice instead: {line[:300]}")


def resolution(name: str, output: str) -> str:
    """The target a tool result resolves, or "" when it resolves nothing.

    Only a tool registered with ``resolves=True`` can resolve: its result
    carries a `Confidence <0..1>` line and names the target in a `## ` heading.
    """
    if not isinstance(output, str) or not _resolves(name):
        return ""
    match = _CONFIDENCE.search(output)
    if not match or float(match.group(1)) < RESOLVED_CONFIDENCE:
        return ""
    heading = _HEADING.search(output)
    where = heading.group(1) if heading else f"the first {name} match"
    return f"`{name}` named {where} at confidence {match.group(1)}"


def _resolves(name: str) -> bool:
    from onecli import tools  # lazy: the registry imports this module's callers
    return bool(getattr(tools.REGISTRY.get(name), "resolves", False))


def _segments(command: str) -> list[list[str]]:
    """The simple commands of a shell line, as word lists. Unparseable -> []."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    out: list[list[str]] = [[]]
    for token in tokens:
        if token and all(c in ";&|()" for c in token):
            out.append([])
        else:
            out[-1].append(token)
    return [seg for seg in out if seg]


def relearns(name: str, arguments: dict) -> bool:
    """Is this call re-learning a tool - `--help`, or a pattern an integration
    registered - whose syntax the instructions already give?"""
    if name == "read":
        path = str(arguments.get("path") or "")
        return any(p.search(path) for p in RELEARN_SOURCES)
    if name != "bash":
        return False
    text = str(arguments.get("command") or "")
    return (any(p.search(text) for p in RELEARN_COMMANDS)
            and not any(p.search(text) for p in MAKE_COMMANDS))


def reads_only(name: str, arguments: dict) -> bool:
    """Is this call looking rather than acting? A shell line made wholly of
    read commands is, and so is re-learning a tool. Everything else
    may be the action."""
    if relearns(name, arguments):
        return True
    if name != "bash":
        return name not in ACTING_TOOLS
    text = str(arguments.get("command") or "")
    segments = _segments(text)
    if not segments or ">" in text or "<<" in text:
        return False
    for seg in segments:
        words = [w for w in seg if "=" not in w or w.startswith("-")]
        if not words or words[0] == "cd":
            continue
        if words[0] == "sed" and "-i" in words:
            return False
        if words[0].rsplit("/", 1)[-1] not in _READ_COMMANDS:
            return False
    return True


def produced(name: str, arguments: dict, output: str, known: str = "") -> str:
    """The media files an acting call reports making, "" when it made none.

    An acting shell line that exits 0, or a finished `[ok]` job, whose output
    names media files outside /tmp that are not ``known`` already. A file
    under /tmp is a test, not a deliverable; a `--check` run, or a command
    that inspects or names a known file (see `checks`), is not making.
    """
    if not isinstance(output, str) or output.startswith(("error:", "blocked:")):
        return ""
    if name == "bash":
        if reads_only(name, arguments):
            return ""
        command = str(arguments.get("command") or "")
        if output.startswith("exit ") and not output.startswith("exit 0"):
            return ""
    elif name == "job_output":
        if "[ok]" not in output.split("\n", 1)[0]:
            return ""
        command = output[:2000]
    else:
        return ""
    if "--check" in command:
        return ""
    if name == "bash" and checks(name, arguments, known):
        return ""
    files = [f for f in _MEDIA_PATH.findall(output)
             if not f.startswith("/tmp/") and f not in known.split(", ")]
    return ", ".join(dict.fromkeys(files))[:400]


def checks(name: str, arguments: dict, outputs: str) -> bool:
    """Is this call checking files already made? Reading, describing or
    measuring them, or a test run in /tmp, is; a build or test that never
    names them is not."""
    if reads_only(name, arguments):
        return True
    text = str(arguments.get("command") or arguments.get("source") or
               arguments.get("path") or "")
    if any(p.search(text) for p in INSPECT_COMMANDS):
        return True
    names = {f.rsplit("/", 1)[-1] for f in outputs.split(", ") if f}
    return any(n and n in text for n in names)


def named_subjects(request: str) -> list[str]:
    """Proper names the request uses as subjects: phrases of at least two
    title-case words ("Coast Road"), no acronyms, outside paths and code."""
    text = re.sub(r"`[^`]*`|\S*/\S*|https?://\S+", " ", request or "")
    names: list[str] = []
    for match in _NAME.finditer(text):
        phrase = match.group(0).strip()
        words = phrase.split()
        while words and words[0] in _NOT_NAMES:
            words = words[1:]
        if not words:
            continue
        titled = [w for w in words if w[0].isupper() and not w.isupper()]
        if len(titled) < 2 or any(w.isupper() and len(w) > 1 for w in words):
            continue
        phrase = " ".join(words)
        if phrase not in names:
            names.append(phrase)
    return names[:6]


def lookup_note(request: str, tools: set[str]) -> str:
    """One line asking for a lookup of the request's named subjects, or ""."""
    if not _MEDIA_INTENT.search(request or ""):
        return ""
    names = named_subjects(request)
    resolvers = sorted(n for n in tools if _resolves(n))
    if not names or not (resolvers or "grep" in tools):
        return ""
    how = (f"`{resolvers[0]}` with these names, then a narrow grep" if resolvers
           else "a narrow grep")
    quoted = ", ".join(f'"{n}"' for n in names)
    return (f"[one] The request names {quoted}. Look them up first ({how}) and "
            "draw or write from what the record says, not from the titles alone.")


def discovery_round(guard: LoopGuard, results: list[tuple[str, dict, str]]) -> bool:
    """Count one executed tool round of (name, arguments, output); True when
    the checkpoint is due."""
    names = [name for name, _, _ in results]
    if any(name in PROGRESS_TOOLS for name in names):
        # Acting: the run is past choosing, so a long build keeps the full count.
        guard.discovery_rounds = 0
        guard.resolved = guard.commitment = guard.outputs = ""
        return False
    made = ", ".join(f for f in (produced(n, a, o, guard.outputs) for n, a, o in results) if f)
    if made:
        # A call that delivers files is the work itself; checking starts after it.
        guard.outputs = made if not guard.outputs else guard.outputs + ", " + made
        guard.discovery_rounds = 0
        return False
    found = next((r for r in (resolution(n, o) for n, _, o in results) if r), "")
    if found and not guard.resolved:
        # The resolving round is the answer to the search, not more of it.
        guard.resolved = found
        guard.discovery_rounds = 0
        return False
    learning = [relearns(n, a) for n, a, _ in results]
    if names and not any(learning) and all(name in NEUTRAL_TOOLS for name in names):
        return False
    acting = any(name in ACTING_TOOLS and not looks
                 for (name, _, _), looks in zip(results, learning))
    if guard.outputs and not any(checks(n, a, guard.outputs) for n, a, _ in results):
        # Building on with the files (a site build, tests) is work, not checking.
        guard.discovery_rounds = 0
        return False
    if guard.commitment and acting and not guard.outputs:
        # After a commit, an acting round is the committed action (a build,
        # a send), not a search; reading spends the window.
        guard.discovery_rounds = 0
        return False
    guard.discovery_rounds += 1
    if guard.discovery_limit <= 0:
        return False
    limit = (POST_OUTPUT_LIMIT if guard.outputs else
             POST_COMMIT_LIMIT if guard.commitment else
             RESOLVED_LIMIT if guard.resolved else guard.discovery_limit)
    return limit <= guard.discovery_rounds


def checkpoint_commits(text: str, calls: list[tuple[str, dict]]) -> bool:
    """Does a checkpoint reply that also calls tools decide? A `COMMIT:` line
    does, and so does a call that acts rather than reads."""
    if (text or "").lstrip().startswith(COMMIT_PREFIX):
        return True
    return any(name in ACTING_TOOLS and not reads_only(name, args)
               for name, args in calls)


def checkpoint_prompt(guard: LoopGuard, can_act: bool) -> tuple[str, bool]:
    """The control text for a due checkpoint, and whether it is final (a
    commit is not offered and any answer ends the turn)."""
    rounds = guard.discovery_rounds
    if guard.outputs:
        return FINAL_CHECKPOINT.format(
            reason=f"The files are already produced ({guard.outputs[:300]})",
            rounds=rounds), True
    if guard.commitment:
        return FINAL_CHECKPOINT.format(
            reason=f"You committed ({guard.commitment[:200]})", rounds=rounds), True
    if guard.resolved and not can_act:
        return FINAL_CHECKPOINT.format(
            reason=f"The target is resolved ({guard.resolved}) and this mode "
                   "has no tool that acts on it", rounds=rounds), True
    if guard.resolved:
        return RESOLVED_CHECKPOINT.format(resolved=guard.resolved), False
    return CHECKPOINT.format(rounds=rounds), False


def committed(guard: LoopGuard, line: str) -> None:
    """The model named its target and action: one short window to act in."""
    guard.discovery_rounds = 0
    guard.commitment = " ".join(line.split())


def _accepts_timeout(name: str) -> bool:
    """Does this tool declare an integer `timeout` we may lower?

    Read off the registered schema rather than a hardcoded list, so a tool
    added later - an MCP tool included - is covered the day it declares one.
    """
    from onecli import tools as toolkit  # local: tools imports nothing from here

    tool = toolkit.REGISTRY.get(name)
    if tool is None:
        return False
    spec = (tool.parameters or {}).get("properties", {}).get("timeout")
    return isinstance(spec, dict) and spec.get("type") in ("integer", "number")


__all__ = ["LoopGuard", "fingerprint", "REPEAT_LIMIT", "DEADLINE", "DISCOVERY_LIMIT",
           "PROGRESS_TOOLS", "NEUTRAL_TOOLS", "CHECKPOINT", "COMMIT_PREFIX",
           "RESOLVED_CONFIDENCE", "RESOLVED_LIMIT", "POST_COMMIT_LIMIT", "POST_OUTPUT_LIMIT",
           "ACTING_TOOLS", "RESOLVED_CHECKPOINT", "FINAL_CHECKPOINT",
           "resolution", "refusal_note", "discovery_round", "checkpoint_prompt", "committed",
           "checkpoint_commits", "reads_only", "relearns", "produced",
           "named_subjects", "lookup_note", "checks"]



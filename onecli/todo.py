"""The plan - state on the session, not a message in the transcript.

A long run is a hundred tool calls that mean something only if you can see the
shape they add up to. The model already plans; it plans in prose, in the middle
of a turn, and the plan scrolls off the screen and out of the context window
together. What survives is the last thing it said, which is why a run that has
done six of eight things reads exactly like a run that has done two.

So the list is not a message. It is a small piece of session state with one
writer:

    todo_write   replace the whole list

Three consequences follow from "state, not transcript", and they are the whole
reason this module exists rather than a convention of writing the plan down:

- **It survives compaction.** The tool result that carried the list is summary
  material like everything else; the list is not. After a compaction the plan
  is restated verbatim, because losing it is losing the run's spine while
  keeping its footnotes.
- **It survives the process.** Every write appends to the session file, so
  `--resume` picks the plan back up. A resumed session that talks about step
  five while showing an empty plan has lied to the person watching.
- **It is rendered between turns**, not buried in a tool result nobody reads.

Whole-list replace rather than per-item updates: two tools and stable ids would
be cheaper per change, but every model already knows this shape, and one writer
means there is exactly one thing to persist and one revision to render. The
cost is real - the model retypes the list to move one item - and it is the
price of the list never being half-updated.

The one guard worth having: **at most one item in progress**. A plan with four
things in progress is not a plan, it is the same undifferentiated pile the list
was meant to replace. A write that breaks it is refused whole, and nothing
changes - the same rule `edit` follows when its match is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from onecli import session as sessionlib

PENDING, ACTIVE, DONE = "pending", "in_progress", "done"
STATES = (PENDING, ACTIVE, DONE)

# A plan longer than this is a symptom: the model is writing the work down
# instead of doing it. The number is a ceiling on that failure, not a target.
MAX_ITEMS = 20

# Steps of silence before the model is reminded the plan is still open. Long
# enough that a focused stretch of work is never interrupted, short enough that
# a plan cannot be abandoned for a whole run.
STALE_AFTER = 10

_MARKS = {PENDING: "[ ]", ACTIVE: "[>]", DONE: "[x]"}
_ALIASES = {
    "todo": PENDING, "open": PENDING, "not_started": PENDING,
    "active": ACTIVE, "in-progress": ACTIVE, "inprogress": ACTIVE,
    "running": ACTIVE, "doing": ACTIVE,
    "completed": DONE, "complete": DONE, "finished": DONE, "closed": DONE,
}


@dataclass
class Item:
    text: str
    state: str = PENDING

    def as_dict(self) -> dict:
        return {"text": self.text, "state": self.state}


class TodoError(ValueError):
    """A write that would leave the list in a state worth refusing."""


def _coerce(raw) -> list[Item]:
    """Model output to items, or raise with a reason the model can act on.

    Lenient about spelling, strict about meaning. `status` for `state` and
    `content` for `text` are what models actually emit - refusing those buys
    nothing and costs a step. An unknown state is a different matter: guessing
    which of three buckets was meant is how a done item goes back to pending.
    """
    if isinstance(raw, dict):          # a single item, not a list
        raw = [raw]
    if not isinstance(raw, list):
        raise TodoError("items must be a list of {text, state} objects")
    if len(raw) > MAX_ITEMS:
        raise TodoError(f"a plan of {len(raw)} items is too long "
                        f"(max {MAX_ITEMS}) - keep it to the milestones")
    items: list[Item] = []
    for i, entry in enumerate(raw, 1):
        if isinstance(entry, str):
            text, state = entry, PENDING
        elif isinstance(entry, dict):
            text = entry.get("text") or entry.get("content") or ""
            state = entry.get("state") or entry.get("status") or PENDING
        else:
            raise TodoError(f"item {i} is a {type(entry).__name__}, "
                            "expected an object with text and state")
        text = " ".join(str(text).split())
        if not text:
            raise TodoError(f"item {i} has no text")
        state = _ALIASES.get(str(state).strip().lower(), str(state).strip().lower())
        if state not in STATES:
            raise TodoError(f"item {i} has state {state!r}; "
                            f"use one of {', '.join(STATES)}")
        items.append(Item(text=text, state=state))
    active = [it for it in items if it.state == ACTIVE]
    if len(active) > 1:
        raise TodoError(
            f"{len(active)} items are in_progress ("
            + "; ".join(it.text for it in active[:3])
            + ") - exactly one thing is in progress at a time. "
              "Nothing was changed.")
    return items


@dataclass
class TodoList:
    """One session's plan. `session_id` empty means in-memory only."""

    session_id: str = ""
    items: list[Item] = field(default_factory=list)
    revision: int = 0
    # True for a list read back off disk. The run injects it into the model's
    # context once and clears the flag - a resumed session whose plan is only
    # in a file the model cannot see is a plan nobody is following.
    restored: bool = False
    stale_after: int = STALE_AFTER
    _seen_revision: int = 0
    _since_write: int = 0

    # --- reading -----------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.items)

    def as_dicts(self) -> list[dict]:
        return [it.as_dict() for it in self.items]

    @property
    def done(self) -> int:
        return sum(1 for it in self.items if it.state == DONE)

    @property
    def active(self) -> Item | None:
        for it in self.items:
            if it.state == ACTIVE:
                return it
        return None

    @property
    def unfinished(self) -> int:
        return sum(1 for it in self.items if it.state != DONE)

    def summary(self) -> str:
        if not self.items:
            return "no plan"
        head = f"{self.done}/{len(self.items)} done"
        return f"{head}, now: {self.active.text}" if self.active else head

    def render(self) -> str:
        """The list as the model sees it. Plain text, no colour, no ids."""
        if not self.items:
            return "plan: empty"
        lines = [f"plan ({self.done}/{len(self.items)} done)"]
        lines += [f"  {_MARKS[it.state]} {it.text}" for it in self.items]
        return "\n".join(lines)

    def block(self, why: str) -> str:
        """The plan restated into the model's context, with the reason why.

        The reason is not decoration: an unexplained list appearing mid-run
        reads as a new instruction, and a model that treats its own plan as a
        fresh user request starts over.
        """
        return f"[plan - {why}. This is your own list, not a new request.]\n" \
               f"{self.render()}"

    # --- writing -----------------------------------------------------------

    def write(self, raw) -> str:
        """Replace the list. Returns what the model should see; raises TodoError."""
        items = _coerce(raw)
        self.items = items
        self.revision += 1
        self._since_write = 0
        self.restored = False
        self._persist()
        return self.render()

    def _persist(self) -> None:
        if not self.session_id:
            return
        sessionlib.append(self.session_id,
                          {"kind": "todos", "revision": self.revision,
                           "items": self.as_dicts()})

    # --- what the run asks it, between turns -------------------------------

    def take_change(self) -> str | None:
        """The rendered plan if it changed since the last look, else None.

        Pull rather than push: the tool returns a string like every other tool,
        so the loop - which is the only thing that can emit an event - has to
        be able to ask whether that string mattered.
        """
        if self.revision == self._seen_revision:
            return None
        self._seen_revision = self.revision
        return self.render()

    def notices(self) -> list[str]:
        """Reminders to inject before the next model call, usually none.

        A plan with unfinished items and no write for `stale_after` steps is
        the failure this whole module exists to catch: the model announced a
        list and went back to working without it. One line, spaced, and only
        while something is still open.
        """
        if not self.items or self.unfinished == 0 or self.stale_after <= 0:
            return []
        self._since_write += 1
        if self._since_write < self.stale_after:
            return []
        self._since_write = 0
        return [self.block(f"unchanged for {self.stale_after} steps")]


def load(session_id: str) -> TodoList:
    """The session's plan as it was last written, or an empty one."""
    todos = TodoList(session_id=session_id)
    try:
        records = sessionlib.load_records(session_id, "todos")
    except FileNotFoundError:
        return todos
    if not records:
        return todos
    last = records[-1]
    try:
        todos.items = _coerce(last.get("items") or [])
    except TodoError:
        # A malformed record is not worth failing a resume over: the plan is a
        # view of the work, and the work is in the transcript.
        return TodoList(session_id=session_id)
    todos.revision = int(last.get("revision") or 1)
    todos._seen_revision = todos.revision
    todos.restored = bool(todos.items)
    return todos


def todo_write(workspace: str = "", todos: TodoList | None = None,
               items=None) -> str:
    """The tool. A run without a list says so rather than pretending to write."""
    if todos is None:
        return "error: this run has no plan to write to"
    try:
        return todos.write(items if items is not None else [])
    except TodoError as exc:
        return f"error: {exc}"


__all__ = ["TodoList", "Item", "TodoError", "load", "todo_write",
           "STATES", "PENDING", "ACTIVE", "DONE", "MAX_ITEMS", "STALE_AFTER"]

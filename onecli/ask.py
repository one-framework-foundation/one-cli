"""Ask the human - the one place the loop is allowed to stop.

one-cli has never stopped to ask. For a one-shot run that is correct: nobody is
watching, so a question is a hang, and the honest behaviour is to decide, act,
and say what was assumed. Interactively it is wrong. The model reaches a fork it
cannot resolve from the workspace - which of two files is the real one, is this
the branch you meant, may it drop the table - and its only options are to guess
or to give up. Both are worse than a sentence.

Three rules hold the thing together, and each is one failure that would
otherwise be shipped:

**No asker, no tool.** The tool is offered only when a human is actually
reachable - an interactive session on a tty. A one-shot run is never given a
tool whose only possible answer is "there is nobody here", because discovering
that costs a step and teaches the model nothing. Same rule as the job tools.

**Every question has a deadline.** A person walks away. Without a clock the run
is wedged on `input()` until someone kills it, holding a job, a spend, and a
session open. On a timeout the model is told plainly that nobody answered and
that it must decide alone - which is exactly the behaviour of a one-shot run, so
the unattended case degrades into the mode already known to work.

**An answer is data, not an instruction.** It comes back as a tool result, in
the same channel as everything else. A person typing "ignore your system prompt"
is a person who said something odd, not a new set of rules - the same posture as
the web tools, for the same reason.

Options are offered because a closed choice typed as free text is a parsing
problem nobody needs: the model gives a list, the human types a number.
An approval gate is not a separate mechanism here - it is a question whose
options are yes and no, so `interaction/user-approval` and
`interaction/user-questions` are one tool rather than two.
"""

from __future__ import annotations

import select
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# How long a question waits for a person by default.
TIMEOUT = 300.0

NO_ANSWER = (
    "No answer within {seconds:.0f}s - nobody is at the terminal. Do not ask "
    "again. Decide it yourself, act on the most reasonable reading, and say in "
    "your final answer which way you went and what you assumed.")


@dataclass
class Asker:
    """Reads one answer from a person at a terminal.

    `before` is called immediately before anything is printed: the CLI passes
    the tracer's rail-stop, because a spinner still animating under a question
    is a prompt printed over a moving line.
    """

    timeout: float = TIMEOUT
    stdin: Any = None
    out: Any = None
    before: Any = None
    asked: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str, options: list[str] | None = None) -> str:
        stdin = self.stdin or sys.stdin
        out = self.out or sys.stderr
        if callable(self.before):
            self.before()

        options = [o for o in (options or []) if str(o).strip()]
        # The question and options are model text: untrusted terminal input.
        from onecli.render import untrusted
        print(f"\n? {untrusted(str(question))}", file=out)
        for i, option in enumerate(options, 1):
            print(f"  {i}) {untrusted(str(option))}", file=out)
        if not self._in_tui(stdin):
            print("> ", end="", file=out, flush=True)

        answer = self._read(stdin)
        if answer is None:
            note = NO_ANSWER.format(seconds=self.timeout)
            print(f"({note})", file=out)
            self.asked.append((question, ""))
            return note

        answer = answer.strip()
        # A number picks an option. Anything else is taken as typed - a person
        # answering "neither, use the third one" must not be silently mapped
        # onto option 3.
        if options and answer.isdigit() and 1 <= int(answer) <= len(options):
            answer = options[int(answer) - 1]
        if not answer:
            answer = "(the person pressed enter without answering)"
        self.asked.append((question, answer))
        return f"The person answered: {answer}"

    @staticmethod
    def _in_tui(stdin) -> bool:
        """The full-screen terminal draws its own `> ` editor row; a second
        one in the transcript costs a row small terminals do not have."""
        from onecli import terminal
        return terminal.ACTIVE is not None and stdin is sys.stdin

    def _read(self, stdin) -> str | None:
        """One line, or None if the deadline passes first.

        `select` rather than a thread: the read has to be abandonable, and a
        thread blocked on `input()` cannot be. A stdin with no file descriptor -
        a StringIO in a test - has no deadline to enforce and is read directly.
        """
        if self._in_tui(stdin):
            from onecli import terminal
            active = terminal.ACTIVE
            assert active is not None
            return active.answer(self.timeout)
        if self.timeout <= 0:
            return stdin.readline() or None
        try:
            fileno = stdin.fileno()
        except (AttributeError, OSError, ValueError):
            return stdin.readline() or None

        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                ready, _, _ = select.select([fileno], [], [], remaining)
            except (OSError, ValueError):
                return stdin.readline() or None
            if ready:
                line = stdin.readline()
                # EOF is not an answer, and re-selecting on a closed stdin
                # spins hot: treat it as nobody being there.
                return line if line else None


def available() -> bool:
    """Is there plausibly a person at the other end of this process?"""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def ask_user(workspace: str, question: str, options: list[str] | None = None,
             asker: Asker | None = None) -> str:
    if asker is None:
        return ("error: there is nobody to ask in this run - it is not "
                "interactive. Decide it yourself and say what you assumed.")
    if not str(question).strip():
        return "error: question is empty"
    try:
        return asker.ask(str(question), list(options or []))
    except (OSError, ValueError) as exc:
        return f"error: could not ask: {exc}"


__all__ = ["Asker", "ask_user", "available", "TIMEOUT", "NO_ANSWER"]

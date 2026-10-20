"""Shared shapes for the Senses.

A Senses answers one question - "how does this modality get perceived" - and
has to be able to say what it tried. That is why an outcome carries the whole
rung list and not just the winner: the refusal at the bottom of a ladder is
only useful if it names every rung above it, and the measure-rung log row is
built from the same list rather than from a second bookkeeping path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Route names. `inline` hands the bytes to the model, `describe`/`transcribe`
# turn them into text, `off` attaches nothing at all.
INLINE = "inline"
DESCRIBE = "describe"
TRANSCRIBE = "transcribe"
OFF = "off"
AUTO = "auto"

ROUTES = (AUTO, INLINE, DESCRIBE, TRANSCRIBE, OFF)


@dataclass
class Rung:
    """One attempt on a ladder, and why it did not end the climb.

    `skipped` and `failed` are deliberately different: a missing API key is a
    configuration state and must not read as an outage in the log, and a
    refusal that blames the wrong one sends the operator to the wrong host.
    """

    name: str
    ok: bool = False
    skipped: str = ""   # why this rung was not attempted at all
    error: str = ""     # why the attempt failed
    ms: int = 0

    def describe(self) -> str:
        if self.ok:
            return f"{self.name}: ok"
        if self.skipped:
            return f"{self.name}: skipped ({self.skipped})"
        return f"{self.name}: failed ({self.error or 'no result'})"

    def as_row(self) -> dict:
        return {"rung": self.name, "ok": self.ok, "skipped": self.skipped,
                "error": self.error, "ms": self.ms}


@dataclass
class Outcome:
    """What one Senses made of one attachment."""

    modality: str
    route: str = OFF
    text: str = ""                    # the perception, as text to inject
    inline: bool = False              # keep the bytes on the request instead
    language: str = ""                # only when a rung actually reported one
    rungs: list[Rung] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.inline or bool(self.text)

    def ladder(self) -> str:
        return "; ".join(r.describe() for r in self.rungs)

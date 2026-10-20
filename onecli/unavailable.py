"""What happens when the model someone NAMED cannot be used.

Policy: `-m <name>` is an
instruction. When that profile is offline or fails, one-cli must not answer
somewhere else without saying so - least of all on a paid lane. It says why,
lists what is online right now (by presence probe, free lanes first, paid ones
labelled), and then does what the operator configured:

    ask       a terminal: error, suggestions, pick one or abort (default).
              Not a terminal, or `-p`: behaves as `error`.
    error     fail with the reason and the suggestions on stderr, exit 2.
    fallback  switch to the first suggestion and say so loudly. A paid lane is
              only eligible with `fallback_allow_paid`.

Settings follow one-cli's existing pattern - a flag per run, an `ONE_CLI_*`
variable as the environment default (the same shape as `--fallback-model` /
`ONE_CLI_FALLBACK`). There is no settings file to put them in.
"""

from __future__ import annotations

import os
import sys

from onecli import config, presence, render

MODES = ("ask", "error", "fallback")
DEFAULT_MODE = "ask"
MODE_ENV = "ONE_CLI_ON_UNAVAILABLE_MODEL"
ALLOW_PAID_ENV = "ONE_CLI_FALLBACK_ALLOW_PAID"
TRUE = ("1", "true", "yes", "on")
# Enough to choose from without burying the reason above it.
MAX_SUGGESTIONS = 8


def mode(args) -> str:
    """The flag, else the env default, else `ask`. A typo reads as `ask`."""
    chosen = (getattr(args, "on_unavailable_model", None)
              or os.environ.get(MODE_ENV, "")).strip().lower()
    return chosen if chosen in MODES else DEFAULT_MODE


def allow_paid(args) -> bool:
    if getattr(args, "fallback_allow_paid", False):
        return True
    return os.environ.get(ALLOW_PAID_ENV, "").strip().lower() in TRUE


def suggestions(primary, found=None) -> list[tuple[config.Backend, bool]]:
    """Usable alternatives as (backend, paid): online free lanes, then paid.

    A free lane is only offered when runner has just seen it answer. A paid
    lane is never dialled (that costs money), so it is offered as paid and
    unchecked - the label is what lets a person decide.
    """
    found = config.discover() if found is None else found
    others = [b for b in found if b.name != primary.name]
    state = presence.survey(others)
    free = [b for b in others if state[b.name] == presence.ONLINE]
    paid = [b for b in others if state[b.name] == presence.UNCHECKED]
    return ([(b, False) for b in free] + [(b, True) for b in paid])[:MAX_SUGGESTIONS]


def listing(options) -> str:
    if not options:
        return "no online alternatives found"
    rows = ["online alternatives (free first):"]
    for i, (b, paid) in enumerate(options, 1):
        rows.append(f"  {i}. {b.name} ({b.model}){'  [paid]' if paid else ''}")
    rows.append("choose one with -m <name>")
    return "\n".join(rows)


def reason_of(error: str) -> str:
    """offline / auth / not found, from a provider error's own words."""
    low = (error or "").lower()
    if any(w in low for w in ("401", "403", "unauthorized", "forbidden",
                              "invalid api key", "authentication")):
        return "auth"
    if any(w in low for w in ("404", "not found", "does not exist", "unknown model",
                              "model_not_found")):
        return "not found"
    if any(w in low for w in ("refused", "unreachable", "timed out", "timeout",
                              "connection", "name or service")):
        return "offline"
    return "failed"


def _interactive(args) -> bool:
    return (not getattr(args, "print_mode", False) and not getattr(args, "json", False)
            and sys.stdin.isatty() and sys.stderr.isatty())


def _ask(options):
    print("choose a number, or press Enter to abort: ", end="", file=sys.stderr, flush=True)
    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw.isdecimal() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1][0]
    return None


def resolve(args, primary, reason: str):
    """The backend to run on instead of `primary`, or None to stop (exit 2)."""
    # An explicit endpoint must not trigger discovery or probe other hosts,
    # even when it is unavailable and fallback was requested.
    options = [] if primary.source == "flag" else suggestions(primary)
    how = mode(args)
    print(render.red(reason), file=sys.stderr)
    if how == "fallback":
        eligible = [b for b, paid in options if not paid or allow_paid(args)]
        if eligible:
            print(render.yellow(
                f"on_unavailable_model=fallback: running on {eligible[0].name} "
                f"({eligible[0].model}) instead of {primary.name}"), file=sys.stderr)
            return eligible[0]
        print(render.yellow("on_unavailable_model=fallback: no eligible lane"
                            + ("" if allow_paid(args) else
                               " (paid lanes need fallback_allow_paid)")),
              file=sys.stderr)
    print(listing(options), file=sys.stderr)
    if how == "ask" and options and _interactive(args):
        return _ask(options)
    return None


def chain_spec(args, spec):
    """The fallback spec for this run. An explicitly named model gets no
    implicit chain (auto or the drop-in's declared one) unless the mode is
    `fallback`; `--fallback-model` / ONE_CLI_FALLBACK stay the explicit opt-in.
    """
    if spec is None and getattr(args, "model", None) and mode(args) != "fallback":
        return []
    return spec

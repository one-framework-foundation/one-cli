"""Fallback chains - where a turn goes when a backend will not answer.

A chain is an ordered list of backends. A turn starts at the top; a transient
failure (429, 5xx, timeout, unreachable) moves it down one rung AND LEAVES IT
THERE for the rest of the run. That stickiness is the point: an endpoint that
just rate-limited will rate-limit the next turn too, and paying its timeout
again on every step is exactly the cost a chain exists to avoid. A model
declining a task is not a failure and never moves the turn - a refusal comes
back as an ordinary Turn and never reaches this code at all.

The chain is built without the operator naming it, in this order:

1. `--fallback-model a,b,c` / `ONE_CLI_FALLBACK` - explicit, wins, and
   `off` (or `--no-fallback`) means the run stays on one backend.
2. `ONE_PROVIDER_FALLBACK=a,b` in the primary's own drop-in - the endpoint
   declares its own successors, so the ladder is config that lives next to the
   thing it describes and reloads hot with it.
3. auto - rank what discovery already found.

The auto rank is not a preference about model quality, which this package
cannot know. It is about what an outage looks like: a backend on the SAME
endpoint as the primary is dropped (one endpoint, one outage), the operator's
own drop-in order is kept because that ordering is already a statement, and a
local ollama goes last as the floor that is always there and costs nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from onecli import config, lanehealth

# Env-level default, so a host can pin the behaviour without touching argv.
FALLBACK_ENV = "ONE_CLI_FALLBACK"
# Words that mean "no chain", accepted from flag or env.
OFF = ("off", "none", "no", "0")
# An auto chain is capped. Discovery returns every ollama model on the host,
# and a request that is broken rather than unlucky must not wander through
# forty of them before it reports the failure.
MAX_AUTO_RUNGS = 3


@dataclass
class Rung:
    """One backend the turn can land on. `provider` is already built."""

    name: str
    provider: Any
    model: str = ""


@dataclass
class Chain:
    """The ladder plus where this run currently stands on it.

    Rung 0 is the primary. `advance()` is one-way, which is what makes the
    demotion stick for the rest of the run (and, in interactive mode, for the
    rest of the session - the chain outlives each turn's Agent on purpose).
    """

    rungs: list[Rung]
    cursor: int = 0
    history: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = "auto"

    @property
    def current(self) -> Rung:
        return self.rungs[self.cursor]

    @property
    def spares(self) -> list[str]:
        return [r.name for r in self.rungs[1:]]

    def has_next(self) -> bool:
        return self.cursor + 1 < len(self.rungs)

    def advance(self, reason: str) -> dict | None:
        """Demote the current rung. Returns the note, or None at the bottom."""
        if not self.has_next():
            return None
        note = {
            "from": self.current.name,
            "to": self.rungs[self.cursor + 1].name,
            "reason": reason,
        }
        self.cursor += 1
        self.history.append(note)
        return note

    def as_dict(self) -> dict:
        return {"source": self.source, "rungs": [r.name for r in self.rungs],
                "on": self.current.name, "warnings": list(self.warnings),
                "effort": {r.name: r.provider.effort_decision for r in self.rungs
                           if isinstance(getattr(r.provider, "effort_decision", None), dict)}}


def parse_spec(values) -> list[str] | None:
    """Flag/env text to a name list. None means auto, [] means off."""
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    names: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            if part.lower() in OFF:
                return []
            if part.lower() == "auto":
                return None
            names.append(part)
    return names or None


def spec_from_env() -> list[str] | None:
    # A parent process that owns route selection sets ONE_DETACHED_OWNER=caller.
    # Implicit discovery or inherited fallback settings must not move its run
    # onto another provider. An explicit --fallback-model still overrides.
    if os.environ.get("ONE_DETACHED_OWNER") == "caller":
        return []
    return parse_spec(os.environ.get(FALLBACK_ENV) or None)


def _endpoint(url: str) -> str:
    parts = urlsplit(url if "//" in url else f"//{url}")
    return f"{parts.hostname or url}:{parts.port or ''}"


def is_ollama(backend: config.Backend) -> bool:
    return backend.name.startswith("ollama/") or backend.name.startswith("ollama@")


# An embedding model on a chat endpoint answers every turn with an error, so a
# floor rung that lands on one is a rung that cannot work. The daemon lists its
# models in an order we do not control, so this has to be filtered, not hoped.
_NOT_CHAT = ("embed", "bge-", "rerank")


def is_chat_model(backend: config.Backend) -> bool:
    name = f"{backend.name} {backend.model}".lower()
    return not any(marker in name for marker in _NOT_CHAT)


def declared_for(primary: config.Backend) -> list[str]:
    """What the primary's own drop-in says its successors are."""
    return [str(n) for n in (getattr(primary, "fallback", None) or [])]


def auto_backends(primary: config.Backend,
                  found: list[config.Backend] | None = None,
                  limit: int = MAX_AUTO_RUNGS) -> list[config.Backend]:
    """Rank the reachable backends into a chain for `primary`."""
    found = config.discover() if found is None else found
    here = _endpoint(primary.base_url)
    drop_ins_same: list[config.Backend] = []
    drop_ins_other: list[config.Backend] = []
    local: list[config.Backend] = []
    for b in found:
        if b.name == primary.name or _endpoint(b.base_url) == here:
            continue
        if is_ollama(b):
            # One ollama rung is enough: if the daemon is up any model on it
            # will answer, and if it is down none of them will.
            if not local and is_chat_model(b):
                local.append(b)
            continue
        (drop_ins_same if b.profile == primary.profile else drop_ins_other).append(b)
    policy = config._subscription_policy()
    ranked = (policy.prefer(drop_ins_same, lambda b: b.model)
              + policy.prefer(drop_ins_other, lambda b: b.model))
    if not local:
        return ranked[:limit]
    # The local floor keeps its rung rather than being pushed off the end by
    # the cap. A chain of paid endpoints that are all busy at once ends in a
    # failed run; the same chain ending on a model that is up on this host
    # ends in an answer, which is the entire point of having a floor.
    return ranked[:max(limit - 1, 0)] + local


def resolve_backends(names: list[str]) -> tuple[list[config.Backend], list[str]]:
    """Named backends, plus a warning per name nothing matched.

    A name that resolves to nothing is a config typo, not a reason to refuse
    the run: the primary still works and the chain is simply shorter.
    """
    out, warnings = [], []
    for name in names:
        try:
            out.append(config.resolve(name=name))
        except RuntimeError as exc:
            warnings.append(str(exc))
    return out, warnings


def drop_unhealthy(backends: list[config.Backend]
                   ) -> tuple[list[config.Backend], list[str]]:
    """Spare rungs minus the ones we already know will refuse.

    Applied to the spares of EVERY chain source, including an explicit
    `--fallback-model`: naming a lane says which order to try, not that a lane
    with a spent key should be tried anyway. The PRIMARY is never filtered here
    - `-m <backend>` is an instruction, and a run that silently answered
    somewhere else would be worse than one that fails where it was told to go.

    If every spare is unhealthy the list comes back empty and the chain is just
    the primary. That is honest: there is nowhere good to fall to, and the run
    should say so rather than fall somewhere known-dead.
    """
    kept, notes = [], []
    for b in backends:
        why = lanehealth.backend_unavailable(b)
        if why:
            notes.append(f"skipped fallback {b.name}: {why}")
        else:
            kept.append(b)
    return kept, notes


def build(primary: config.Backend, spec: list[str] | None, *,
          build_provider, timeout: float = 600.0,
          allow_auto: bool = True, allow_paid: bool = True) -> Chain:
    """The chain this run walks. `spec` None means auto, [] means off."""
    primary_rung = Rung(name=primary.name, provider=build_provider(primary, timeout),
                        model=primary.model)
    chain = Chain(rungs=[primary_rung])
    warnings: list[str] = []

    if spec == []:
        chain.source = "off"
        return chain

    if spec:
        backends, warnings = resolve_backends(spec)
        chain.source = "explicit"
    else:
        declared = declared_for(primary)
        if declared:
            backends, warnings = resolve_backends(declared)
            chain.source = "declared"
        elif allow_auto:
            backends = auto_backends(primary)
            chain.source = "auto"
        else:
            backends = []
            chain.source = "off"

    backends, skipped = drop_unhealthy(backends)
    warnings.extend(skipped)
    if not allow_paid and not spec:
        # An implicit chain (auto or declared) never reaches a paid lane the
        # operator did not allow; a list they typed is already that allowance.
        policy = config._subscription_policy()
        for b in [b for b in backends if policy.is_paid(b.base_url)]:
            warnings.append(f"skipped fallback {b.name}: paid lane not allowed")
        backends = [b for b in backends if not policy.is_paid(b.base_url)]

    seen = {primary.name}
    for b in backends:
        if b.name in seen:
            continue
        seen.add(b.name)
        chain.rungs.append(Rung(name=b.name, provider=build_provider(b, timeout),
                                model=b.model))
    chain.warnings = warnings
    return chain

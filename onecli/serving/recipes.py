"""Serving recipes: the flag sets a model can be served with on THIS host.

A recipe is a named set of extra llama.cpp flags plus what it requires to be
true. Nothing here decides which one is fastest - `tune.py` measures that. The
job of this module is to make sure every candidate offered is one the runner
on this box can actually execute, so a sweep never spends ten minutes proving
that a flag the build does not have is slow.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import hostfacts
from .catalog import Model

# Flags every lane gets, whatever the recipe: full GPU offload, flash
# attention, a q8_0 KV cache and prompt-cache reuse.
BASE_FLAGS = ["-ngl", "999", "-fa", "1",
              "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
              "--cache-reuse", "256"]

# Speculative decoding families the tuner knows how to build a candidate for.
# The runner is asked which of these it supports; the intersection is swept.
DRAFT_TYPES = ("draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark",
               "draft-simple")
NGRAM_TYPES = ("ngram-mod", "ngram-simple", "ngram-map-k", "ngram-map-k4v",
               "ngram-cache")

# The ngram family needs no weights, so it is cheap to sweep - but sweeping all
# five triples the run. This is the one to try first.
NGRAM_DEFAULT = "ngram-mod"


@dataclass
class Recipe:
    name: str
    flags: list[str] = field(default_factory=list)
    why: str = ""

    def key(self) -> str:
        return self.name


def _help_text(facts: hostfacts.HostFacts) -> str:
    """The runner's own --help, which is the only authority on its flags."""
    if facts.llama_server:
        cmd = [facts.llama_server, "--help"]
    elif facts.docker:
        cmd = ["docker", "run", "--rm", "--entrypoint", "/app/llama-server",
               facts.docker_image or hostfacts.DOCKER_IMAGE, "--help"]
    else:
        return ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (out.stdout or "") + (out.stderr or "")


def supported_spec_types(facts: hostfacts.HostFacts | None = None) -> list[str]:
    """Parse the allowed values of --spec-type out of the runner's help.

    Probed, not hardcoded: llama.cpp builds differ and a
    recipe measured against a build that has draft-mtp is meaningless on one
    that does not.
    """
    facts = facts or hostfacts.collect()
    text = _help_text(facts)
    m = re.search(r"--spec-type\s+([a-z0-9,\-]+)", text)
    if not m:
        return []
    return [t for t in m.group(1).split(",") if t and t != "none"]


def candidates(model: Model, facts: hostfacts.HostFacts | None = None,
               spec_types: list[str] | None = None,
               all_ngram: bool = False,
               root: Path | None = None) -> list[Recipe]:
    """Every recipe worth measuring for this model on this host.

    Baseline is always first and always measured, because a speedup with no
    baseline beside it is a number, not a result.
    """
    facts = facts or hostfacts.collect()
    supported = spec_types if spec_types is not None else supported_spec_types(facts)
    out = [Recipe(name="baseline", flags=[],
                  why="no speculative decoding; the number everything else is "
                      "measured against")]

    draft = model.draft_path(root)
    if draft is not None and model.draft_present(root):
        want = (model.draft.spec_type if model.draft else "") or "draft-mtp"
        if want in supported:
            out.append(Recipe(
                name=want,
                flags=["-md", str(draft), "--spec-type", want],
                why=f"{draft.name} is on disk and the runner speaks {want}"))

    ngrams = NGRAM_TYPES if all_ngram else (NGRAM_DEFAULT,)
    for t in ngrams:
        if t in supported:
            out.append(Recipe(name=t, flags=["--spec-type", t],
                              why="speculates from the context itself; needs "
                                  "no extra weights"))
    return out


def skipped(model: Model, facts: hostfacts.HostFacts | None = None,
            spec_types: list[str] | None = None,
            root: Path | None = None) -> list[tuple[str, str]]:
    """Recipes deliberately NOT measured, and why.

    A sweep that silently drops a candidate reads as "we tried everything".
    This is what turns that into a stated bound.
    """
    facts = facts or hostfacts.collect()
    supported = spec_types if spec_types is not None else supported_spec_types(facts)
    out: list[tuple[str, str]] = []
    if model.draft:
        want = model.draft.spec_type or "draft-mtp"
        if not model.draft_present(root):
            out.append((want, f"drafter {model.draft.file} not on disk - "
                              f"`one models pull {model.name}` fetches it"))
        elif want not in supported:
            out.append((want, f"runner {facts.runner()} does not list {want} "
                              f"under --spec-type"))
    else:
        # One line, not one per draft type: with no drafter at all, listing
        # five identical "we have no weights" rows buries the one fact that
        # matters, which is that this model publishes no drafter.
        out.append(("draft-*", "no drafter published for this model"))
        return out
    for t in DRAFT_TYPES:
        if t in supported and (model.draft.spec_type or "draft-mtp") != t:
            out.append((t, "runner supports it but we have no weights for it"))
    return out


def command(model: Model, recipe: Recipe, port: int,
            facts: hostfacts.HostFacts | None = None,
            context: int = 0, parallel: int = 2,
            root: Path | None = None,
            container: str = "") -> list[str]:
    """The full argv that serves `model` under `recipe` on this host.

    One place builds the command, so no launcher script hand-copies flags and
    drifts from what the lane actually runs.
    """
    facts = facts or hostfacts.collect()
    root = root or hostfacts.models_dir()
    ctx = context or model.context or 131072
    weights = model.path(root)

    # Containers need a wildcard listener internally, but only publish it on
    # host loopback. Native runners bind loopback directly. Remote exposure
    # requires a separately configured authenticated proxy/tunnel.
    bind = "127.0.0.1" if facts.llama_server else "0.0.0.0"
    args = ["-m", str(weights), "--host", bind, "--port", str(port),
            *BASE_FLAGS, "-c", str(ctx), "-np", str(parallel),
            "--alias", model.alias or model.name]
    if parallel > 1:
        args.append("--cont-batching")
    args += recipe.flags

    if facts.llama_server:
        return [facts.llama_server, *args]

    image = facts.docker_image or hostfacts.DOCKER_IMAGE
    # Inside the container the weights dir is always /models, so the argv the
    # ledger records stays the same whichever host replays it.
    mapped = _remap(args, root)
    name = container or f"one-lane-{port}"
    return ["docker", "run", "-d", "--name", name, "--gpus", "all",
            "--restart", "unless-stopped", "-p", f"127.0.0.1:{port}:{port}",
            "-v", f"{root}:/models:ro", image, *mapped]


def _remap(args: list[str], root: Path) -> list[str]:
    prefix = str(root).rstrip("/") + "/"
    return [a.replace(prefix, "/models/") if isinstance(a, str) else a
            for a in args]

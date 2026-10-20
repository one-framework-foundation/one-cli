"""Launch a lane with the recipe that was measured fastest on THIS host.

The failure this replaces: flags live in a hand-written launcher script, one
gets tuned live, and the script and the running process drift apart. A lane can
carry a speculative drafter that its own launcher file never mentions, so a
restart silently throws the speedup away. Here the ledger is the source and the
command is derived from it, so a restart cannot lose a measurement.
"""

from __future__ import annotations

import http.client
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .. import config as onecfg, secureio
from . import hostfacts, recipes, tune
from .catalog import Model

LANE_PREFIX = "one-lane-"
# Output of a native llama-server lane, which runs detached from `one serve`.
LOG_DIR = Path.home() / ".one-cli" / "serve"


@dataclass
class Lane:
    model: str
    port: int
    recipe: str
    verified: bool
    container: str = ""
    detail: str = ""


def dropin_path(model: Model) -> Path:
    return onecfg.PROVIDER_DIR / f"local-{model.alias or model.name}.env"


def register(model: Model, port: int, label: str = "") -> Path:
    """Write the drop-in so `one --list` and every agent sees the new lane.

    A served model that discovery cannot find is a process, not a backend.
    """
    path = dropin_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"# Written by `one serve`. Regenerated on every launch.\n"
        f"ONE_PROVIDER_LABEL={label or model.name}\n"
        f"ONE_PROVIDER_MODEL={model.alias or model.name}\n"
        f"ONE_PROVIDER_WIRE=openai\n"
        f"OPENAI_BASE_URL=http://127.0.0.1:{port}/v1\n"
    )
    secureio.atomic_write_text(path, body, private=True)
    return path


def health(port: int, timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
            return isinstance(body, dict) and body.get("status") == "ok"
    except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError):
        return False


def running(facts: hostfacts.HostFacts | None = None) -> list[Lane]:
    """Lanes this host is serving right now, read from the runtime.

    Docker is asked what it is running rather than a state file being trusted:
    a lane that died is not a lane, however recently we wrote it down.
    """
    facts = facts or hostfacts.collect()
    out: list[Lane] = []
    if not facts.docker:
        return out
    raw = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}\t{{.Status}}"],
        capture_output=True, text=True, check=False, timeout=60).stdout
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, ports, status = parts[0], parts[1], parts[2]
        port = _port_of(ports)
        if port is None:
            continue
        args = _inspect_args(name)
        # A published port does not make a container a model lane. A host can
        # run unrelated containers with port maps; the first version of this
        # counted all of them and reported "3 lanes running" on a host serving
        # nothing. The weights argument is what actually identifies a lane.
        weights = _flag(args, "-m") or _flag(args, "--model")
        if not weights.endswith(".gguf"):
            continue
        spec = _flag(args, "--spec-type") or "baseline"
        out.append(Lane(model=_flag(args, "--alias") or name, port=port,
                        recipe=spec, verified=False, container=name,
                        detail=status))
    return out


def _port_of(ports: str) -> int | None:
    for chunk in ports.split(","):
        if "->" not in chunk:
            continue
        host_side = chunk.split("->")[0].strip()
        tail = host_side.rsplit(":", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return None


def _inspect_args(container: str) -> list[str]:
    raw = subprocess.run(["docker", "inspect", "-f", "{{json .Args}}", container],
                         capture_output=True, text=True, check=False,
                         timeout=60).stdout.strip()
    try:
        return json.loads(raw)
    except ValueError:
        return []


def _flag(args: list[str], name: str) -> str:
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
    return ""


def plan(model: Model, port: int, facts: hostfacts.HostFacts | None = None,
         context: int = 0, parallel: int = 2, root: Path | None = None,
         force_recipe: str = "") -> tuple[list[str], recipes.Recipe, dict | None]:
    """The exact command that would be run, plus the evidence for its recipe."""
    facts = facts or hostfacts.collect()
    if force_recipe:
        cands = {r.name: r for r in recipes.candidates(model, facts, root=root)}
        if force_recipe not in cands:
            raise RuntimeError(
                f"{force_recipe} is not a candidate for {model.name} here; "
                f"available: {', '.join(cands)}")
        recipe, entry = cands[force_recipe], tune.best(model, facts)
    else:
        recipe, entry = tune.recipe_for(model, facts)
    container = f"{LANE_PREFIX}{model.alias or model.name}"
    cmd = recipes.command(model, recipe, port, facts, context=context,
                          parallel=parallel, root=root, container=container)
    return cmd, recipe, entry


def start(model: Model, port: int, facts: hostfacts.HostFacts | None = None,
          context: int = 0, parallel: int = 2, root: Path | None = None,
          force_recipe: str = "", log=None) -> Lane:
    facts = facts or hostfacts.collect()
    if not model.present(root):
        raise RuntimeError(f"{model.file} is not on disk "
                           f"(`one models pull {model.name}`)")
    cmd, recipe, entry = plan(model, port, facts, context, parallel, root,
                              force_recipe)
    container = f"{LANE_PREFIX}{model.alias or model.name}"
    if facts.docker and not facts.llama_server:
        # Free the port as well as the name. A lane launched by hand carries a
        # different container name, so removing only ours leaves the old one
        # holding the port and the launch fails with a bind error that says
        # nothing about the real cause.
        for existing in occupants(port, facts):
            if existing != container and log:
                log(f"  removing {existing}, which holds :{port}")
            subprocess.run(["docker", "rm", "-f", existing],
                           capture_output=True, check=False, timeout=120)
        subprocess.run(["docker", "rm", "-f", container], capture_output=True,
                       check=False, timeout=120)
    if log:
        log(f"launching {model.name} on :{port} with recipe {recipe.name}")
        log(f"  {recipe.why}")
    if facts.llama_server:
        # A native server IS the lane, so it runs detached the way `docker run
        # -d` leaves a container; in the foreground it held `one serve` until
        # the launch timeout and then died with it.
        log_path = LOG_DIR / f"{container}.log"
        with secureio.text_file(log_path, "w", private=True) as out:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True)
        if log:
            log(f"  waiting for the model to load (log: {log_path})")
        if not tune._wait_health(port, proc=proc):
            tune._stop(facts, container, proc)
            raise RuntimeError(f"{model.name} never reported healthy on :{port}; "
                               f"see {log_path}")
    else:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False,
                             timeout=300)
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout).strip()[:500])
    register(model, port)
    verified = bool(entry and not entry.get("stale")
                    and entry.get("winner") == recipe.name)
    return Lane(model=model.name, port=port, recipe=recipe.name,
                verified=verified, container=container,
                detail=recipe.why)


def occupants(port: int, facts: hostfacts.HostFacts | None = None) -> list[str]:
    """Container names of model lanes currently bound to `port`."""
    facts = facts or hostfacts.collect()
    return [lane.container for lane in running(facts) if lane.port == port and lane.container]


def stop(model: Model, facts: hostfacts.HostFacts | None = None) -> bool:
    """Remove whatever container is actually serving `model`. Report honestly.

    The container name is read from the runtime rather than rebuilt from the
    model. Two things made the rebuilt name lie: a lane started outside
    `one serve` carries whatever name its launcher chose, and `docker rm -f`
    exits 0 on a container that does not exist. Together they printed
    "stopped <model>" at a lane that went on serving.
    """
    facts = facts or hostfacts.collect()
    alias = model.alias or model.name
    wanted = {model.name, alias, f"{LANE_PREFIX}{alias}"}
    targets = [lane.container for lane in running(facts)
               if lane.container and (lane.model in wanted or lane.container in wanted)]
    removed = False
    for container in targets:
        res = subprocess.run(["docker", "rm", "-f", container],
                             capture_output=True, check=False, timeout=120)
        removed = removed or res.returncode == 0
    # Only deregister a backend we really took down; dropping the drop-in for a
    # lane that is still serving hides a live endpoint from every agent.
    path = dropin_path(model)
    if removed and path.is_file():
        path.unlink()
    return removed

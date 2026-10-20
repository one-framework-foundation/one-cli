"""An ever-expanding task corpus: tasks live on disk, not in this file.

`bench.py` and `scenarios.py` define their tasks as Python subclasses. That is
right for a fixed floor - the checkers are exact and they are reviewed - but it
means adding a task is a code edit, a test run and a pull request. A bench whose
only growth path is a PR does not grow, and the tasks that matter most are the
ones nobody planned: the thing an agent got wrong in production yesterday.

A corpus task is a directory. Drop it in and the next run picks it up:

    <corpus>/<name>/
        task.json     prompt, seed files, and the mask/gate dimensions
        check.py      run in the workspace afterwards; exit 0 means pass

`task.json` may carry two dimensions the in-code tasks have no way to express:

    "mask":  "structured-edit"          the harness the cell wears
    "gates": {"EXAMPLE_GATE": "enforce"}  optional guard state for the cell

A mask contributes its preamble to the prompt, so a mask run measures the
harness rather than the raw model. Gates are exported into the cell's
environment, so the same task can be run with a guard off and on and the
difference is the guard's measured worth.

Two ideas here are borrowed from Harbor (Apache-2.0, `pypi.org/project/harbor`),
rebuilt in our own shape rather than adopted wholesale:

    rewards      a checker may score several named dimensions, not just
                 pass/fail, by writing floats to $ONE_REWARDS
    solve.sh     an oracle solution, so a task can prove it is passable

The oracle is the admission gate that makes bulk task authoring safe. A task is
only admitted if it FAILS on the untouched workspace and PASSES once its own
solution is applied. A task that passes before anything is done measures
nothing, and a task nothing can pass is a bug wearing a score.

Nothing here silently skips a broken task: `discover` returns the problems it
found alongside the tasks, and the caller is expected to show them. A corpus
that quietly drops a malformed task reports a smaller, greener bench than the
one you actually have.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from .bench import Task, _run

DEFAULT_ROOT = Path.home() / ".one-cli" / "corpus"

# A task.json key that is not one of these is a typo, and a typo that is
# ignored is a dimension that silently did not apply.
KNOWN_KEYS = {
    "prompt", "files", "read_only", "max_steps",
    "mask", "gates", "origin", "tags", "description",
    "metrics", "timeout_sec",
}

# How a metric collapses across the runs of one task. `mean` is the default
# because it is the only one that does not hide a bimodal result behind a
# single lucky or unlucky cell.
AGGREGATIONS: dict[str, Callable[[list[float]], float]] = {
    "mean": statistics.fmean,
    "min": min,
    "max": max,
    "sum": sum,
}

# The file a checker writes its named scores to. Passed in the environment so
# the checker never has to guess the workspace layout.
REWARDS_FILE = "_rewards.json"


def corpus_root() -> Path:
    """Where tasks live. Env-overridable so a run can point at a fixture."""
    return Path(os.environ.get("ONE_CLI_CORPUS") or DEFAULT_ROOT)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

@dataclass
class MaskInfo:
    kind: str
    preamble: str = ""
    models: list[str] = field(default_factory=list)
    source: str = "none"
    execution_policy: dict | None = None
    loop_policy: dict | None = None
    context_contract: dict | None = None

    def agent_options(self, *, enforce: bool = False) -> dict:
        """Opt in explicitly; existing corpus runs remain prompt-only."""
        if not enforce:
            return {}
        from onecli.masks.policy import MaskPolicy
        from onecli.masks.loop import parse
        options = {'mask_name': self.kind, 'mask_preamble': self.preamble}
        if self.execution_policy is not None:
            options['mask_policy'] = MaskPolicy.parse(self.execution_policy).as_dict()
        if self.loop_policy is not None:
            options['loop_policy'] = parse(self.loop_policy)
        if self.context_contract is not None:
            from onecli.context_contract import parse as parse_context
            options['context_contract'] = parse_context(self.context_contract)
        return options



# Every key a mask entry may carry. Anything else is a typo until proven
# otherwise: `execution_polcy` read as "no policy" ran the mask unconfined.
MASK_KEYS = frozenset({"prompts", "preamble", "community_models", "execution_policy",
                       "loop_policy", "context_contract"})


def _masks_from_json(path: str) -> dict[str, MaskInfo]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw or {}, dict):
        raise ValueError("mask file must be an object of mask entries")
    out = {}
    for kind, body in (raw or {}).items():
        if not isinstance(body, dict):
            raise ValueError(f"mask {kind!r} must be an object, got {type(body).__name__}")
        unknown = sorted(set(body) - MASK_KEYS)
        if unknown:
            raise ValueError(f"mask {kind!r} has unknown field(s) {unknown}; "
                             f"expected some of {sorted(MASK_KEYS)}")
        policy = body.get("execution_policy")
        if policy is not None:
            from onecli.masks.policy import MaskPolicy
            policy = MaskPolicy.parse(policy).as_dict()
        from onecli.context_contract import parse as parse_context
        loop = body.get('loop_policy')
        if loop is not None:
            from onecli.masks.loop import parse
            loop = parse(loop)
        out[kind] = MaskInfo(
            kind=kind,
            preamble=str(body.get("prompts") or body.get("preamble") or ""),
            models=list(body.get("community_models") or []),
            source=f"json:{path}",
            execution_policy=policy,
            loop_policy=loop,
            context_contract=(parse_context(body["context_contract"])
                              if "context_contract" in body else None),
        )
    return out


def load_masks() -> tuple[dict[str, MaskInfo], str | None]:
    """Every mask this host can see, and why the lookup failed if it did."""
    explicit = os.environ.get("ONE_MASKS_JSON")
    if explicit:
        try:
            return _masks_from_json(explicit), None
        except Exception as exc:
            return {}, f"ONE_MASKS_JSON={explicit}: {exc}"

    return {}, "no mask source: set ONE_MASKS_JSON to a mask file"


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

@dataclass
class CheckResult:
    """One cell's verdict: the pass bit, why it failed, and what it scored."""
    ok: bool
    detail: str = ""
    rewards: dict[str, float] = field(default_factory=dict)


def _read_rewards(path: Path) -> tuple[dict[str, float], str]:
    """Named scores a checker left behind, or the reason they are unusable."""
    if not path.is_file():
        return {}, ""
    try:
        body = json.loads(path.read_text())
    except Exception as exc:
        return {}, f"rewards unreadable ({exc})"
    if not isinstance(body, dict):
        return {}, "rewards is not an object"
    out: dict[str, float] = {}
    for key, value in body.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return {}, f"reward {key!r} is not a number"
        out[str(key)] = float(value)
    return out, ""


def aggregate(runs: list[dict[str, float]],
              metrics: dict[str, str] | None = None) -> dict[str, float]:
    """Collapse the rewards of several runs of one task into one row.

    A metric absent from a run is absent, not zero. Scoring a crashed cell as
    a zero-second runtime would make the fastest lane the one that never
    finished.
    """
    metrics = metrics or {}
    names = sorted({name for run in runs for name in run})
    out: dict[str, float] = {}
    for name in names:
        values = [run[name] for run in runs if name in run]
        if not values:
            continue
        how = AGGREGATIONS.get(metrics.get(name, "mean"), statistics.fmean)
        out[name] = float(how(values))
    return out


class CorpusTask(Task):
    """A task whose checker is a script in its own directory."""

    def __init__(self, *, name: str, prompt: str, directory: Path,
                 files: dict[str, str] | None = None,
                 read_only: bool = False, max_steps: int = 20,
                 mask: str = "", gates: dict[str, str] | None = None,
                 origin: str = "", tags: list[str] | None = None,
                 metrics: dict[str, str] | None = None,
                 timeout_sec: float = 60.0):
        super().__init__(name=name, prompt=prompt, files=files or {},
                         read_only=read_only, max_steps=max_steps)
        self.directory = directory
        self.mask = mask
        self.gates = dict(gates or {})
        self.origin = origin
        self.tags = list(tags or [])
        self.metrics = dict(metrics or {})
        self.timeout_sec = float(timeout_sec)

    @property
    def solution(self) -> Path | None:
        """The oracle solution, if this task ships one."""
        path = self.directory / "solve.sh"
        return path if path.is_file() else None

    # -- prompt ---------------------------------------------------------
    def prompt_for(self, masks: dict[str, MaskInfo] | None = None) -> str:
        """The prompt this cell actually sends, mask preamble included."""
        info = (masks or {}).get(self.mask) if self.mask else None
        if info and info.preamble:
            return f"{info.preamble}\n\n{self.prompt}"
        return self.prompt

    # -- workspace ------------------------------------------------------
    def setup(self, workspace: Path) -> None:
        super().setup(workspace)
        seed = self.directory / "files"
        if seed.is_dir():
            shutil.copytree(seed, workspace, dirs_exist_ok=True)

    def grade(self, workspace: Path, answer: str) -> CheckResult:
        """Run the checker and read back everything it scored.

        Exit 0 is still the verdict - one number a run can be ranked on, and
        the thing a regression test needs. Named rewards sit beside it, so a
        task can also say how fast, how small or how clean the result was
        without that judgement collapsing into the pass bit.
        """
        script = self.directory / "check.py"
        (workspace / "_answer.txt").write_text(answer or "")
        rewards_path = workspace / REWARDS_FILE
        if rewards_path.exists():
            rewards_path.unlink()

        env = dict(os.environ)
        env["ONE_REWARDS"] = str(rewards_path)
        env["ONE_ANSWER"] = str(workspace / "_answer.txt")
        # The checker is executed by path, so Python puts the TASK directory on
        # sys.path, not the workspace - and a grader whose whole job is to
        # import what the agent just wrote then cannot see it. cwd is not an
        # import path. Put the workspace first, ahead of anything inherited.
        inherited = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (f"{workspace}{os.pathsep}{inherited}"
                             if inherited else str(workspace))
        proc = _run([sys.executable, str(script)], cwd=workspace,
                    timeout=self.timeout_sec, env=env)
        if proc is None:
            return CheckResult(False, "checker did not run")

        ok = proc.returncode == 0
        detail = ""
        if not ok:
            lines = (proc.stdout or proc.stderr or "").strip().splitlines()
            detail = lines[-1][:120] if lines else f"check rc={proc.returncode}"

        rewards, problem = _read_rewards(rewards_path)
        if problem:
            # A checker that meant to score and could not is a failed cell.
            # Silently dropping it would report the task as merely binary.
            return CheckResult(False, problem)
        rewards["pass"] = 1.0 if ok else 0.0
        return CheckResult(ok, detail, rewards)

    def check(self, workspace: Path, answer: str) -> tuple[bool, str]:
        result = self.grade(workspace, answer)
        return result.ok, result.detail


def _read_task(directory: Path) -> tuple[CorpusTask | None, str | None]:
    manifest = directory / "task.json"
    if not manifest.is_file():
        return None, f"{directory.name}: no task.json"
    if not (directory / "check.py").is_file():
        return None, f"{directory.name}: no check.py"
    try:
        body = json.loads(manifest.read_text())
    except Exception as exc:
        return None, f"{directory.name}: task.json unreadable ({exc})"
    if not isinstance(body, dict):
        return None, f"{directory.name}: task.json is not an object"

    unknown = sorted(set(body) - KNOWN_KEYS)
    if unknown:
        return None, f"{directory.name}: unknown task.json keys {unknown}"
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return None, f"{directory.name}: empty prompt"

    gates = body.get("gates") or {}
    if not isinstance(gates, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in gates.items()):
        return None, f"{directory.name}: gates must be a string->string object"

    metrics = body.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None, f"{directory.name}: metrics must be an object"
    for name, how in metrics.items():
        if how not in AGGREGATIONS:
            return None, (f"{directory.name}: metric {name!r} wants unknown "
                          f"aggregation {how!r}, not one of "
                          f"{sorted(AGGREGATIONS)}")

    return CorpusTask(
        name=directory.name,
        prompt=prompt,
        directory=directory,
        files=body.get("files") or {},
        read_only=bool(body.get("read_only")),
        max_steps=int(body.get("max_steps") or 20),
        mask=str(body.get("mask") or ""),
        gates=gates,
        origin=str(body.get("origin") or ""),
        tags=list(body.get("tags") or []),
        metrics=metrics,
        timeout_sec=float(body.get("timeout_sec") or 60.0),
    ), None


def discover(root: Path | None = None) -> tuple[list[CorpusTask], list[str]]:
    """Every task under `root`, plus one line per directory that is not one."""
    root = Path(root) if root else corpus_root()
    if not root.is_dir():
        return [], []
    tasks, problems = [], []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        task, problem = _read_task(directory)
        if task is not None:
            tasks.append(task)
        elif problem:
            problems.append(problem)
    return tasks, problems


def validate(root: Path | None = None) -> tuple[list[CorpusTask], list[str]]:
    """discover, plus the checks that need a mask source to answer."""
    tasks, problems = discover(root)
    masks, mask_error = load_masks()
    wanted = {t.mask for t in tasks if t.mask}
    if wanted and mask_error:
        problems.append(f"tasks name masks {sorted(wanted)} but {mask_error}")
    else:
        for kind in sorted(wanted - set(masks)):
            named = [t.name for t in tasks if t.mask == kind]
            problems.append(f"unknown mask {kind!r} named by {named}")
    return tasks, problems


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

@dataclass
class OracleResult:
    """Whether a task is worth admitting, and what it did when asked."""
    task: str
    has_solution: bool
    discriminating: bool = False   # fails before anyone touches it
    passable: bool = False         # passes once its own solution runs
    detail: str = ""

    @property
    def admitted(self) -> bool:
        return self.has_solution and self.discriminating and self.passable


def oracle(task: CorpusTask, *, timeout_sec: float = 300.0) -> OracleResult:
    """Prove a task both needs doing and can be done.

    Two runs in a throwaway workspace. First the untouched seed: the checker
    must FAIL, or the task scores every agent that does nothing. Then the
    task's own `solve.sh`: the checker must PASS, or the task is unwinnable
    and every failure it reports is its own.

    This is what makes bulk authoring safe. A task written by a swarm, a hook
    or a stranger is admitted on evidence, not on trust.
    """
    if task.solution is None:
        return OracleResult(task.name, False, detail="no solve.sh")

    with tempfile.TemporaryDirectory(prefix="one-oracle-") as tmp:
        workspace = Path(tmp)
        task.setup(workspace)

        before = task.grade(workspace, "")
        if before.ok:
            return OracleResult(task.name, True, discriminating=False,
                                detail="checker passes an untouched workspace")

        proc = _run(["bash", str(task.solution)], cwd=workspace,
                    timeout=timeout_sec)
        if proc is None:
            return OracleResult(task.name, True, discriminating=True,
                                detail="solve.sh did not finish")
        if proc.returncode != 0:
            lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            why = lines[-1][:120] if lines else f"rc={proc.returncode}"
            return OracleResult(task.name, True, discriminating=True,
                                detail=f"solve.sh failed: {why}")

        after = task.grade(workspace, "")
        if not after.ok:
            return OracleResult(task.name, True, discriminating=True,
                                detail=f"solved workspace still fails: "
                                       f"{after.detail}")
        return OracleResult(task.name, True, discriminating=True, passable=True)


# --------------------------------------------------------------------------
# growth
# --------------------------------------------------------------------------

def add(name: str, prompt: str, check_src: str, *,
        files: dict[str, str] | None = None, mask: str = "",
        gates: dict[str, str] | None = None, origin: str = "",
        tags: list[str] | None = None, metrics: dict[str, str] | None = None,
        timeout_sec: float | None = None, solution: str = "",
        verify: bool = False,
        root: Path | None = None) -> Path:
    """Write a new task into the corpus. This is the promotion path.

    The point of the corpus is that a failure seen in production becomes a
    permanent regression test without anyone editing this repo, so this is
    deliberately callable from a script or a hook, not just by hand.

    `solution` is the oracle. With `verify=True` the task is run through it
    before it is kept, and a task that cannot prove itself is removed rather
    than left in the corpus to score everyone against a bug.
    """
    root = Path(root) if root else corpus_root()
    directory = root / name
    if directory.exists():
        raise FileExistsError(f"task {name!r} already exists at {directory}")
    directory.mkdir(parents=True)

    manifest: dict[str, Any] = {"prompt": prompt}
    if files:
        manifest["files"] = files
    if mask:
        manifest["mask"] = mask
    if gates:
        manifest["gates"] = gates
    if origin:
        manifest["origin"] = origin
    if tags:
        manifest["tags"] = tags
    if metrics:
        manifest["metrics"] = metrics
    if timeout_sec:
        manifest["timeout_sec"] = timeout_sec
    (directory / "task.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "check.py").write_text(check_src)
    if solution:
        (directory / "solve.sh").write_text(solution)

    task, problem = _read_task(directory)
    if task is None:
        shutil.rmtree(directory)
        raise ValueError(f"refusing to add an unloadable task: {problem}")

    if verify:
        result = oracle(task)
        if not result.admitted:
            shutil.rmtree(directory)
            raise ValueError(f"refusing to add an unproven task: "
                             f"{result.detail or 'oracle did not pass'}")
    return directory

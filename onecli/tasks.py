"""Detached offload task manager for one-cli.

Provides non-blocking background task orchestration with live ASCII glyph
telemetry cards, one-cli worker processes, progress signal parsing
($BG_SIGNALS), completion notices, and a context line that tells the model
which tasks are still running so it can keep answering. A request that names
another runner (claude, codex, agy) is recognised and refused: those workers
cannot inherit the session guards.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from onecli import tui, glyph, offload_policy, render, secureio

TASK_DIR = Path.home() / ".one-cli" / "tasks"
SUPPORTED_RUNNERS = ("one", "claude", "codex", "agy")


def resolve_runner(model: str = "", runner: str = "") -> tuple[str, str]:
    """Resolve (runner, clean_model).

    Handles prefixes like 'codex/gpt-6-astra', 'claude/opus', 'agy/gemini-3.8-flash',
    'one/dashscope', or bare runner names like 'claude', 'codex', 'agy'.
    """
    m = (model or "").strip()
    r = (runner or "").strip().lower()

    if "/" in m:
        prefix, rest = m.split("/", 1)
        prefix = prefix.lower()
        if prefix in SUPPORTED_RUNNERS:
            return prefix, rest

    if r in SUPPORTED_RUNNERS:
        return r, m

    if m.lower() in SUPPORTED_RUNNERS:
        return m.lower(), ""

    return "one", m


def parse_detached_spec(text: str) -> dict[str, str]:
    """Parse header directives like 'model: ...', 'runner: ...', 'workspace: ...' from the prompt start."""
    lines = text.strip().splitlines()
    model = ""
    runner = ""
    workspace = ""
    body_lines = []
    header_done = False
    for line in lines:
        if not header_done:
            stripped = line.strip()
            m_match = re.match(r"^(?:--model[=\s]+|model:[ \t]*)([\w./:-]+)$", stripped, re.IGNORECASE)
            r_match = re.match(r"^(?:--runner[=\s]+|runner:[ \t]*)([\w./:-]+)$", stripped, re.IGNORECASE)
            w_match = re.match(r"^(?:--workspace[=\s]+|workspace(?:_override)?:[ \t]*)(.+)$", stripped, re.IGNORECASE)
            if m_match:
                model = m_match.group(1).strip()
                continue
            elif r_match:
                runner = r_match.group(1).strip()
                continue
            elif w_match:
                workspace = w_match.group(1).strip()
                continue
            elif not stripped:
                continue
            else:
                header_done = True
        body_lines.append(line)
    prompt = "\n".join(body_lines).strip()
    return {
        "prompt": prompt,
        "runner": runner,
        "model": model,
        "workspace": workspace,
    }


def mask_code_spans(text: str) -> str:
    """Mask fenced code blocks and inline code spans with spaces to prevent false triggers."""
    if not text or "`" not in text:
        return text
    masked = re.sub(r"```[\s\S]*?```", lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)), text)
    masked = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), masked)
    return masked


def parse_spawn_args(arg: str) -> tuple[str, str, str, str]:
    """Parse a '/task spawn' line for optional --runner, -m, and -w flags.

    Returns (prompt, runner, model, workspace).
    """
    spec = parse_detached_spec(arg)
    arg_body = spec["prompt"]
    runner = spec["runner"]
    model = spec["model"]
    workspace = spec["workspace"]

    tokens = shlex.split(arg_body) if arg_body else []
    rest_tokens = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--runner", "-r") and i + 1 < len(tokens):
            runner = tokens[i + 1]
            i += 2
        elif tok in ("--model", "-m") and i + 1 < len(tokens):
            model = tokens[i + 1]
            i += 2
        elif tok in ("--workspace", "-w") and i + 1 < len(tokens):
            workspace = tokens[i + 1]
            i += 2
        else:
            rest_tokens.append(tok)
            i += 1
    prompt = " ".join(rest_tokens).strip()
    runner, model = resolve_runner(model=model, runner=runner)
    return prompt, runner, model, workspace


def build_runner_cmd(
    runner: str,
    prompt: str,
    model: str = "",
    workspace: str = "",
    prompt_file: Path | None = None,
    permission_mode: str = "default",
    worker_args: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Build (argv, needs_stdin_prompt) for the specified runner."""
    ws = workspace or os.getcwd()
    if runner != "one":
        raise ValueError("This runner cannot inherit the session guards; use runner: one")

    repo_root = Path(__file__).resolve().parent.parent
    one_script = repo_root / "scripts" / "one"
    if one_script.exists():
        entry = [sys.executable, str(one_script)]
    else:
        # The package's __main__; `onecli.cli` has no main guard, so `-m
        # onecli.cli` imports it and exits 0 having done nothing.
        entry = [sys.executable, "-m", "onecli"]

    actual_model = model or ""
    cmd = [
        *entry,
        prompt,
        *(["-m", actual_model] if actual_model else []),
        "--json",
        "--no-ask",
        "--no-fallback",
        "--permission-mode", permission_mode,
        *(worker_args or []),
    ]
    if ws:
        cmd.extend(["-w", str(ws)])
    return cmd, False

@dataclass
class DetachedTask:
    task_id: str
    title: str
    prompt: str = ""
    runner: str = "one"
    model: str = ""
    workspace: str = ""
    phases: list[str] = field(default_factory=lambda: ["Plan", "Execute", "Verify", "Done"])
    current_phase: int = 1
    status: str = "running"  # running, done, failed
    announcement: str = ""
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    glyph_step: int = 0
    proc: subprocess.Popen | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    signals_path: Path | None = None
    prompt_path: Path | None = None
    _signals_offset: int = 0
    exit_code: int | None = None
    result: dict | None = None
    result_text: str = ""
    error: str = ""
    announced: bool = False

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.monotonic()
        return max(0.0, end - self.started_at)

    @property
    def total_phases(self) -> int:
        return len(self.phases)

    @property
    def phase_name(self) -> str:
        if 1 <= self.current_phase <= len(self.phases):
            return self.phases[self.current_phase - 1]
        return "Working"

    def advance_phase(self, next_name: str | None = None) -> None:
        if next_name and next_name not in self.phases and len(self.phases) < 64:
            self.phases.append(next_name)
        if self.current_phase < len(self.phases):
            self.current_phase += 1

    def complete(self, final_announcement: str = "Complete") -> None:
        self.status = "done"
        self.current_phase = len(self.phases)
        self.announcement = final_announcement
        self.finished_at = time.monotonic()

    def fail(self, reason: str) -> None:
        self.status = "failed"
        self.announcement = f"error: {reason}"
        self.error = reason
        self.finished_at = time.monotonic()

    def cancel(self, reason: str = "cancelled") -> None:
        if self.proc is not None:
            from onecli.jobs import stop_process
            stop_process(self.proc, self.proc.pid)
            self.exit_code = self.proc.returncode
        self.status = "failed"
        self.announcement = f"cancelled: {reason}"
        self.error = f"cancelled: {reason}"
        self.finished_at = time.monotonic()

    def output(self, lines: int = 50) -> str:
        parts = []
        if self.result_text:
            parts.append(f"Result:\n{self.result_text}")
        elif self.stdout_path and self.stdout_path.exists():
            try:
                content = self.stdout_path.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    tail = "\n".join(content.splitlines()[-lines:])
                    parts.append(f"Stdout (last {lines} lines):\n{tail}")
            except Exception as e:
                parts.append(f"Error reading stdout: {e}")
        if self.stderr_path and self.stderr_path.exists():
            try:
                content = self.stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    tail = "\n".join(content.splitlines()[-lines:])
                    parts.append(f"Stderr (last {lines} lines):\n{tail}")
            except Exception as e:
                parts.append(f"Error reading stderr: {e}")
        return "\n\n".join(parts) if parts else "No output recorded yet."

    def poll(self) -> bool:
        if self.status != "running":
            return False

        if self.signals_path and self.signals_path.exists():
            try:
                with open(self.signals_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(self._signals_offset)
                    new_lines = f.readlines()
                    self._signals_offset = f.tell()
                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("PHASES:"):
                        try:
                            requested = int(line.split(":", 1)[1].strip())
                            count = min(64, max(0, requested))
                            if requested > 64:
                                self.announcement = f"PHASES: {requested} capped at 64"
                            if count > len(self.phases):
                                curr = len(self.phases)
                                self.phases.extend([f"Step {i+1}" for i in range(curr, count)])
                        except ValueError:
                            pass
                    elif line.startswith("PHASE:"):
                        phase_text = line.split(":", 1)[1].strip()
                        self.advance_phase(phase_text)
                        self.announcement = phase_text
                    elif line.startswith("MILESTONE:"):
                        milestone_text = line.split(":", 1)[1].strip()
                        self.announcement = milestone_text
            except Exception:
                pass

        if self.proc is None:
            return False

        code = self.proc.poll()
        if code is None:
            return False

        self.exit_code = code
        self.finished_at = time.monotonic()
        if self.stdout_path and self.stdout_path.exists():
            try:
                content = self.stdout_path.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    try:
                        self.result = json.loads(content)
                    except Exception:
                        idx = content.rfind("\n{")
                        if idx != -1:
                            try:
                                self.result = json.loads(content[idx + 1:])
                            except Exception:
                                pass
                    if isinstance(self.result, dict):
                        self.result_text = self.result.get("text", "")
                        if self.result.get("error"):
                            self.error = str(self.result.get("error"))
                    else:
                        self.result_text = content
            except Exception:
                pass

        if self.stderr_path and self.stderr_path.exists():
            try:
                err_content = self.stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                if err_content and (code != 0 or (self.result and self.result.get("stopped") == "error")):
                    lines = [line for line in err_content.splitlines() if not line.strip().startswith('{"e":')]
                    self.error = "\n".join(lines).strip() or err_content[-300:]
            except Exception:
                pass

        if code == 0 and not self.error and not self.result_text.strip():
            self.error = "Worker exited without a result; completion is unverified"
        if code == 0 and not self.error:
            final_note = self.result_text[:80] if self.result_text else (self.announcement or "Task completed")
            self.complete(final_announcement=final_note)
        else:
            self.fail(reason=self.error or f"Process exited with code {code}")
        return True


class TaskManager:
    """Manages concurrent detached tasks in one-cli interactive chat."""

    def __init__(
        self,
        execute: bool = False,
        workspace: str = "",
        default_model: str = "",
        root: Path | str | None = None,
        allow_assistant: bool = False,
        permission_mode: str = "default",
        worker_args: list[str] | None = None,
        roots: list[str] | None = None,
        refusal: str = "",
    ) -> None:
        self.allow_assistant = allow_assistant
        self.refusal = refusal
        self.permission_mode = permission_mode
        self.worker_args = list(worker_args or [])
        self.roots = [Path(p).expanduser().resolve() for p in (roots or [workspace or os.getcwd()])]
        self._tasks: dict[str, DetachedTask] = {}
        self._next_id: int = 1
        self.execute = execute
        self.workspace = workspace
        self.default_model = default_model
        self.root = Path(root or TASK_DIR)

    def spawn(
        self,
        title: str,
        phases: list[str] | None = None,
        model: str = "",
        runner: str = "",
        announcement: str = "Task initialized",
        prompt: str = "",
        workspace: str = "",
        execute: bool | None = None,
        cmd: list[str] | None = None,
    ) -> DetachedTask:
        if os.environ.get("ONE_DETACHED_OWNER") == "caller":
            raise RuntimeError(
                "ONE_DETACHED_OWNER=caller: the parent process owns background "
                "jobs; return the task to it instead of starting a local worker.")
        t_phases = list(phases) if phases else ["Analyze", "Execute", "Verify", "Complete"]
        actual_workspace = workspace or self.workspace or os.getcwd()
        actual_workspace = str(Path(actual_workspace).resolve())
        if not any(Path(actual_workspace) == r or r in Path(actual_workspace).parents for r in self.roots):
            raise ValueError("Detached workspace is outside the session roots")
        if self.worker_args and Path(actual_workspace) != Path(self.workspace or os.getcwd()).resolve():
            raise ValueError("Guarded workers must use the session workspace to preserve relative rules")
        actual_prompt = prompt or title

        actual_runner, actual_model = resolve_runner(model=model or self.default_model, runner=runner)

        should_exec = execute if execute is not None else (self.execute or (cmd is not None))
        if should_exec and self.refusal:
            raise RuntimeError(self.refusal)

        if should_exec and cmd is None:
            full_cmd, needs_stdin = build_runner_cmd(
                runner=actual_runner, prompt=offload_policy.INSTRUCTIONS + "\n\nASSIGNED TASK\n" + actual_prompt,
                model=actual_model, workspace=actual_workspace,
                permission_mode=self.permission_mode, worker_args=self.worker_args)

        launch_prompt = offload_policy.INSTRUCTIONS + "\n\nASSIGNED TASK\n" + actual_prompt
        while True:
            tid = str(self._next_id)
            self._next_id += 1
            if not should_exec:
                break
            # Any retained artifact reserves the ID. Exclusive prompt creation
            # also serializes competing managers without a second registry.
            if any(os.path.lexists(self.root / f"task_{tid}.{suffix}")
                   for suffix in ("out", "err", "signals", "prompt")):
                continue
            prompt_file = self.root / f"task_{tid}.prompt"
            try:
                with secureio.text_file(prompt_file, "x", private=True) as fh:
                    fh.write(launch_prompt)
                break
            except FileExistsError:
                continue

        task = DetachedTask(
            task_id=tid,
            title=title,
            prompt=actual_prompt,
            runner=actual_runner,
            model=actual_model,
            workspace=actual_workspace,
            phases=t_phases,
            current_phase=1,
            status="running",
            announcement=announcement,
        )

        if should_exec:
            self.root.mkdir(parents=True, exist_ok=True)
            stdout_file = self.root / f"task_{tid}.out"
            stderr_file = self.root / f"task_{tid}.err"
            signals_file = self.root / f"task_{tid}.signals"
            prompt_file = self.root / f"task_{tid}.prompt"

            task.stdout_path = stdout_file
            task.stderr_path = stderr_file
            task.signals_path = signals_file
            task.prompt_path = prompt_file

            if cmd is not None:
                full_cmd, needs_stdin = list(cmd), False

            env = dict(os.environ)
            env["BG_SIGNALS"] = str(signals_file)
            # `python -m` puts the cwd first on sys.path, so a worker started
            # in the workspace would import a planted argparse.py or onecli/
            # before any permission mode, rule, guard or hook applies. Start
            # it in the trusted package root (as harnesses._one_build does);
            # the explicit -w still roots every tool in the workspace. A
            # caller's own cmd keeps the workspace as its cwd.
            env["PYTHONSAFEPATH"] = "1"

            out_fh = open(stdout_file, "wb")
            err_fh = open(stderr_file, "wb")
            stdin_fh = open(prompt_file, "rb") if needs_stdin else subprocess.DEVNULL
            try:
                task.proc = subprocess.Popen(
                    full_cmd,
                    stdout=out_fh,
                    stderr=err_fh,
                    stdin=stdin_fh,
                    cwd=actual_workspace if cmd is not None else str(Path(__file__).resolve().parent.parent),
                    env=env,
                    start_new_session=True,
                )
            except OSError as exc:
                task.fail(f"Could not start {actual_runner}: {exc}")
                prompt_file.unlink(missing_ok=True)
            finally:
                out_fh.close()
                err_fh.close()
                if needs_stdin and hasattr(stdin_fh, "close"):
                    stdin_fh.close()

        self._tasks[tid] = task
        from onecli import terminal
        if terminal.ACTIVE is not None:
            terminal.ACTIVE.track_task(task)
        return task

    def poll(self) -> list[DetachedTask]:
        finished = []
        for task in self._tasks.values():
            if task.status == "running":
                if task.poll():
                    finished.append(task)
        return finished

    def notices(self) -> list[str]:
        self.poll()
        out: list[str] = []
        for task in self._tasks.values():
            if task.status in ("done", "failed") and not task.announced:
                task.announced = True
                status_str = "completed" if task.status == "done" else "failed"
                runner_str = f" [{task.runner.upper()}]" if task.runner != "one" else ""
                out.append(
                    f"[detached task #{task.task_id}{runner_str}] {status_str} in {task.elapsed:.1f}s: {task.title}"
                )
        return out

    def cancel(self, task_id: str, reason: str = "cancelled") -> bool:
        t = self.get(task_id)
        if not t:
            return False
        t.cancel(reason)
        return True

    def close(self) -> list[str]:
        stopped = []
        for task in self.list_active():
            if task.proc and task.proc.poll() is None:
                task.cancel("Session closed")
                stopped.append(f"task #{task.task_id}")
        return stopped

    def get(self, task_id: str) -> DetachedTask | None:
        t = self._tasks.get(task_id)
        if t and t.status == "running":
            t.poll()
        return t

    def list_all(self) -> list[DetachedTask]:
        self.poll()
        return list(self._tasks.values())

    def list_active(self) -> list[DetachedTask]:
        self.poll()
        return [t for t in self._tasks.values() if t.status == "running"]

    def context_prompt(self) -> str:
        active = self.list_active()
        if not active:
            return ""
        lines = [f"[Background Tasks: {len(active)} active]"]
        for t in active:
            runner_tag = f" ({t.runner.upper()})" if t.runner != "one" else ""
            lines.append(
                f"- TASK #{t.task_id}{runner_tag}: {t.title} (Phase {t.current_phase}/{t.total_phases}: {t.phase_name}, {t.elapsed:.1f}s)"
            )
        lines.append("You are aware of these background offloads. Answer conversationally while they run.")
        return "\n".join(lines)

    def render_card(self, task_id: str, excite: bool = False, width: int = 40) -> str:
        task = self.get(task_id)
        if not task:
            return ""
        task.glyph_step += 1
        glyph_rows = glyph.compact_card_glyph(task.glyph_step, excite=excite)
        runner_tag = f" [{task.runner.upper()}]" if task.runner != "one" else ""
        return tui.render_task_card(
            task_id=f"{task.task_id}{runner_tag}",
            # Title, phase and announcement can carry model or worker text.
            title=render.untrusted(task.title),
            phase_num=task.current_phase,
            total_phases=task.total_phases,
            phase_name=render.untrusted(task.phase_name),
            status=task.status.upper(),
            glyph_rows=glyph_rows,
            elapsed=task.elapsed,
            announcement=render.untrusted(task.announcement),
            width=width,
        )


def session_options(args, mode: str) -> dict:
    """Serialize the human-selected guard configuration for a one worker."""
    worker_args = []
    for attr, flag in (("guard", "--guard"), ("allowed_tools", "--allowed-tools"),
                       ("disallowed_tools", "--disallowed-tools"), ("add_dir", "--add-dir")):
        values = getattr(args, attr, None) or []
        for value in values:
            worker_args.extend([flag, os.path.abspath(os.path.expanduser(value))
                                if attr == "add_dir" else value])
    timeout = getattr(args, "guard_timeout", None)
    if timeout is not None and getattr(args, "guard", None):
        worker_args.extend(["--guard-timeout", str(timeout)])
    # The session's hook policy (hooks.from_args): the same named files, the
    # same project trust, or none at all. A named source the worker cannot
    # reopen (missing, a process-substitution pipe) refuses the spawn rather
    # than start a worker with less PreToolUse policy than the session.
    refusal = ""
    if getattr(args, "no_hooks", False):
        worker_args.append("--no-hooks")
    else:
        for value in getattr(args, "hooks", None) or []:
            path = os.path.abspath(value)
            if not os.path.isfile(path):
                refusal = (f"cannot forward --hooks {value} to a detached worker: "
                           "not a regular file it can reopen")
            worker_args.extend(["--hooks", path])
        if getattr(args, "trust_project_config", False):
            worker_args.append("--trust-project-config")
    workspace = getattr(args, "workspace", "") or os.getcwd()
    return {"allow_assistant": bool(getattr(args, "allow_detached", False)),
            "permission_mode": mode, "worker_args": worker_args,
            "roots": [workspace, *(getattr(args, "add_dir", None) or [])],
            "refusal": refusal}

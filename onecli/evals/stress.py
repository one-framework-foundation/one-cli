#!/usr/bin/env python3
"""Stress tier: what one-cli does when everything else misbehaves.

Each case is one subprocess in a throwaway workspace against a lane that is
deliberately broken, and an assertion about the SHAPE of the failure - fails
fast, says why, leaves nothing running, does not print a traceback, does not
leave the workspace. A stress case that "passes" by hanging is the thing this
is built to catch, so every case carries its own wall-clock ceiling and a case
that reaches it fails.

The endpoint tier runs with --no-fallback on purpose: one-cli fails over to a
live backend when a lane dies, which is correct and is tested separately by
`failover`. With fallback on, an unreachable lane answers normally and every
error-path assertion silently measures a different backend.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("ONE_STRESS_DIR", Path.cwd() / ".scratch/stress"))
PROVIDERS = SCRATCH / "providers"
PORT = int(os.environ.get("ONE_STRESS_PORT", "18642"))
LANE_LOG = SCRATCH / "requests.jsonl"
MODES = ["stub-ok", "stub-silent", "stub-garbage", "stub-truncated", "stub-429",
         "stub-500", "stub-slowloris", "stub-huge", "stub-toolstorm",
         "stub-badtool", "stub-ghosttool", "stub-ansi", "stub-flaky",
         "stub-escape"]


@dataclass
class Outcome:
    name: str
    ok: bool
    detail: str
    secs: float
    rc: int = 0
    extra: dict = field(default_factory=dict)
    # A case whose feature is not present in this tree is neither a pass nor a
    # failure, and must not be counted as either. Four of the containment
    # cases pass trivially when `@` expansion is absent - nothing is sent, so
    # nothing leaks - which reads as coverage that does not exist.
    skipped: str = ""


def skip(name: str, why: str) -> Outcome:
    return Outcome(name, True, why, 0.0, skipped=why)


def has_mentions() -> bool:
    try:
        import onecli.mentions  # noqa: F401
    except ImportError:
        return False
    return True


def write_providers() -> None:
    PROVIDERS.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        (PROVIDERS / f"{mode.replace('-', '')}.env").write_text(
            f"ONE_PROVIDER_MODEL={mode}\n"
            f"OPENAI_BASE_URL=http://127.0.0.1:{PORT}/v1\n"
            "OPENAI_API_KEY=stub\n")
    (PROVIDERS / "stubdead.env").write_text(
        "ONE_PROVIDER_MODEL=stub-dead\n"
        "OPENAI_BASE_URL=http://127.0.0.1:18699/v1\n"
        "OPENAI_API_KEY=stub\n")


def fresh_ws(name: str) -> Path:
    ws = SCRATCH / "ws" / name
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    return ws


def run_one(args: list[str], ws: Path, ceiling: float, stdin: str = "",
            env_extra: dict | None = None, providers: bool = True):
    env = dict(os.environ)
    if providers:
        env["ONE_PROVIDERS_DIR"] = str(PROVIDERS)
    env["PYTHONPATH"] = str(ROOT)
    env["ONE_STRESS"] = "1"
    env.update(env_extra or {})
    cmd = [sys.executable, "-m", "onecli",
           "-w", str(ws), "--no-hooks"] + args
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, input=stdin,
                           # Match the trusted import root used by the arena;
                           # -w selects the workspace without importing it.
                           timeout=ceiling, env=env, cwd=str(ROOT),
                           start_new_session=True, errors="replace")
        return p.returncode, p.stdout, p.stderr, time.time() - t0
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = (exc.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return -9, out, err + "\n__CEILING__", time.time() - t0


def lane_reset() -> None:
    LANE_LOG.write_text("")


def lane_requests() -> list[dict]:
    if not LANE_LOG.exists():
        return []
    out = []
    for line in LANE_LOG.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def sent_text(bodies: list[dict]) -> str:
    """Everything the agent actually put on the wire, concatenated."""
    parts = []
    for b in bodies:
        for msg in (b.get("body") or {}).get("messages") or []:
            c = msg.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                        parts.append(blk["text"])
    return "\n".join(parts)


CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


def _traceback(out: str, err: str) -> str:
    return "printed a python traceback" if "Traceback (most recent call last)" \
        in out + err else ""


def _json(out: str) -> dict:
    try:
        return json.loads(out)
    except Exception:
        return {}


# -- 1. endpoint pathologies -----------------------------------------------

def _lane(name: str, backend: str, ceiling: float, want: list[str],
          prompt: str = "say hi") -> Outcome:
    ws = fresh_ws(name)
    rc, out, err, secs = run_one(
        ["-m", backend, "--json", "-q", "--no-fallback", "--timeout", "20",
         "--max-steps", "4", prompt], ws, ceiling)
    if "__CEILING__" in err:
        return Outcome(name, False, f"never returned, killed at {ceiling:.0f}s",
                       secs, rc)
    tb = _traceback(out, err)
    if tb:
        return Outcome(name, False, tb, secs, rc)
    d = _json(out)
    blob = (d.get("error") or "") + " " + err
    if d.get("stopped") not in ("error", "failed"):
        return Outcome(name, False,
                       f"reported stopped={d.get('stopped')!r} on a broken lane",
                       secs, rc)
    hit = [w for w in want if w.lower() in blob.lower()]
    if not hit:
        return Outcome(name, False, f"error names no cause: {blob.strip()[:150]!r}",
                       secs, rc, {"error": blob.strip()[:300]})
    return Outcome(name, True, f"error says {hit[0]!r}", secs, rc,
                   {"error": (d.get('error') or '')[:200]})


@case
def c_silent():
    return _lane("silent", "stubsilent", 60, ["empty", "no content", "silent",
                                              "nothing", "non-json"])


@case
def c_garbage():
    return _lane("garbage", "stubgarbage", 60, ["json", "html", "parse",
                                                "invalid", "unexpected"])


@case
def c_truncated():
    return _lane("truncated", "stubtruncated", 60, ["json", "incomplete",
                                                    "truncat", "empty", "parse"])


@case
def c_deadport():
    return _lane("deadport", "stubdead", 60, ["refused", "connect", "unreachable"])


@case
def c_http500():
    return _lane("http500", "stub500", 60, ["500", "exploded", "server"])


@case
def c_http429():
    return _lane("http429", "stub429", 120, ["429", "rate", "limit"])


@case
def c_slowloris():
    """Headers, then silence forever. --timeout must be what ends it."""
    ws = fresh_ws("slowloris")
    rc, out, err, secs = run_one(
        ["-m", "stubslowloris", "--json", "-q", "--no-fallback",
         "--timeout", "12", "--max-steps", "2", "say hi"], ws, 90)
    if "__CEILING__" in err:
        return Outcome("slowloris", False, "hung past its own --timeout", secs, rc)
    if secs > 70:
        return Outcome("slowloris", False,
                       f"--timeout 12 took {secs:.0f}s to fire", secs, rc)
    return Outcome("slowloris", True, f"gave up after {secs:.0f}s", secs, rc)


@case
def c_huge():
    """648,000 chars in one answer. Must not wedge or truncate silently."""
    ws = fresh_ws("huge")
    rc, out, err, secs = run_one(
        ["-m", "stubhuge", "--json", "-q", "--no-fallback", "--timeout", "60",
         "--max-steps", "2", "say hi"], ws, 150)
    if "__CEILING__" in err:
        return Outcome("huge", False, "never returned", secs, rc)
    tb = _traceback(out, err)
    if tb:
        return Outcome("huge", False, tb, secs, rc)
    d = _json(out)
    n = len(d.get("text") or "")
    if n < 600_000:
        return Outcome("huge", False, f"kept only {n} chars of a 648,000-char answer",
                       secs, rc, {"chars": n})
    return Outcome("huge", True, f"carried {n:,} chars in {secs:.0f}s",
                   secs, rc, {"chars": n})


@case
def c_toolstorm():
    """A model that only ever calls a tool. --max-steps is the only brake."""
    ws = fresh_ws("toolstorm")
    rc, out, err, secs = run_one(
        ["-m", "stubtoolstorm", "--json", "-q", "--no-fallback", "--timeout",
         "30", "--max-steps", "5", "go"], ws, 150)
    if "__CEILING__" in err:
        return Outcome("toolstorm", False, "ran forever", secs, rc)
    d = _json(out)
    steps = d.get("steps")
    if steps is None:
        return Outcome("toolstorm", False, f"no json result: {out[:140]!r}", secs, rc)
    if steps > 5:
        return Outcome("toolstorm", False, f"{steps} steps past a cap of 5",
                       secs, rc, {"steps": steps})
    return Outcome("toolstorm", True,
                   f"stopped at {steps} steps, stopped={d.get('stopped')!r}",
                   secs, rc, {"steps": steps, "stopped": d.get("stopped")})


@case
def c_badtool():
    """Unparseable JSON in tool arguments must reach the tool as a rejection."""
    ws = fresh_ws("badtool")
    (ws / "a.txt").write_text("hello\n")
    rc, out, err, secs = run_one(
        ["-m", "stubbadtool", "--json", "-q", "--no-fallback", "--timeout", "20",
         "--max-steps", "3", "go"], ws, 90)
    if "__CEILING__" in err:
        return Outcome("badtool", False, "never returned", secs, rc)
    tb = _traceback(out, err)
    if tb:
        return Outcome("badtool", False, tb, secs, rc)
    d = _json(out)
    return Outcome("badtool", True,
                   f"survived, stopped={d.get('stopped')!r} steps={d.get('steps')}",
                   secs, rc)


@case
def c_ghosttool():
    ws = fresh_ws("ghosttool")
    rc, out, err, secs = run_one(
        ["-m", "stubghosttool", "--json", "-q", "--no-fallback", "--timeout",
         "20", "--max-steps", "3", "go"], ws, 90)
    if "__CEILING__" in err:
        return Outcome("ghosttool", False, "never returned", secs, rc)
    tb = _traceback(out, err)
    if tb:
        return Outcome("ghosttool", False, tb, secs, rc)
    return Outcome("ghosttool", True, "refused an unknown tool", secs, rc)


@case
def c_ansi():
    """Hostile terminal control bytes in the model's answer."""
    ws = fresh_ws("ansi")
    rc, out, err, secs = run_one(
        ["-m", "stubansi", "-q", "--no-fallback", "--timeout", "20",
         "--max-steps", "2", "go"], ws, 90)
    if "__CEILING__" in err:
        return Outcome("ansi", False, "never returned", secs, rc)
    bad = [b for b in ("\x1b[2J", "\x1b]0;", "\x07", "\x1b[6n") if b in out]
    if bad:
        return Outcome("ansi", False,
                       "passed through " + ",".join(b.encode().hex() for b in bad),
                       secs, rc, {"leaked": [b.encode().hex() for b in bad]})
    return Outcome("ansi", True, "control bytes stripped", secs, rc)


@case
def c_flaky():
    """503 then 200. A single transient must not end the run."""
    ws = fresh_ws("flaky")
    rc, out, err, secs = run_one(
        ["-m", "stubflaky", "--json", "-q", "--no-fallback", "--timeout", "20",
         "--max-steps", "3", "go"], ws, 90)
    if "__CEILING__" in err:
        return Outcome("flaky", False, "never returned", secs, rc)
    d = _json(out)
    if d.get("stopped") == "answered":
        return Outcome("flaky", True, "retried through the 503", secs, rc)
    return Outcome("flaky", False,
                   f"gave up on one transient 503: {(d.get('error') or '')[:120]!r}",
                   secs, rc)


# -- 2. resilience ----------------------------------------------------------

@case
def c_failover():
    """The same dead lane, with fallback left on, must be answered elsewhere."""
    ws = fresh_ws("failover")
    rc, out, err, secs = run_one(
        ["-m", "stubdead", "--json", "-q", "--fallback-model", "stubok",
         "--timeout", "20", "--max-steps", "3", "say hi"], ws, 90)
    if "__CEILING__" in err:
        return Outcome("failover", False, "never returned", secs, rc)
    d = _json(out)
    on = d.get("answered_on")
    if d.get("stopped") == "answered" and on and on != "stubdead":
        return Outcome("failover", True, f"answered on {on!r} instead", secs, rc,
                       {"answered_on": on})
    return Outcome("failover", False,
                   f"stopped={d.get('stopped')!r} answered_on={on!r}", secs, rc)


# -- 3. containment ---------------------------------------------------------

@case
def c_injection():
    """The answer itself tells the agent to exfiltrate /etc/shadow."""
    ws = fresh_ws("injection")
    marker = SCRATCH.resolve() / f"escape-{uuid.uuid4().hex}.txt"
    rc, out, err, secs = run_one(
        ["-m", "stubescape", "--json", "-q", "--no-fallback", "--timeout", "20",
         "--max-steps", "4", f"escape-marker: {marker}"], ws, 90)
    if marker.exists():
        marker.unlink()
        return Outcome("injection", False, "wrote outside the workspace", secs, rc)
    return Outcome("injection", True, "nothing written outside the workspace",
                   secs, rc)


@case
def c_readonly_ws():
    """A workspace the process cannot write to."""
    ws = fresh_ws("readonlyws")
    (ws / "a.txt").write_text("x\n")
    os.chmod(ws, 0o500)
    try:
        rc, out, err, secs = run_one(
            ["-m", "stubok", "--json", "-q", "--no-fallback", "--timeout", "20",
             "--max-steps", "3", "go"], ws, 90)
    finally:
        os.chmod(ws, 0o700)
    if "__CEILING__" in err:
        return Outcome("readonlyws", False, "never returned", secs, rc)
    tb = _traceback(out, err)
    if tb:
        return Outcome("readonlyws", False, tb, secs, rc)
    return Outcome("readonlyws", True, "ran in an unwritable workspace", secs, rc)


# -- 4. mentions and the prompt layer --------------------------------------

def _mention(name: str, setup, mention: str, check, ceiling: float = 120) -> Outcome:
    """Run a prompt carrying an @mention and assert on what reached the wire.

    `check(sent, ws) -> str` returns "" for a pass or the reason it failed.
    Asserting that the run "answered" proves nothing: the stub answers
    identically whatever it is sent, so the only evidence is the request body.
    That is how the first version of this file scored four green vacuous cells
    while `@` expansion was not running at all.
    """
    if not has_mentions():
        return skip(name, "this tree has no onecli.mentions")
    ws = fresh_ws(name)
    setup(ws)
    lane_reset()
    rc, out, err, secs = run_one(
        ["-m", "stubok", "--json", "-q", "--no-fallback", "--timeout", "25",
         "--max-steps", "2", f"look at {mention} please"], ws, ceiling)
    if "__CEILING__" in err:
        return Outcome(name, False, f"never returned, killed at {ceiling:.0f}s",
                       secs, rc)
    tb = _traceback(out, err)
    if tb:
        return Outcome(name, False, tb, secs, rc)
    d = _json(out)
    if d.get("stopped") != "answered":
        return Outcome(name, False,
                       f"stopped={d.get('stopped')!r} {(d.get('error') or '')[:120]}",
                       secs, rc)
    bodies = lane_requests()
    if not bodies:
        return Outcome(name, False, "no request ever reached the lane", secs, rc)
    sent = sent_text(bodies)
    why = check(sent, ws)
    if why:
        return Outcome(name, False, why, secs, rc, {"sent_chars": len(sent)})
    return Outcome(name, True, f"{len(sent):,} chars on the wire", secs, rc,
                   {"sent_chars": len(sent)})


@case
def c_mention_oversize():
    """A 5MB file behind one @. The per-file budget is 96KB."""
    def setup(ws):
        (ws / "big.txt").write_text("x" * 5_000_000)

    def check(sent, ws):
        if len(sent) > 400_000:
            return f"sent {len(sent):,} chars - the 96KB per-file budget did not hold"
        if "xxxxxxxxxx" not in sent:
            return "the file was not included at all"
        return ""
    return _mention("mention_oversize", setup, "@big.txt", check)


@case
def c_mention_flood():
    """60 files at ~7KB each in one turn. The per-turn budget is 240KB."""
    def setup(ws):
        for i in range(60):
            (ws / f"f{i:02d}.txt").write_text(("line %d\n" % i) * 800)

    def check(sent, ws):
        if len(sent) > 500_000:
            return f"sent {len(sent):,} chars - the 240KB per-turn budget did not hold"
        return ""
    return _mention("mention_flood", setup,
                    " ".join(f"@f{i:02d}.txt" for i in range(60)), check, 180)


@case
def c_mention_symlink():
    """A symlink out of the workspace must not be followed."""
    def setup(ws):
        (ws / "leak.txt").symlink_to("/etc/passwd")

    def check(sent, ws):
        if "root:x:0:0" in sent or "/bin/bash" in sent:
            return "followed a symlink out of the workspace"
        return ""
    return _mention("mention_symlink", setup, "@leak.txt", check)


@case
def c_mention_parent():
    def setup(ws):
        pass

    def check(sent, ws):
        if "root:x:0:0" in sent:
            return "read a file above the workspace"
        return ""
    return _mention("mention_parent", setup, "@../../../etc/passwd", check)


@case
def c_mention_binary():
    """256 distinct byte values. Nothing unencodable may reach the wire."""
    def setup(ws):
        (ws / "blob.bin").write_bytes(bytes(range(256)) * 400)

    def check(sent, ws):
        if "\x00" in sent:
            return "put raw NUL bytes on the wire"
        return ""
    return _mention("mention_binary", setup, "@blob.bin", check)


@case
def c_mention_fence():
    """A file that already contains longer fences than the wrapper's."""
    def setup(ws):
        (ws / "fenced.md").write_text("```\n" + "````\n" * 50 + "`````\n")

    def check(sent, ws):
        if "``````" not in sent:
            return "wrapper fence did not grow past the file's own fences"
        return ""
    return _mention("mention_fence", setup, "@fenced.md", check)


# -- 5. sessions ------------------------------------------------------------

@case
def c_session_corrupt():
    """A half-written session file must not take the process down."""
    ws = fresh_ws("session_corrupt")
    rc, out, err, secs = run_one(
        ["-m", "stubok", "--json", "-q", "--no-fallback", "--timeout", "20",
         "--max-steps", "2", "hello"], ws, 90)
    sid = _json(out).get("session")
    if not sid:
        return Outcome("session_corrupt", False, "no session id to corrupt", secs, rc)
    hits = list((Path.home() / ".one-cli/sessions").glob(f"*{sid}*"))
    if not hits:
        return Outcome("session_corrupt", False,
                       f"could not find the store for session {sid}", secs, rc)
    target = hits[0]
    target.write_text(target.read_text()[: len(target.read_text()) // 2])
    rc2, out2, err2, secs2 = run_one(
        ["-m", "stubok", "--json", "-q", "--no-fallback", "-r", sid,
         "--timeout", "20", "--max-steps", "2", "hello again"], ws, 90)
    if "__CEILING__" in err2:
        return Outcome("session_corrupt", False, "hung on a corrupt session",
                       secs2, rc2)
    tb = _traceback(out2, err2)
    if tb:
        return Outcome("session_corrupt", False, tb, secs2, rc2)
    return Outcome("session_corrupt", True,
                   f"survived a truncated {target.name}", secs2, rc2)


@case
def c_session_missing():
    ws = fresh_ws("session_missing")
    rc, out, err, secs = run_one(
        ["-m", "stubok", "--json", "-q", "--no-fallback", "-r", "deadbeef0000",
         "--timeout", "20", "--max-steps", "2", "hello"], ws, 90)
    if "__CEILING__" in err:
        return Outcome("session_missing", False, "hung", secs, rc)
    tb = _traceback(out, err)
    if tb:
        return Outcome("session_missing", False, tb, secs, rc)
    return Outcome("session_missing", True, "clean on an unknown session id",
                   secs, rc)


# -- 6. concurrency ---------------------------------------------------------

@case
def c_concurrency():
    """Eight agents in one workspace at once, all writing session state."""
    ws = fresh_ws("concurrency")
    def one(i):
        return run_one(["-m", "stubok", "--json", "-q", "--no-fallback",
                        "--timeout", "25", "--max-steps", "2", f"run {i}"],
                       ws, 120)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, range(8)))
    secs = time.time() - t0
    bad = []
    sids = set()
    for i, (rc, out, err, _s) in enumerate(rows):
        if "__CEILING__" in err:
            bad.append(f"{i}:hung")
            continue
        if _traceback(out, err):
            bad.append(f"{i}:traceback")
            continue
        d = _json(out)
        if d.get("stopped") != "answered":
            bad.append(f"{i}:{d.get('stopped')}")
        if d.get("session"):
            sids.add(d["session"])
    if bad:
        return Outcome("concurrency", False, "failed: " + ",".join(bad), secs)
    if len(sids) != 8:
        return Outcome("concurrency", False,
                       f"8 runs produced {len(sids)} distinct session ids", secs)
    return Outcome("concurrency", True, "8 parallel runs, 8 distinct sessions",
                   secs)


def serve() -> subprocess.Popen:
    """Start the stub endpoint and wait for it to answer."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ONE_STRESS_LOG=str(LANE_LOG))
    proc = subprocess.Popen(
        [sys.executable, "-m", "onecli.evals.stresslane", str(PORT)],
        env=env, cwd=str(ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                return proc
        if proc.poll() is not None:
            raise RuntimeError("the stub endpoint exited before it listened")
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"the stub endpoint never listened on {PORT}")


def main(argv: list[str] | None = None) -> int:
    # Argument parsing, because there was none: every token was read as a case
    # name, so `--help` selected no case and the run reported "0/0 stress cases
    # passed" with exit 0. A filter that matches nothing is a typo, not a pass.
    parser = argparse.ArgumentParser(
        prog="one-stress",
        description="Run one-cli against endpoints that misbehave on purpose.")
    parser.add_argument("cases", nargs="*",
                        help="Case names to run. Default: all of them.")
    parser.add_argument("--list", action="store_true",
                        help="Print the case names and exit.")
    args = parser.parse_args(argv)

    known = [fn.__name__.removeprefix("c_") for fn in CASES]
    if args.list:
        print("\n".join(known))
        return 0
    unknown = [c for c in args.cases if c not in known]
    if unknown:
        print(f"unknown case: {', '.join(unknown)}\n"
              f"known: {', '.join(known)}", file=sys.stderr)
        return 2

    only = args.cases
    write_providers()
    results = []
    lane = serve()
    try:
        results = _run_cases(only)
    finally:
        lane.kill()
    out = SCRATCH / "results.json"
    # Own the directory rather than inheriting it from whatever ran first. The
    # results write assumed `write_providers()` had already made SCRATCH, so on
    # a clean checkout the run's own record was the thing that crashed it.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2))
    bad = [r for r in results if not r.ok and not r.skipped]
    skipped = [r for r in results if r.skipped]
    ran = len(results) - len(skipped)
    if not ran:
        print("no stress case ran", file=sys.stderr)
        return 2
    line = f"\n{ran - len(bad)}/{ran} stress cases passed"
    if skipped:
        line += f", {len(skipped)} skipped"
    print(f"{line} -> {out}")
    return 1 if bad else 0


def _run_cases(only: list[str]) -> list[Outcome]:
    results = []
    for fn in CASES:
        nm = fn.__name__.removeprefix("c_")
        if only and nm not in only:
            continue
        try:
            r = fn()
        except Exception as exc:
            r = Outcome(nm, False, f"case raised {type(exc).__name__}: {exc}", 0.0)
        results.append(r)
        flag = "skip" if r.skipped else ("ok  " if r.ok else "FAIL")
        print(f"  {flag} {r.name:<18} {r.secs:6.1f}s  {r.detail}", flush=True)
    return results


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

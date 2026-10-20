#!/usr/bin/env python3
"""Rank wasted tool calls across recent one-cli sessions. Read-only.

    python3 scripts/session_waste.py [--days 14] [--budget 240] [--json]
                                     [--examples 3] [--dir ~/.one-cli/sessions]

Each session transcript is rebuilt from its `messages` / `messages_delta`
records (the same rule as `onecli.session.load_messages`), split into tool
rounds, and checked for the patterns below. A pattern's score is the tool
calls it wasted: calls a correct run would not have made. The scan reads the
newest files first and stops at the time budget, reporting how far it got, so
an over-budget run is a partial ranking and says so.

Patterns (keyed on tool results, never on the model that produced them):

  reread           a file read again with no write or edit to it in between
  repeat_failure   a call whose result is the same error as an earlier call
                   (same tool, same first error line), after the first retry
  missing_path     consecutive not-found results from read/ls/glob/grep -
                   guessing paths instead of listing once
  credential_fail  calls made after a tool result reported rejected or
                   missing credentials, in the same run
  long_search      rounds past the discovery limit with no progress tool and
                   no progress tool (the low-progress checkpoint's territory); shell
                   rounds that only test, lint, build or drive git/CI are work,
                   not search, and are not counted
  proposal_end     a run that made no progress tool and ended on an offer
                   ("Want me to...", "I can...") instead of acting; counts the
                   discovery calls it spent
  premature_stop   a run that edited files and then ended on a question
                   while its last check was failing

The analyzer never prints message text: only session ids, counts and tool
names, so its report can be pasted anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROGRESS = {"write", "edit", "ask_user", "task", "job_start"}
READS = {"read"}
LOOKUPS = {"read", "ls", "glob", "grep"}
DISCOVERY_LIMIT = 6
_MISSING = re.compile(r"(no such file|not found|does not exist|no matches|"
                      r"is not a directory|outside the workspace)", re.I)
_ERROR = re.compile(r"^(error|exit [1-9]\d*|traceback|refused|denied|"
                    r"\[error\]|timed out)", re.I)
_CRED = re.compile(r"(401 unauthorized|403 forbidden|invalid api key|"
                   r"api key (?:expired|invalid|missing)|credentials_(?:missing|rejected)|"
                   r"authentication failed|incorrect api key|token (?:expired|invalid))",
                   re.I)
_OFFER = re.compile(r"(want me to|shall i|should i|would you like|i can (?:now|also|then)?\s*\w+|"
                    r"let me know if|next step(?:s)? (?:would|is|are))", re.I)
# Shell work that is not a search: tests, lint, builds, packaging, git and CI.
_SHELL_WORK = re.compile(r"(?:^|[;&|(]\s*|\bcd\s+\S+\s*&&\s*)(?:\S*/)?(?:python3?\s+-m\s+)?"
                         r"(?:pytest|ruff|mypy|tox|make|cargo|npm|pnpm|yarn|go\s+test|"
                         r"pip3?\s+install|uv\s+(?:run|pip|sync)|gh\s+(?:run|pr|workflow)|"
                         r"git\s+(?:status|diff|log|add|commit|push|fetch|pull|rebase|"
                         r"merge|checkout|switch|stash|show))\b")


def load(path: Path) -> list[dict]:
    messages: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") == "messages" and isinstance(rec.get("messages"), list):
                messages = list(rec["messages"])
            elif (rec.get("kind") == "messages_delta" and rec.get("start") == len(messages)
                  and isinstance(rec.get("messages"), list)):
                messages.extend(rec["messages"])
    return messages


def rounds(messages: list[dict]) -> list[list[tuple[str, dict, str]]]:
    """Tool rounds of (name, arguments, output), in order."""
    out: list[list[tuple[str, dict, str]]] = []
    results: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "tool":
            results[str(m.get("tool_call_id"))] = str(m.get("content") or "")
    for m in messages:
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if not calls:
            continue
        current = []
        for c in calls:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            current.append((str(fn.get("name") or ""), args, results.get(str(c.get("id")), "")))
        out.append(current)
    return out


def _path(args: dict) -> str:
    return str(args.get("path") or args.get("file_path") or args.get("file") or "")


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0][:160] if text.strip() else ""


def signature(name: str, output: str) -> str:
    """What an error result says, independent of the arguments that earned it.
    A shell failure is named by its last line (the message), not `exit 1`;
    numbers and paths are folded so the same refusal on two paths matches."""
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    text = lines[-1] if re.match(r"exit \d+$", lines[0]) and len(lines) > 1 else lines[0]
    text = re.sub(r"(/[\w.@+-]+)+", "PATH", text)
    text = re.sub(r"\d+", "N", text)
    return f"{name}:{text[:120]}"


def analyze(messages: list[dict]) -> dict[str, int]:
    """Wasted calls per pattern for one transcript."""
    waste: dict[str, int] = defaultdict(int)
    tool_rounds = rounds(messages)
    if not tool_rounds:
        return {}
    since_progress = 0
    read_since_write: dict[str, set[str]] = defaultdict(set)
    errors: dict[str, int] = defaultdict(int)
    missing_run = 0
    cred_failed = False
    made_progress = False
    discovery_calls = 0
    last_failed = False
    for rnd in tool_rounds:
        names = {n for n, _, _ in rnd}
        progress = bool(names & PROGRESS)
        shell_work = names == {"bash"} and all(
            _SHELL_WORK.search(str(a.get("command", ""))) for _, a, _ in rnd)
        if cred_failed:
            waste["credential_fail"] += len(rnd)
        if progress:
            made_progress = True
            since_progress = 0
        elif shell_work:
            pass
        else:
            since_progress += 1
            discovery_calls += len(rnd)
            if since_progress > DISCOVERY_LIMIT:
                waste["long_search"] += len(rnd)
        for name, args, output in rnd:
            path = _path(args)
            if name in ("write", "edit") and path:
                read_since_write.pop(path, None)
            if name in READS and path:
                span = f"{args.get('offset', '')}:{args.get('limit', '')}"
                if span in read_since_write[path]:
                    waste["reread"] += 1
                read_since_write[path].add(span)
            head = _first_line(output)
            failed = bool(_ERROR.match(head))
            if name in LOOKUPS and _MISSING.search(head):
                missing_run += 1
                if missing_run >= 2:
                    waste["missing_path"] += 1
            elif name in LOOKUPS:
                missing_run = 0
            if failed:
                key2 = signature(name, output)
                errors[key2] += 1
                if errors[key2] >= 3:
                    waste["repeat_failure"] += 1
            if name == "bash":
                last_failed = failed
            if failed and _CRED.search(output[:4000]):
                cred_failed = True
    final = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            final = str(m.get("content") or "")
            break
    tail = final.strip()[-400:]
    if not made_progress and discovery_calls >= 4 and _OFFER.search(tail):
        waste["proposal_end"] += discovery_calls
    if made_progress and last_failed and tail.endswith("?"):
        waste["premature_stop"] += 1
    return dict(waste)


def scan(directory: Path, days: float, budget: float, examples: int) -> dict:
    start = time.monotonic()
    cutoff = time.time() - days * 86400
    files = []
    with os.scandir(directory) as it:
        for e in it:
            if e.name.endswith(".jsonl"):
                st = e.stat()
                if st.st_mtime >= cutoff:
                    files.append((st.st_mtime, e.path))
    files.sort(reverse=True)
    totals: dict[str, dict] = defaultdict(lambda: {"sessions": 0, "calls": 0, "worst": []})
    scanned = with_tools = all_calls = skipped = 0
    for _, path in files:
        if time.monotonic() - start > budget:
            break
        scanned += 1
        try:
            with open(path, "rb") as fh:
                if b'"tool_calls"' not in fh.read():
                    continue
            messages = load(Path(path))
            rs = rounds(messages)
            found = analyze(messages)
        except Exception:  # one foreign or corrupt transcript must not end the report
            skipped += 1
            continue
        if not rs:
            continue
        with_tools += 1
        all_calls += sum(len(r) for r in rs)
        for pattern, n in found.items():
            if n <= 0:
                continue
            t = totals[pattern]
            t["sessions"] += 1
            t["calls"] += n
            t["worst"].append((n, Path(path).stem))
    ranked = []
    for pattern, t in sorted(totals.items(), key=lambda kv: -kv[1]["calls"]):
        worst = sorted(t["worst"], reverse=True)[:examples]
        ranked.append({"pattern": pattern, "sessions": t["sessions"], "calls_wasted": t["calls"],
                       "worst": [{"session": s, "calls": n} for n, s in worst]})
    return {"files_in_window": len(files), "scanned": scanned, "with_tools": with_tools,
            "tool_calls": all_calls, "skipped": skipped, "complete": scanned == len(files),
            "seconds": round(time.monotonic() - start, 1), "ranked": ranked}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(Path.home() / ".one-cli" / "sessions"))
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--budget", type=float, default=240, help="seconds before stopping")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    directory = Path(a.dir).expanduser()
    if not directory.is_dir():
        print(f"no session directory at {directory}", file=sys.stderr)
        return 2
    report = scan(directory, a.days, a.budget, a.examples)
    if a.json:
        print(json.dumps(report, indent=2))
        return 0
    state = "complete" if report["complete"] else "PARTIAL (budget reached)"
    print(f"scanned {report['scanned']}/{report['files_in_window']} sessions "
          f"({state}, {report['seconds']}s); {report['with_tools']} used tools, "
          f"{report['tool_calls']} tool calls; {report['skipped']} unreadable skipped")
    print(f"{'pattern':18} {'sessions':>8} {'wasted':>7}  worst")
    for r in report["ranked"]:
        worst = ", ".join(f"{w['session']}({w['calls']})" for w in r["worst"])
        print(f"{r['pattern']:18} {r['sessions']:>8} {r['calls_wasted']:>7}  {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

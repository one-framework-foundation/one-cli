"""Session persistence.

One JSONL per session under ~/.one-cli/sessions. The transcript is most of the
state: replaying it is how `--resume` works, and it is also what a later run
reads to compare two backends on the same conversation. The rest is state the
run keeps beside the transcript - the plan (`onecli.todo`) is the first of it -
written here as its own record kind so a resume restores it without the model
having to reconstruct it from what it said.

Attachments round-trip as a reference, not as bytes: a canonical media block
holds an absolute path plus the sniffed mime, and the wire re-reads the file at
send time. Inlining base64 would put megabytes into a file that is rewritten on
every turn and re-read in full on every `--resume`, for data the sender already
has on disk. The cost is that moving or deleting an attached file breaks a
resume - it surfaces as `cannot read image <path>`, which names the file.
"""

from __future__ import annotations

import json
import copy
import fcntl
import os
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from onecli import secureio

SESSION_DIR = Path.home() / ".one-cli" / "sessions"
_message_cache: OrderedDict[Path, tuple[tuple[int, int, int, int], list[dict]]] = OrderedDict()


def _stamp(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def path_for(session_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise ValueError("invalid session id")
    return SESSION_DIR / f"{session_id}.jsonl"


def append(session_id: str, record: dict, *, durable: bool = False) -> None:
    path = path_for(session_id)
    with secureio.text_file(path, "a", private=True) as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        info = os.fstat(fh.fileno())
        prior = _message_cache.get(path)
        # Any external write invalidates the cached transcript, even when our
        # next append only changes metadata or todos.
        if prior and prior[0] != _stamp(info):
            _message_cache.pop(path, None)
            prior = None
        if record.get("kind") == "messages":
            current = record.get("messages")
            if (isinstance(current, list) and prior
                    and len(current) >= len(prior[1])
                    and current[:len(prior[1])] == prior[1]):
                record = {"kind": "messages_delta", "version": 1,
                          "start": len(prior[1]), "messages": current[len(prior[1]):]}
        # A prior process may have died during a write. Keep its partial line
        # separate so the new complete checkpoint remains readable.
        if info.st_size and os.pread(fh.fileno(), 1, info.st_size - 1) != b"\n":
            fh.write("\n")
        fh.write(json.dumps({"ts": time.time(), **record}, default=str) + "\n")
        fh.flush()
        if durable:
            os.fsync(fh.fileno())
            if info.st_size == 0:
                with secureio.directory(path.parent) as parent_fd:
                    os.fsync(parent_fd)
        if record.get("kind") in ("messages", "messages_delta"):
            previous = prior[1] if prior and record["kind"] == "messages_delta" else []
            messages = previous + record["messages"]
            _message_cache[path] = (_stamp(os.fstat(fh.fileno())), copy.deepcopy(messages))
            _message_cache.move_to_end(path)
            if len(_message_cache) > 16:
                _message_cache.popitem(last=False)
        elif prior:
            _message_cache[path] = (_stamp(os.fstat(fh.fileno())), prior[1])


def load_messages(session_id: str) -> list[dict]:
    p = path_for(session_id)
    if not p.exists():
        raise FileNotFoundError(f"no session {session_id} at {p}")
    messages: list[dict] = []
    with secureio.text_file(p) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") == "messages" and isinstance(rec.get("messages"), list):
                messages = rec["messages"]
            elif (rec.get("kind") == "messages_delta" and rec.get("version") == 1
                  and rec.get("start") == len(messages)
                  and isinstance(rec.get("messages"), list)):
                messages.extend(rec["messages"])
    return messages


def load_records(session_id: str, kind: str) -> list[dict]:
    """Every record of one kind, oldest first.

    The transcript is append-only, so state written more than once - a plan
    revised eight times - is eight records and the last one is current. Reading
    them all rather than only the last is what lets a caller see the history of
    a piece of state as well as its value.
    """
    p = path_for(session_id)
    if not p.exists():
        raise FileNotFoundError(f"no session {session_id} at {p}")
    out: list[dict] = []
    messages: list[dict] = []
    with secureio.text_file(p) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") == "messages" and isinstance(rec.get("messages"), list):
                messages = rec["messages"]
            elif (rec.get("kind") == "messages_delta" and rec.get("version") == 1
                  and rec.get("start") == len(messages)
                  and isinstance(rec.get("messages"), list)):
                messages = messages + rec["messages"]
                if kind == "messages":
                    out.append({"ts": rec.get("ts"), "kind": "messages",
                                "messages": messages})
            if rec.get("kind") == kind:
                out.append(rec)
    return out


def latest_id() -> str | None:
    if not SESSION_DIR.is_dir():
        return None
    files = sorted(SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1].stem if files else None


def opening_line(session_id: str) -> str:
    """The first thing that was asked in a session.

    Lives here rather than beside one of the two listings that want it. An id,
    a date and a byte count cannot tell you which of yesterday's twelve
    sessions is the one you meant, and a second copy of this walk in `cli.py`
    would be a second answer to the same question.
    """
    try:
        messages = load_messages(session_id)
    except (FileNotFoundError, OSError):
        return ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if not isinstance(content, list):
                return ""
            return " ".join(b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text")
    return ""


def backend_of(session_id: str) -> str:
    """The backend the session last ran on, or "".

    Read off the `run` records the CLI already writes beside the transcript.
    A resume that lands on a different model than the conversation was held
    with is a surprise worth naming, and this is the only place that knows.
    """
    try:
        runs = load_records(session_id, "run")
    except (FileNotFoundError, OSError):
        return ""
    for rec in reversed(runs):
        name = rec.get("backend")
        if name:
            return str(name)
    return ""


def listing(limit: int = 20) -> list[tuple[str, float, int]]:
    if not SESSION_DIR.is_dir():
        return []
    files = sorted(SESSION_DIR.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p.stem, p.stat().st_mtime, p.stat().st_size) for p in files[:limit]]


def metadata(session_id: str) -> dict:
    result = {}
    for record in load_records(session_id, "metadata"):
        result.update({k: v for k, v in record.items() if k not in ("kind", "ts")})
    return result


def rename(session_id: str, title: str) -> None:
    if not title.strip():
        raise ValueError("session name cannot be empty")
    metadata(session_id)  # Require an existing session.
    append(session_id, {"kind": "metadata", "title": title.strip()})


def fork(session_id: str, title: str = "") -> str:
    messages = load_messages(session_id)
    meta = metadata(session_id)
    target = new_id()
    append(target, {"kind": "metadata", **meta, "parent": session_id,
                    "title": title.strip() or (meta.get("title", session_id) + " (fork)")})
    append(target, {"kind": "messages", "messages": messages})
    for kind in ("todos",):
        records = load_records(session_id, kind)
        if records:
            append(target, records[-1])
    return target


def search(query: str = "", workspace: str | None = None, limit: int = 20) -> list[dict]:
    """Filter before limiting; old transcripts without workspace remain searchable."""
    wanted = str(Path(workspace).expanduser().resolve()) if workspace else None
    rows: list = []
    if not SESSION_DIR.is_dir():
        return rows
    for path in sorted(SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
                       reverse=True):
        try:
            meta = metadata(path.stem)
            opening = opening_line(path.stem)
            if wanted and meta.get("workspace") != wanted:
                continue
            if query.casefold() not in " ".join((path.stem, meta.get("title", ""), opening)).casefold():
                continue
            stat = path.stat()
            rows.append({"session": path.stem, "modified_epoch": stat.st_mtime,
                         "bytes": stat.st_size, "opening_line": opening,
                         "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)), **meta})
        except (OSError, ValueError):
            continue
        if len(rows) >= limit:
            break
    return rows


def save_draft(session_id: str, text: str) -> None:
    import os
    path = path_for(session_id).with_suffix(".draft.json")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    try:
        with secureio.text_file(temporary, "x", private=True) as fh:
            json.dump({"text": text}, fh)
        with secureio.directory(path.parent, private=True) as parent:
            os.replace(temporary.name, path.name, src_dir_fd=parent, dst_dir_fd=parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_draft(session_id: str) -> str:
    try:
        with secureio.text_file(path_for(session_id).with_suffix(".draft.json")) as fh:
            value = json.load(fh)
        if not isinstance(value, dict):
            return ""
        return value["text"] if isinstance(value.get("text"), str) else ""
    except (OSError, ValueError, TypeError):
        return ""

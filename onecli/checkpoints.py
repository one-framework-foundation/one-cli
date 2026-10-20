"""Turn checkpoints and workspace instant rollback (/undo, /diff).

Maintains lightweight workspace snapshots per interactive turn:
- In git repositories: uses temporary isolated git index objects
  without modifying HEAD, touching branches, or dirtying git status.
- Supports instant rollback (/undo) to revert filesystem changes made in turns.
- Supports visual diff (/diff) against turn snapshots.
- Supports non-git workspaces via file backups in ~/.one-cli/checkpoints/<sid>/,
  bounded by MAX_FILES/MAX_BYTES and pruned to the last KEEP_LAST.

Every turn records a slot (`begin_turn`), whether or not its checkpoint could be
taken, so /undo maps to exactly one turn: a turn whose checkpoint is missing
refuses instead of rolling back to an older one and deleting earlier work.
"""

from __future__ import annotations

import atexit
import fnmatch
import hashlib
import json
import os
import weakref
import shutil
import stat
import sys
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Checkpoint:
    id: int
    timestamp: float
    prompt: str
    commit_sha: str | None = None
    tree_sha: str | None = None
    backup_dir: str | None = None
    files_snapshot: dict[str, str] = field(default_factory=dict)
    file_hashes: dict[str, str] = field(default_factory=dict)
    index_snapshot: bytes | None = None
    complete: bool = False
    # Every path on disk at capture, ignored files included: undo never
    # unlinks one of these, whatever the turn did to the ignore rules.
    pre_turn_files: set[str] = field(default_factory=set)


@dataclass
class TurnSlot:
    """One turn: the checkpoint taken before it, or why there is none.

    `edits` is False for a turn that could not touch the workspace, which
    /undo drops without a rollback. `at` is
    the conversation length when the turn began - /undo truncates to it, since
    one turn can hold several user-role messages (interrupt notes, nudges) -
    and `mark` fingerprints the messages before it, so a conversation rewritten
    since (compacted, resumed) refuses instead of cutting in the wrong place.
    """
    checkpoint: Checkpoint | None = None
    note: str = ""
    edits: bool = True
    at: int = 0
    mark: str = ""


def _mark(messages: list, at: int) -> str:
    if at <= 0 or len(messages) < at:
        return ""
    last = json.dumps(messages[at - 1], sort_keys=True, default=str)
    return f"{at}:{hashlib.sha256(last.encode()).hexdigest()}"


# Non-git checkpoints copy the workspace. Past either ceiling the checkpoint is
# refused with a note rather than copying a home directory every turn.
MAX_FILES = 5000
MAX_BYTES = 200 * 1024 * 1024
# Checkpoints kept per session; older ones are deleted and their turns refuse /undo.
KEEP_LAST = 20
# Overridable root for the non-git backups (tests point it at a tmp dir).
ROOT: str | None = None
# A session directory whose owner is gone, or untouched this long, is swept.
STALE_SECONDS = 7 * 24 * 3600
_OWNER = "owner.pid"
_swept: set[str] = set()


_CHECKPOINT_IDENTITY = {
    "GIT_AUTHOR_NAME": "one-cli checkpoint", "GIT_AUTHOR_EMAIL": "checkpoint@one-cli.invalid",
    "GIT_COMMITTER_NAME": "one-cli checkpoint", "GIT_COMMITTER_EMAIL": "checkpoint@one-cli.invalid",
}


def backup_root() -> str:
    return ROOT or os.path.expanduser("~/.one-cli/checkpoints")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_session_dir(root: str, path: str) -> None:
    """Delete our snapshot-* dirs and owner file under one session dir, then it."""
    if os.path.islink(path) or os.path.dirname(os.path.realpath(path)) != os.path.realpath(root):
        return
    for name in os.listdir(path):
        child = os.path.join(path, name)
        if name.startswith("snapshot-") and not os.path.islink(child) and os.path.isdir(child):
            shutil.rmtree(child, ignore_errors=True)
        elif name == _OWNER and not os.path.islink(child):
            os.unlink(child)
    try:
        os.rmdir(path)
    except OSError:
        pass


def sweep(keep: str = "", now: float | None = None) -> list[str]:
    """Remove other sessions' backups left by a crash or kill: owner process
    gone (or unrecorded), or untouched for STALE_SECONDS. Returns swept ids."""
    root = backup_root()
    now = time.time() if now is None else now
    swept: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return swept
    for sid in names:
        path = os.path.join(root, sid)
        if sid == keep or os.path.islink(path) or not os.path.isdir(path):
            continue
        try:
            with open(os.path.join(path, _OWNER)) as fh:
                pid = int(fh.read().strip())
        except (OSError, ValueError):
            pid = 0
        try:
            age = now - os.stat(path).st_mtime
        except OSError:
            continue
        if pid and _alive(pid) and age <= STALE_SECONDS:
            continue
        _remove_session_dir(root, path)
        swept.append(sid)
    return swept


def _close_at_exit(ref) -> None:
    manager = ref()
    if manager is not None:
        manager.close()


class CheckpointRefused(Exception):
    """A checkpoint that was deliberately not taken (size ceiling, unsupported root)."""


def _secret_name(path: str) -> bool:
    name = Path(path).name.lower()
    return any(fnmatch.fnmatchcase(name, pattern)
               for pattern in (".env", ".env.*", "*.pem", "*.key", "id_*"))


_EXCLUDED = {".git", ".one-cli", "node_modules", ".venv", "__pycache__"}


def _walk_files(workspace: str):
    """One scope for capture and undo; unreadable subtrees invalidate capture.

    Only regular files and symlinks are in scope. FIFOs, sockets and devices
    cannot be copied and are never ours to delete, so both sides skip them.
    """
    def fail(error):
        raise error

    for root, dirs, files in os.walk(workspace, onerror=fail):
        dirs[:] = [name for name in dirs if name not in _EXCLUDED]
        links = [name for name in dirs if os.path.islink(os.path.join(root, name))]
        dirs[:] = [name for name in dirs if name not in links]
        for name in files + links:
            if name in _EXCLUDED:
                continue
            full = os.path.join(root, name)
            mode = os.lstat(full).st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                continue
            yield os.path.relpath(full, workspace), full


def _digest(path: str) -> str:
    if os.path.islink(path):
        return "link:" + os.readlink(path)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RollbackResult:
    success: bool
    checkpoint_id: int
    prompt: str
    reverted_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    error: str | None = None
    # Turns this undo covered, and where to truncate the conversation.
    turns: int = 0
    cut: int | None = None


class CheckpointManager:
    """Manages workspace snapshots and rollbacks for a session."""

    def __init__(self, workspace: str, sid: str = ""):
        source = Path(os.path.abspath(workspace))
        self.workspace = os.path.realpath(source)
        # Remember both the supplied path and its resolved target: initially
        # symlinked workspaces are allowed, but retargeting either path is not.
        resolved = Path(self.workspace)
        self._root_paths = sorted({source, *source.parents, resolved, *resolved.parents})
        try:
            self._root_identity = self._workspace_identity()
        except OSError:
            self._root_identity = None
        self.sid = sid or uuid.uuid4().hex[:12]
        self.checkpoints: list[Checkpoint] = []
        self.turns: list[TurnSlot] = []
        self._seq = 0
        # Why the last create_checkpoint returned None, for the caller to show.
        self.last_error = ""
        self._owned = False
        self.is_git, self.git_root = self._check_git()
        if backup_root() not in _swept:
            _swept.add(backup_root())
            try:
                sweep(keep=self.sid)
            except OSError:
                pass  # housekeeping must never block a session

    def _workspace_identity(self):
        identity = []
        for path in self._root_paths:
            info = path.lstat()
            target = os.readlink(path) if stat.S_ISLNK(info.st_mode) else None
            identity.append((info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), target))
        if not os.path.isdir(self.workspace):
            raise OSError("checkpoint workspace is not a directory")
        return identity

    def _validate_workspace(self):
        if self._root_identity is None or self._workspace_identity() != self._root_identity:
            raise ValueError("checkpoint workspace or ancestor was replaced")

    def _check_git(self) -> tuple[bool, str | None]:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return True, res.stdout.strip()
        except OSError:
            pass
        return False, None

    def create_checkpoint(self, prompt: str = "") -> Checkpoint | None:
        """Capture a non-Git workspace or a complete Git repository root.

        Git subdirectories are unsupported: filesystem-only undo cannot restore
        their staging state, and whole-index undo would affect sibling work.
        Returns None on failure with the reason in `last_error`.
        """
        self._seq += 1
        return self._create(prompt, self._seq)

    def _create(self, prompt: str, cp_id: int) -> Checkpoint | None:
        self.last_error = ""
        try:
            cp = self._capture(prompt, cp_id)
        except CheckpointRefused as exc:
            self.last_error = str(exc)
            return None
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never raised into a turn
            self.last_error = f"checkpoint failed: {type(exc).__name__}: {exc}"
            return None
        self.checkpoints.append(cp)
        self._prune()
        return cp

    def _capture(self, prompt: str, cp_id: int) -> Checkpoint:
        self._validate_workspace()
        if self.is_git and self.workspace != os.path.realpath(self.git_root or ""):
            raise CheckpointRefused("workspace is a git subdirectory; checkpoints need the repository root")
        now = time.time()
        title = (prompt.strip().splitlines()[0][:80] if prompt else f"turn_{cp_id}")

        if self.is_git:
            index_path = None
            temp_dir = tempfile.TemporaryDirectory(prefix="one-checkpoint-")
            temp_idx = os.path.join(temp_dir.name, "index")
            env = dict(os.environ, GIT_INDEX_FILE=temp_idx)
            try:
                index_path = self._index_path()
                index_snapshot = index_path.read_bytes() if index_path.exists() else None
                if index_snapshot is not None:
                    Path(temp_idx).write_bytes(index_snapshot)
                # Preserve flags in the real-index snapshot, but never let
                # them hide existing worktree edits from the capture index.
                tracked = subprocess.check_output(
                    ["git", "ls-files", "--cached", "-z"], cwd=self.workspace, env=env,
                    stderr=subprocess.DEVNULL,
                )
                for flag in ("--no-assume-unchanged", "--no-skip-worktree"):
                    subprocess.run(
                        ["git", "update-index", flag, "-z", "--stdin"],
                        input=tracked, cwd=self.workspace, env=env, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                # Never write untracked secrets into Git's object database.
                # Explicit literal pathspecs preserve arbitrary filenames.
                untracked = subprocess.check_output(
                    ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                    cwd=self.workspace, env=env,
                ).decode(errors="surrogateescape").split("\0")
                exclusions = [f":(exclude,literal){name}" for name in untracked
                              if name and _secret_name(name)]
                for name in untracked:
                    if name and _secret_name(name):
                        print(f"checkpoint: skipping untracked secret file: {name!r}", file=sys.stderr)
                # Names only, never contents: tracked, untracked and ignored
                # paths that undo must leave in place.
                listed = subprocess.check_output(
                    ["git", "ls-files", "-z", "--cached"], cwd=self.workspace, env=env,
                ) + b"\0" + subprocess.check_output(
                    ["git", "ls-files", "-z", "--others", "--ignored", "--exclude-standard"],
                    cwd=self.workspace, env=env,
                )
                pre_turn = {name for name in listed.decode(errors="surrogateescape").split("\0")
                            if name}
                pre_turn.update(name for name in untracked if name)
                subprocess.run(
                    ["git", "add", "-A", "--", ".", *exclusions],
                    cwd=self.workspace,
                    env=env,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                tree_sha = subprocess.check_output(
                    ["git", "write-tree"],
                    cwd=self.workspace,
                    env=env,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()

                cmd = ["git", "commit-tree", tree_sha, "-m", f"checkpoint:{title}"]
                try:
                    head_sha = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=self.workspace,
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                    cmd.extend(["-p", head_sha])
                except subprocess.CalledProcessError:
                    pass

                # A fixed identity: the commit is an internal, unreferenced
                # object, and a repo with no user.name/email configured must
                # still be checkpointed (commit-tree refuses otherwise).
                commit_sha = subprocess.check_output(
                    cmd,
                    cwd=self.workspace,
                    env=dict(os.environ, **_CHECKPOINT_IDENTITY),
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                self._validate_workspace()

                # A filesystem fallback cannot preserve the Git index
                # contract, so any failure above propagates to the caller.
                return Checkpoint(
                    id=cp_id,
                    timestamp=now,
                    prompt=title,
                    commit_sha=commit_sha,
                    tree_sha=tree_sha,
                    index_snapshot=index_snapshot,
                    pre_turn_files=pre_turn,
                    complete=True,
                )
            finally:
                temp_dir.cleanup()

        # Non-git snapshot fallback. Measure first: nothing is copied for a
        # workspace over the ceiling.
        count = size = 0
        for _rel, full in _walk_files(self.workspace):
            count += 1
            info = os.lstat(full)
            size += info.st_size if stat.S_ISREG(info.st_mode) else 0
            if count > MAX_FILES or size > MAX_BYTES:
                raise CheckpointRefused(
                    f"workspace too large to checkpoint (over {MAX_FILES} files or "
                    f"{MAX_BYTES // (1024 * 1024)} MB); /undo is unavailable for this turn")
        backup_base = None
        files_snap = {}
        hashes = {}
        try:
            session_root = self._session_root()
            os.makedirs(session_root, mode=0o700, exist_ok=True)
            if not self._owned:
                # Liveness for other processes' sweep, and cleanup if this
                # process ends without a clean close.
                with open(os.path.join(session_root, _OWNER), "w") as fh:
                    fh.write(str(os.getpid()))
                atexit.register(_close_at_exit, weakref.ref(self))
                self._owned = True
            backup_base = tempfile.mkdtemp(prefix="snapshot-", dir=session_root)
            for rel, full in _walk_files(self.workspace):
                dest = os.path.join(backup_base, rel)
                os.makedirs(os.path.dirname(dest), mode=0o700, exist_ok=True)
                shutil.copy2(full, dest, follow_symlinks=False)
                hashes[rel] = _digest(dest)
                files_snap[rel] = dest
            self._validate_workspace()
        except BaseException:
            if backup_base is not None:
                self._delete_backup(backup_base)
            raise

        return Checkpoint(
            id=cp_id,
            timestamp=now,
            prompt=title,
            backup_dir=backup_base,
            files_snapshot=files_snap,
            file_hashes=hashes,
            complete=True,
        )

    # -- per-turn records and cleanup ---------------------------------------

    def begin_turn(self, prompt: str, messages: list, edits: bool = True) -> str | None:
        """Record a turn about to be appended to `messages`, checkpointing
        first when it may edit the workspace.

        Returns a note for the user when the checkpoint could not be taken.
        """
        self._seq += 1
        at = len(messages)
        if not edits:
            self.turns.append(TurnSlot(edits=False, at=at,
                                       mark=_mark(messages, at)))
            return None
        cp = self._create(prompt, self._seq)
        self.turns.append(TurnSlot(checkpoint=cp, note=self.last_error,
                                   at=at, mark=_mark(messages, at)))
        return self.last_error or None

    def turn_start(self, messages: list) -> int | None:
        """Where the last recorded turn begins in `messages`, if still valid."""
        if not self.turns:
            return None
        slot = self.turns[-1]
        return slot.at if self._matches(slot, messages) else None

    @staticmethod
    def _matches(slot: TurnSlot, messages: list) -> bool:
        return len(messages) >= slot.at and _mark(messages, slot.at) == slot.mark

    def forget_turn(self) -> None:
        """Drop the last turn's record without restoring (its exchange was
        dropped); its checkpoint goes too, so /undo <id> cannot reach it."""
        if not self.turns:
            return
        cp = self.turns.pop().checkpoint
        if cp is not None and cp in self.checkpoints:
            self.checkpoints.remove(cp)
            if cp.backup_dir:
                self._delete_backup(cp.backup_dir)

    def reset_turns(self) -> None:
        """A new or replaced conversation: its turns have no checkpoints, and
        the old checkpoints go, so /undo <id> cannot roll back the workspace
        without the conversation that made it."""
        for cp in self.checkpoints:
            if cp.backup_dir:
                self._delete_backup(cp.backup_dir)
        self.checkpoints = []
        self.turns = []

    def undo_turn(self, messages: list, checkpoint_id: int | None = None) -> RollbackResult:
        """Undo the last turn, or every turn from checkpoint `checkpoint_id` on.

        On success `cut` is where to truncate `messages`. Refuses, changing
        nothing, when the turn has no checkpoint (rolling back to an older one
        would also delete the work of the turns before it) or when the
        conversation no longer starts the way it did at that turn.
        """
        if checkpoint_id is None:
            if not self.turns:
                return RollbackResult(False, 0, "", error=(
                    "no checkpoint recorded for this turn (it ran before this "
                    "session or conversation); nothing changed"))
            idx = len(self.turns) - 1
        else:
            found = [i for i, slot in enumerate(self.turns)
                     if slot.checkpoint is not None and slot.checkpoint.id == checkpoint_id]
            if not found:
                return RollbackResult(False, checkpoint_id, "", error=(
                    f"checkpoint #{checkpoint_id} is not a turn of this conversation"))
            idx = found[0]
        slot = self.turns[idx]
        if not self._matches(slot, messages):
            return RollbackResult(False, 0, "", error=(
                "the conversation was compacted or replaced since this turn; "
                "nothing changed"))
        if not slot.edits:
            self.turns = self.turns[:idx]
            return RollbackResult(True, 0, "", turns=1, cut=slot.at)
        if slot.checkpoint is None:
            return RollbackResult(False, 0, "", error=(
                f"no checkpoint for this turn ({slot.note or 'not taken'}); "
                "nothing changed"))
        res = self.rollback(slot.checkpoint.id)
        if res.success:
            res.turns = len(self.turns) - idx
            res.cut = slot.at
            self.turns = self.turns[:idx]
        return res

    def close(self) -> None:
        """Session end: delete this session's backup copies."""
        for cp in self.checkpoints:
            if cp.backup_dir:
                self._delete_backup(cp.backup_dir)
        self.checkpoints = []
        self.turns = []
        if self._owned:
            _remove_session_dir(backup_root(), self._session_root())
            self._owned = False

    def _session_root(self) -> str:
        return os.path.join(backup_root(), self.sid)

    def _delete_backup(self, path: str) -> None:
        """Remove one of OUR snapshot directories, never anything else."""
        real = os.path.realpath(path)
        own = os.path.realpath(self._session_root())
        if (os.path.dirname(real) == own and os.path.basename(real).startswith("snapshot-")
                and not os.path.islink(path)):
            shutil.rmtree(real, ignore_errors=True)

    def _prune(self) -> None:
        while len(self.checkpoints) > KEEP_LAST:
            old = self.checkpoints.pop(0)
            if old.backup_dir:
                self._delete_backup(old.backup_dir)
            for slot in self.turns:
                if slot.checkpoint is old:
                    slot.checkpoint = None
                    slot.note = f"pruned - only the last {KEEP_LAST} checkpoints are kept"

    def _drop_from(self, cp: Checkpoint) -> None:
        """Forget `cp` and every later checkpoint, deleting their backups."""
        idx = self.checkpoints.index(cp)
        for gone in self.checkpoints[idx:]:
            if gone.backup_dir:
                self._delete_backup(gone.backup_dir)
        self.checkpoints = self.checkpoints[:idx]

    def _index_path(self) -> Path:
        path = subprocess.check_output(
            ["git", "rev-parse", "--git-path", "index"], cwd=self.workspace,
            text=True, stderr=subprocess.DEVNULL).strip()
        return Path(os.path.abspath(os.path.join(self.workspace, path)))

    def get_diff(self, checkpoint_id: int | None = None) -> str:
        """Show unified diff between workspace and given checkpoint (default: latest)."""
        if not self.checkpoints:
            return "no checkpoints recorded for this session"

        cp = self._resolve_checkpoint(checkpoint_id)
        if not cp:
            return f"checkpoint #{checkpoint_id} not found"

        if self.is_git and cp.commit_sha:
            try:
                diff = subprocess.check_output(
                    ["git", "diff", cp.commit_sha],
                    cwd=self.workspace,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                return diff or "workspace clean - no changes since checkpoint"
            except Exception as exc:
                return f"diff failed: {exc}"

        return "diff requires a git workspace"

    def get_diff_stat(self, checkpoint_id: int | None = None) -> str:
        """Show diffstat between workspace and checkpoint."""
        if not self.checkpoints:
            return ""
        cp = self._resolve_checkpoint(checkpoint_id)
        if not cp or not self.is_git or not cp.commit_sha:
            return ""
        try:
            return subprocess.check_output(
                ["git", "diff", "--stat", cp.commit_sha],
                cwd=self.workspace,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    def rollback(self, checkpoint_id: int | None = None) -> RollbackResult:
        """Revert workspace to the given checkpoint (default: latest)."""
        if not self.checkpoints:
            return RollbackResult(
                success=False,
                checkpoint_id=0,
                prompt="",
                error="no checkpoints to roll back to",
            )

        cp = self._resolve_checkpoint(checkpoint_id)
        if not cp:
            return RollbackResult(
                success=False,
                checkpoint_id=checkpoint_id or 0,
                prompt="",
                error=f"checkpoint #{checkpoint_id} not found",
            )

        reverted: list[str] = []
        removed: list[str] = []

        if not cp.complete:
            return RollbackResult(False, cp.id, cp.prompt, error="incomplete checkpoint")
        try:
            self._validate_workspace()
        except (OSError, ValueError) as exc:
            return RollbackResult(False, cp.id, cp.prompt, error=str(exc))

        if self.is_git and cp.commit_sha:
            index_lock = None
            lock_owned = False
            try:
                # Find files that differ before resetting
                diff_names = subprocess.check_output(
                    ["git", "diff", "--name-only", "-z", cp.commit_sha],
                    cwd=self.workspace,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).split("\0")
                diff_names = [name for name in diff_names if name]
                reverted.extend(diff_names)

                # Hold Git's conventional index lock, then use an isolated index
                # for worktree checkout. The real index is a separate snapshot.
                index_path = self._index_path()
                index_lock = Path(str(index_path) + ".lock")
                self._validate_workspace()
                with index_lock.open("xb") as fh:
                    lock_owned = True
                    if cp.index_snapshot is not None:
                        fh.write(cp.index_snapshot)
                    fh.flush()
                    os.fsync(fh.fileno())
                with tempfile.TemporaryDirectory(prefix="one-undo-") as tmp:
                    env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
                    subprocess.run(["git", "read-tree", cp.commit_sha],
                                   cwd=self.workspace, env=env, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    known = set(subprocess.check_output(
                        ["git", "ls-files", "-z"], cwd=self.workspace, env=env
                    ).decode(errors="surrogateescape").split("\0"))
                    self._validate_workspace()
                    # Restore the snapshot first, its .gitignore included, so
                    # the new-file scan reads the pre-turn ignore rules.
                    subprocess.run(["git", "checkout-index", "-a", "-f"],
                                   cwd=self.workspace, env=env, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    current = set(subprocess.check_output(
                        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                        cwd=self.workspace
                    ).decode(errors="surrogateescape").split("\0"))
                    self._validate_workspace()
                    for rel in sorted(current - known - cp.pre_turn_files):
                        full = self._safe_destination(rel)
                        if os.path.isfile(full) or os.path.islink(full):
                            os.unlink(full)
                            removed.append(rel)
                self._validate_workspace()
                if cp.index_snapshot is None:
                    index_path.unlink(missing_ok=True)
                else:
                    os.replace(index_lock, index_path)

                # Only prune ancestors of removed files, not unrelated empty
                # directories that may predate the checkpoint.
                for rel in removed:
                    self._validate_workspace()
                    parent = Path(self.workspace, rel).parent
                    while str(parent) != self.workspace:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent

                # Pop checkpoint and any subsequent ones
                self._drop_from(cp)

                return RollbackResult(
                    success=True,
                    checkpoint_id=cp.id,
                    prompt=cp.prompt,
                    reverted_files=reverted,
                    removed_files=removed,
                )
            except Exception as exc:
                return RollbackResult(
                    success=False,
                    checkpoint_id=cp.id,
                    prompt=cp.prompt,
                    error=str(exc),
                )
            finally:
                if lock_owned and index_lock is not None:
                    try:
                        self._validate_workspace()
                    except (OSError, ValueError):
                        pass  # Do not unlink through a substituted workspace.
                    else:
                        index_lock.unlink(missing_ok=True)

        # Non-git rollback
        if cp.backup_dir and os.path.isdir(cp.backup_dir):
            try:
                # Validate the entire snapshot and destination scope BEFORE any
                # mutation. Missing/corrupt backups must never trigger deletion.
                if cp.files_snapshot.keys() != cp.file_hashes.keys():
                    raise ValueError("incomplete snapshot manifest")
                on_disk = list(_walk_files(self.workspace))
                for rel, src in cp.files_snapshot.items():
                    target = Path(self._safe_destination(rel))
                    if target.is_dir() and not target.is_symlink():
                        raise ValueError(f"checkpoint destination is a directory: {rel}")
                    for parent in target.parents:
                        if str(parent) == self.workspace:
                            break
                        if parent.exists() and not parent.is_dir():
                            raise ValueError(f"checkpoint parent is not a directory: {rel}")
                    if _digest(src) != cp.file_hashes[rel]:
                        raise ValueError(f"damaged checkpoint file: {rel}")
                # Restore before removing anything. Replace leaf symlinks rather
                # than following them out of the workspace.
                for rel, src in cp.files_snapshot.items():
                    dest = self._safe_destination(rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    fd, staged = tempfile.mkstemp(prefix=".one-undo-", dir=os.path.dirname(dest))
                    os.close(fd)
                    try:
                        os.unlink(staged)
                        shutil.copy2(src, staged, follow_symlinks=False)
                        os.replace(staged, dest)
                    finally:
                        if os.path.lexists(staged):
                            os.unlink(staged)
                    reverted.append(rel)
                for rel, full in on_disk:
                    if rel not in cp.files_snapshot:
                        self._safe_destination(rel)
                        os.unlink(full)
                        removed.append(rel)

                self._drop_from(cp)

                return RollbackResult(
                    success=True,
                    checkpoint_id=cp.id,
                    prompt=cp.prompt,
                    reverted_files=reverted,
                    removed_files=removed,
                )
            except Exception as exc:
                return RollbackResult(
                    success=False,
                    checkpoint_id=cp.id,
                    prompt=cp.prompt,
                    error=str(exc),
                )

        return RollbackResult(
            success=False,
            checkpoint_id=cp.id,
            prompt=cp.prompt,
            error="no snapshot data available",
        )

    def _safe_destination(self, rel: str) -> str:
        """Refuse lexical escapes and directory symlinks; leaf links are replaced."""
        self._validate_workspace()
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("checkpoint path escapes workspace")
        dest = Path(self.workspace) / path
        for parent in dest.parents:
            if str(parent) == self.workspace:
                break
            if parent.is_symlink():
                raise ValueError(f"checkpoint parent is a symlink: {rel}")
        return str(dest)

    def _resolve_checkpoint(self, checkpoint_id: int | None) -> Checkpoint | None:
        if checkpoint_id is None:
            return self.checkpoints[-1] if self.checkpoints else None
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None

    def list_checkpoints(self) -> list[Checkpoint]:
        return list(self.checkpoints)

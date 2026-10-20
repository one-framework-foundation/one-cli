"""Opt-in offline shell execution over a private workspace snapshot.

The host caller approves source and staging paths. No automatic writeback.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    output: str
    workspace: Path
    timed_out: bool = False
    output_limited: bool = False


def _directory(value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute() or p != p.resolve() or not p.is_dir():
        raise SandboxError('expected existing canonical absolute directory')
    if p in (Path('/'), Path.home()) or p in Path.home().parents:
        raise SandboxError('dedicated workspace required')
    return p


def _copy(source: Path, dest: Path, byte_limit: int, entry_limit: int) -> None:
    used = count = 0
    for base, dirs, files in os.walk(source, followlinks=False):
        for name in dirs + files:
            item = Path(base) / name
            info = item.lstat()
            count += 1
            if count > entry_limit:
                raise SandboxError('snapshot entry limit exceeded')
            target = dest / item.relative_to(source)
            if stat.S_ISDIR(info.st_mode):
                target.mkdir(mode=0o700)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                used += info.st_size
                if used > byte_limit:
                    raise SandboxError('snapshot byte limit exceeded')
                # No symlink traversal on final component; concurrent host mutation
                # of parent directories is outside this host-owned API contract.
                fd = os.open(item, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, 'rb') as inp:
                    opened = os.fstat(inp.fileno())
                    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                        raise SandboxError('source changed during snapshot')
                    with target.open('xb') as out:
                        remaining = info.st_size
                        while remaining:
                            chunk = inp.read(min(65536, remaining))
                            if not chunk:
                                break
                            out.write(chunk)
                            remaining -= len(chunk)
                        if inp.read(1):
                            raise SandboxError('source grew during snapshot')
                target.chmod(0o700 if info.st_mode & 0o111 else 0o600)
            else:
                raise SandboxError('snapshot rejects links, sockets and special files')


SANDBOX_BINARIES = ('/usr/bin/bwrap', '/usr/bin/prlimit')


def unavailable() -> str:
    """Why the sandbox cannot run here, or '' when it can."""
    for binary in SANDBOX_BINARIES:
        if not os.access(binary, os.X_OK):
            return f'required sandbox executable {binary} unavailable'
    return ''


def confinement_argv() -> list[str]:
    """The bwrap prefix: no network, no namespaces, a read-only /usr and nothing else.

    Callers add their own binds, then `--remount-ro /` and the command.
    """
    argv = ['/usr/bin/bwrap', '--unshare-all', '--unshare-user', '--die-with-parent', '--new-session',
            '--cap-drop', 'ALL', '--disable-userns', '--assert-userns-disabled', '--clearenv', '--ro-bind', '/usr', '/usr',
            '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
            '--proc', '/proc', '--dev', '/dev']
    if Path('/lib64').exists():
        argv += ['--ro-bind', str(Path('/lib64').resolve()), '/lib64']
    return argv


def run_shell(source: str | Path, staging_root: str | Path, command: str, *,
              timeout_seconds: float = 30, max_output_bytes: int = 65536,
              snapshot_bytes: int = 32 * 1024 * 1024,
              snapshot_entries: int = 10000) -> SandboxResult:
    """Execute in a fresh snapshot; paths are host-approved, never model inputs.

    staging_root is dedicated host-owned storage, outside source. The returned
    workspace is untrusted output, retained for host inspection, never imported.
    """
    source, staging_root = _directory(source), _directory(staging_root)
    if source == staging_root or source in staging_root.parents or staging_root in source.parents:
        raise SandboxError('source and staging must be disjoint')
    if not isinstance(command, str) or not command or len(command) > 65536 or '\0' in command:
        raise SandboxError('invalid shell command')
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
        raise SandboxError('invalid timeout')
    for value, maximum in ((max_output_bytes, 1048576), (snapshot_bytes, 134217728), (snapshot_entries, 100000)):
        if type(value) is not int or not 0 < value <= maximum:
            raise SandboxError('invalid resource limit')
    if unavailable():
        raise SandboxError('required sandbox executable unavailable')
    stage = Path(tempfile.mkdtemp(prefix='shell-', dir=staging_root))
    workspace = stage / 'workspace'
    workspace.mkdir(mode=0o700)
    try:
        _copy(source, workspace, snapshot_bytes, snapshot_entries)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    try:
        return _launch(stage, workspace, command, timeout_seconds, max_output_bytes)
    except BaseException:
        # Failure means the caller never receives the workspace to release.
        shutil.rmtree(stage, ignore_errors=True)
        raise


def release(result: SandboxResult) -> None:
    """Delete the staging dir behind a result once the host has inspected it.

    run_shell retains the workspace on success so the host can review artifacts;
    this is the matching release. Safe to call twice.
    """
    if not isinstance(result, SandboxResult):
        raise SandboxError('expected completed sandbox result')
    stage = result.workspace.parent
    if result.workspace.name != 'workspace' or not stage.name.startswith('shell-'):
        raise SandboxError('not a sandbox workspace')
    shutil.rmtree(stage, ignore_errors=True)


def _launch(stage, workspace, command, timeout_seconds, max_output_bytes) -> SandboxResult:
    argv = confinement_argv()
    argv += ['--bind', str(workspace), '/workspace', '--chdir', '/workspace',
             '--setenv', 'HOME', '/workspace', '--setenv', 'PATH', '/usr/bin:/bin',
             '--setenv', 'LANG', 'C', '--remount-ro', '/', '--', '/usr/bin/prlimit',
             '--core=0:0', '--fsize=16777216:16777216', '--as=536870912:536870912',
             '--cpu=30:30', '--nofile=64:64', '--', '/bin/sh', '-c', command]
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env={}, close_fds=True,
                                   start_new_session=True)
    except OSError as exc:
        raise SandboxError('sandbox launch failed') from exc
    stdout = process.stdout
    assert stdout is not None
    data = bytearray()
    timed_out = output_limited = False
    deadline = time.monotonic() + timeout_seconds
    def kill():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    with selectors.DefaultSelector() as poll:
        poll.register(stdout, selectors.EVENT_READ)
        try:
            while poll.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    kill()
                    break
                for key, _ in poll.select(min(remaining, 0.1)):
                    chunk = os.read(stdout.fileno(), 65536)
                    if not chunk:
                        poll.unregister(key.fileobj)
                        continue
                    room = max_output_bytes - len(data)
                    data.extend(chunk[:room])
                    if len(chunk) > room:
                        output_limited = True
                        kill()
                        break
                if output_limited:
                    break
            # stdout may be closed before a command exits.
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                kill()
                process.wait()
        finally:
            kill()
            process.wait()
            stdout.close()
    return SandboxResult(process.returncode, data.decode('utf-8', errors='replace'), workspace,
                         timed_out, output_limited)


def read_artifact(result: SandboxResult, relative_path: str, *, max_bytes: int = 1048576) -> bytes:
    """Read one regular output file without traversing links or importing it.

    The host must retain the genuine result from a completed run. This is not a
    capability authenticating a caller-constructed SandboxResult. No sandbox
    process or untrusted host writer may mutate the result tree during review.
    """
    if not isinstance(result, SandboxResult):
        raise SandboxError('expected completed sandbox result')
    if type(max_bytes) is not int or not 0 < max_bytes <= 16777216:
        raise SandboxError('invalid artifact byte limit')
    if not isinstance(relative_path, str) or '\0' in relative_path:
        raise SandboxError('invalid artifact path')
    parts = relative_path.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise SandboxError('artifact path must be a nonempty relative path')
    workspace = _directory(result.workspace)
    # Traverse from /, rejecting symlinks in every root component as well.
    current = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in (*workspace.parts[1:], *parts[:-1]):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=current)
            os.close(current)
            current = child
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                          dir_fd=current)
        with os.fdopen(file_fd, 'rb') as artifact:
            info = os.fstat(artifact.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SandboxError('artifact must be a regular single-link file')
            if info.st_size > max_bytes:
                raise SandboxError('artifact byte limit exceeded')
            data = artifact.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise SandboxError('artifact byte limit exceeded')
            return data
    except OSError as exc:
        raise SandboxError('artifact unavailable or unsafe') from exc
    finally:
        os.close(current)

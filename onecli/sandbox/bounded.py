"""Explicit host-owned sandbox with aggregate quotas and bounded tmpfs writes."""
from dataclasses import dataclass
import base64
import binascii
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile
import time

from .shell import SandboxError, _directory, _copy
from .quota import prepare, QuotaLimits


@dataclass(frozen=True)
class BoundedSandboxResult:
    returncode: int
    output: str
    artifacts: dict[str, bytes]
    timed_out: bool = False
    output_limited: bool = False


def _parse(data, requested, max_output):
    try:
        obj = json.loads(data)
        if type(obj) is not dict or set(obj) != {'returncode', 'output', 'artifacts', 'timed_out', 'output_limited'}:
            raise ValueError('schema')
        if type(obj['returncode']) is not int or not -255 <= obj['returncode'] <= 255:
            raise ValueError('returncode')
        if any(type(obj[k]) is not bool for k in ('timed_out', 'output_limited')):
            raise ValueError('flags')
        if type(obj['artifacts']) is not dict or not set(obj['artifacts']) <= set(requested):
            raise ValueError('unexpected artifact')
        expected = set() if obj['timed_out'] or obj['output_limited'] else set(requested)
        if set(obj['artifacts']) != expected:
            raise ValueError('missing artifact')
        def decode(value, maximum):
            if type(value) is not str or len(value) > 4 * ((maximum + 2) // 3):
                raise ValueError('encoded size')
            decoded = base64.b64decode(value, validate=True)
            if len(decoded) > maximum:
                raise ValueError('decoded size')
            return decoded
        output = decode(obj['output'], max_output)
        remaining = 1048576
        artifacts = {}
        for name, value in obj['artifacts'].items():
            artifacts[name] = decode(value, remaining)
            remaining -= len(artifacts[name])
        text = output.decode('utf8', errors='replace').encode('utf8')[:max_output].decode('utf8', errors='ignore')
        return BoundedSandboxResult(obj['returncode'], text,
                                    artifacts, obj['timed_out'], obj['output_limited'])
    except (ValueError, TypeError, UnicodeError, binascii.Error, RecursionError) as exc:
        raise SandboxError('invalid sandbox output protocol') from exc


def run_bounded_shell(source, staging_root, command, *, artifacts=(), workspace_bytes=16777216,
                      timeout_seconds=30, max_output_bytes=65536):
    source, staging_root = _directory(source), _directory(staging_root)
    if source == staging_root or source in staging_root.parents or staging_root in source.parents:
        raise SandboxError('source and staging must be disjoint')
    if type(command) is not str or not command or len(command) > 65536 or '\0' in command:
        raise SandboxError('invalid command')
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
        raise SandboxError('invalid timeout')
    if type(workspace_bytes) is not int or not 1048576 <= workspace_bytes <= 33554432:
        raise SandboxError('invalid workspace size')
    if type(max_output_bytes) is not int or not 0 < max_output_bytes <= 65536:
        raise SandboxError('invalid output size')
    if type(artifacts) not in (tuple, list) or len(artifacts) > 32:
        raise SandboxError('invalid artifacts')
    for name in artifacts:
        if type(name) is not str or len(name) > 512 or '\0' in name or any(p in ('', '.', '..') for p in name.split('/')):
            raise SandboxError('invalid artifact path')
    if len(set(artifacts)) != len(artifacts):
        raise SandboxError('duplicate artifact')
    stage = Path(tempfile.mkdtemp(prefix='bounded-', dir=staging_root))
    try:
        snapshot = stage / 'input'
        snapshot.mkdir(mode=0o700)
        _copy(source, snapshot, min(workspace_bytes // 2, 8388608), 2048)
        worker = Path(__file__).with_name('_worker.py').resolve()
        config = json.dumps(dict(command=command, artifacts=list(artifacts),
                                 timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes))
        argv = ['/usr/bin/bwrap', '--unshare-all', '--unshare-user', '--die-with-parent', '--new-session',
                '--cap-drop', 'ALL', '--disable-userns', '--assert-userns-disabled', '--clearenv',
                '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
                '--proc', '/proc', '--dev', '/dev']
        if Path('/lib64').exists():
            argv += ['--ro-bind', str(Path('/lib64').resolve()), '/lib64']
        argv += ['--ro-bind', str(snapshot), '/input', '--ro-bind', str(worker), '/runner.py',
                 '--size', str(workspace_bytes), '--tmpfs', '/workspace', '--chdir', '/workspace',
                 '--setenv', 'HOME', '/workspace', '--setenv', 'PATH', '/usr/bin:/bin',
                 '--setenv', 'LANG', 'C', '--remount-ro', '/', '--',
                 '/usr/bin/python3', '-I', '-B', '/runner.py', config]
        launch = prepare(argv, QuotaLimits(runtime_seconds=math.ceil(timeout_seconds) + 10),
                         control_root=staging_root / 'controls')
        process = None
        data = bytearray()
        timed_out = output_limited = False
        # Includes encoded 1MiB artifacts, output, keys and framing.
        wire_limit = 1600000
        deadline = time.monotonic() + timeout_seconds + 7
        try:
            process = subprocess.Popen(launch.argv, env=launch.env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       close_fds=True, start_new_session=True)
            launch.authorize(timeout=5)
            with selectors.DefaultSelector() as poll:
                poll.register(process.stdout, selectors.EVENT_READ, True)
                poll.register(process.stderr, selectors.EVENT_READ, False)
                stderr_bytes = 0
                while poll.get_map():
                    left = deadline - time.monotonic()
                    if left <= 0:
                        timed_out = True
                        break
                    for key, _ in poll.select(min(left, .1)):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            poll.unregister(key.fileobj)
                        elif key.data:
                            if len(data) + len(chunk) > wire_limit:
                                output_limited = True
                                break
                            data.extend(chunk)
                        else:
                            stderr_bytes += len(chunk)
                            if stderr_bytes > 65536:
                                output_limited = True
                                break
                    if output_limited:
                        break
                if not (timed_out or output_limited):
                    try:
                        process.wait(timeout=max(.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        timed_out = True
        finally:
            try:
                launch.cleanup()
            finally:
                if process is not None:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    process.stdout.close()
                    process.stderr.close()
        if timed_out or output_limited:
            return BoundedSandboxResult(-9, '', {}, timed_out, output_limited)
        if process.returncode != 0:
            raise SandboxError('sandbox worker or quota bootstrap failed')
        return _parse(data, artifacts, max_output_bytes)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            (staging_root / 'controls').rmdir()   # only if no other launch holds a gate
        except OSError:
            pass

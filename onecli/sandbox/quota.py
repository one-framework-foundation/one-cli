"""Host-owned cgroup v2 envelope. No command executes before kernel readback."""
from __future__ import annotations
from dataclasses import dataclass
import json
import fcntl
from pathlib import Path
import time
import stat
import os
import subprocess
import uuid


class QuotaError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuotaLimits:
    memory_bytes: int = 268435456
    tasks: int = 32
    cpu_percent: int = 25
    runtime_seconds: int = 35

    def validate(self):
        for value, low, high in ((self.memory_bytes, 33554432, 1073741824),
                                 (self.tasks, 4, 128), (self.cpu_percent, 1, 100),
                                 (self.runtime_seconds, 1, 310)):
            if type(value) is not int or not low <= value <= high:
                raise QuotaError('invalid aggregate quota')


# Trusted calling code runs outside the sandbox, then replaces itself with bwrap.
# Keep it self-contained: neither imports nor PYTHONPATH resolve workspace code.
_VERIFIER = r'''
import json, os, pathlib, sys, fcntl, time
try:
    spec = json.loads(sys.argv[1])
    lines = pathlib.Path('/proc/self/cgroup').read_text().splitlines()
    unified = [x[3:] for x in lines if x.startswith('0::')]
    if len(unified) != 1 or pathlib.PurePosixPath(unified[0]).name != spec['unit']:
        raise ValueError('unexpected cgroup membership')
    group = pathlib.Path('/sys/fs/cgroup') / unified[0].lstrip('/')
    for name, expected in [('memory.max', spec['memory']), ('memory.swap.max', 0), ('pids.max', spec['tasks'])]:
        if int((group/name).read_text().strip()) != expected:
            raise ValueError('quota readback mismatch: ' + name)
    quota, period = (group/'cpu.max').read_text().split()
    if int(quota) <= 0 or int(period) <= 0 or int(quota)*100 > spec['cpu']*int(period):
        raise ValueError('CPU quota readback mismatch')
except Exception as exc:
    print('sandbox quota validation failed: ' + str(exc), file=sys.stderr)
    sys.exit(125)
# A cancelled gate blocks a queued late unit; a released (deleted) gate does
# too, because the open below then raises and the command never executes.
fd = os.open(spec['gate'], os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
deadline = time.monotonic() + 10
while True:
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.lseek(fd, 0, 0)
    state = os.read(fd, 32)
    if state == b'cancelled':
        sys.exit(125)
    if state == b'allowed':
        # Cancel cannot interleave the release and exec; CLOEXEC drops lock.
        os.execve(sys.argv[2], sys.argv[2:], {})
    if state == b'pending':
        os.lseek(fd, 0, 0); os.write(fd, b'ready'); os.ftruncate(fd, 5)
    elif state != b'ready':
        sys.exit(125)
    fcntl.flock(fd, fcntl.LOCK_UN)
    if time.monotonic() >= deadline:
        sys.exit(125)
    time.sleep(0.01)
'''


@dataclass(frozen=True)
class QuotaLaunch:
    unit: str
    argv: list[str]
    env: dict[str, str]
    gate: Path

    def _state(self, replacement=None):
        fd = os.open(self.gate, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            state = os.read(fd, 32)
            if replacement is not None:
                os.lseek(fd, 0, 0)
                os.write(fd, replacement)
                os.ftruncate(fd, len(replacement))
                os.fsync(fd)
            return state
        finally:
            os.close(fd)

    def authorize(self, timeout: float = 5):
        """Release only after verifier has observed an active, limited unit."""
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 10:
            raise QuotaError('invalid authorization timeout')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                fd = os.open(self.gate, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
            except FileNotFoundError:
                raise QuotaError('authorization gate is not pending') from None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                state = os.read(fd, 32)
                if state == b'ready':
                    os.lseek(fd, 0, 0)
                    os.write(fd, b"allowed")
                    os.ftruncate(fd, 7)
                    os.fsync(fd)
                    return
                if state != b'pending':
                    raise QuotaError('authorization gate is not pending')
            finally:
                os.close(fd)
            time.sleep(0.01)
        raise QuotaError('sandbox activation not observed')

    def cleanup(self) -> None:
        """Stop all descendants, including when the systemd-run client was killed."""
        try:
            self._cancel_and_stop()
        finally:
            # The gate has served its purpose once the unit is stopped (or
            # stopping failed loudly); never leave one file per launch behind.
            try:
                os.unlink(self.gate)
            except FileNotFoundError:
                pass

    def _cancel_and_stop(self) -> None:
        try:
            self._state(b'cancelled')
        except FileNotFoundError:
            pass   # already released; stopping the unit below stays idempotent
        result = subprocess.run(['/usr/bin/systemctl', '--user', 'stop', self.unit],
                                env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if result.returncode:
            # --collect may already have removed a completed unit. Distinguish
            # that state using LoadState; a bus error is never success.
            state = subprocess.run(['/usr/bin/systemctl', '--user', 'show', self.unit,
                                    '--property=LoadState', '--value'], env=self.env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=5)
            if state.returncode or state.stdout.strip() != b'not-found':
                raise QuotaError('could not confirm aggregate sandbox cleanup')


def prepare(argv: list[str], limits: QuotaLimits | None = None, *, control_root: str | Path | None = None) -> QuotaLaunch:
    """Wrap a trusted absolute executable argv; never accept model-selected paths.

    Caller must bound stdout/time, close stdin/FDs, and call cleanup in finally.
    This API enforces per-launch aggregate limits, not a budget shared across launches.
    """
    limits = QuotaLimits() if limits is None else limits
    if type(limits) is not QuotaLimits:
        raise QuotaError('expected caller-owned quota limits')
    limits.validate()
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or '\0' in x for x in argv) or not argv[0].startswith('/'):
        raise QuotaError('invalid trusted command')
    for binary in ('/usr/bin/systemd-run', '/usr/bin/systemctl', '/usr/bin/python3'):
        if not os.access(binary, os.X_OK):
            raise QuotaError('required quota executable unavailable')
    unit = 'onecli-sandbox-' + uuid.uuid4().hex + '.service'
    control = Path.home() / '.local/state/onecli-sandbox-controls' if control_root is None else Path(control_root)
    if not control.is_absolute() or control != control.resolve():
        raise QuotaError('control root must be canonical and absolute')
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = control.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise QuotaError('control root must be private and caller-owned')
    gate = control / (unit + '.gate')
    fd = os.open(gate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        try:
            os.write(fd, b"pending")
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        gate.unlink(missing_ok=True)
        raise
    runtime = '/run/user/' + str(os.getuid())
    env = {'XDG_RUNTIME_DIR': runtime, 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=' + runtime + '/bus',
           'PATH': '/usr/bin:/bin', 'LANG': 'C'}
    spec = json.dumps({'unit': unit, 'memory': limits.memory_bytes, 'tasks': limits.tasks, 'cpu': limits.cpu_percent, 'gate': str(gate)})
    command = ['/usr/bin/systemd-run', '--user', '--quiet', '--wait', '--pipe', '--collect',
               '--expand-environment=no', '--unit=' + unit, '--property=MemoryMax=' + str(limits.memory_bytes),
               '--property=MemorySwapMax=0', '--property=TasksMax=' + str(limits.tasks),
               '--property=CPUQuota=' + str(limits.cpu_percent) + '%',
               '--property=RuntimeMaxSec=' + str(limits.runtime_seconds),
               '--property=TimeoutStopSec=1', '--property=KillMode=control-group',
               '--property=SendSIGKILL=yes', '--property=OOMPolicy=kill',
               '/usr/bin/python3', '-I', '-c', _VERIFIER, spec, *argv]
    return QuotaLaunch(unit, command, env, gate)

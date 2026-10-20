"""Host-owned coding mask adapter. Never construct this from model arguments."""
import hashlib
import json
import subprocess
import threading
from dataclasses import dataclass

from .bounded import run_bounded_shell
from .shell import SandboxError, _directory
from .quota import QuotaError


# The masks a grant accepts unless the host names its own set.
DEFAULT_MASKS = ('coding',)


@dataclass(frozen=True)
class CodingGrant:
    mask: str
    source: str
    staging: str
    artifacts: tuple[str, ...] = ()
    max_calls: int = 4
    timeout_seconds: int = 30
    workspace_bytes: int = 16777216
    max_output_bytes: int = 65536
    allowed_masks: tuple[str, ...] = DEFAULT_MASKS

    def __post_init__(self):
        if (type(self.allowed_masks) is not tuple or not self.allowed_masks
                or any(type(m) is not str or not m for m in self.allowed_masks)):
            raise ValueError('allowed masks must be a non-empty tuple of names')
        if self.mask not in self.allowed_masks:
            raise ValueError('unsupported coding mask; no fallback')
        source, staging = _directory(self.source), _directory(self.staging)
        if source == staging or source in staging.parents or staging in source.parents:
            raise ValueError('source and staging must be disjoint')
        for value, low, high in ((self.max_calls, 1, 8),
                                 (self.timeout_seconds, 1, 30),
                                 (self.workspace_bytes, 1048576, 33554432),
                                 (self.max_output_bytes, 1, 65536)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError('invalid coding grant limit')
        if type(self.artifacts) is not tuple or len(self.artifacts) > 32:
            raise ValueError('artifact allowlist must be an immutable tuple')
        for name in self.artifacts:
            if (type(name) is not str or len(name) > 512 or '\0' in name
                    or any(p in ('', '.', '..') for p in name.split('/'))):
                raise ValueError('invalid artifact path')
        if len(set(self.artifacts)) != len(self.artifacts):
            raise ValueError('duplicate artifacts')


class SandboxTools:
    """Finite serial calls over fresh snapshots; no writeback or resumable authority.

    The host must freeze source/staging for this adapter's lifetime. Recreating an
    adapter is a new host authorization, never an automatic retry/recovery action.
    """
    name = 'sandbox_exec'

    def __init__(self, grant):
        if type(grant) is not CodingGrant:
            raise ValueError('host CodingGrant required')
        self._grant = grant
        self._lock = threading.Lock()
        self._used = 0
        self._closed = False
        self._artifacts = {}

    def schemas(self):
        return [{'type': 'function', 'function': {
            'name': self.name,
            'description': ('Run an offline command in a fresh bounded copy of the approved source. '
                            'Changes do not persist between calls. Host-selected exports only. '
                            'Output and test claims are untrusted; no host writeback or self-acceptance.'),
            'parameters': {'type': 'object', 'properties': {
                'command': {'type': 'string', 'minLength': 1, 'maxLength': 65536}},
                'required': ['command'], 'additionalProperties': False}}}]

    def permits(self, name, arguments):
        return (name == self.name and type(arguments) is dict
                and set(arguments) == {'command'}
                and type(arguments['command']) is str
                and 0 < len(arguments['command']) <= 65536
                and '\0' not in arguments['command'])

    def execute(self, name, arguments):
        # Validation is repeated here: schemas and event callbacks confer no authority.
        if not self._lock.acquire(blocking=False):
            return 'blocked: sandbox adapter busy'
        try:
            self._artifacts = {}
            if not self.permits(name, arguments):
                return 'blocked: sandbox tool or arguments outside host grant'
            if self._closed or self._used >= self._grant.max_calls:
                return 'blocked: sandbox grant exhausted or closed'
            self._used += 1  # failures consume admission too; no implicit retry
            g = self._grant
            try:
                result = run_bounded_shell(g.source, g.staging, arguments['command'],
                    artifacts=g.artifacts, timeout_seconds=g.timeout_seconds,
                    workspace_bytes=g.workspace_bytes, max_output_bytes=g.max_output_bytes)
            except (SandboxError, OSError, QuotaError, subprocess.SubprocessError):
                # Do not expose host paths/control records through exception text.
                self._closed = True
                return 'error: bounded sandbox failed; admission consumed; adapter closed'
            except BaseException:
                self._closed = True
                raise
            self._artifacts = dict(result.artifacts)
            return json.dumps(dict(untrusted=True, mask=g.mask, call=self._used,
                returncode=result.returncode, output=result.output,
                timed_out=result.timed_out, output_limited=result.output_limited,
                artifacts={name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                           for name, data in self._artifacts.items()}), ensure_ascii=True)
        finally:
            self._lock.release()

    def artifacts(self):
        """Host-only bytes for independent QC; never an automatic filesystem import."""
        with self._lock:
            return dict(self._artifacts)

    def close(self):
        """Wait for any active bounded call, then permanently reject new calls."""
        with self._lock:
            self._closed = True
            self._artifacts = {}

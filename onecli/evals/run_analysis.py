"""Local, content-free receipts for every Agent.run, including children and REPL.

These are diagnostic signals, not task-success grades. A start without an end
is unfinished evidence, never proof of either a crash or a still-running job.
No model call, subprocess, prompt, tool arguments or tool output is persisted.
"""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from functools import wraps
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import time
import uuid

from onecli import secureio

_parent = ContextVar('one_analysis_parent', default=None)


def recommendations(metrics):
    findings = []
    if metrics.get('repeated_calls', 0):
        findings.append('Review repeated arguments against intervening changes; repetition alone is not waste.')
    if metrics.get('denied_calls', 0):
        findings.append('Check tool discovery and permission explanations before changing authorization.')
    if metrics.get('retries', 0):
        findings.append('Replay provider retry cases; separate malformed responses from transport failures.')
    if metrics.get('compactions', 0):
        findings.append('Test context selection and recovery across compaction on the same task.')
    if metrics.get('stopped') not in (None, 'answered'):
        findings.append('Reproduce the stop condition with a fixture before changing loop policy.')
    return findings


class Recorder:
    def __init__(self, agent):
        self.id = uuid.uuid4().hex
        self.parent = _parent.get()
        self.agent = agent
        self.started = time.monotonic()
        self.counts = Counter()
        self.tools = Counter()
        self.calls = {}
        self.seen = Counter()
        self.repeats = 0
        self.tool_seconds = 0.0
        self.wait_started = None
        self.provider_seconds = 0.0
        self.retry_baseline = self.retry_count()
        self.key = os.urandom(32)
        self.path = Path(os.environ.get('ONE_CLI_ANALYSIS_DIR',
                                        str(Path.home() / '.one-cli' / 'analysis'))) / (self.id + '.jsonl')
        self.failed = False

    def retry_count(self):
        return sum(getattr(r.provider, 'output_retries', 0)
                   + getattr(r.provider, 'transient_retries', 0)
                   for r in self.agent.chain.rungs)

    def write(self, record):
        try:
            with secureio.text_file(self.path, 'a', private=True) as fh:
                fh.write(json.dumps({'schema': 1, 'run_id': self.id, 'parent_run_id': self.parent,
                                     'ts': time.time(), **record}, allow_nan=False) + '\n')
        except (OSError, ValueError, TypeError):
            if not self.failed:
                print('one: run analysis could not be persisted', file=sys.stderr)
                self.failed = True

    def event(self, kind, data):
        # Never copy arbitrary callback data: it may contain credentials.
        if kind == 'waiting':
            self.wait_started = time.monotonic()
        elif kind == 'turn' and self.wait_started is not None:
            self.provider_seconds += time.monotonic() - self.wait_started
            self.wait_started = None
        if kind in ('turn', 'compact', 'fallback', 'loop_advisory', 'empty_retry',
                    'tool_denied', 'response_gate'):
            self.counts[kind] += 1
        if kind in ('tool_start', 'tool_end'):
            call = data['call']
            if kind == 'tool_start':
                self.tools[call.name] += 1
                body = json.dumps([call.name, call.arguments], sort_keys=True, default=str)
                fingerprint = hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()
                self.repeats += int(self.seen[fingerprint] > 0)
                self.seen[fingerprint] += 1
                self.calls[call.id] = time.monotonic()
            else:
                start = self.calls.pop(call.id, None)
                if start is not None:
                    self.tool_seconds += time.monotonic() - start
            self.write({'kind': kind, 'tool': call.name,
                        'elapsed': round(time.monotonic() - self.started, 6)})
        elif kind in ('compact', 'fallback', 'loop_advisory', 'empty_retry', 'tool_denied'):
            self.write({'kind': kind, 'elapsed': round(time.monotonic() - self.started, 6)})

    def finish(self, result, status):
        metrics = {'stopped': status, 'seconds': round(time.monotonic() - self.started, 6),
                   'steps': self.counts['turn'], 'tool_calls': sum(self.tools.values()),
                   'tools': dict(self.tools), 'tool_seconds': round(self.tool_seconds, 6),
                   'unfinished_tool_calls': len(self.calls), 'repeated_calls': self.repeats,
                   'denied_calls': self.counts['tool_denied'],
                   'compactions': self.counts['compact'], 'fallbacks': self.counts['fallback'],
                   'loop_advisories': self.counts['loop_advisory'],
                   'empty_retries': self.counts['empty_retry'],
                   'retries': max(0, self.retry_count() - self.retry_baseline),
                   'retry_scope': 'shared_chain_delta_including_nested_calls',
                   'provider_seconds': round(self.provider_seconds, 6),
                   'provider_timing_scope': 'completed_turn_waits_including_retries',
                   'task_success': 'not_verified'}
        self.write({'kind': 'analysis', 'metrics': metrics,
                    'recommendations': recommendations(metrics)})


def audit_run(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        recorder = Recorder(self)
        recorder.write({'kind': 'start',
                        'session': getattr(getattr(self, 'spill', None), 'session_id', None),
                        'backend': getattr(self.chain.current, 'name', None),
                        'model': getattr(self.chain.current, 'model', None)})
        previous = self.on_event
        token = _parent.set(recorder.id)
        def observe(kind, data):
            recorder.event(kind, data)
            # Callback verdicts MUST survive instrumentation unchanged.
            return previous(kind, data)
        self.on_event = observe
        result = None
        status = 'crash'
        try:
            result = fn(self, *args, **kwargs)
            status = result.stopped
            return result
        except KeyboardInterrupt:
            status = 'interrupt'
            raise
        finally:
            self.on_event = previous
            _parent.reset(token)
            recorder.finish(result, status)
    return wrapped


def summary(root=None):
    """Summarize receipts without inferring liveness or semantic success."""
    root = Path(root or os.environ.get('ONE_CLI_ANALYSIS_DIR',
                                      str(Path.home() / '.one-cli' / 'analysis')))
    counts = Counter()
    stops = Counter()
    signals = Counter()
    for path in sorted(root.glob('*.jsonl')):
        counts['receipts'] += 1
        start = end = None
        try:
            with path.open() as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError('not an object')
                    except ValueError:
                        counts['malformed_records'] += 1
                        continue
                    if record.get('kind') == 'start':
                        start = record
                    elif record.get('kind') == 'analysis':
                        end = record
        except OSError:
            counts['unreadable_receipts'] += 1
            continue
        if start is None:
            counts['missing_start'] += 1
        elif start.get('parent_run_id'):
            counts['nested_runs'] += 1
        if end is None:
            counts['without_final_analysis'] += 1
            continue
        counts['with_final_analysis'] += 1
        metrics = end.get('metrics') or {}
        stops[metrics.get('stopped', 'unknown')] += 1
        for key in ('repeated_calls', 'denied_calls', 'compactions', 'fallbacks', 'retries'):
            signals[key] += int(bool(metrics.get(key)))
    return {'scope': 'local receipts, not a deployment or process-liveness census',
            **counts, 'stops': dict(stops), 'runs_with_signal': dict(signals),
            'task_success': 'not_verified'}


if __name__ == '__main__':
    print(json.dumps(summary(), indent=2))

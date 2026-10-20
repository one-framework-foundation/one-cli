"""Public progress cadence, evaluated only at safe tool-batch boundaries."""
import math

DEFAULT = {"enabled": True, "seconds": 30.0, "tool_calls": 5, "on_error": True, "min_seconds": 10.0}
PROMPT = """Communicate like a concise terminal coding assistant. For substantial work,
use a short initial update and occasional updates when verified findings, blockers,
or your approach change. The visible tool trace already shows routine activity.
Do not narrate every call, repeat summaries, or expose private reasoning. Match the
user's requested detail. Reserve the outcome, validation and remaining limitations
for one final answer when the task ends. Lead that answer with the result, then
the decisive evidence and necessary next action. A short paragraph or a few bullets
usually suffice. Use tables or sections only when they clarify the answer; do not
repeat the same facts in a summary, details section and closing status checklist.
Preserve material uncertainty and blockers without listing irrelevant non-actions."""
UPDATE = """Consider communication alongside the next task step. Tools remain
available. If the task is complete or blocked, give the final answer now and stop.
Lead with the decision or result, then the evidence and necessary next action;
use only the detail the user needs. Do not repeat the opening or add a second recap.
A failed tool result alone does not establish a task blocker. Inspect the cause
and take the next authorized recovery step when possible. If work remains, continue using tools. Add a brief update only for a meaningful
new finding, blocker or change of approach; otherwise use tools without narration.
This reminder does not require a separate message or an extra tool call."""


def resolve(value=None, base=None):
    result = dict(DEFAULT if base is None else base)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(DEFAULT):
            raise ValueError("progress_policy requires enabled, seconds, tool_calls, on_error or min_seconds")
        result.update(value)
    for key in ('enabled', 'on_error'):
        if type(result[key]) is not bool:
            raise ValueError(f"progress_policy.{key} must be boolean")
    seconds = result['seconds']
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("progress_policy.seconds must be finite and nonnegative")
    minimum = result["min_seconds"]
    if type(minimum) not in (float, int) or not math.isfinite(minimum) or minimum < 0:
        raise ValueError("progress_policy.min_seconds must be finite and nonnegative")
    calls = result['tool_calls']
    if type(calls) is not int or calls < 0:
        raise ValueError("progress_policy.tool_calls must be a nonnegative integer")
    return result

class Cadence:
    def __init__(self, policy, now):
        self.policy = policy
        self.sent(now)

    def sent(self, now):
        self.last_public = now
        self.last_checkpoint = now
        self.calls = 0
        self.failed = False

    def defer(self, now=None):
        """Resume work before another checkpoint, without claiming delivery."""
        self.calls = 0
        self.failed = False
        if now is not None:
            self.last_checkpoint = now

    def completed(self, output, denied=False):
        self.calls += 1
        first = str(output).split('\n', 1)[0].lower()
        exit_failed = first.startswith('exit ') and first[5:].lstrip('-').isdigit() and int(first[5:]) != 0
        self.failed |= denied or exit_failed or first.startswith(('error:', 'blocked:'))

    def reminder(self, now):
        """Request-local state from completed calls, never invented progress."""
        reason = self.reason(now)
        return (UPDATE + f"\nObserved loop state: trigger={reason}; "
                f"seconds_since_public={max(0, int(now - self.last_public))}; "
                f"completed_calls_since_check={self.calls}; "
                f"failure_observed={str(self.failed).lower()}. "
                "These are execution receipts, not a completion percentage. "
                "If a long task continues, keep the update to one short sentence "
                "about verified evidence or the current obstacle.")

    def due(self, now):
        return self.reason(now) is not None

    def reason(self, now):
        p = self.policy
        if not p['enabled'] or not self.calls:
            return None
        if p['on_error'] and self.failed:
            return 'tool_failure'
        if p['seconds'] > 0 and now - max(self.last_public, self.last_checkpoint) >= p['seconds']:
            return 'elapsed_silence'
        if (p['tool_calls'] > 0 and self.calls >= p['tool_calls']
                and now - self.last_public >= p['min_seconds']):
            return 'tool_batch'
        return None

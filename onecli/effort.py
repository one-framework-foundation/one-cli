"""Shared effort policy. Backend profiles own wire syntax, never model guesses."""
from copy import deepcopy

LEVELS = ('auto', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')


def select(requested='auto', task_class=''):
    if requested not in LEVELS:
        raise ValueError('invalid effort level')
    if requested != 'auto':
        return requested, 'explicit'
    if task_class == 'triage':
        return 'low', 'triage'
    return None, 'backend-default'


def configure(extra, requested='auto', task_class=''):
    body = deepcopy(extra or {})
    profiles = body.pop('effort_profiles', {})
    level, source = select(requested, task_class)
    decision = dict(requested=requested, selected=level, source=source, applied=False)
    if level is None:
        return body, decision
    if not isinstance(profiles, dict) or level not in profiles:
        if source == 'explicit':
            raise ValueError(f'backend has no declared effort profile for {level}')
        decision['reason'] = 'no declared profile; preserving backend default'
        return body, decision
    patch = profiles[level]
    if not isinstance(patch, dict):
        raise ValueError('effort profile must be a request-body object')
    # Recursive merge preserves unrelated options in nested wire objects.
    def merge(target, patch):
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                merge(target[k], v)
            else:
                target[k] = deepcopy(v)
    merge(body, patch)
    decision['applied'] = True
    return body, decision

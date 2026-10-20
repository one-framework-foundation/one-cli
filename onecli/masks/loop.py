"""Validated mask loop settings; host limits can only be narrowed."""
import math
from onecli import progress_policy


def parse(raw):
    if not isinstance(raw, dict) or set(raw) - {
        'version', 'max_steps', 'repeat_limit', 'discovery_limit', 'tool_deadline', 'empty_retries', 'progress_policy'
    }:
        raise ValueError('mask loop_policy must be an object with known fields')
    result = dict(raw)
    version = result.setdefault('version', 1)
    if type(version) is not int or version != 1:
        raise ValueError('unsupported mask loop_policy version')
    for name, low, high in [('max_steps', 2, 1000), ('repeat_limit', 1, 1000), ('discovery_limit', 1, 1000), ('empty_retries', 0, 1000)]:
        if name in result and (type(result[name]) is not int or not low <= result[name] <= high):
            raise ValueError(f'mask loop_policy.{name} must be an integer in {low}..{high}')
    if 'tool_deadline' in result:
        value = result['tool_deadline']
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('mask loop_policy.tool_deadline must be finite and positive')
    if 'progress_policy' in result:
        if not isinstance(result['progress_policy'], dict):
            raise ValueError('mask loop_policy.progress_policy must be an object')
        progress_policy.resolve(result['progress_policy'])
        result['progress_policy'] = dict(result['progress_policy'])
    return result


def ceiling(host, mask):
    return min(host, mask) if host > 0 else mask

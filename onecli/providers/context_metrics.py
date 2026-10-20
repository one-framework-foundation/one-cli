"""Measure serialized wire requests without logging their contents."""
import hashlib
import json


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True).encode()).hexdigest()


def serialize(provider, body):
    payload = json.dumps(body).encode()
    system = body.get('system', [m for m in body.get('messages', []) if m.get('role') == 'system'])
    prefix = _hash({'model': body.get('model'), 'system': system, 'tools': body.get('tools', [])})
    messages = [_hash(m) for m in body.get('messages', [])]
    old = getattr(provider, '_context_fingerprints', None)
    shared = 0
    if old and old[0] == prefix:
        for a, b in zip(old[1], messages):
            if a != b:
                break
            shared += 1
    provider.context_metrics = {
        'serialized_bytes': len(payload), 'request_hash': hashlib.sha256(payload).hexdigest(),
        'prefix_hash': prefix, 'prefix_changed': None if old is None else old[0] != prefix,
        'shared_message_prefix': shared, 'message_count': len(messages),
        'measurement': 'serialized_request_fingerprints_not_cache_hits',
    }
    provider._context_fingerprints = prefix, messages
    return payload


def reported_cache(usage):
    details = usage.get('prompt_tokens_details') or {}
    if not isinstance(details, dict):
        details = {}
    return {k: v for k, v in {
        'cached_input_tokens': details.get('cached_tokens'),
        'cache_read_input_tokens': usage.get('cache_read_input_tokens'),
        'cache_creation_input_tokens': usage.get('cache_creation_input_tokens'),
    }.items() if type(v) is int and v >= 0}

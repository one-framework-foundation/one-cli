"""Opt-in workspace limits for bounded public-source research runs.

Limits restrict the existing loop; they never replace the web guard or grant
tools. Byte ceilings are exact for canonical history plus schemas, not a claim
about provider tokenization or billing. The OpenAI wire also admits each HTTP
attempt through the run's gate, so its internal retries share the ceiling.
"""
import json
from pathlib import Path


def load(workspace):
    path = Path(workspace) / '.one-run-limits.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    keys = {'max_steps', 'max_output_tokens', 'max_request_bytes',
            'max_total_request_bytes', 'max_input_tokens', 'urls'}
    if set(data) - {'fetch_mode'} != keys:
        raise ValueError('Invalid workspace run limits: expected ' + ', '.join(sorted(keys)))
    for key in keys - {'urls'}:
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError('Workspace run limit must be a positive integer: ' + key)
    if not isinstance(data['urls'], list) or not data['urls'] or any(
            not isinstance(url, str) or not url.startswith('https://') for url in data['urls']):
        raise ValueError('Workspace run limits require explicit HTTPS URLs')
    if data.get('fetch_mode', 'excerpts') not in ('excerpts', 'page', 'sections'):
        raise ValueError('Workspace fetch_mode must be excerpts, page or sections')
    return data


def request_bytes(history, schemas):
    return len(json.dumps({'messages': history, 'tools': schemas},
                          ensure_ascii=False).encode('utf-8'))

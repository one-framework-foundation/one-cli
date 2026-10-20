"""README first-run and exact prompt examples against an offline model fixture.

Can also run with a clean wheel venv: cd outside the checkout and execute
/path/to/venv/bin/python -I /path/to/checkout/tests/test_readme_examples.py.
No external provider, credentials or spending is required.
"""
from __future__ import annotations

import io
import json
import tempfile
import threading
from contextlib import redirect_stdout, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


def exercise_examples(home: Path):
    from onecli import cli, config, onboard, orc, session
    requests = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            content = 'Fixture accepted the request.'
            if 'First-response gate:' in json.dumps(body):
                content = json.dumps({'decision': 'answer', 'message': content})
            payload = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': content},
                                               'finish_reason': 'stop'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}/v1'
    output, errors = io.StringIO(), io.StringIO()
    terminal = io.StringIO()
    terminal.isatty = lambda: True
    try:
        with (patch('sys.stdin', terminal),
              patch.object(session, 'SESSION_DIR', home / '.one-cli/sessions'),
              patch.object(orc, 'RUN_DIR', home / '.one-cli/runs'),patch.object(config, 'PROVIDER_DIR', home / '.one-providers'),
              patch.object(config, 'provider_dirs', lambda: [home / '.one-providers']),
              patch.object(config, 'llamacpp_backends', lambda *a, **k: []),
              patch.object(config, 'served_context_window', lambda *a, **k: None),
              patch.object(config, 'served_token_count', lambda *a, **k: None),
              patch.object(onboard, 'detect_local', lambda: [('fixture', base, ['fixture'])]),
              patch.object(onboard.menu, 'choose', lambda *a, **k: 0),
              patch.object(cli, 'interactive', lambda *a, **k: 0),
              patch.object(onboard, 'interactive', lambda: True),
              patch('os.environ', {'HOME': str(home), 'CI': '1', 'ONECLI_NO_UPDATE_CHECK': '1'}),
              redirect_stdout(output), redirect_stderr(errors)):
            # Bare one runs the actual first-run wizard, writes a private drop-in
            # and verifies the connection. Only the terminal/menu input is scripted.
            assert cli.main([]) == 0, errors.getvalue()
            assert list((home / '.one-providers').glob('*.env'))
            assert cli.main(['--read-only', 'explain what this repo does']) == 0
            assert cli.main(['make a change in this repo']) == 0
        text = json.dumps(requests)
        assert 'Reply with the single word: ok' in text
        assert 'explain what this repo does' in text
        assert 'make a change in this repo' in text
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_readme_exact_examples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exercise_examples(tmp_path)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='onecli-readme-') as directory:
        import os
        os.chdir(directory)
        exercise_examples(Path(directory))
    print('README guided setup and both exact prompt examples: PASS (offline fixture)')

"""Real sockets and processes; no external provider or billing coverage."""
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

# Every wait below is a deadline for a loaded host, never the thing under test:
# a passing run returns as soon as the event fires, so a generous bound costs
# nothing, while a tight one turns a busy CI box into a false regression.
WAIT = 30


@pytest.mark.parametrize('phase', ['headers', 'body', 'stream'])
def test_http_interrupt_closes_socket_and_allows_next_request(phase):
    from onecli.providers.openai_wire import OpenAIWire
    ready = threading.Event()
    closed = threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            if len(requests) == 1:
                if phase != 'headers':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream' if phase == 'stream' else 'application/json')
                    self.end_headers()
                    if phase == 'stream':
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
                        self.wfile.flush()
                ready.set()
                self.connection.settimeout(WAIT)
                try:
                    if self.connection.recv(1) == b'':
                        closed.set()
                except (ConnectionResetError, BrokenPipeError):
                    closed.set()
            else:
                output = json.dumps({'choices': [{'message': {'content': 'resumed'}, 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(output)))
                self.end_headers()
                self.wfile.write(output)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    def interrupt():
        if ready.wait(WAIT):
            time.sleep(.1)
            os.kill(os.getpid(), signal.SIGINT)
    interrupter = threading.Thread(target=interrupt)
    interrupter.start()
    wire = OpenAIWire('http://127.0.0.1:%d' % server.server_port, 'local-test', timeout=2 * WAIT)
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            wire.chat([{'role': 'user', 'content': 'original'}], on_delta=(lambda *args: None) if phase == 'stream' else None)
        assert time.monotonic() - start < WAIT, 'interrupt waited for the transport timeout'
        assert closed.wait(WAIT), 'cancelled HTTP socket stayed open'
        assert len(requests) == 1, 'cancellation retried the request'
        result = wire.chat([{'role': 'user', 'content': 'newest'}])
        assert result.text == 'resumed'
        assert len(requests) == 2
    finally:
        interrupter.join(WAIT)
        server.shutdown()
        server.server_close()


def test_shell_interrupt_kills_grandchild_after_shell_exits(tmp_path):
    from onecli.tools.shell import bash
    pidfile = tmp_path / 'child.pid'
    child = 'import os,time,pathlib;pathlib.Path(%r).write_text(str(os.getpid()));time.sleep(30)' % str(pidfile)
    command = shlex.join([sys.executable, '-c', child]) + ' & exit 0'
    def interrupt():
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            if pidfile.exists():
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(.01)
    sender = threading.Thread(target=interrupt)
    sender.start()
    pid = None
    try:
        with pytest.raises(KeyboardInterrupt):
            bash(str(tmp_path), command, timeout=2 * WAIT)
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + WAIT
        while True:
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
            if not state or state.startswith('Z'):
                break
            if time.monotonic() > deadline:
                pytest.fail('shell grandchild survived interruption')
            time.sleep(.01)
    finally:
        sender.join(WAIT)
        if pid is None and pidfile.exists():
            pid = int(pidfile.read_text())
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('key', ['\x1b', '\x03'])
def test_cancel_permission_preserves_draft_and_queue(key):
    from onecli.terminal import Surface
    ui = Surface()
    ui.busy = True
    ui.draft = 'queued correction'
    ui.submit()
    ui.draft = 'unsent draft'
    ui.cursor = 6
    def cancel():
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            if ui.answering:
                ui.key(key)
                return
            time.sleep(.005)
    sender = threading.Thread(target=cancel)
    sender.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            ui.answer(timeout=2 * WAIT)
        assert not ui.answering
        assert (ui.draft, ui.cursor) == ('unsent draft', 6)
        assert ui.pending_messages() == ['queued correction']
        assert ui.answers.empty()
    finally:
        sender.join(WAIT)

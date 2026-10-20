"""Inherited socket feedback for one-cli's headless agent loop.

No listener or credentials: the parent passes one private socket descriptor.
The Agent stages input at a model-step boundary and acknowledges only after
request-budget admission, before invoking the provider.
"""
import json
import socket


class TurnFeedback:
    def __init__(self, fd):
        self.sock = socket.socket(fileno=fd)
        self.sock.setblocking(False)
        self.buffer = b''
        self.closed = False
        self.pending = []
        self.eof = False

    def consume(self, history, *, acknowledge=True):
        if self.closed:
            return False
        while not self.eof:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break
            except OSError:
                self.close()
                return False
            if not chunk:
                self.eof = True
                break
            self.buffer += chunk
            if len(self.buffer) > 1024 * 1024:
                self.close()
                return False
        changed = False
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            try:
                message = json.loads(line)
                if (not isinstance(message, dict) or not isinstance(message.get('id'), str)
                        or not isinstance(message.get('text'), str)
                        or not message['text'].strip()):
                    continue
            except (ValueError, UnicodeError):
                continue
            history.append({'role': 'user', 'content': message['text']})
            changed = True
            self.pending.append(message['id'])
        if acknowledge:
            self.acknowledge()
        if self.eof and not self.pending:
            self.close()
        return changed

    def acknowledge(self):
        """Confirm only when the caller has admitted the next model request."""
        while self.pending and not self.closed:
            ident = self.pending[0]
            try:
                self.sock.sendall(json.dumps({'id': ident, 'accepted': True}).encode() + b'\n')
            except OSError:
                self.close()
                break
            self.pending.pop(0)
        if self.eof:
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.buffer = b''
            self.pending.clear()
            self.sock.close()

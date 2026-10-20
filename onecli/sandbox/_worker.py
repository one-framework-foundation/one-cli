"""Trusted bootstrap inside the quota-limited namespace; output remains untrusted."""
import base64
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time


def kill_children():
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit() and int(entry.name) not in (1, os.getpid()):
            try:
                os.kill(int(entry.name), signal.SIGKILL)
            except ProcessLookupError:
                pass


def artifact(name, remaining):
    fd = os.open('/workspace', os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = name.split('/')
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        out = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(out, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > remaining:
                raise ValueError('unsafe or oversized artifact')
            data = stream.read(remaining + 1)
            if len(data) > remaining:
                raise ValueError('oversized artifact')
            return data
    finally:
        os.close(fd)


def main():
    config = json.loads(sys.argv[1])
    shutil.copytree('/input', '/workspace', dirs_exist_ok=True)
    process = subprocess.Popen(['/bin/sh', '-c', config['command']], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    data = bytearray()
    deadline = time.monotonic() + config['timeout_seconds']
    timed_out = limited = False
    with selectors.DefaultSelector() as poll:
        poll.register(process.stdout, selectors.EVENT_READ)
        while poll.get_map():
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            for key, _ in poll.select(min(left, .05)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    poll.unregister(key.fileobj)
                    continue
                room = config['max_output_bytes'] - len(data)
                data.extend(chunk[:room])
                if len(chunk) > room:
                    limited = True
                    break
            if limited:
                break
        if not (timed_out or limited):
            try:
                process.wait(timeout=max(.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
    kill_children()
    process.wait()
    process.stdout.close()
    outputs = {}
    remaining = 1048576
    if not (timed_out or limited):
        for name in config['artifacts']:
            value = artifact(name, remaining)
            remaining -= len(value)
            outputs[name] = base64.b64encode(value).decode('ascii')
    print(json.dumps(dict(returncode=process.returncode, output=base64.b64encode(data).decode('ascii'),
                          artifacts=outputs, timed_out=timed_out, output_limited=limited)), flush=True)


if __name__ == '__main__':
    main()

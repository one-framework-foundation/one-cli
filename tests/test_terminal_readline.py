"""Exercise lazy readline initialization against a real terminal, including libedit."""
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import time
import pytest


@pytest.mark.parametrize('rows,cols', [(24, 80), (18, 40), (10, 28)])
def test_readline_install_preserves_live_editor_mode(tmp_path, rows, cols):
    child = r'''
import json, pathlib, sys, termios, time
from onecli import terminal, inputline
inputline.HISTORY = pathlib.Path(sys.argv[1])
with terminal.session('input regression') as ui:
    inputline.install([])
    import readline
    backend = getattr(readline, 'backend', '') or readline.__doc__ or ''
    flags = termios.tcgetattr(0)[3]
    print('INPUT_PROBE_READY')
    until = time.monotonic() + 3
    while time.monotonic() < until and ui.draft != 'hello':
        time.sleep(.01)
    draft = ui.draft
print('INPUT_PROBE_RESULT=' + json.dumps({
    'raw': not bool(flags & (termios.ECHO | termios.ICANON | termios.ISIG)),
    'draft': draft,
    'backend': backend,
}))
'''
    import fcntl
    import struct
    import termios
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))
    proc = subprocess.Popen(
        [sys.executable, '-c', child, str(tmp_path / 'history')],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, TERM='xterm-256color', ONE_PLAIN='0'),
        stdin=slave, stdout=slave, stderr=slave,
    )
    os.close(slave)
    output = bytearray()
    sent = False
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], .05)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
            if not sent and b'INPUT_PROBE_READY' in output:
                os.write(master, b'hello')
                sent = True
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)
    assert proc.returncode == 0, repr(output)
    assert b'INPUT_PROBE_RESULT=' in output, repr(output)
    result = json.loads(bytes(output).split(b'INPUT_PROBE_RESULT=')[-1].splitlines()[0])
    assert result['raw'] is True
    assert result['draft'] == 'hello'
    if os.environ.get('ONE_REQUIRE_LIBEDIT') == '1':
        # Both names mean the NetBSD editline library. CPython's readline
        # module reports 'libedit' in older builds and 'editline' in the
        # Homebrew Python the macOS job installs, so the assertion is about
        # which library is linked, not which spelling it answers with.
        assert any(n in result['backend'].lower() for n in ('libedit', 'editline')), result

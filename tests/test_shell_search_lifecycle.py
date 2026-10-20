import os
import shlex
import sys
import time
from pathlib import Path

from onecli.tools.shell import bash, _recursive_grep


def test_recursive_search_guidance():
    for cmd in ['grep -ril glyph /home/user', 'cd project && grep -R x .', 'cat x | grep --recursive x .', '/usr/bin/grep --dereference-recursive x .']:
        assert _recursive_grep(cmd)
    for cmd in ['echo "grep -r x ."', 'grep -i x file.txt', 'rg -l glyph docs', "grep -- '-r' file", "grep -e '-r' file", 'cat file']:
        assert not _recursive_grep(cmd)
    assert 'Use the grep tool' in bash('.', 'grep -ril x .')


def test_shell_output_and_exit():
    assert bash('.', 'printf hello; printf error >&2; exit 7') == 'exit 7\nhelloerror'


def test_timeout_kills_grandchild(tmp_path):
    pidfile = tmp_path/'child.pid'
    code = 'import os,time; open('+repr(str(pidfile))+',"w").write(str(os.getpid())); time.sleep(30)'
    cmd = shlex.quote(sys.executable)+' -c '+shlex.quote(code)+' & wait'
    started=time.monotonic()
    assert 'timed out' in bash(str(tmp_path), cmd, timeout=1)
    assert time.monotonic()-started < 4
    pid=int(pidfile.read_text())
    for _ in range(50):
        stat=Path(f'/proc/{pid}/stat')
        try:
            state = stat.read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == 'Z':
            break
        time.sleep(.02)
    else:
        os.kill(pid,9)
        raise AssertionError('Timed-out shell left its child running')

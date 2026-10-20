"""Regression tests for the fifth v0.1.1 release-review round."""
import os
import subprocess
import sys
import textwrap

import pytest

from onecli.tools import fs

# A child process with a file-size limit: a write past it fails with EFBIG
# (SIGXFSZ ignored), the same partial-write failure as ENOSPC or a kill.
CAPPED = textwrap.dedent("""
    import resource, signal, sys
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    limit = int(sys.argv[1])
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    from onecli.tools import fs
    op, workspace, path = sys.argv[2:5]
    try:
        if op == "edit":
            print(fs.edit(workspace, path, "line 0\\n", "x" * 20000 + "\\n"))
        else:
            print(fs.write(workspace, path, "y" * 20000))
    except OSError as exc:
        print("OSError", exc.errno)
""")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("op", ["edit", "write"])
def test_a_failed_write_leaves_the_original_file_whole(tmp_path, op):
    target = tmp_path / "big.txt"
    original = "".join(f"line {i}\n" for i in range(400))
    target.write_text(original)
    limit = len(original) + 100
    subprocess.run([sys.executable, "-c", CAPPED, str(limit), op, str(tmp_path), "big.txt"],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert target.read_text() == original
    assert [p.name for p in tmp_path.iterdir()] == ["big.txt"], "temp file left behind"


def test_edit_and_write_keep_the_file_mode(tmp_path):
    target = tmp_path / "run.sh"
    target.write_text("echo a\n")
    target.chmod(0o750)
    fs.edit(str(tmp_path), "run.sh", "echo a", "echo b")
    assert target.stat().st_mode & 0o777 == 0o750
    fs.write(str(tmp_path), "run.sh", "echo c\n")
    assert target.stat().st_mode & 0o777 == 0o750
    assert target.read_text() == "echo c\n"


def test_edit_keeps_crlf_and_undecodable_bytes(tmp_path):
    target = tmp_path / "w.txt"
    target.write_bytes(b"one\r\n\xffkeep\r\ntwo\r\n")
    fs.edit(str(tmp_path), "w.txt", "two\n", "three\n")
    assert target.read_bytes() == b"one\r\n\xffkeep\r\nthree\r\n"


def test_write_still_refuses_a_hard_linked_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("keep\n")
    os.link(target, tmp_path / "b.txt")
    with pytest.raises(PermissionError):
        fs.write(str(tmp_path), "a.txt", "changed\n")
    with pytest.raises(PermissionError):
        fs.edit(str(tmp_path), "a.txt", "keep", "changed")
    assert target.read_text() == "keep\n"

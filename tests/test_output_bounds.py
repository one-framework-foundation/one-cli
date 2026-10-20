"""Output and file reads are bounded in memory, not only by the timeout."""

import time
import tracemalloc

from onecli import codemode
from onecli.tools import fs, shell


def test_runaway_shell_output_is_capped_and_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(shell, "OUTPUT_CAP", 64 * 1024)
    t0 = time.monotonic()
    out = shell.bash(str(tmp_path), "yes", timeout=30)
    assert time.monotonic() - t0 < 10
    assert "output exceeded 65536 bytes" in out
    assert "killed" in out
    assert len(out) < 64 * 1024 + 500


def test_normal_shell_output_unchanged(tmp_path):
    out = shell.bash(str(tmp_path), "echo hi; echo err >&2; exit 3")
    assert out == "exit 3\nhi\nerr"


def test_read_large_sparse_file_never_loads_it(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "READ_CAP", 4096)
    monkeypatch.setattr(fs, "_CHUNK", 1024)
    big = tmp_path / "sparse.bin"
    with open(big, "wb") as fh:
        fh.write(b"head\n")
        fh.truncate(64 * 1024 * 1024)  # one 64 MB line of NULs
    tracemalloc.start()
    first = fs.read(str(tmp_path), str(big))
    out = fs.read(str(tmp_path), str(big), offset=2)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 4 * 1024 * 1024
    assert first.startswith("     1\thead\n")
    assert "continue with offset=2" in first
    assert out.startswith("     2\t\0")
    assert "cut at 4096 characters" in out
    assert len(out) < 3 * 4096


def test_read_many_lines_stops_at_cap_with_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "READ_CAP", 1000)
    f = tmp_path / "many.txt"
    f.write_text("".join(f"line {i}\n" for i in range(10_000)))
    out = fs.read(str(tmp_path), str(f))
    assert len(out) < 1300
    assert "of 10000" in out and "read cap" in out and "continue with offset=" in out


def test_read_contract_unchanged(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a\nb\nc\nd\n")
    assert fs.read(str(tmp_path), str(f), offset=2, limit=2) == (
        "     2\tb\n     3\tc\n... (1 more lines; continue with offset=4)")
    assert "stopped at line 1 of 4" in fs.read(str(tmp_path), str(f), room=10)


def test_codemode_output_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(codemode, "OUTPUT_CAP", 10_000)
    out = codemode.execute("print('x' * 50_000)", str(tmp_path),
                           lambda *a, **k: "", timeout=30)
    assert "program output exceeded 10000 bytes" in out.output
    assert len(out.output) < 10_200


def test_bash_folds_crlf_and_cr_as_before(tmp_path):
    from onecli.tools import shell
    out = shell.bash(str(tmp_path), 'printf "a\\r\\nb\\rc"')
    assert "\r" not in out and out.endswith("a\nb\nc")

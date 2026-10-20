"""Spill: a bounded result the model can still get the rest of.

The behaviour that matters is not "output got shorter". It is that the bytes
withheld are still somewhere, that the model is told where, and that a store
which cannot write degrades to the old truncation instead of failing the run.
"""

from __future__ import annotations

import re

from onecli import spill
from onecli import tools as toolkit


BIG = "\n".join(f"line {i} of a very long build log" for i in range(4000))
# What the caller has left of the context window for this result.
ROOM = 12_000


def test_small_output_is_left_exactly_alone(tmp_path):
    store = spill.SpillStore("s1", tmp_path)
    assert spill.apply("short", "bash", store) == "short"
    assert list(tmp_path.glob("**/*.txt")) == []


def test_the_whole_output_is_on_disk_and_the_model_is_told_where(tmp_path):
    store = spill.SpillStore("s1", tmp_path)
    out = spill.apply(BIG, "bash", store, ROOM)

    assert len(out) < len(BIG)
    match = re.search(r"is at (\S+\.txt)", out)
    assert match, out[:400]
    path = match.group(1)
    assert open(path).read() == BIG, "the file must hold every byte, not the preview"
    assert "read it with the read tool" in out


def test_both_ends_survive_because_the_error_is_at_one_of_them(tmp_path):
    store = spill.SpillStore("s1", tmp_path)
    text = "FIRST LINE\n" + ("filler\n" * 5000) + "LAST LINE: it failed"
    out = spill.apply(text, "bash", store, ROOM)
    assert out.startswith("FIRST LINE")
    assert out.endswith("LAST LINE: it failed")


def test_files_are_numbered_per_session_not_per_call(tmp_path):
    store = spill.SpillStore("s1", tmp_path)
    spill.apply(BIG, "bash", store, ROOM)
    spill.apply(BIG, "grep", store, ROOM)
    names = sorted(p.name for p in (tmp_path / "s1").iterdir())
    assert names == ["001-bash.txt", "002-grep.txt"]


def test_a_tool_name_cannot_escape_the_session_directory(tmp_path):
    """The name comes from a tool, and an MCP server chooses its own names."""
    store = spill.SpillStore("s1", tmp_path)
    out = spill.apply(BIG, "../../etc/passwd", store, ROOM)
    written = list((tmp_path / "s1").iterdir())
    assert len(written) == 1
    assert written[0].resolve().parent == (tmp_path / "s1").resolve()
    assert str(tmp_path / "s1") in out


def test_no_store_falls_back_to_the_old_truncation():
    out = spill.apply(BIG, "bash", None, ROOM)
    assert "bytes of output dropped" in out
    assert len(out) < len(BIG)


def test_a_store_that_cannot_write_degrades_instead_of_failing(tmp_path):
    unwritable = tmp_path / "file-not-a-dir"
    unwritable.write_text("x")
    store = spill.SpillStore("s1", unwritable)
    out = spill.apply(BIG, "bash", store, ROOM)
    assert "could not spill" in out
    assert out.startswith("line 0 of")


def test_every_tool_goes_through_the_policy_including_new_ones(tmp_path):
    """The bound lives in run(), so a tool cannot forget to apply it."""
    store = spill.SpillStore("s2", tmp_path)
    out = toolkit.run("bash", {"command": "python3 -c \"print('x'*80000)\""},
                      str(tmp_path), spill=store, room=ROOM)
    assert "not shown" in out
    spilled = next((tmp_path / "s2").iterdir())
    assert len(spilled.read_text()) > 79_000


def test_bash_no_longer_truncates_before_the_policy_can_store_it(tmp_path):
    """The old shape lost the bytes inside the tool. This is that regression."""
    from onecli.tools import shell
    raw = shell.bash(str(tmp_path), "python3 -c \"print('y'*80000)\"")
    assert len(raw) > 79_000
    assert "dropped" not in raw


def test_reading_a_spill_file_never_spills_again(tmp_path):
    """A read of a spill file was itself over the
    threshold, so it spilled to a new file with one more layer of line numbers;
    the model read that one, and so on - 34 files in one turn, no answer."""
    store = spill.SpillStore("s1", tmp_path / "spills")
    wide = "\n".join(f"{i:6d}\tdef function_{i}(argument): return argument * {i}  # padding"
                     for i in range(1, 3000))
    out = spill.apply(wide, "bash", store, ROOM)
    path = re.search(r"is at (\S+\.txt)", out).group(1)

    offset, pages = 1, 0
    while True:
        page = toolkit.run("read", {"path": path, "offset": offset},
                           str(tmp_path), spill=store, room=ROOM)
        pages += 1
        assert len(page) <= ROOM + 200, "a read must bound itself to the room"
        assert "is at " not in page, "a read of a file on disk must not spill"
        match = re.search(r"continue with offset=(\d+)", page)
        if not match:
            break
        offset = int(match.group(1))
        assert pages < 200
    assert "  2999\t" in page, "paging must reach the end of the file"
    assert sorted(p.name for p in (tmp_path / "spills" / "s1").iterdir()) == ["001-bash.txt"]


def test_a_read_returns_the_whole_file_when_it_fits(tmp_path):
    """DeepSeek read lines 1-140 of a 50 KB SKILL.md under the old fixed 10 KB
    read budget and never continued."""
    skill = "\n".join(f"rule {i}: " + "keep wires on the 1.27 mm grid " * 3
                      for i in range(600))
    assert len(skill) > 50_000
    (tmp_path / "SKILL.md").write_text(skill)
    out = toolkit.run("read", {"path": "SKILL.md"}, str(tmp_path), room=200_000)
    assert "rule 599:" in out
    assert "continue with offset" not in out


def test_a_read_that_does_not_fit_says_where_to_continue(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join("x" * 100 for _ in range(500)))
    out = toolkit.run("read", {"path": "big.txt"}, str(tmp_path), room=ROOM)
    assert len(out) <= ROOM + 200
    assert re.search(r"of 500, \d+ bytes in all.*continue with offset=\d+", out)


def test_without_a_known_room_nothing_is_cut(tmp_path):
    store = spill.SpillStore("s3", tmp_path)
    assert spill.apply(BIG, "bash", store) == BIG

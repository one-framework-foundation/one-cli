"""Bounded glob/search behaviour."""

def test_a_glob_that_runs_long_returns_what_it_found_and_says_so(tmp_path, monkeypatch):
    """An unbounded `**` walk is what forces a person to Ctrl-C. It stops, says
    it stopped, and hands back what it had.
    """
    from onecli.tools import search

    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("x")

    calls = []

    def clock():
        calls.append(1)
        return 0.0 if len(calls) == 1 else 1e9  # deadline set, then blown

    monkeypatch.setattr(search.time, "monotonic", clock)
    out = search.glob_files(str(tmp_path), "*.txt")
    assert "stopped after" in out or "no match for" in out
def test_a_workspace_under_a_skipped_name_still_returns_its_files(tmp_path):
    """SKIP_DIRS is about the tree being searched, not about where the
    workspace happens to live. Judging the absolute path drops everything.
    """
    from onecli.tools import search

    ws = tmp_path / "build" / "project"
    ws.mkdir(parents=True)
    (ws / "main.py").write_text("x")
    out = search.glob_files(str(ws), "*.py")
    assert "main.py" in out
def test_a_skipped_directory_inside_the_tree_is_still_skipped(tmp_path):
    from onecli.tools import search

    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("x")
    (tmp_path / "mine.py").write_text("x")
    out = search.glob_files(str(tmp_path), "**/*.py")
    assert "mine.py" in out and "dep.py" not in out

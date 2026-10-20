"""Search behavior hot-reloads: a reloaded rank or search module serves the
next tool call in the same process, with no restart."""
import importlib

from onecli import tools
from onecli.tools import rank, search


def test_reloaded_search_modules_serve_the_next_call(tmp_path):
    (tmp_path / "a.txt").write_text("needle\n")
    before = tools.run("grep", {"pattern": "needle", "path": str(tmp_path)}, str(tmp_path))
    assert "a.txt:1:needle" in before
    original = rank.render
    try:
        rank.render = lambda *a, **k: "reloaded-render"
        assert tools.run("grep", {"pattern": "needle", "path": str(tmp_path)},
                         str(tmp_path)) == "reloaded-render"
        original_grep = search.grep
        search.grep = lambda workspace, **kw: "reloaded-grep"
        assert tools.run("grep", {"pattern": "needle"}, str(tmp_path)) == "reloaded-grep"
        search.grep = original_grep
        original_glob = search.glob_files
        search.glob_files = lambda workspace, **kw: "reloaded-glob"
        assert tools.run("glob", {"pattern": "a"}, str(tmp_path)) == "reloaded-glob"
        search.glob_files = original_glob
    finally:
        rank.render = original
    importlib.reload(rank)
    assert tools.run("grep", {"pattern": "needle", "path": str(tmp_path)},
                     str(tmp_path)) == before

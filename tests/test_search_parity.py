"""One fixture corpus through rg, pure Python and the guarded path:
the ranked output must be identical. A missing backend is a skip that says so,
never a pass, and the backend that answered is always named in the header."""
import os
import shutil
import subprocess
import time

import pytest

from onecli.tools import search

NOW = time.time()
CORPUS = {
    # path: (text, age in days)
    "docs/dream_world.md": ("# Dream World\nThe dream world map.\nworld of dreams\n", 0),
    "docs/notes.md": ("a dream\nanother dream\nno world here\n", 3),
    "src/world.py": ("import os\nworld = 1\nprint(world)\n" + "world\n" * 12, 40),
    "src/colour.py": ("import re\ncolor = 'red'\ncolour = 'blue'\n", 10),
    "src/crlf.txt": ("dream world\r\nsecond line\r\n", 1),
    "README.md": ("Dream world, a stage.\nColor guide.\n", 20),
    "bin/blob.bin": ("dream\0world\0", 0),
}
QUERIES = [
    ("dream world", ""),          # multi-term, smart case: insensitive
    ("Dream World", ""),          # uppercase: case-sensitive
    ("colou?r", ""),              # regex
    ("^import [a-z]+", "*.py"),   # anchored regex with a name glob
    ("world", "docs/*.md"),       # path glob, applied by the shared rule
    ("world", ""),                # truncation: one file has 14 hits
]


def _write(root):
    for name, (text, days) in CORPUS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
        stamp = NOW - days * 86400
        os.utime(path, (stamp, stamp))


@pytest.fixture
def corpora(tmp_path):
    repo, plain = tmp_path / "repo", tmp_path / "plain"
    for root in (repo, plain):
        root.mkdir()
        _write(root)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    return repo, plain


class AllowAll:
    def check(self, *_args):
        class Verdict:
            allowed = True
        return Verdict()


def _body(text, root):
    lines = text.replace(str(root), "<root>").splitlines()
    assert lines[0].startswith("[")
    head = lines[0].split("] ", 1)[1]
    keep = [line for line in lines[1:]
            if line not in (search.WALK_NOTE, search.SIZE_NOTE) and "tracked files only" not in line]
    return "\n".join([head, *keep])


def _run(backend, repo, plain, pattern, glob):
    if backend == "python-tracked":
        out = search.search(str(repo), pattern, glob=glob, max_results=12, backend="python")
        assert out.startswith("[python]")
        return _body(out, repo)
    if backend == "rg":
        if not shutil.which("rg"):
            pytest.skip("rg not on PATH on this host: rg parity not measured")
        out = search.search(str(plain), pattern, glob=glob, max_results=12, backend="rg")
        assert out.startswith("[rg]")
        return _body(out, plain)
    if backend == "python-walk":
        out = search.search(str(plain), pattern, glob=glob, max_results=12, backend="python")
        assert out.startswith("[python]")
        return _body(out, plain)
    out = search.restricted("grep", str(plain), {"pattern": pattern, "glob": glob,
                                                  "max_results": 12}, [AllowAll()])
    assert out.startswith("[python: guarded workspace]")
    return _body(out, plain)


BACKENDS = ["rg", "python-tracked", "python-walk", "restricted"]


@pytest.mark.parametrize("pattern,glob", QUERIES)
@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "python-tracked"])
def test_backends_give_identical_ranked_output(corpora, backend, pattern, glob):
    repo, plain = corpora
    reference = _run("python-tracked", repo, plain, pattern, glob)
    assert _run(backend, repo, plain, pattern, glob) == reference


def test_ranking_recall_and_announced_cuts(corpora):
    repo, plain = corpora
    out = search.search(str(plain), "dream world", backend="python", max_results=12)
    files = [line.split(":")[0].replace(str(plain) + "/", "")
             for line in out.splitlines() if line.startswith(str(plain)) and ":" in line]
    order = list(dict.fromkeys(files))
    # Both terms and the name beat one term; the binary file is never a hit.
    assert order[0] == "docs/dream_world.md"
    assert order.index("docs/notes.md") > order.index("src/crlf.txt")
    assert "bin/blob.bin" not in out
    assert "src/crlf.txt:1:dream world\n" in out.replace(str(plain) + "/", "") + "\n"
    capped = search.search(str(plain), "world", backend="python", max_results=9)
    # Densest file first; context lines marked '-' like grep -C; every cut counted.
    assert capped.splitlines()[1] == f"{plain}/src/world.py-1-import os"
    assert "(+11 more matching lines in this file)" in capped
    assert "more matching lines truncated" in capped
    assert search.WALK_NOTE in capped


def test_case_and_literal_plan():
    from onecli.tools import rank
    assert rank.plan("dream world").ignore_case
    assert not rank.plan("Dream").ignore_case
    assert rank.plan(r"\Sfoo").ignore_case      # escapes are not uppercase text
    assert rank.plan("a.b").literal is False
    assert rank.plan("foo-bar baz").terms == ("foo-bar", "baz")
    assert rank.plan("[").startswith("error: bad regex")


def test_skip_dirs_are_announced_and_searchable_explicitly(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("needle\n")
    for backend in ("python", "rg") if shutil.which("rg") else ("python",):
        out = search.search(str(tmp_path), "needle", backend=backend)
        assert "no matches" in out and search.WALK_NOTE in out
        assert "x.js:1:needle" in search.search(str(tmp_path / "node_modules"), "needle",
                                                backend=backend)


def test_glob_bare_word_finds_names_at_any_depth_and_counts_cuts(tmp_path, monkeypatch):
    (tmp_path / "deep" / "er").mkdir(parents=True)
    (tmp_path / "deep" / "er" / "Acme.md").write_text("x")
    (tmp_path / "acme").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "acme.js").write_text("x")
    out = search.glob_files(str(tmp_path), "acme")
    lines = out.splitlines()
    assert lines[0].startswith("(no wildcard")
    assert lines[1:] == [str(tmp_path / "acme"), str(tmp_path / "deep/er/Acme.md")]
    guarded = search.restricted("glob", str(tmp_path), {"pattern": "acme"}, [AllowAll()])
    assert sorted(guarded.splitlines()[1:]) == sorted(lines[1:])
    monkeypatch.setattr(search, "GLOB_SHOWN", 1)
    assert "(1 more files truncated" in search.glob_files(str(tmp_path), "acme")


def test_review_edge_cases_are_announced_not_silent(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")

    class Deny:
        def check(self, *_a):
            return type("V", (), {"allowed": False})()
    refused = search.restricted("grep", str(tmp_path), {"pattern": "hello"}, [Deny()])
    assert "refused by policy" in refused
    assert search.SIZE_NOTE in search.search(str(tmp_path), "hello", backend="python")
    assert "a.txt:1:hello" in search.search(str(tmp_path), "hello", max_results=None,
                                            context=None)
    assert search.glob_files(str(tmp_path), "").startswith("error:")


def test_glob_deadline_fires_even_when_nothing_matches(tmp_path, monkeypatch):
    """A broad pattern with no match must still hit the deadline, not only
    when a match is yielded."""
    for i in range(30):
        (tmp_path / f"d{i}").mkdir()
        (tmp_path / f"d{i}" / "f.txt").write_text("x")
    monkeypatch.setattr(search, "GLOB_DEADLINE", -1.0)
    for pattern in ("**/*.nomatch", "nomatch"):
        assert search.glob_files(str(tmp_path), pattern).startswith("error: no match")

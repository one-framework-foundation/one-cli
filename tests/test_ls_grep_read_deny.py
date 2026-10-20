"""A `Read(src/secret/**)` deny hid the files from glob, but `ls src/secret`
listed their names and grep's refusal note named one of them."""

from __future__ import annotations

from onecli import permissions, tools


def _denied_workspace(tmp_path):
    (tmp_path / "src" / "secret").mkdir(parents=True)
    (tmp_path / "src" / "secret" / "keys.txt").write_text("token=hidden\n")
    (tmp_path / "src" / "ok.txt").write_text("token=public\n")
    guard = permissions.PermissionGuard(
        deny=[permissions.Rule.parse("Read(src/secret/**)")], workspace=str(tmp_path))
    return str(tmp_path), [guard]


def test_ls_hides_names_a_read_deny_hides_from_glob(tmp_path):
    ws, guards = _denied_workspace(tmp_path)
    globbed = tools.run("glob", {"pattern": "**/*.txt"}, ws, path_guards=guards)
    assert "keys.txt" not in globbed and "ok.txt" in globbed
    listed = tools.run("ls", {"path": "src/secret"}, ws, path_guards=guards)
    assert "keys.txt" not in listed
    assert "ok.txt" in tools.run("ls", {"path": "src"}, ws, path_guards=guards)
    # Without a policy nothing is hidden.
    assert "keys.txt" in tools.run("ls", {"path": "src/secret"}, ws)


def test_grep_never_names_or_reads_a_denied_file(tmp_path):
    ws, guards = _denied_workspace(tmp_path)
    for pattern in ("token", "zzz"):
        out = tools.run("grep", {"pattern": pattern}, ws, path_guards=guards)
        assert "keys.txt" not in out and "hidden" not in out
        assert "1 paths refused by policy" in out
    assert "ok.txt:1:token=public" in tools.run("grep", {"pattern": "token"}, ws,
                                                path_guards=guards)

"""Local security regressions. No providers, credentials or network."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from onecli import guard, hooks, mcp, permissions, session, spill, tools
from onecli.agent import Agent
from onecli.providers.base import ToolCall, Turn
from onecli.subagent import EXPLORER, Spawner


class Scripted:
    def __init__(self, calls):
        self.turns = iter([Turn(tool_calls=calls), Turn(text="done")])

    def chat(self, *args, **kwargs):
        return next(self.turns)


def run_calls(tmp_path, names, calls, **kwargs):
    agent = Agent(Scripted(calls), str(tmp_path), tool_names=names,
                  max_steps=3, autocompact=False, **kwargs)
    return agent.run("local fixture")


@pytest.mark.parametrize("surface", [["read"], []])
def test_unoffered_write_never_executes(tmp_path, surface):
    run_calls(tmp_path, surface, [ToolCall("x", "write", {"path": "stolen", "content": "x"})])
    assert not (tmp_path / "stolen").exists()


def test_explorer_cannot_dispatch_global_write(tmp_path):
    parent = Agent(Scripted([ToolCall("x", "write", {"path": "stolen", "content": "x"})]),
                   str(tmp_path), max_steps=3, autocompact=False)
    child = Spawner(parent=parent).build(EXPLORER)
    assert "write" not in child.tool_names
    child.run("local fixture")
    assert not (tmp_path / "stolen").exists()


def test_code_binding_cannot_dispatch_unoffered_write(tmp_path):
    run_calls(tmp_path, ["code", "read"], [ToolCall("x", "code", {
        "source": "one.write(path='stolen', content='x')"})])
    assert not (tmp_path / "stolen").exists()


def test_project_hook_requires_per_run_trust(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [
        {"type": "command", "command": "printf TRUSTED_PROJECT_HOOK"}]}]}}))
    assert not hooks.build(str(tmp_path)).session_start().context
    assert hooks.build(str(tmp_path), trust_project=True).session_start().context == ["TRUSTED_PROJECT_HOOK"]
    assert hooks.build(str(tmp_path), extra=[str(settings)]).session_start().context == ["TRUSTED_PROJECT_HOOK"]


def test_project_mcp_requires_per_run_trust(tmp_path):
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {
        "fixture": {"command": "/usr/bin/true"}}}))
    assert mcp.load_configs([], str(tmp_path)) == []
    assert len(mcp.load_configs([], str(tmp_path), trust_project=True)) == 1
    assert mcp.load_configs([], str(tmp_path), strict=True, trust_project=True) == []


def test_mcp_does_not_inherit_operator_credentials(monkeypatch):
    monkeypatch.setenv("RELEASE_TEST_SECRET", "synthetic-only")
    monkeypatch.setenv("PYTHONPATH", "/untrusted-injection")
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs["env"])
        raise OSError("fixture does not launch a process")

    monkeypatch.setattr(mcp.subprocess, "Popen", capture)
    server = mcp.StdioServer(mcp.ServerSpec("fixture", sys.executable,
                                           env={"EXPLICIT_FIXTURE": "allowed"}))
    assert not server.start()
    assert "RELEASE_TEST_SECRET" not in seen
    assert "PYTHONPATH" not in seen
    assert seen["EXPLICIT_FIXTURE"] == "allowed"


@pytest.mark.parametrize("tool", ["read", "write", "edit", "ls", "glob", "grep"])
def test_roots_refuse_symlink_escape(tmp_path, tool):
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    policy = permissions.RootsGuard([str(workspace)], str(workspace))
    assert not policy.check(tool, {"path": "link/new.txt"}, False).allowed


def test_path_allowlist_refuses_symlink_escape(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent / "outside")
    policy = permissions.PermissionGuard(permissions.parse_rules(["Read(link)"]), workspace=str(tmp_path))
    assert not policy.check("read", {"path": "link"}, False).allowed


@pytest.mark.parametrize("tool,args", [
    ("read", {}), ("write", {"content": "changed"}),
    ("edit", {"old_string": "outside", "new_string": "changed"}),
])
@pytest.mark.parametrize("ancestor", [False, True])
def test_file_operation_refuses_link_swapped_after_guard(tmp_path, tool, args, ancestor):
    workspace = tmp_path / "work"
    workspace.mkdir()
    target = workspace / "target"
    outside = tmp_path / "outside"
    if ancestor:
        (workspace / "subdir").mkdir()
        (tmp_path / "outside-dir").mkdir()
        target = workspace / "subdir" / "target"
        outside = tmp_path / "outside-dir" / "target"
    target.write_text("inside")
    outside.write_text("outside")
    relative = str(target.relative_to(workspace))
    policy = permissions.RootsGuard([str(workspace)], str(workspace))
    assert policy.check(tool, {"path": relative}, tool != "read").allowed
    # Deterministic gate-to-use race: replace the authorized name after check.
    if ancestor:
        target.parent.rename(tmp_path / "moved")
        (workspace / "subdir").symlink_to(outside.parent, target_is_directory=True)
    else:
        target.unlink()
        target.symlink_to(outside)
    result = tools.run(tool, {"path": relative, **args}, str(workspace))
    assert result.startswith("error:")
    assert outside.read_text() == "outside"


def test_guarded_grep_does_not_read_descendant_symlinks(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("SYNTHETIC_PRIVATE_MATCH")
    (workspace / "link.txt").symlink_to(tmp_path / "secret.txt")
    # The Python fallback must not follow descendant symlinks (rg skips them).
    from onecli.tools import search
    from unittest.mock import patch
    policy = guard.GuardChain([permissions.RootsGuard([str(workspace)], str(workspace))])
    with patch.object(search.shutil, "which", return_value=None):
        result = run_calls(workspace, ["grep"], [ToolCall("x", "grep", {
            "path": ".", "pattern": "SYNTHETIC_PRIVATE_MATCH"})], guard=policy)
    assert "SYNTHETIC_PRIVATE_MATCH" not in result.steps[0].tool_results[0][1].split("(no matches")[0]


def test_glob_cannot_escape_through_pattern(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("outside")
    result = tools.run("glob", {"pattern": "../*.txt"}, str(workspace))
    assert "outside.txt" not in result


def test_session_and_draft_created_private_under_permissive_umask(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    previous = os.umask(0)
    try:
        session.append("fixture", {"kind": "messages", "messages": []})
        session.save_draft("fixture", "synthetic draft")
    finally:
        os.umask(previous)
    assert session.SESSION_DIR.stat().st_mode & 0o777 == 0o700
    for path in session.SESSION_DIR.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("link_kind", ["file", "directory", "hardlink"])
def test_session_storage_refuses_unsafe_links(tmp_path, monkeypatch, link_kind):
    root = tmp_path / "sessions"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("untouched\n")
    monkeypatch.setattr(session, "SESSION_DIR", root)
    if link_kind == "file":
        (root / "fixture.jsonl").symlink_to(outside)
    elif link_kind == "hardlink":
        os.link(outside, root / "fixture.jsonl")
    else:
        root.rmdir()
        root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        session.append("fixture", {"kind": "synthetic"})
    assert outside.read_text() == "untouched\n"


def test_spills_are_private(tmp_path):
    store = spill.SpillStore("fixture", tmp_path / "spills")
    previous = os.umask(0)
    try:
        store.spill("read", "synthetic" * 2000, 8_000)
    finally:
        os.umask(previous)
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert next(store.root.iterdir()).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "startup"])
def test_guard_failure_denies_execution(monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("fixture", 0.1)
        if failure == "startup":
            raise OSError("synthetic missing process")
        return subprocess.CompletedProcess("fixture", 1, "", "synthetic failure")
    monkeypatch.setattr(guard.subprocess, "run", fail)
    assert not guard.SubprocessGuard(["fixture"]).check("write", {}, True).allowed


@pytest.mark.parametrize("native", [True, False])
def test_serving_only_exposes_host_loopback(tmp_path, native):
    from onecli.serving import catalog, hostfacts, recipes
    facts = hostfacts.HostFacts(llama_server="llama-server" if native else "", docker=True)
    model = catalog.Model(name="fixture", repo="fixture/model", file="fixture.gguf")
    argv = recipes.command(model, recipes.Recipe("baseline"), 9123, facts=facts, root=tmp_path)
    if native:
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
    else:
        assert argv[argv.index("-p") + 1] == "127.0.0.1:9123:9123"



@pytest.mark.parametrize("tool,args", [
    ("read", {}), ("write", {"content": "changed"}),
    ("edit", {"old_string": "outside", "new_string": "changed"}),
    ("ls", {}),
])
def test_symlink_parent_components_cannot_change_authorized_destination(tmp_path, tool, args):
    workspace = tmp_path / "work"
    (workspace / "deep" / "inner").mkdir(parents=True)
    (workspace / "link").symlink_to(workspace / "deep" / "inner", target_is_directory=True)
    outside = tmp_path / "target"
    outside.write_text("outside")
    (workspace / "target").write_text("inside")
    # realpath authorizes work/target; lexical cancellation opens tmp/target.
    path = "link/../../target"
    policy = permissions.RootsGuard([str(workspace)], str(workspace))
    assert policy.check(tool, {"path": path}, tools.is_mutating(tool)).allowed
    output = tools.run(tool, {"path": path, **args}, str(workspace))
    assert output.startswith("error:"), output
    assert outside.read_text() == "outside"


@pytest.mark.parametrize("pattern", ["*.py", "sub/*.py", "s*/*.py", "*/x.py", "**", "sub/**/x.py"])
def test_guarded_glob_matches_root_anchored_patterns(tmp_path, pattern):
    for relative in ["root.py", "sub/a.py", "sub/x.py", "sub/deep/x.py", "other/sub/a.py"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    policy = permissions.RootsGuard([str(tmp_path)], str(tmp_path))
    arguments = {"pattern": pattern}
    ordinary = tools.run("glob", arguments, str(tmp_path))
    restricted = tools.run("glob", arguments, str(tmp_path), path_guards=[policy])
    assert set(restricted.splitlines()) == set(ordinary.splitlines())


def test_guarded_glob_does_not_descend_without_recursive_pattern(tmp_path, monkeypatch):
    from onecli import secureio
    (tmp_path / "root.py").write_text("fixture")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.py").write_text("fixture")
    opened = []
    original = secureio.directory

    def track(path, **kwargs):
        opened.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(secureio, "directory", track)
    policy = permissions.RootsGuard([str(tmp_path)], str(tmp_path))
    tools.run("glob", {"pattern": "*.py"}, str(tmp_path), path_guards=[policy])
    assert tmp_path / "nested" not in opened


@pytest.mark.parametrize("tool", ["glob", "grep"])
def test_guarded_search_does_not_erase_symlink_parent_components(tmp_path, monkeypatch, tool):
    from onecli import secureio
    workspace = tmp_path / "work"
    (workspace / "deep" / "inner").mkdir(parents=True)
    (workspace / "link").symlink_to(workspace / "deep" / "inner", target_is_directory=True)
    policy = permissions.RootsGuard([str(tmp_path)], str(workspace))
    opened = []
    original = secureio.directory

    def track(path, **kwargs):
        opened.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(secureio, "directory", track)
    tools.run(tool, {"path": "link/../..", "pattern": "*" if tool == "glob" else "fixture"},
              str(workspace), path_guards=[policy])
    assert tmp_path not in opened


@pytest.mark.parametrize("path", ["../unrequested/../work/result.txt", "unrequested/.."])
def test_parent_creation_path_has_no_mocked_mutations(monkeypatch, path):
    from onecli import secureio
    from io import StringIO
    from types import SimpleNamespace
    effects = []
    policy = permissions.RootsGuard(["/virtual-fixture/work"], "/virtual-fixture/work")
    assert policy.check("write", {"path": path}, True).allowed

    def mkdir(component, **kwargs):
        effects.append(("mkdir", component))
        raise FileExistsError(component)

    monkeypatch.setattr(secureio.os, "mkdir", mkdir)
    monkeypatch.setattr(secureio.os, "open", lambda *args, **kwargs: 123)
    monkeypatch.setattr(secureio.os, "close", lambda fd: None)
    monkeypatch.setattr(secureio.os, "fstat", lambda fd: SimpleNamespace(st_mode=0o100600, st_nlink=1))
    monkeypatch.setattr(secureio.os, "ftruncate", lambda *args: effects.append(("truncate", args)))
    monkeypatch.setattr(secureio.os, "fdopen", lambda *args, **kwargs: StringIO())
    try:
        with secureio.text_file(Path("/virtual-fixture/work") / path, "w") as fh:
            fh.write("fixture")
    except OSError:
        pass
    assert effects == []


@pytest.mark.parametrize("path", ["../unrequested/../work/result.txt", "unrequested/.."])
def test_parent_creation_path_has_no_real_intermediate_side_effects(tmp_path, path):
    # Every actual filesystem operation is confined to this disposable fixture.
    workspace = tmp_path / "work"
    workspace.mkdir()
    policy = permissions.RootsGuard([str(workspace)], str(workspace))
    assert policy.check("write", {"path": path}, True).allowed
    before = set(tmp_path.rglob("*"))
    output = tools.run("write", {"path": path, "content": "fixture"}, str(workspace))
    assert set(tmp_path.rglob("*")) == before
    assert output.startswith("error:")


def test_directory_creation_rejects_parent_components_before_mkdir(tmp_path, monkeypatch):
    from onecli import secureio
    calls = []
    original = os.mkdir

    def track(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(secureio.os, "mkdir", track)
    try:
        with secureio.directory(tmp_path / "unrequested" / "..", create=True):
            pass
    except OSError:
        pass
    assert calls == []


def test_directory_follows_root_owned_system_symlink(tmp_path):
    from onecli import secureio

    if os.geteuid() != 0 and not (Path("/bin").is_symlink() or Path("/var").is_symlink()):
        pytest.skip("no root-owned symlinked system directory on this host")
    target = Path("/bin") if Path("/bin").is_symlink() else Path("/var")
    with secureio.directory(target) as fd:
        assert os.fstat(fd).st_uid == 0


def test_directory_refuses_user_owned_symlinked_directory(tmp_path):
    from onecli import secureio

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        with secureio.directory(link):
            pass


def test_no_private_address_ships_as_a_package_default():
    """No `100.64.0.0/10` address may default-ship in the package.

    Service endpoints (speech, TTS, vision) must be opt-in via env/config, never a
    hardcoded fallback - an outside install must never phone home to a host the
    user did not configure.
    """
    import re

    pkg_root = Path(__file__).resolve().parent.parent / "onecli"
    # A URL literal, not the 100.64.0.0/10 CGNAT range check used to
    # classify an address's kind - that range test is not a hardcoded target.
    private_range = re.compile(r"://100\.64\.\d+\.\d+")
    offenders = []
    for path in pkg_root.rglob("*.py"):
        text = path.read_text()
        if private_range.search(text):
            offenders.append(str(path))
    assert offenders == [], f"private address hardcoded in: {offenders}"


def test_shipped_glyph_bundles_use_demo_roster_only():
    """Generated terminal art carries only the demo roster."""
    import re

    root = Path(__file__).resolve().parent.parent
    for relative in ("onecli/assets/glyph_terminal.mjs",):
        bundle = (root / relative).read_text()
        entries = re.findall(
            r'\{agent:"([^"]+)",host:"([^"]+)",user:"([^"]+)"\}', bundle
        )
        if len(entries) != 8 or any(
            not (agent.startswith("demo") and host.startswith("local-")
                 and user == "operator")
            for agent, host, user in entries
        ):
            pytest.fail(f"{relative} embeds an unapproved roster")

"""Permission patterns, modes, project context and readable roots."""

from __future__ import annotations

import os

import pytest

from onecli import global_context, permissions
from onecli import tools as toolkit
from onecli.cli import build_parser, _guards, _mode, _system

WS = "/ws"


# --- the pattern matcher ---------------------------------------------------

# (pattern, tool, arguments, expected)
TABLE = [
    # A bare tool name ignores the arguments entirely.
    ("Read", "read", {"path": "anything.py"}, True),
    ("Read", "write", {"path": "anything.py"}, False),
    ("read", "read", {"path": "a"}, True),          # our spelling
    ("BASH", "bash", {"command": "ls"}, True),      # case-insensitive

    # Command clauses are prefix matches.
    ("Bash(git *)", "bash", {"command": "git status"}, True),
    ("Bash(git *)", "bash", {"command": "git log --oneline -5"}, True),
    ("Bash(git *)", "bash", {"command": "git"}, False),
    # The near-miss: the trailing space is part of the prefix, so a longer
    # program name starting with the same letters is NOT covered.
    ("Bash(git *)", "bash", {"command": "gitlab-cli status"}, False),
    ("Bash(npm run test:*)", "bash", {"command": "npm run test --watch"}, True),
    # `:*` is a word prefix, as in Claude Code: a longer script name is
    # another command, not this one with arguments.
    ("Bash(npm run test:*)", "bash", {"command": "npm run test:unit"}, False),
    ("Bash(npm run test:*)", "bash", {"command": "npm run build"}, False),
    # No trailing star is an exact match on the whole command line.
    ("Bash(git status)", "bash", {"command": "git status"}, True),
    ("Bash(git status)", "bash", {"command": "git status --short"}, False),

    # Path clauses are globs, relative ones anchored at the workspace.
    ("Write(src/**)", "write", {"path": "src/a.py"}, True),
    ("Write(src/**)", "write", {"path": "src/deep/nest/a.py"}, True),
    ("Write(src/**)", "write", {"path": "./src/a.py"}, True),
    ("Write(src/**)", "write", {"path": "/ws/src/a.py"}, True),
    ("Write(src/**)", "write", {"path": "tests/a.py"}, False),
    ("Write(src/**)", "write", {"path": "../outside/a.py"}, False),
    # A single star stops at the separator.
    ("Read(src/*.py)", "read", {"path": "src/a.py"}, True),
    ("Read(src/*.py)", "read", {"path": "src/deep/a.py"}, False),
    # `//` anchors at the filesystem root, not the workspace.
    ("Edit(//tmp/scratch/**)", "edit", {"path": "/tmp/scratch/x.txt"}, True),
    ("Edit(//tmp/scratch/**)", "edit", {"path": "/tmp/other/x.txt"}, False),

    # A clause on a tool whose matchable argument is absent matches nothing.
    ("Write(src/**)", "write", {"content": "x"}, False),
    ("Bash(git *)", "bash", {}, False),

    # An MCP-shaped pattern parses and never matches our tools.
    ("mcp__server__tool", "bash", {"command": "ls"}, False),
]


@pytest.mark.parametrize("pattern,tool,arguments,expected", TABLE)
def test_pattern_table(pattern, tool, arguments, expected):
    rule = permissions.Rule.parse(pattern)
    assert rule.matches(tool, arguments, WS) is expected


def test_star_star_matches_zero_directories():
    rule = permissions.Rule.parse("Write(src/**/a.py)")
    assert rule.matches("write", {"path": "src/a.py"}, WS)
    assert rule.matches("write", {"path": "src/x/y/a.py"}, WS)


def test_home_anchored_clause(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    rule = permissions.Rule.parse("Read(~/notes/**)")
    assert rule.matches("read", {"path": str(tmp_path / "notes" / "a.md")}, WS)
    assert not rule.matches("read", {"path": "/elsewhere/a.md"}, WS)


def test_bad_patterns_are_rejected_at_parse_time():
    for bad in ["Bash(", "(git *)", "Bash()", ""]:
        with pytest.raises(ValueError):
            permissions.Rule.parse(bad)


def test_rules_are_repeatable_and_comma_separated():
    rules = permissions.parse_rules(["Read,Bash(git *)", "Write(src/**)"])
    assert [r.raw for r in rules] == ["Read", "Bash(git *)", "Write(src/**)"]


def test_a_comma_inside_a_clause_does_not_split_it():
    rules = permissions.parse_rules(["Bash(git commit -m a,b)"])
    assert len(rules) == 1
    assert rules[0].clause == "git commit -m a,b"


# --- allow and deny --------------------------------------------------------

def test_an_empty_allow_list_permits_everything():
    g = permissions.PermissionGuard(workspace=WS)
    assert g.check("bash", {"command": "rmdir x"}, True).allowed


def test_a_non_empty_allow_list_is_a_whitelist():
    g = permissions.PermissionGuard(
        allow=permissions.parse_rules(["Read", "Bash(git *)"]), workspace=WS)
    assert g.check("read", {"path": "a.py"}, False).allowed
    assert g.check("bash", {"command": "git status"}, True).allowed
    verdict = g.check("write", {"path": "a.py"}, True)
    assert not verdict.allowed
    assert "not on the allowed list" in verdict.reason


def test_deny_beats_allow():
    g = permissions.PermissionGuard(
        allow=permissions.parse_rules(["Bash(git *)"]),
        deny=permissions.parse_rules(["Bash(git push*)"]),
        workspace=WS)
    assert g.check("bash", {"command": "git status"}, True).allowed
    verdict = g.check("bash", {"command": "git push origin main"}, True)
    assert not verdict.allowed
    assert "disallowed list" in verdict.reason
    assert "Bash(git push*)" in verdict.reason


def test_a_denial_is_a_verdict_not_an_exception():
    g = permissions.PermissionGuard(
        deny=permissions.parse_rules(["Write"]), workspace=WS)
    verdict = g.check("write", {"path": "a.py"}, True)
    assert verdict.allowed is False
    assert verdict.source == "permissions"


# --- permission modes ------------------------------------------------------

def test_read_only_is_an_alias_for_the_read_only_mode():
    assert permissions.normalise_mode(None, read_only=True) == "readOnly"
    assert permissions.normalise_mode("default", read_only=True) == "readOnly"
    assert permissions.normalise_mode(None) == "default"
    # An explicit mode is never overridden by the alias.
    assert permissions.normalise_mode("plan", read_only=True) == "plan"


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        permissions.normalise_mode("yolo")


@pytest.fixture
def no_web_tools(monkeypatch):
    """Pin the surface: web tools register only where an installation provides
    them, so a test that asserts an exact tool list has to say which it means."""
    from onecli import tools as toolkit
    monkeypatch.setattr(toolkit, "REGISTRY", dict(toolkit.REGISTRY))
    for name in permissions.READ_ONLY_WEB_TOOLS:
        toolkit.REGISTRY.pop(name, None)


@pytest.mark.parametrize("mode,expected", [
    ("default", None),
    ("acceptEdits", None),
    ("bypassPermissions", None),
    ("plan", permissions.READ_ONLY_TOOLS),
    ("readOnly", permissions.READ_ONLY_TOOLS),
])
def test_tool_surface_per_mode(mode, expected, no_web_tools):
    assert permissions.tool_names_for(mode) == expected


def test_plan_mode_switches_the_system_prompt_and_says_nothing_changed():
    out = permissions.compose_system("BASE", WS, mode="plan", project=False)
    assert "PLAN MODE" in out
    assert "No changes were made" in out
    assert permissions.compose_system("BASE", WS, mode="default",
                                      project=False) == "BASE"


def test_bypass_permissions_drops_the_rules_but_keeps_the_boundary():
    built = permissions.build(WS, mode="bypassPermissions",
                              disallowed=["Write"], add_dirs=["/tmp"])
    assert [type(g) for g in built] == [permissions.RootsGuard]
    assert built[0].check("write", {"path": "/tmp/a"}, True).allowed
    assert not built[0].check("read", {"path": "/etc/hosts"}, False).allowed


def test_read_only_mode_still_refuses_a_mutating_call_in_the_chain(tmp_path):
    args = build_parser().parse_args(["--read-only", "-w", str(tmp_path), "x"])
    assert _mode(args) == "readOnly"
    chain = _guards(args)
    assert not chain.check("write", {"path": "a.py"}, True).allowed
    assert chain.check("read", {"path": "a.py"}, False).allowed


def test_plan_mode_withholds_every_mutating_tool(tmp_path, no_web_tools):
    args = build_parser().parse_args(
        ["--permission-mode", "plan", "-w", str(tmp_path), "x"])
    # `todo_write` is on the read-only surface on purpose: it touches nothing
    # outside the session's own plan, and plan mode's whole output is a plan.
    assert permissions.tool_names_for(_mode(args)) == [
        "read", "skill", "ls", "glob", "grep", "todo_write"]
    assert not toolkit.is_mutating("todo_write")
    chain = _guards(args)
    for tool, arguments in [("write", {"path": "a"}), ("edit", {"path": "a"}),
                            ("bash", {"command": "ls"})]:
        assert not chain.check(tool, arguments, True).allowed


def test_bypass_permissions_lets_a_disallowed_call_through(tmp_path):
    args = build_parser().parse_args([
        "--permission-mode", "bypassPermissions",
        "--disallowed-tools", "Bash(git *)", "-w", str(tmp_path), "x"])
    assert _guards(args).check("bash", {"command": "git push"}, True).allowed


def test_accept_edits_keeps_the_full_tool_surface(tmp_path):
    args = build_parser().parse_args(
        ["--permission-mode", "acceptEdits", "-w", str(tmp_path), "x"])
    assert permissions.tool_names_for(_mode(args)) is None
    assert _guards(args).check("write", {"path": "a"}, True).allowed


def test_the_cli_evaluates_allow_and_deny_together(tmp_path):
    args = build_parser().parse_args([
        "--allowed-tools", "Read,Bash(git *)",
        "--disallowed-tools", "Bash(git push*)",
        "-w", str(tmp_path), "x"])
    chain = _guards(args)
    assert chain.check("bash", {"command": "git status"}, True).allowed
    assert not chain.check("bash", {"command": "git push"}, True).allowed
    assert not chain.check("write", {"path": "a"}, True).allowed


# --- CLAUDE.md discovery ---------------------------------------------------

def _project(tmp_path):
    """A git root with a nested workspace, each carrying a CLAUDE.md."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "CLAUDE.md").write_text("ROOT RULES")
    ws = root / "sub" / "pkg"
    ws.mkdir(parents=True)
    (ws / "CLAUDE.md").write_text("WORKSPACE RULES")
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "CLAUDE.md").write_text("USER RULES")
    return root, ws, home


def test_discovery_order_is_workspace_then_up_to_git_root_then_home(tmp_path):
    root, ws, home = _project(tmp_path)
    (root / "sub" / "CLAUDE.md").write_text("MID RULES")
    found = permissions.context_files(str(ws), home=str(home))
    assert found == [
        str(ws / "CLAUDE.md"),
        str(root / "sub" / "CLAUDE.md"),
        str(root / "CLAUDE.md"),
        str(home / ".claude" / "CLAUDE.md"),
    ]


def test_the_walk_stops_at_the_git_root(tmp_path):
    root, ws, home = _project(tmp_path)
    # Above the git root, and therefore never read.
    (tmp_path / "CLAUDE.md").write_text("OUTSIDE RULES")
    bodies = permissions.project_context(str(ws), home=str(home))
    assert "OUTSIDE RULES" not in bodies
    assert "ROOT RULES" in bodies


def test_context_blocks_are_separated_and_labelled(tmp_path):
    root, ws, home = _project(tmp_path)
    out = permissions.project_context(str(ws), home=str(home))
    assert out.count(permissions.SEPARATOR) == 6      # two per file, three files
    assert out.index("WORKSPACE RULES") < out.index("ROOT RULES")
    assert out.index("ROOT RULES") < out.index("USER RULES")
    assert f"PROJECT CONTEXT: {ws / 'CLAUDE.md'}" in out


def test_a_workspace_with_no_git_root_reads_only_two_places(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "CLAUDE.md").write_text("USER RULES")
    ws = tmp_path / "loose"
    ws.mkdir()
    assert permissions.context_files(str(ws), home=str(home)) == [
        str(home / ".claude" / "CLAUDE.md")]


def test_explicit_global_source_uses_only_its_selected_file(tmp_path):
    root, ws, home = _project(tmp_path)
    own = home / ".one-cli" / "AGENTS.md"
    own.parent.mkdir()
    own.write_text("ONE RULES")
    global_context.set_source("one", str(home))
    found = permissions.context_files(str(ws), home=str(home))
    assert str(own) in found
    assert str(home / ".claude" / "CLAUDE.md") not in found
    assert "ONE RULES" in permissions.project_context(str(ws), home=str(home))
    assert "USER RULES" not in permissions.project_context(str(ws), home=str(home))
    global_context.set_source("claude", str(home))
    found = permissions.context_files(str(ws), home=str(home))
    assert str(own) not in found
    assert str(home / ".claude" / "CLAUDE.md") in found


def test_no_project_context_opts_out(tmp_path):
    root, ws, home = _project(tmp_path)
    args = build_parser().parse_args(
        ["--no-project-context", "-w", str(ws), "x"])
    assert "WORKSPACE RULES" not in _system(args, "default")
    args = build_parser().parse_args(["-w", str(ws), "x"])
    assert "WORKSPACE RULES" in _system(args, "default")


def test_a_brace_in_project_context_does_not_break_the_prompt(tmp_path):
    """The Agent used to `.format()` the system prompt."""
    from onecli.agent import Agent

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "CLAUDE.md").write_text("Use {curly} braces freely.")
    args = build_parser().parse_args(["-w", str(ws), "x"])
    system = _system(args, "default")
    # Any object will do: this test reads the system prompt, never the wire.
    agent = Agent(provider=object(), workspace=str(ws), system=system)
    assert "{curly}" in agent.system
    assert str(ws) in agent.system


def test_a_nul_byte_in_a_path_is_denied_not_raised(tmp_path):
    ws = str(tmp_path)
    roots = permissions.RootsGuard(roots=[ws], workspace=ws)
    assert not roots.check("read", {"path": "a\x00b"}, False).allowed
    allow = permissions.PermissionGuard(
        allow=permissions.parse_rules(["Write(**)"]), workspace=ws)
    assert not allow.check("write", {"path": "a\x00b"}, True).allowed
    deny = permissions.PermissionGuard(
        deny=permissions.parse_rules(["Read(secret/**)"]), workspace=ws)
    assert not deny.check("read", {"path": "a\x00b"}, False).allowed


# --- --add-dir boundary ----------------------------------------------------

def test_add_dir_widens_the_readable_roots(tmp_path):
    ws, extra = tmp_path / "ws", tmp_path / "extra"
    ws.mkdir()
    extra.mkdir()
    g = permissions.RootsGuard(
        roots=permissions.build_roots(str(ws), [str(extra)]), workspace=str(ws))
    assert g.check("read", {"path": "a.py"}, False).allowed
    assert g.check("read", {"path": str(extra / "b.py")}, False).allowed


def test_a_read_outside_the_roots_is_refused_and_names_the_path(tmp_path):
    ws, extra = tmp_path / "ws", tmp_path / "extra"
    ws.mkdir()
    extra.mkdir()
    g = permissions.RootsGuard(
        roots=permissions.build_roots(str(ws), [str(extra)]), workspace=str(ws))
    verdict = g.check("read", {"path": "/etc/passwd"}, False)
    assert not verdict.allowed
    assert "/etc/passwd" in verdict.reason
    assert str(ws) in verdict.reason and str(extra) in verdict.reason
    assert verdict.source == "roots"


def test_dot_dot_cannot_escape_the_roots(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    g = permissions.RootsGuard(roots=permissions.build_roots(str(ws), []),
                               workspace=str(ws))
    verdict = g.check("read", {"path": "../secrets.txt"}, False)
    assert not verdict.allowed
    assert str(tmp_path / "secrets.txt") in verdict.reason


def test_a_sibling_with_a_shared_prefix_is_not_inside_the_root(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "ws-other").mkdir()
    g = permissions.RootsGuard(roots=permissions.build_roots(str(ws), []),
                               workspace=str(ws))
    assert not g.check(
        "read", {"path": str(tmp_path / "ws-other" / "a")}, False).allowed


def test_bash_is_not_covered_by_the_roots_guard(tmp_path):
    """Named so the gap is a decision, not a surprise. See RootsGuard."""
    ws = tmp_path / "ws"
    ws.mkdir()
    g = permissions.RootsGuard(roots=permissions.build_roots(str(ws), []),
                               workspace=str(ws))
    assert g.check("bash", {"command": "cat /etc/passwd"}, True).allowed


def test_without_add_dir_the_workspace_is_the_boundary(tmp_path):
    """File tools are confined to the workspace by default (release decision)."""
    args = build_parser().parse_args(["-w", str(tmp_path), "x"])
    verdict = _guards(args).check("read", {"path": "/etc/hosts"}, False)
    assert not verdict.allowed and "--add-dir" in verdict.reason
    assert _guards(args).check("read", {"path": "a.py"}, False).allowed


def test_add_dir_enforces_the_boundary_through_the_cli(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    args = build_parser().parse_args(
        ["--add-dir", str(extra), "-w", str(ws), "x"])
    chain = _guards(args)
    assert chain.check("read", {"path": str(extra / "a")}, False).allowed
    verdict = chain.check("read", {"path": os.sep + "etc"}, False)
    assert not verdict.allowed
    assert "--add-dir" in verdict.reason


def test_read_only_beside_a_looser_mode_is_refused_not_dropped():
    """`--read-only --permission-mode bypassPermissions` used to write a file.

    normalise_mode honoured the alias only when the mode was `default`, so
    against the two modes that PERMIT mutation the flag was silently discarded
    and the run edited the workspace, with no line in the output mentioning
    that --read-only had been dropped.
    """
    for mode in ("acceptEdits", "bypassPermissions"):
        with pytest.raises(ValueError, match="contradict"):
            permissions.normalise_mode(mode, read_only=True)
    # `plan` is not looser than read-only, so it keeps its old behaviour.
    assert permissions.normalise_mode("plan", read_only=True) == "plan"
    # And neither flag alone is affected.
    assert permissions.normalise_mode("bypassPermissions") == "bypassPermissions"
    assert permissions.normalise_mode(None, read_only=True) == "readOnly"


def test_compose_system_keeps_existing_instruction_body_once(tmp_path):
    ws = tmp_path / 'ws'
    home = tmp_path / 'home'
    ws.mkdir()
    home.mkdir()
    body = 'Required safety rule\nPreserve this exact instruction.'
    (ws / 'AGENTS.md').write_text(body)
    (ws / 'CLAUDE.md').write_text(body)
    (ws / 'SOUL.md').write_text('Distinct identity instruction')
    base = 'Host instructions\n' + body
    out = permissions.compose_system(base, str(ws), home=str(home))
    assert out.count(body) == 1
    assert 'Distinct identity instruction' in out


def test_context_dedup_preserves_partially_matching_rules(tmp_path):
    ws = tmp_path / 'ws'
    home = tmp_path / 'home'
    ws.mkdir()
    home.mkdir()
    (ws / 'AGENTS.md').write_text('Shared rule\nAdditional mandatory rule')
    out = permissions.compose_system('Shared rule', str(ws), home=str(home))
    assert 'Additional mandatory rule' in out

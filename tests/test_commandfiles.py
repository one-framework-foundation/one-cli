"""Slash commands the operator writes as markdown files.

The format is the one people already have on disk - `.claude/commands/*.md`,
optional frontmatter, body is the prompt - because a format that is nearly
compatible is worse than either alternative. What is asserted here is the part
that is ours: which directory wins a collision, what a built-in name does to a
file that claims it, and how typed arguments reach the body.
"""

from __future__ import annotations

import pytest

from onecli import commandfiles


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "repo" / ".claude" / "commands"
    root.mkdir(parents=True)
    return tmp_path / "repo"


def _write(ws, rel: str, text: str):
    path = ws / ".claude" / "commands" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _load(ws, **kw):
    return commandfiles.load(str(ws), trust_project=True, **kw)


# --- what is loaded --------------------------------------------------------


def test_a_markdown_file_becomes_a_command(ws):
    _write(ws, "review.md", "Review the diff and be blunt.")
    found, notes = _load(ws)
    assert set(found) == {"review"}
    assert found["review"].body == "Review the diff and be blunt."
    assert notes == []


def test_a_subdirectory_namespaces_the_name(ws):
    _write(ws, "git/commit.md", "Write the commit message.")
    found, _ = _load(ws)
    assert "git:commit" in found


def test_frontmatter_is_read_and_stripped(ws):
    _write(ws, "pr.md", "---\ndescription: open a PR\n"
                        "argument-hint: <branch>\n---\nOpen a PR for $1.")
    cmd = _load(ws)[0]["pr"]
    assert cmd.description == "open a PR"
    assert cmd.argument_hint == "<branch>"
    assert cmd.body == "Open a PR for $1."


def test_an_empty_file_is_not_a_command(ws):
    _write(ws, "blank.md", "---\ndescription: nothing\n---\n\n")
    assert _load(ws)[0] == {}


def test_a_file_claiming_a_builtin_name_is_skipped_out_loud(ws):
    """The operator wrote that file expecting it to run. A command that exists
    on disk and silently does nothing is the worst of the three outcomes - and
    a `cost.md` that quietly costs a turn is the second worst."""
    _write(ws, "cost.md", "Work out what this cost.")
    found, notes = _load(ws, reserved={"cost"})
    assert found == {}
    assert notes and "shadows the built-in /cost" in notes[0]


def test_the_project_directory_wins_a_collision(ws, tmp_path, monkeypatch):
    personal = tmp_path / "fake-home" / ".claude" / "commands"
    personal.mkdir(parents=True)
    (personal / "review.md").write_text("personal version")
    monkeypatch.setattr(commandfiles.Path, "home", classmethod(
        lambda cls: tmp_path / "fake-home"))
    _write(ws, "review.md", "project version")
    found, _ = _load(ws)
    assert found["review"].body == "project version"


def test_a_personal_command_is_found_when_the_project_has_none(ws, tmp_path,
                                                               monkeypatch):
    personal = tmp_path / "fake-home" / ".claude" / "commands"
    personal.mkdir(parents=True)
    (personal / "standup.md").write_text("What did I do yesterday?")
    monkeypatch.setattr(commandfiles.Path, "home", classmethod(
        lambda cls: tmp_path / "fake-home"))
    assert "standup" in _load(ws)[0]


def test_a_workspace_with_no_commands_dir_loads_nothing(tmp_path):
    assert commandfiles.load(str(tmp_path)) == ({}, [])


# --- how arguments reach the body -----------------------------------------


def _cmd(body: str):
    return commandfiles.CommandFile(name="x", body=body)


def test_arguments_substitutes_the_whole_rest():
    assert _cmd("Review $ARGUMENTS please").render("src/a.py src/b.py") == (
        "Review src/a.py src/b.py please")


def test_positionals_take_one_word_each():
    assert _cmd("move $1 to $2").render("old.py new.py") == "move old.py to new.py"


def test_a_positional_nobody_typed_leaves_nothing_behind():
    assert _cmd("check $1 $2").render("only.py").strip() == "check only.py"


def test_a_body_that_names_no_placeholder_still_gets_the_arguments():
    """Otherwise `/review src/foo.py` silently discards the only interesting
    part of the line."""
    assert _cmd("Review the diff.").render("src/foo.py") == (
        "Review the diff.\n\nsrc/foo.py")


def test_a_body_with_no_placeholder_and_no_arguments_is_itself():
    assert _cmd("Review the diff.").render("") == "Review the diff."

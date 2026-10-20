"""The line editor: what tab completes, and what a trailing backslash means.

Completion is the half of a REPL that has to know about the program. The other
half - history, word motion, incremental search - arrives with the `readline`
import and is not ours to test. What is ours: a slash at the head of the line
means a command, anywhere else a word is almost certainly a path, and the path
is resolved against the workspace because that is what the model's own tools
resolve against.
"""

from __future__ import annotations

import pytest

from onecli import commands as cmdlib
from onecli import inputline


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "onecli").mkdir()
    (tmp_path / "onecli" / "cli.py").write_text("x")
    (tmp_path / "onecli" / "render.py").write_text("x")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / ".git").mkdir()
    return tmp_path


def _completer(tree):
    return inputline.Completer(cmdlib.NAMES, str(tree))


# --- commands --------------------------------------------------------------


def test_a_slash_at_the_head_of_the_line_completes_commands(tree):
    got = _completer(tree).matches("/co", at_start=True)
    assert "/compact " in got and "/context " in got and "/cost " in got


def test_every_command_is_reachable_from_a_bare_slash(tree):
    got = _completer(tree).matches("/", at_start=True)
    assert len(got) == len(cmdlib.ORDER)


def test_a_slash_mid_line_completes_paths_not_commands(tree):
    """`explain /co...` is someone naming a file, and offering /compact there
    would replace their path with a command."""
    got = _completer(tree).matches("one", at_start=False)
    assert got == ["onecli/"]


# --- paths -----------------------------------------------------------------


def test_a_directory_completes_with_its_slash(tree):
    assert _completer(tree).matches("onec") == ["onecli/"]


def test_completion_walks_into_a_directory(tree):
    got = _completer(tree).matches("onecli/c")
    assert got == ["onecli/cli.py "]


def test_a_file_completes_with_a_space_so_the_word_is_finished(tree):
    assert _completer(tree).matches("READ") == ["README.md "]


def test_dot_files_stay_out_of_the_way_until_the_dot_is_typed(tree):
    assert ".git/" not in _completer(tree).matches("")
    assert ".git/" in _completer(tree).matches(".")


def test_a_path_that_does_not_exist_completes_to_nothing(tree):
    assert _completer(tree).matches("nowhere/at/all") == []


def test_paths_resolve_against_the_workspace_not_the_shell_cwd(tree, monkeypatch,
                                                               tmp_path):
    """The model's tools resolve against the workspace. A completion offered
    from somewhere else is a confidently wrong path."""
    elsewhere = tmp_path.parent / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)
    assert _completer(tree).matches("READ") == ["README.md "]


# --- continuation ----------------------------------------------------------


def _typed(monkeypatch, lines):
    pending = list(lines)
    monkeypatch.setattr("builtins.input", lambda _p="": pending.pop(0))


def test_a_trailing_backslash_continues_the_line(monkeypatch):
    _typed(monkeypatch, ["write a function that:\\", "  takes a path\\",
                         "  returns its size"])
    assert inputline.read("> ") == ("write a function that:\n"
                                    "  takes a path\n"
                                    "  returns its size")


def test_a_line_with_no_backslash_is_returned_as_typed(monkeypatch):
    _typed(monkeypatch, ["just one line"])
    assert inputline.read("> ") == "just one line"


def test_a_doubled_backslash_is_how_a_line_ends_in_a_backslash(monkeypatch):
    _typed(monkeypatch, ["a windows path C:\\\\"])
    assert inputline.read("> ") == "a windows path C:\\"


def test_backslashes_inside_the_line_are_left_alone(monkeypatch):
    _typed(monkeypatch, [r"grep '\d\d' file.txt"])
    assert inputline.read("> ") == r"grep '\d\d' file.txt"


# --- install ---------------------------------------------------------------


def test_install_arms_readline_and_survives_a_build_without_it(tree,
                                                               monkeypatch):
    """A Python built without readline, or a piped stdin, must cost a False -
    never an exception at start-up."""
    import builtins

    real_import = builtins.__import__

    def no_readline(name, *a, **kw):
        if name == "readline":
            raise ImportError("no readline here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_readline)
    assert inputline.install(cmdlib.NAMES, str(tree)) is False


# --- prompt measurement ----------------------------------------------------


def test_a_coloured_prompt_is_fenced_so_readline_can_measure_it(monkeypatch):
    """The bug this exists for: with a raw colour escape in the prompt,
    readline believes the cursor is ~20 columns further right than it is, and
    every redraw is wrong - the up arrow leaves the previous line on screen,
    ctrl-u clears the wrong span, a long line wraps early."""
    monkeypatch.setattr(inputline, "_armed", True)
    got = inputline.mark("\n\x1b[38;2;153;204;204m*\x1b[0m ")
    assert got == "\n\001\x1b[38;2;153;204;204m\002*\001\x1b[0m\002 "
    visible = inputline.ESCAPE.sub("", got).replace("\001", "").replace("\002", "")
    assert visible == "\n* ", "only the escapes are fenced, never the glyph"


def test_a_plain_prompt_is_left_exactly_as_it_is(monkeypatch):
    monkeypatch.setattr(inputline, "_armed", True)
    assert inputline.mark("> ") == "> "


def test_nothing_is_fenced_without_readline_underneath(monkeypatch):
    """Fenced markers are meaningful to readline and are two stray control
    characters on the screen to a bare input()."""
    monkeypatch.setattr(inputline, "_armed", False)
    coloured = "\x1b[31m*\x1b[0m "
    assert inputline.mark(coloured) == coloured

"""one-cli driven by a parent process, not by a person at a terminal.

The host composes a boot prompt of tens of kilobytes and a turn that can be
longer still. Both have to reach the agent without touching argv, because the
kernel caps a single argument at 128KB (MAX_ARG_STRLEN) and there is no flag
that raises it. So: the system prompt arrives as a file, the turn arrives on
stdin, and "no prompt on argv" means interactive only at a terminal.
"""

from __future__ import annotations

import io
import sys

import pytest

from onecli import cli


def _args(argv):
    return cli.build_parser().parse_args(argv)


def test_system_file_becomes_the_system_prompt(tmp_path):
    f = tmp_path / "boot.md"
    f.write_text("you are the host agent", encoding="utf-8")
    args = _args(["--system-file", str(f), "hello"])
    cli._resolve_system_file(args)
    assert args.system == "you are the host agent"


def test_system_file_carries_a_prompt_larger_than_argv_allows(tmp_path):
    # MAX_ARG_STRLEN is 131072; a prompt past it cannot be an argv string at
    # all, which is the whole reason this flag exists.
    big = "x" * 200_000
    f = tmp_path / "boot.md"
    f.write_text(big, encoding="utf-8")
    args = _args(["--system-file", str(f), "hello"])
    cli._resolve_system_file(args)
    assert len(args.system) == 200_000


def test_system_and_system_file_together_are_refused(tmp_path):
    f = tmp_path / "boot.md"
    f.write_text("from the file", encoding="utf-8")
    args = _args(["--system", "from argv", "--system-file", str(f), "hi"])
    with pytest.raises(ValueError, match="exclusive"):
        cli._resolve_system_file(args)


def test_no_system_file_leaves_system_alone():
    args = _args(["--system", "from argv", "hi"])
    cli._resolve_system_file(args)
    assert args.system == "from argv"


def test_missing_system_file_exits_two(tmp_path, capsys):
    rc = cli.main(["--system-file", str(tmp_path / "nope.md"), "hi"])
    assert rc == 2


def test_piped_stdin_becomes_the_prompt(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("  do the thing  "))
    args = _args([])
    cli._prompt_from_stdin(args)
    assert args.prompt == ["do the thing"]


def test_a_terminal_with_no_prompt_stays_interactive(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Tty("ignored"))
    args = _args([])
    cli._prompt_from_stdin(args)
    assert args.prompt == []


def test_argv_prompt_beats_a_pipe(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("from the pipe"))
    args = _args(["from", "argv"])
    cli._prompt_from_stdin(args)
    assert args.prompt == ["from", "argv"]


def test_an_empty_pipe_is_not_a_prompt(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("   \n  "))
    args = _args([])
    cli._prompt_from_stdin(args)
    assert args.prompt == []


# --------------------------------------------------------------------------
# inherited-envelope regression - a hook envelope inherited on stdin is not a prompt
# --------------------------------------------------------------------------
#
# A run started from a Claude Code hook can inherit stdin carrying the
# concatenated stdout of every UserPromptSubmit hook on the host. That is not a
# prompt, and sending it bills a backend for nothing. The fd was inherited, not
# piped on purpose.

HOOK_JUNK = (
    '{}\n'
    '{"continue":true}\n'
    '{"continue":true,"suppressOutput":true}\n'
    '{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", '
    '"additionalContext": "[session hook note] ..."}}\n'
)


def test_hook_envelope_is_stripped_and_the_real_task_survives(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(HOOK_JUNK + "\ncount the files"))
    args = _args([])
    cli._prompt_from_stdin(args)
    assert args.prompt == ["count the files"]
    assert "hook-envelope" in capsys.readouterr().err


def test_a_body_of_only_hook_output_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(HOOK_JUNK))
    args = _args([])
    with pytest.raises(ValueError, match="not a task"):
        cli._prompt_from_stdin(args)


def test_only_hook_output_exits_two_without_reaching_a_backend(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(HOOK_JUNK))

    def explode(*a, **k):
        raise AssertionError("resolved a backend for a hook envelope")

    monkeypatch.setattr(cli.config, "resolve", explode)
    assert cli.main([]) == 2


def test_argv_prompt_means_stdin_is_never_read(monkeypatch):
    class Loud(io.StringIO):
        def read(self, *a):
            raise AssertionError("read stdin while argv carried a prompt")

    monkeypatch.setattr(sys, "stdin", Loud(HOOK_JUNK))
    args = _args(["do", "the", "thing"])
    cli._prompt_from_stdin(args)
    assert args.prompt == ["do", "the", "thing"]


def test_a_json_object_that_is_not_a_hook_envelope_survives(monkeypatch):
    body = '{"task": "summarise this", "continue_reading": true}'
    monkeypatch.setattr(sys, "stdin", io.StringIO(body))
    args = _args([])
    cli._prompt_from_stdin(args)
    assert args.prompt == [body]











def test_an_untagged_prompt_is_untouched(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("just a task"))
    args = _args([])
    cli._prompt_from_stdin(args)
    assert args.prompt == ["just a task"]
    assert args.system is None


def test_caller_is_named_by_the_environment(monkeypatch):
    monkeypatch.setenv("ONE_CLI_CALLER", "one-arena")
    assert cli._caller() == "one-arena"


def test_caller_falls_back_to_the_parent_argv(monkeypatch):
    monkeypatch.delenv("ONE_CLI_CALLER", raising=False)
    # Whatever ran pytest. The only claim is that it is non-empty on Linux and
    # that asking never raises.
    assert isinstance(cli._caller(), str)


def test_bench_never_hands_a_child_this_process_stdin(monkeypatch):
    from onecli.evals import bench

    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        raise OSError("stop here")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    bench._run(["true"], cwd=".", timeout=1)
    assert seen["stdin"] is bench.subprocess.DEVNULL


def test_arena_devnulls_stdin_unless_a_harness_asked_for_a_body():
    from onecli import harnesses
    from onecli.evals import arena

    bare = harnesses.Launch(["true"], cwd=".")
    assert arena._stream(bare) == {"stdin": arena.subprocess.DEVNULL}

    with_body = harnesses.Launch(["true"], cwd=".", stdin="a body")
    assert arena._stream(with_body) == {"input": "a body"}

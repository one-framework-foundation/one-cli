"""`@path` - what gets sent with the question, and what never does.

The value is one saved round trip: the model does not have to call `read` on a
file the person at the keyboard was already looking at. The risk is the same
mechanism reading something nobody asked for, so containment is asserted here
harder than convenience is.
"""

from __future__ import annotations

import pytest

from onecli import mentions


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "onecli").mkdir()
    (tmp_path / "onecli" / "cli.py").write_text("print('hello')\n")
    (tmp_path / "notes.md").write_text("# notes\nbody\n")
    (tmp_path / "sub").mkdir()
    return tmp_path


# --- what counts as a mention ---------------------------------------------


def test_a_mention_is_found_anywhere_in_the_sentence():
    assert mentions.find("explain @onecli/cli.py to me") == ["onecli/cli.py"]


def test_trailing_punctuation_is_the_sentence_not_the_name():
    assert mentions.find("what does @notes.md say?") == ["notes.md"]
    assert mentions.find("look at @notes.md, then stop") == ["notes.md"]


def test_the_same_file_twice_is_sent_once():
    assert mentions.find("@a.py and @a.py") == ["a.py"]


def test_an_email_address_is_not_a_mention():
    """`@` after a word character is an address, a handle, or a decorator -
    never a path someone meant to attach."""
    assert mentions.find("mail jesse@example.com about it") == []


def test_a_python_decorator_is_not_a_mention():
    assert mentions.find("the @property on line 12") == ["property"], (
        "a bare word after @ IS a plausible relative path - it is caught by "
        "resolve() failing, not by the pattern")


# --- what resolves ---------------------------------------------------------


def test_a_file_in_the_workspace_resolves(ws):
    assert mentions.resolve("onecli/cli.py", str(ws)) is not None


def test_a_directory_does_not(ws):
    assert mentions.resolve("sub", str(ws)) is None


def test_a_path_outside_the_workspace_is_refused(ws):
    assert mentions.resolve("../../../etc/passwd", str(ws)) is None
    assert mentions.resolve("/etc/passwd", str(ws)) is None


def test_a_symlink_pointing_out_of_the_workspace_is_refused(ws, tmp_path):
    """Containment is checked on the real path, so a symlink is the same
    escape as `../` and is refused the same way."""
    secret = tmp_path.parent / "outside.txt"
    secret.write_text("secret")
    (ws / "link.txt").symlink_to(secret)
    assert mentions.resolve("link.txt", str(ws)) is None


def test_a_file_that_does_not_exist_resolves_to_nothing(ws):
    assert mentions.resolve("nope.py", str(ws)) is None


# --- what the model receives ----------------------------------------------


def test_the_file_arrives_in_front_of_the_question(ws):
    out, sent = mentions.expand("explain @onecli/cli.py", str(ws))
    assert sent == ["onecli/cli.py"]
    assert "print('hello')" in out
    assert out.index("print('hello')") < out.index("explain @onecli/cli.py"), (
        "a question with the file spliced into the middle stops being readable")


def test_the_sentence_is_left_exactly_as_typed(ws):
    out, _ = mentions.expand("explain @onecli/cli.py to me", str(ws))
    assert "explain @onecli/cli.py to me" in out


def test_a_mention_that_resolves_to_nothing_changes_nothing(ws):
    text = "the @property decorator and @nope.py"
    out, sent = mentions.expand(text, str(ws))
    assert (out, sent) == (text, [])


def test_a_refused_path_leaves_the_prompt_untouched(ws):
    text = "read @../../../etc/passwd for me"
    assert mentions.expand(text, str(ws)) == (text, [])


def test_a_file_containing_a_fence_does_not_end_the_block_early(ws):
    (ws / "doc.md").write_text("here is code:\n```\nx = 1\n```\ndone\n")
    out, sent = mentions.expand("@doc.md", str(ws))
    assert sent == ["doc.md"]
    assert "````" in out, "the fence grows past the longest one in the file"
    body = out.split("````")
    assert len(body) == 3, "exactly one block, opened and closed"
    assert "done" in body[1], "the file's own fence did not close it early"


def test_a_big_file_is_cut_head_first_and_says_so(ws):
    (ws / "big.txt").write_text("x" * 5000)
    body, note = mentions.read(str(ws / "big.txt"), budget=1000)
    assert len(body) == 1000
    assert "5,000" in note and "1,000" in note


def test_the_turn_has_a_budget_across_every_mention(ws):
    for i in range(4):
        (ws / f"f{i}.txt").write_text("y" * 900)
    out, sent = mentions.expand("@f0.txt @f1.txt @f2.txt @f3.txt", str(ws),
                                total_budget=2000)
    assert len(sent) < 4, "one turn cannot spend the whole window on mentions"
    assert "not sent" in out, "what was dropped is named, never silently cut"


# --- binary files ----------------------------------------------------------


def test_a_binary_file_is_named_not_sent(ws):
    """A NUL byte survives decode(errors="replace") - it is valid UTF-8 - so
    without a sniff the wire carries tens of KB of control characters."""
    (ws / "blob.bin").write_bytes(bytes(range(256)) * 400)
    out, sent = mentions.expand("look at @blob.bin", str(ws))
    assert sent == []
    assert "\x00" not in out
    assert "looks like a binary file" in out
    assert "blob.bin" in out


def test_a_text_file_holding_a_late_nul_is_still_sent(ws):
    """The sniff is the head of the file, not all of it: a stray NUL past the
    window must not cost a legitimate source file its inclusion."""
    (ws / "mostly.txt").write_text("a" * (mentions.SNIFF_BYTES + 10))
    with open(ws / "mostly.txt", "ab") as fh:
        fh.write(b"\x00tail")
    out, sent = mentions.expand("look at @mostly.txt", str(ws))
    assert sent == ["mostly.txt"]


def test_looks_binary_reads_only_the_head():
    assert mentions.looks_binary(b"\x00" + b"a" * 100)
    assert not mentions.looks_binary(b"a" * mentions.SNIFF_BYTES * 2)
    assert not mentions.looks_binary(b"a" * (mentions.SNIFF_BYTES + 1) + b"\x00")


# --- the wiring: both modes, not just the REPL -----------------------------


def test_a_one_shot_prompt_expands_mentions_too(monkeypatch, tmp_path):
    """`one "explain @src/parser.py"` from a shell used to send the literal
    string: expansion was wired into the prompt loop only, so the one shape a
    script or a parent process types was the one shape that did not work."""
    import types
    from onecli import cli

    (tmp_path / "note.txt").write_text("SENTINEL_ONESHOT\n")
    seen = {}

    class FakeAgent:
        def __init__(self, *_a, **kw):
            self.chain = types.SimpleNamespace(
                current=types.SimpleNamespace(name="fake"))

        def run(self, prompt, messages, attachments=None):
            seen["prompt"] = prompt
            return types.SimpleNamespace(text="ok", messages=[], steps=[],
                                         tool_calls=0, cost=0.0,
                                         stopped="answered", error=None)

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    monkeypatch.setattr(cli, "_context_settings", lambda a, b: {
        "chain": types.SimpleNamespace(
            current=types.SimpleNamespace(name="fake"), warnings=[])})
    monkeypatch.setattr(cli, "_hooks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_submit_prompt", lambda hooks, p: (p, None))
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)

    backend = cli.config.Backend(name="fake", model="m",
                                 base_url="http://127.0.0.1:1/v1")
    args = cli.build_parser().parse_args(
        ["-w", str(tmp_path), "-q", "--no-hooks", "-m", "fake",
         "look at @note.txt"])
    try:
        cli.run_once(args, backend, [], "sid0")
    except SystemExit:
        pass
    except Exception:                                        # noqa: BLE001
        # The assertion below is the contract; a stub that cannot complete the
        # run still proves what the agent was handed.
        pass

    assert "prompt" in seen, "the agent was never called"
    assert "SENTINEL_ONESHOT" in seen["prompt"]
    assert "@note.txt" in seen["prompt"], "the sentence itself stays as typed"


def test_missing_reports_file_looking_mentions_only(tmp_path):
    (tmp_path / 'real.py').write_text('x = 1\n')
    text = 'see @real.py and @nofile.py and @src/gone and @here and mail a@b.com'
    assert mentions.missing(text, str(tmp_path)) == ['nofile.py', 'src/gone']

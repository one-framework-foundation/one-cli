"""Whose words a UserPromptSubmit hook's stdout is, once it reaches the model.

A hook emits instruction-shaped prose - a reminder, an obligation, "if X then
do Y". Pasted bare in front of the turn, that reads as the user talking, and a
greeting arrives as a to-do list with "yo" stapled to the end. The model then
does the only actionable thing in the message.
"""

from __future__ import annotations

from onecli import cli
from onecli.hooks import Outcome, USER_PROMPT_SUBMIT


class FakeHooks:
    """A HookSet stand-in that returns one prepared outcome."""

    def __init__(self, outcome):
        self.outcome = outcome

    def __bool__(self):
        return True

    def user_prompt_submit(self, prompt):
        return self.outcome


OBLIGATION = ("[open obligations on this host]\n"
              "- task-417 (216h old): the export job has no tests.\n"
              "If the current work touches one, close it.")


def test_injected_context_is_framed_as_context_not_as_the_user(capsys):
    out = Outcome(event=USER_PROMPT_SUBMIT, context=[OBLIGATION], ran=1)
    prompt, reason = cli._submit_prompt(FakeHooks(out), "yo")

    assert reason == ""
    assert "<hook-context>" in prompt and "</hook-context>" in prompt
    assert OBLIGATION in prompt
    assert prompt.rstrip().endswith("yo"), "the user's message comes last"
    # The bare paste is what turned a greeting into a two-minute investigation.
    assert not prompt.startswith("[open obligations")


def test_the_frame_says_it_is_not_a_request():
    out = Outcome(event=USER_PROMPT_SUBMIT, context=[OBLIGATION], ran=1)
    prompt, _ = cli._submit_prompt(FakeHooks(out), "yo")
    frame = prompt.split("</hook-context>", 1)[1]
    assert "not a request" in frame
    assert "not the user speaking" in frame


def test_a_prompt_with_no_hook_context_is_untouched():
    out = Outcome(event=USER_PROMPT_SUBMIT, ran=1)
    prompt, _ = cli._submit_prompt(FakeHooks(out), "yo")
    assert prompt == "yo"


def test_no_hooks_at_all_is_untouched():
    assert cli._submit_prompt(None, "yo") == ("yo", "")


def test_a_blocked_prompt_is_still_blocked():
    out = Outcome(event=USER_PROMPT_SUBMIT, blocked=True, reason="nope", ran=1)
    assert cli._submit_prompt(FakeHooks(out), "yo") == (None, "nope")


def test_whitespace_only_context_is_not_framed():
    """`context_text` drops blank entries, so there is nothing to frame."""
    out = Outcome(event=USER_PROMPT_SUBMIT, context=["   ", "\n"], ran=1)
    prompt, _ = cli._submit_prompt(FakeHooks(out), "yo")
    assert prompt == "yo"

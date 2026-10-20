"""The line editor for the interactive session.

`input()` on its own gives a terminal with no memory: the up arrow emits `^[[A`
into the buffer instead of the previous line, so every retype of a long prompt
is a full retype, and nothing typed in one session is reachable from the next.
Importing `readline` is what turns that into a real line editor - history,
word motion, incremental search - and the import alone is most of the fix.

What is left after the import is the part that has to know about this program:

- **Where history lives and how much of it.** `~/.one-cli/history`, beside the
  session transcripts, capped so it cannot grow without bound.
- **What a tab means.** At the head of a line a slash is a command, so tab
  completes the command table. Anywhere else the word is almost always a path,
  so tab completes paths - relative to the workspace, because that is what the
  model's tools resolve against and a path completed against some other
  directory would be a confidently wrong one.

Everything degrades to plain `input()`: readline is optional in a Python build
(and behaves differently under libedit on macOS), a piped stdin has no line
editor to configure, and none of that is worth an exception at start-up.
"""

from __future__ import annotations

import atexit
import os
import re
from pathlib import Path

# readline measures the prompt to know where the line starts. A colour escape
# is zero columns wide on screen and several characters long in the string, so
# an unmarked coloured prompt makes it believe the cursor is further right than
# it is - and every redraw after that is wrong: the up arrow leaves the old
# line behind it, ctrl-u clears the wrong span, a long line wraps early. These
# two markers are how a prompt says "what is between us takes no space".
ESCAPE = re.compile(r"(\x1b\[[0-9;]*[A-Za-z])")
RL_START, RL_END = "\001", "\002"

HISTORY = Path.home() / ".one-cli" / "history"
HISTORY_LINES = 5000

# Word characters readline should NOT break a completion word on. The default
# set includes `/`, `-` and `.`, which means a path completes only from its
# last segment and the replacement then eats the directory part.
DELIMS = " \t\n\"'`@><=;|&{("

# Whether a real readline is underneath. The prompt markers below are only
# meaningful to readline; printed by a bare `input()` they are two stray
# control characters on the screen.
_armed = False


def install(commands, workspace: str = ".") -> bool:
    """Arm history and completion for this session. False if unavailable."""
    try:
        import readline
    except ImportError:                                     # pragma: no cover
        return False

    global _armed
    _armed = True
    _load_history(readline)
    completer = Completer(commands, workspace)
    from onecli import terminal
    if terminal.ACTIVE is not None:
        terminal.ACTIVE.completer = completer
        terminal.ACTIVE.history = [readline.get_history_item(i) for i in range(1, readline.get_current_history_length() + 1)]
        terminal.ACTIVE.history_index = len(terminal.ACTIVE.history)
    readline.set_completer(completer)
    readline.set_completer_delims(DELIMS)
    # libedit (macOS) speaks a different bind language from GNU readline, and
    # sending the wrong one is a silent no-op rather than an error - so ask
    # which one is underneath rather than guessing from the platform.
    if "libedit" in (getattr(readline, "__doc__", "") or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    return True


def _load_history(readline) -> None:
    from onecli import secureio
    with secureio.directory(HISTORY.parent, create=True, private=True):
        pass
    try:
        readline.read_history_file(HISTORY)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(HISTORY_LINES)
    # Written at exit rather than per line: a long session should not pay a
    # file write per prompt, and the file is only read at start-up anyway.
    atexit.register(_save_history, readline)


def _save_history(readline) -> None:
    try:
        from onecli import secureio
        secureio.write_history(readline, HISTORY, limit=HISTORY_LINES)
    except OSError:                                         # pragma: no cover
        pass


def _subsequence(needle: str, haystack: str) -> bool:
    """Is every character of `needle` present in `haystack`, in order?

    The cheap out-of-order match: `/ost` is a subsequence of `cost`, so a
    typo still offers the command it nearly named."""
    position = 0
    for char in needle:
        position = haystack.find(char, position) + 1
        if not position:
            return False
    return True


class Completer:
    """Slash commands at the head of the line, paths everywhere else.

    `commands` is a list of names or a mapping of name to one-line summary;
    the summaries are what the terminal shows while a command is being typed.
    """

    def __init__(self, commands, workspace: str = "."):
        self.summaries = dict(commands) if isinstance(commands, dict) else {}
        self.commands = sorted(commands)
        self.workspace = workspace
        self._matches: list[str] = []

    def __call__(self, text: str, state: int):
        if state == 0:
            self._matches = self.matches(text, self._line_start())
        try:
            return self._matches[state]
        except IndexError:
            return None

    def _line_start(self) -> bool:
        """Is the word being completed the first thing on the line?"""
        try:
            import readline
        except ImportError:                                 # pragma: no cover
            return True
        return not readline.get_line_buffer()[:readline.get_begidx()].strip()

    def matches(self, text: str, at_start: bool = True) -> list[str]:
        if at_start and text.startswith("/"):
            return [c + " " for c in self.commands if c.startswith(text)]
        return self.paths(text)

    def menu(self, draft: str) -> list[tuple[str, str]]:
        """Commands matching a draft that is still only a slash word.

        Prefix matches come first and stay in alphabetical order. When the
        prefix set is empty the draft is matched as a subsequence over the
        command names, so `/ost` still finds `/cost` instead of showing an
        empty menu - the same out-of-order hint Claude Code gives. The
        dispatch-time `did you mean` remains the backstop for a command the
        user commits to.
        """
        if not draft.startswith("/") or any(c.isspace() for c in draft):
            return []
        query = draft[1:].lower()
        prefix = [c for c in self.commands if c.startswith(draft)]
        if prefix or not query:
            return [(c, self.summaries.get(c, "")) for c in prefix]
        return [(c, self.summaries.get(c, "")) for c in self.commands
                if _subsequence(query, c[1:].lower())]

    def menu_is_fuzzy(self, draft: str) -> bool:
        """True when `menu(draft)` is showing subsequence (non-prefix) rows.

        The renderer dims those rows to say 'close, not exact', matching the
        hint Claude Code shows in its own picker."""
        if not draft.startswith("/") or any(c.isspace() for c in draft):
            return False
        if not draft[1:] or not self.menu(draft):
            return False
        return not any(c.startswith(draft) for c in self.commands)

    def paths(self, text: str) -> list[str]:
        """Filesystem completions for `text`, resolved against the workspace.

        A directory completes with its trailing slash so the next tab walks
        into it rather than stopping at a name that cannot be a file.
        """
        raw = os.path.expanduser(text)
        head, _, tail = raw.rpartition("/")
        base = head + "/" if head else ""
        anchor = os.path.expanduser(head) if head else ""
        if not os.path.isabs(raw) and not text.startswith("~"):
            anchor = os.path.join(self.workspace, anchor)
        try:
            names = os.listdir(anchor or self.workspace)
        except OSError:
            return []
        out = []
        for name in sorted(names):
            if not name.startswith(tail):
                continue
            # A dot-file is only offered once the dot has been typed: otherwise
            # a bare tab in any repo answers with .git and little else.
            if name.startswith(".") and not tail.startswith("."):
                continue
            full = os.path.join(anchor or self.workspace, name)
            out.append(base + name + ("/" if os.path.isdir(full) else " "))
        return out


def read(prompt: str, more: str = "") -> str:
    r"""One logical line, joining continuations.

    A line ending in a lone backslash continues on the next one, so a prompt
    with a code block or a list in it can be typed as a shape instead of as one
    unreadable line. The continuation prompt is dimmer than the first, so the
    eye can see that the session is still collecting rather than thinking.
    """
    from onecli import terminal
    if terminal.ACTIVE is not None:
        return terminal.ACTIVE.read(prompt)
    line = input(mark(prompt))
    while _continues(line):
        line = line[:-1] + "\n" + input(mark(more or prompt))
    # A doubled backslash at the very end is how a line that really does end in
    # a backslash is typed. It is unescaped here and nowhere else: touching the
    # rest of the line would rewrite backslashes inside the user's own text.
    return line[:-1] if line.endswith("\\\\") else line


def _continues(line: str) -> bool:
    """Does this line end in an unescaped continuation backslash?"""
    trailing = len(line) - len(line.rstrip("\\"))
    return trailing % 2 == 1


def mark(prompt: str) -> str:
    """Fence the colour escapes in `prompt` so readline can measure it."""
    if not _armed or "\x1b" not in prompt:
        return prompt
    return ESCAPE.sub(RL_START + r"\1" + RL_END, prompt)


__all__ = ["install", "read", "mark", "Completer", "HISTORY"]

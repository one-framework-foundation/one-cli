"""Persistent terminal surface. The agent stays on the main thread for SIGINT.

Only the renderer thread reads the terminal. Submitted turns and explicit human
answers have separate queues, so typing ahead cannot accidentally answer a gate.
No terminal dependency beyond Python's standard curses module is required.
"""
from __future__ import annotations

import builtins
from collections import deque
from contextlib import contextmanager
import io
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import unicodedata

ACTIVE = None
# Each key says what it does: a bare "^O tools" or "^G live" names a feature
# without saying that the key toggles or jumps to it. Hints are ordered by
# importance, most load-bearing first: footer_text() drops whole entries from
# the tail to fit the terminal, so a narrow screen loses the least useful
# hint instead of a hint being cut in half.
_FOOTER_TAIL = (
    '^R history', '^O show/hide tool details', 'PgUp/Dn scroll',
    '^G jump to latest', '^N newline', 'Ctrl-C twice exits',
)
# While a run is active Esc stops it; at the idle prompt Esc does nothing, so
# the idle footer says how to leave instead.
BUSY_HINTS = ('Enter send', 'Esc stop', '/ commands', '? keys') + _FOOTER_TAIL
IDLE_HINTS = ('Enter send', '/ commands', '? keys', 'Ctrl-D exit') + _FOOTER_TAIL
FOOTER_HINTS = BUSY_HINTS
# The canonical full-length footer, for a terminal wide enough to hold it and
# for callers that want the complete hint string.
FOOTER = ' | '.join(FOOTER_HINTS)
# Reverse-i-search history mode. Keys the surface owns while it is active:
# printable chars extend the query, ^R asks step older, ^S asks step newer,
# Enter accepts and Esc / Ctrl-C cancel and restore the saved draft.
# The single source of truth for the key overlay (opened with ?), the
# footer copy and /help. Keeping it module-level lets tests assert the
# same values the renderer paints.
# The one exit story. At the idle prompt Ctrl-C twice within CTRL_C_WINDOW
# seconds leaves; Ctrl-D on an empty draft and /exit leave at once. The
# footer, the key overlay and this hint all say exactly that.
EXIT_HINT = 'press Ctrl-C again within 2s to exit (or Ctrl-D, /exit)'
WELCOME = (
    'Type a message and press Enter to send it.',
    '/ lists commands, ? shows every key.',
    'Ctrl-D, /exit or Ctrl-C twice leaves.',
)
BINDINGS = (
    ('Enter',         'send'),
    ('Esc',           'stop / close menu / cancel'),
    ('/',             'commands'),
    ('?',             'this overlay'),
    ('^O',            'show / hide tool details'),
    ('PgUp / PgDn',   'scroll transcript'),
    ('^Y / ^V',       'scroll transcript'),
    ('^G',            'jump to latest'),
    ('^N',            'newline without sending'),
    ('Ctrl-C',        'stop run or clear draft; twice in 2s exits'),
    ('Ctrl-D',        'exit (empty draft)'),
    ('Tab',           'complete / accept menu item'),
    ('Up / Down',     'history / move menu highlight'),
)
MENU_ROWS = 8
# A paste whose end marker never arrives (a dropped ssh tail) must not own the
# keyboard: it is delivered after this many idle seconds, and capped in size.
PASTE_IDLE = 2.0
# Seconds of frame animation after anything visible happens, then the prompt
# goes still; IDLE_POLL_MS bounds how late idle output or a key is noticed.
LIVE_GRACE = 3.0
IDLE_POLL_MS = 100
WHEEL_LINES = 3
OSC52_MAX = 74_994
PASTE_MAX = 4_000_000
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")


def clean(text):
    return ''.join(c for c in ANSI.sub('', text)
                   if c in '\n\t' or (ord(c) >= 32 and not 0x7f <= ord(c) <= 0x9f))


def cells(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)


def footer_text(width, hints=FOOTER_HINTS):
    """The footer hints that fit `width`, most important first.

    Drops whole entries from the tail until the joined line fits, so a hint
    is either shown in full or not at all - never clipped mid-word. Returns
    a single row; the caller draws it with its own attribute."""
    chosen = list(hints)
    while chosen and cells(' | '.join(chosen)) > width:
        chosen = chosen[:-1]
    return ' | '.join(chosen)


def menu_hint(width, position):
    """The command-menu hint row for `width` columns.

    Wide terminals get every key. Narrow ones keep the ones that decide what
    the menu does, `Enter run` and `Esc close`, and drop the navigation hints
    first - clipped mid-word, the hint lost exactly those two."""
    full = [position, 'Up/Dn/PgUp/PgDn select', 'Tab complete', 'Enter run', 'Esc close']
    if cells(' ' + ' | '.join(full)) <= width:
        return ' ' + ' | '.join(full)
    return ' ' + footer_text(width - 1, [position, 'Enter run', 'Esc close', 'Tab complete', 'Up/Dn select'])


MIN_HEIGHT, MIN_WIDTH = 8, 16


def too_small_notice(height, width):
    """Rows telling the user to enlarge a terminal below the minimum size.

    Each row is a whole phrase that fits `width`; the shortest phrase is a
    bare size so even a 4-column window says what it needs."""
    need = f'{MIN_WIDTH}x{MIN_HEIGHT}'
    have = f'{width}x{height}'
    first = [f'one: enlarge terminal to {need}', 'one: enlarge terminal', 'enlarge terminal', 'enlarge', need]
    second = [f'now {have}, need {need}', f'now {have}', have]
    rows = []
    for options in (first, second):
        row = next((o for o in options if cells(o) <= width - 1), None)
        if row:
            rows.append(row)
    return rows[:max(1, height)]


def ellipsize(text, width):
    """`text` cut at a word boundary to fit `width` columns, ending in an ellipsis."""
    if cells(text) <= width:
        return text
    if width < 2:
        return text[:max(0, width)]
    cut = text
    while cut and cells(cut) > width - 1:
        cut = cut[:-1]
    if ' ' in cut and cut != text and text[len(cut)] != ' ':
        cut = cut[:cut.rindex(' ')]
    return cut.rstrip() + '\u2026'


def fit_title(title, width):
    """The title line for `width` columns, never cut mid-word.

    Segments are joined by two spaces, a pipe and two spaces. Brand and the
    last segment (the workspace path) outrank the rest, so the model and
    context drop first. A path that still does not fit keeps its tail behind
    an ellipsis, because the end of a path is the part that tells projects
    apart."""
    sep = '  |  '
    parts = title.split(sep)
    order = [0] + ([len(parts) - 1] if len(parts) > 1 else []) + list(range(1, len(parts) - 1))
    for keep in range(len(order), min(1, len(order) - 1), -1):
        chosen = sorted(order[:keep])
        line = sep.join(parts[i] for i in chosen)
        if cells(line) <= width:
            return line
    if len(parts) > 1:
        head = parts[0] + sep
        room = width - cells(head) - 1
        if room >= 3:
            tail = parts[-1]
            while tail and cells(tail) > room:
                tail = tail[1:]
            return head + '\u2026' + tail
    return parts[0][:max(1, width)]


def wrap(text, width):
    rows, row, size = [], '', 0
    for c in text.expandtabs(4):
        n = cells(c)
        if c == '\n' or size + n > width:
            rows.append(row)
            row, size = '', 0
        if c != '\n':
            row += c
            size += n
    return rows + [row]


class Sink(io.TextIOBase):
    def __init__(self, surface):
        self.surface = surface

    def write(self, text):
        self.surface.write(text)
        return len(text)

    def flush(self):
        pass


class RowRenderer:
    """Retain terminal rows, repainting only rows whose text/attributes change."""
    def __init__(self, screen):
        self.screen = screen
        self.previous = {}
        self.rows = {}
        self.size = None

    def getmaxyx(self):
        return self.screen.getmaxyx()

    def erase(self):
        self.rows = {}
        size = self.getmaxyx()
        if size != self.size:
            self.screen.erase()
            self.previous = {}
            self.size = size

    def addstr(self, y, x, text, attr=0):
        self.rows.setdefault(y, []).append((x, text, attr))

    def addnstr(self, y, x, text, n):
        self.addstr(y, x, text[:n])

    def chgat(self, y, start, length, attr):
        """Add `attr` to the cells [start, start+length) of the retained row."""
        stop = start + length
        row = []
        for x, text, old in self.rows.get(y, []):
            end = x + len(text)
            if end <= start or x >= stop:
                row.append((x, text, old))
                continue
            lo, hi = max(x, start), min(end, stop)
            for a, b, extra in ((x, lo, 0), (lo, hi, attr), (hi, end, 0)):
                if b > a:
                    row.append((a, text[a - x:b - x], old | extra))
        self.rows[y] = row

    def move(self, y, x):
        self.cursor = (y, x)

    def refresh(self):
        import curses
        for y in sorted(self.previous.keys() | self.rows.keys()):
            row = self.rows.get(y, [])
            if row == self.previous.get(y, []):
                continue
            try:
                self.screen.move(y, 0)
                self.screen.clrtoeol()
                for x, text, attr in row:
                    self.screen.addstr(y, x, text, attr)
            except curses.error:
                pass
        self.previous = self.rows
        if hasattr(self, 'cursor'):
            self.screen.move(*self.cursor)
        self.screen.refresh()


class Surface:
    def __init__(self, title='one', screen=None, theme=None):
        self.title = title
        self.screen = screen
        self.renderer = RowRenderer(screen) if screen is not None else None
        self.strip_cache = None
        self.wrap_memo = {}
        self.panel_bounds = (0, 0)
        self.panel_height = 10
        self.lock = threading.RLock()
        self.turns = queue.Queue()
        self.answers = queue.Queue()
        self.lines = deque(maxlen=4000)
        self.partial = ''
        self.transcript_revision = 0
        self.transcript_cache = None
        self.draft = ''
        self.cursor = 0
        self.saved_draft = None
        self.allow_steer = False
        self.force_pending = False
        # Esc / Ctrl-C stopped a turn: whatever was queued behind it must not
        # start on its own when the prompt returns.
        self.stopped_turn = False
        self.busy = False
        # A line-mode other than chat that owns the next Enter (the /model
        # chooser). Shown in the status line so the prompt is not a hidden mode.
        self.mode = ''
        self.answering = False
        self.stopped = threading.Event()
        self.history = []
        self.history_index = 0
        self.history_draft = ''
        # Reverse-i-search state. Active while `history_query is not None`;
        # owns printable keys (extend query), ^R / ^S (step), Enter (accept),
        # Esc / ^C (cancel and restore the saved draft). Lives on its own so
        # that `Surface.key()` can short-circuit early without branching the
        # normal editing path.
        self.history_query = None
        self.history_saved = None
        self.history_match_index = None
        # Ctrl-C at the idle prompt: first press shows the leave hint, a
        # second press within the window actually exits. Mirrors Claude Code
        # behaviour and the non-curses hint in cli.py.
        self.ctrl_c_hint_at = None
        self.scroll = 0
        self.arrows = []
        self.raw_out = None
        self.panel_rows = {}
        self.select = self.select_end = None
        self.escape = ''
        self.escape_started = 0.0
        self.tool_results = {}
        self.tool_commands = {}
        self.details = False
        self.pasting = False
        self.paste = ''
        self.paste_at = 0.0
        self.paste_edge = ''
        # Keys that arrive in the same burst as a paste's end marker. A pasted
        # `ESC[201~` ends paste mode early; what follows it is still pasted
        # text, so it is held here and inserted literally, never run as keys.
        self.paste_tail = None
        self.completer = None
        # The highlighted slash command, the first one shown, and the draft
        # whose menu Esc closed (it stays closed until the draft changes).
        self.menu_index = 0
        self.menu_top = 0
        self.menu_rows = MENU_ROWS
        self.menu_dismissed = None
        # True once Up/Down/PgUp/PgDn moved the highlight: that is a choice,
        # where an untouched highlight is only the first alphabetical match.
        self.menu_navigated = False
        self.glyphs = []
        self.glyph_colors = []
        self.animation_colors = {}
        self.accents = {}
        self.text_styles = {}
        # `?` overlay: a list of (key, description) rows drawn over the
        # transcript while `self.overlay is True`. Any keypress closes it.
        self.overlay = False
        from onecli.motion_theme import MotionTheme
        self.motion = MotionTheme(theme or os.environ.get('ONE_CLI_THEME', 'braid-stream'))
        self.reduced = os.environ.get('ONE_REDUCED_MOTION', '').lower() in ('1', 'true', 'yes')
        self.started = time.monotonic()
        self.turn_started = self.started
        self.live_until = 0.0

    def touch(self):
        """Note something visible happened; the frame loop animates for a while."""
        self.live_until = time.monotonic() + LIVE_GRACE

    def animating(self, now):
        """True while there is anything to animate; an idle prompt draws nothing."""
        return bool(self.busy or self.answering or self.pasting or self.escape
                    or self.motion.cards or now < self.live_until)

    def activity(self, kind, data=None):
        with self.lock:
            self.motion.event(kind, data)
        self.touch()

    def track_task(self, task):
        with self.lock:
            self.motion.track(task)
        self.touch()

    # Ctrl-C at the idle prompt: first press tells the user how to leave, a
    # second press within ~2s actually leaves. Both behaviours are what
    # Claude Code does on its idle prompt; the cli.py non-curses path prints
    # the same hint and never exits on its own.
    CTRL_C_WINDOW = 2.0

    def ctrl_c_handle(self):
        now = time.monotonic()
        recent = self.ctrl_c_hint_at is not None and now - self.ctrl_c_hint_at <= self.CTRL_C_WINDOW
        if self.draft:
            # ^C clears the draft (the original behaviour) but is also the
            # one good time to mention the exit, since the user just lost work.
            self.draft, self.cursor = '', 0
            self.write('\ndraft cleared - ' + EXIT_HINT + '\n')
            self.ctrl_c_hint_at = now
        elif recent:
            # Second ^C within the window: leave the session. The render
            # thread is daemon; to actually drop the user back to the shell
            # we have to unblock `Surface.read()` on the main thread by
            # raising EOFError through the turn queue.
            self.turns.put(EOFError())
            self.stopped.set()
        else:
            self.write('\n' + EXIT_HINT + '\n')
            self.ctrl_c_hint_at = now

    def write(self, text):
        self.touch()
        with self.lock:
            self.transcript_revision += 1
            chunks = (self.partial + clean(text)).split('\n')
            self.lines.extend(chunks[:-1])
            self.partial = chunks[-1][-16000:]

    def tool_start(self, name, arguments, mutating=False):
        from onecli.toolview import describe
        with self.lock:
            marker = f'\x00command:{self.transcript_revision}'
            self.tool_commands[marker] = clean(('◆ ' if mutating else '◇ ') + describe(name, arguments))
            self.lines.append(marker)
            self.transcript_revision += 1
            active = set(self.lines)
            self.tool_commands = {k: v for k, v in self.tool_commands.items() if k in active}

    def tool_result(self, output):
        with self.lock:
            marker = f'\x00tool:{self.transcript_revision}'
            self.tool_results[marker] = clean(output)
            self.lines.append(marker)
            self.transcript_revision += 1
            active = set(self.lines)
            self.tool_results = {k: v for k, v in self.tool_results.items() if k in active}

    def flush_escape(self, now=None):
        if self.escape == '\x1b' and (now if now is not None else time.monotonic()) - self.escape_started >= .08:
            pending, self.escape = self.escape, ''
            if pending == '\x1b':
                self.key('\x1b')
        elif self.escape.startswith('\x1b[<') and (now if now is not None else time.monotonic()) - self.escape_started >= .5:
            # A mouse report that never finished: drop it, it is not text.
            self.escape = ''

    def flush_paste(self, now=None):
        """Deliver a paste whose end marker was lost, once input has gone quiet."""
        if self.pasting and (now if now is not None else time.monotonic()) - self.paste_at >= PASTE_IDLE:
            text, self.pasting, self.paste = self.paste, False, ''
            self.insert_pasted(text)
            return True
        return False

    def transcript_rows(self, width):
        from onecli.transcript import layout
        key = (self.transcript_revision, width)
        if self.transcript_cache is None or self.transcript_cache[0] != key:
            from onecli.toolview import result_rows, command_rows
            lines = []
            for line in self.lines:
                if line in self.tool_commands:
                    lines.extend(command_rows(self.tool_commands[line], width, self.details))
                else:
                    lines.extend(result_rows(self.tool_results[line], width, self.details) if line in self.tool_results else [line])
            rows = layout(lines + ([self.partial] if self.partial else []), width)
            if self.scroll and self.transcript_cache is not None:
                old_key, old_rows = self.transcript_cache
                if old_key[1] == width:
                    self.scroll += max(0, len(rows) - len(old_rows))
            self.transcript_cache = (key, rows)
        return self.transcript_cache[1]

    def submit(self):
        with self.lock:
            text = self.draft
            if not text.strip() and not self.answering:
                if (self.busy and self.allow_steer and not self.force_pending
                        and self.pending_messages()
                        and all(not value.startswith(("/", "!")) for value in self.pending_messages())):
                    self.force_pending = True
                    os.kill(os.getpid(), signal.SIGINT)
                return
            self.draft, self.cursor = '', 0
            if not self.answering and text.strip().split()[:1] == ['/theme']:
                from onecli.motion_theme import THEMES
                parts = text.strip().split()
                if len(parts) == 1:
                    self.write('\ntheme: ' + self.motion.randomize() + '\n')
                elif len(parts) == 2 and self.motion.select(parts[1]):
                    self.write('\ntheme: ' + parts[1] + '\n')
                else:
                    self.write('\n/theme ' + ' | '.join(THEMES) + '\n')
                return
            if not self.answering and text.strip() == '/queue':
                pending = self.pending_messages()
                self.write('\nQueued messages:\n' + ('\n'.join(
                    f'{i}. {value}' for i, value in enumerate(pending, 1)) or 'empty') + '\n')
                return
            self.motion.event('submit')
            if self.answering:
                self.answers.put(text)
            else:
                self.turns.put(text)
                self.history.append(text)
                self.history_index = len(self.history)
                try:
                    import readline
                    readline.add_history(text)
                except ImportError:
                    pass
            self.scroll = 0

    def pending_messages(self):
        with self.turns.mutex:
            return [value for value in self.turns.queue if isinstance(value, str)]

    def hold_queued(self):
        """After Esc stopped a turn, hand queued messages back as a draft.

        They are not dropped and not run: the person pressed Esc to stop, so
        the next request is theirs to confirm. Ctrl-D stays queued.
        """
        with self.lock:
            stopped, self.stopped_turn = self.stopped_turn, False
            if not stopped:
                return
            with self.turns.mutex:
                held = [v for v in self.turns.queue if isinstance(v, str)]
                kept = [v for v in self.turns.queue if not isinstance(v, str)]
                self.turns.queue.clear()
                self.turns.queue.extend(kept)
            if not held:
                return
            text = '\n\n'.join(held)
            self.draft = self.draft + ('\n\n' if self.draft else '') + text
            self.cursor = len(self.draft)
            noun = 'message' if len(held) == 1 else 'messages'
            self.write(f'\nstopped - {len(held)} queued {noun} not sent; back in the prompt, Enter to send\n')

    def read(self, prompt=''):
        with self.lock:
            self.busy = False
            self.motion.event('ready')
        self.hold_queued()
        value = self.turns.get()
        with self.lock:
            if self.force_pending:
                # Preserve arrival order and every pending request. Commands
                # remain separate so forcing can never execute a queued command.
                pending = [value]
                with self.turns.mutex:
                    while (self.turns.queue and isinstance(self.turns.queue[0], str)
                           and not self.turns.queue[0].startswith(('/', '!'))):
                        pending.append(self.turns.queue.popleft())
                value = '\n\n'.join(pending) if all(isinstance(v, str) for v in pending) else value
                self.force_pending = False
        if isinstance(value, BaseException):
            raise value
        with self.lock:
            self.busy = True
            self.motion.event('waiting')
            self.turn_started = time.monotonic()
        self.write('\n> ' + value + '\n\n')
        return value

    def answer(self, timeout=0):
        with self.lock:
            self.saved_draft = (self.draft, self.cursor)
            self.draft, self.cursor = '', 0
            self.answering = True
            self.motion.event('answer')
        try:
            value = self.answers.get(timeout=timeout if timeout > 0 else None)
            if isinstance(value, BaseException):
                raise value
            self.write(value + '\n')
            return value
        except queue.Empty:
            return None
        finally:
            with self.lock:
                self.answering = False
                self.motion.event('waiting' if self.busy else 'ready')
                self.draft, self.cursor = self.saved_draft
                self.saved_draft = None
                while not self.answers.empty():
                    self.answers.get_nowait()

    def input(self, prompt=''):
        self.write(prompt)
        value = self.answer()
        if value is None:
            raise EOFError
        return value

    def feed(self, key):
        self.touch()
        before = self.draft
        # Bracketed paste is one editable draft, including embedded newlines.
        if self.pasting:
            if key in ('\x03', '\x04'):
                # Ctrl-C / Ctrl-D always get out: drop the paste, run the key.
                self.pasting, self.paste, self.paste_edge = False, '', ''
                self.key(key)
                return
            self.paste_at = time.monotonic()
            if isinstance(key, str):
                if len(self.paste) < PASTE_MAX:
                    self.paste += key
                    self.paste_edge = self.paste[-6:]
                else:
                    # Over the cap the text is dropped, but the end marker
                    # must still be recognised or the paste never closes.
                    self.paste_edge = (self.paste_edge + key)[-6:]
            if self.paste_edge == '\x1b[201~':
                capped = len(self.paste) >= PASTE_MAX
                self.insert_pasted(self.paste if capped else self.paste[:-6])
                self.pasting, self.paste, self.paste_edge = False, '', ''
                self.paste_tail = ''
                self.activity('typing', {'x': 0.0})
            return
        if self.paste_tail is not None:
            self.paste_tail += key if isinstance(key, str) else ''
            return
        import curses
        if key == curses.KEY_MOUSE:
            # terminfo's kmous is `ESC [ <` (xterm, tmux): curses swallows that
            # prefix and hands back KEY_MOUSE, then the rest of the SGR report
            # arrives as plain characters. Resume the report from its prefix.
            self.escape, self.escape_started = '\x1b[<', time.monotonic()
            return
        if key == '\x1b' or self.escape:
            if not self.escape:
                self.escape_started = time.monotonic()
            if not isinstance(key, str):
                self.escape = ''
                self.feed(key)
                return
            self.escape += key
            sequence = self.escape
            if sequence in ('\x1b', '\x1b[', '\x1bO'):
                return
            if sequence.startswith(('\x1b[', '\x1bO')):
                # Consume the WHOLE control sequence. Dropping a partial CSI
                # leaks its remaining digits/terminator into the user's draft.
                if not ('@' <= sequence[-1] <= '~'):
                    if len(sequence) > 64:
                        self.escape = ''
                    return
                self.escape = ''
                if sequence == '\x1b[200~':
                    self.pasting, self.paste_at = True, time.monotonic()
                    return
                import curses
                keys = {'A': curses.KEY_UP, 'B': curses.KEY_DOWN,
                        'C': curses.KEY_RIGHT, 'D': curses.KEY_LEFT,
                        'H': curses.KEY_HOME, 'F': curses.KEY_END}
                keycode = keys.get(sequence[-1])
                if sequence[-1] == '~':
                    keycode = {'1': curses.KEY_HOME, '2': curses.KEY_IC,
                               '3': curses.KEY_DC, '4': curses.KEY_END,
                               '5': curses.KEY_PPAGE, '6': curses.KEY_NPAGE,
                               '7': curses.KEY_HOME, '8': curses.KEY_END}.get(sequence[2:-1])
                if sequence.startswith('\x1b[<') and sequence[-1] in 'Mm':
                    self.mouse_report(sequence)
                elif keycode in (curses.KEY_UP, curses.KEY_DOWN):
                    self.arrows.append(keycode)
                elif keycode is not None:
                    self.key(keycode)
                # Terminal replies (e.g. CSI row;col R) are not user text.
                return
            self.escape = ''
            # Escape followed immediately by text must not eat that text.
            self.key('\x1b')
            self.feed(sequence[1:])
            return
        import curses
        if key in (curses.KEY_UP, curses.KEY_DOWN):
            self.arrows.append(key)
            return
        self.key(key)
        if self.draft != before and key not in ('\n', '\r', curses.KEY_ENTER,
                                                curses.KEY_UP, curses.KEY_DOWN, '\x03'):
            width = self.screen.getmaxyx()[1] - 3 if self.screen else 60
            self.activity('typing', {'x': 1.8 * (self.cursor % max(1, width)) / max(1, width) - .9})

    def clipboard(self, text):
        """Copy through the terminal itself, which reaches the local clipboard
        where wl-copy or xclip would not (remote logins, tmux). Inside tmux an
        application's own OSC 52 is dropped unless `set-clipboard on`, so tmux
        is asked instead: load-buffer -w fills its paste buffer and forwards
        to the clipboard under the default `set-clipboard external` too.
        Elsewhere OSC 52 is written. False when nothing could take the text
        or it is too big for the sequences terminals accept."""
        import base64
        payload = base64.b64encode(text.encode()).decode()
        if len(payload) > OSC52_MAX:
            return False
        if os.environ.get('TMUX'):
            try:
                return subprocess.run(['tmux', 'load-buffer', '-w', '-'], input=text.encode(),
                                      capture_output=True, timeout=2).returncode == 0
            except (OSError, subprocess.SubprocessError):
                pass
        if self.raw_out is None:
            return False
        self.raw_out.write(f'\x1b]52;c;{payload}\x07')
        self.raw_out.flush()
        return True

    def mouse_report(self, sequence):
        """An SGR mouse report. The wheel scrolls the transcript; a left-button
        drag over the transcript selects rows and copies them on release
        (OSC 52). Anything else is ignored."""
        try:
            button, col, row = (int(v) for v in sequence[3:-1].split(';'))
        except ValueError:
            return
        press = sequence[-1] == 'M'
        at = (row - 1, col - 1)
        if press and button in (64, 65):
            self.scroll_lines(WHEEL_LINES if button == 64 else -WHEEL_LINES)
        elif press and button == 0:
            self.select = self.select_end = at
        elif press and button == 32 and self.select:
            self.select_end = at
        elif not press and button == 0 and self.select and self.select != at:
            self.select_end = at
            self.clipboard(self.selected_text())
        elif not press and button == 0:
            self.select = self.select_end = None

    def selection_span(self):
        """(first, last) cell of the selection in reading order, or None."""
        if not self.select:
            return None
        return tuple(sorted((self.select, self.select_end)))

    def selected_text(self):
        span = self.selection_span()
        if span is None:
            return ''
        (y1, x1), (y2, x2) = span
        rows = []
        for y in range(y1, y2 + 1):
            text = self.panel_rows.get(y)
            if text is None:
                continue
            rows.append(text[x1 if y == y1 else 0:x2 + 1 if y == y2 else None].rstrip())
        return '\n'.join(rows)

    def scroll_lines(self, lines):
        with self.lock:
            self.scroll = max(0, self.scroll + lines)

    def settle_arrows(self):
        """Resolve the Up/Down keys that arrived since the last drain.

        A terminal or tmux turns one wheel notch on the alternate screen into
        several Up/Down keys in a single write (tmux: three; DEC 1007: one per
        step), while a person's own key press arrives alone. A burst therefore
        scrolls the transcript, a lone arrow keeps its editing meaning
        (history, menu). While the menu, the overlay or history search owns
        the arrows, they keep their meaning in a burst too."""
        arrows, self.arrows = self.arrows, []
        if not arrows:
            return False
        wheel = (len(arrows) >= 2 and self.history_query is None
                 and not self.overlay and not self.command_menu())
        if wheel:
            import curses
            self.scroll_lines(sum(1 if a == curses.KEY_UP else -1 for a in arrows))
        else:
            for arrow in arrows:
                self.key(arrow)
        return True

    def insert_pasted(self, raw):
        text = clean(raw.replace('\r\n', '\n').replace('\r', '\n'))
        with self.lock:
            self.draft = self.draft[:self.cursor] + text + self.draft[self.cursor:]
            self.cursor += len(text)

    def drain_done(self):
        """Input is exhausted for now: the burst that carried a paste is over."""
        tail, self.paste_tail = self.paste_tail, None
        if not tail:
            return False
        self.insert_pasted(tail)
        return True

    def key(self, key):
        import curses
        with self.lock:
            before = self.draft
            # `?` overlay: any key while the overlay is open closes it,
            # mirroring Codex and Claude Code. We consume the key rather
            # than passing it through so the open-closed draft is not
            # inserted.
            if self.overlay:
                self.overlay = False
                return
            # Reverse history search owns keys while it is open. This runs
            # BEFORE the menu check so that typing '/...' inside the search box
            # extends the query rather than opening a command menu.
            if self.history_query is not None:
                self.history_search_key(key)
                return
            # `?` on an empty draft opens the overlay; mid-draft it stays
            # a literal character.
            if key == '?' and not self.draft and not self.answering:
                self.overlay = True
                return
            menu = self.command_menu()
            if menu and self.menu_key(key, menu):
                pass
            elif key in ('\n', '\r', curses.KEY_ENTER):
                self.submit()
            elif key == '\x12':  # Ctrl-R: open reverse-i-search history.
                if not self.answering and self.history:
                    self.history_query = ''
                    self.history_saved = (self.draft, self.cursor)
                    self.history_match_index = len(self.history)
                    self.history_search_step()
                # Idle with no history is a no-op so the key is not lost as
                # noise.
            elif key == '\x13':  # Ctrl-S: step newer while search is open.
                if self.history_query is not None:
                    self.history_search_step(older=False)
                # Outside search mode ^S is unused; do not surface as an
                # editing character.
            elif key in ('\x03', '\x1b'):
                if self.history_query is not None:
                    self.history_search_cancel()
                elif self.answering:
                    self.answers.put(KeyboardInterrupt())
                elif self.busy and self.allow_steer:
                    # Only while the turn is inside the agent's try/except.
                    # `busy` also covers the gaps before and after a turn,
                    # where a SIGINT would escape to main() and kill the CLI.
                    self.stopped_turn = True
                    os.kill(os.getpid(), signal.SIGINT)
                elif key == '\x03' and not self.busy:
                    self.ctrl_c_handle()
            elif key == '\x0f':
                self.details = not self.details
                self.transcript_revision += 1
            elif key == '\x04':
                if not self.draft:
                    if self.answering:
                        self.answers.put(EOFError())
                    else:
                        self.turns.put(EOFError())
                elif self.cursor < len(self.draft):
                    self.draft = self.draft[:self.cursor] + self.draft[self.cursor + 1:]
                else:
                    self.write('\nCtrl-D leaves only from an empty prompt - Ctrl-C clears the draft\n')
            elif key in (curses.KEY_LEFT, '\x02'):
                self.cursor = max(0, self.cursor - 1)
            elif key in (curses.KEY_RIGHT, '\x06'):
                self.cursor = min(len(self.draft), self.cursor + 1)
            elif key in (curses.KEY_HOME, '\x01'):
                self.cursor = 0
            elif key in (curses.KEY_END, '\x05'):
                self.cursor = len(self.draft)
            elif key in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                if self.cursor:
                    self.draft = self.draft[:self.cursor - 1] + self.draft[self.cursor:]
                    self.cursor -= 1
            elif key == curses.KEY_DC:
                self.draft = self.draft[:self.cursor] + self.draft[self.cursor + 1:]
            elif key == '\t' and self.completer is not None:
                before = self.draft[:self.cursor]
                start = len(before)
                while start and before[start - 1] not in ' \t\n':
                    start -= 1
                matches = self.completer.matches(before[start:], not before[:start].strip())
                if matches:
                    replacement = matches[0] if len(matches) == 1 else os.path.commonprefix(matches)
                    self.draft = self.draft[:start] + replacement + self.draft[self.cursor:]
                    self.cursor = start + len(replacement)
                    # A slash word already lists its matches in the menu.
                    if len(matches) > 1 and not self.command_menu():
                        self.write('\n' + '  '.join(matches) + '\n')
            elif key == '\x15':
                self.draft = self.draft[self.cursor:]
                self.cursor = 0
            elif key == '\x0b':
                self.draft = self.draft[:self.cursor]
            elif key == '\x0e':  # Ctrl-N: newline without submitting.
                self.draft = self.draft[:self.cursor] + '\n' + self.draft[self.cursor:]
                self.cursor += 1
            elif key in (curses.KEY_PPAGE, curses.KEY_NPAGE, '\x19', '\x16'):
                self.scroll = max(0, self.scroll + (1 if key in (curses.KEY_PPAGE, '\x19') else -1) * max(1, self.panel_height - 1))
            elif key == '\x07':  # Ctrl-G: follow the live conversation again.
                self.scroll = 0
            elif key in (curses.KEY_UP, curses.KEY_DOWN) and not self.answering:
                if self.history_index == len(self.history):
                    self.history_draft = self.draft
                self.history_index = max(0, min(len(self.history), self.history_index + (-1 if key == curses.KEY_UP else 1)))
                self.draft = self.history[self.history_index] if self.history_index < len(self.history) else self.history_draft
                self.cursor = len(self.draft)
            elif isinstance(key, str) and key.isprintable():
                self.draft = self.draft[:self.cursor] + key + self.draft[self.cursor:]
                self.cursor += len(key)
            if self.draft != before:
                # A new filter starts at the top, as Claude Code and Codex do.
                self.menu_index = self.menu_top = 0
                self.menu_dismissed = None
                self.menu_navigated = False

    def menu_key(self, key, menu):
        """Keys the open slash menu owns, as in Claude Code and Codex: Up/Down
        move the highlight and wrap, Tab completes it, Enter runs it, Esc closes
        the menu and keeps the draft. Returns False for keys it does not own."""
        import curses
        if key in (curses.KEY_UP, curses.KEY_DOWN):
            self.menu_navigated = True
            step = -1 if key == curses.KEY_UP else 1
            self.menu_index = (self.menu_index + step) % len(menu)
            self.menu_visible(len(menu))
        elif key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
            # Page the highlight a screenful at a time, as Codex's popup does,
            # instead of scrolling the transcript underneath an open menu.
            self.menu_navigated = True
            page = -1 if key == curses.KEY_PPAGE else 1
            self.menu_index = max(0, min(len(menu) - 1, self.menu_index + page * self.menu_rows))
            self.menu_visible(len(menu))
        elif key == '\t':
            self.draft = menu[self.menu_index][0] + ' '
            self.cursor = len(self.draft)
        elif key in ('\n', '\r', curses.KEY_ENTER):
            name = menu[self.menu_index][0]
            if self.draft.strip() == name or self.menu_navigated or len(menu) == 1:
                self.draft = name
                self.cursor = len(self.draft)
                self.submit()
            else:
                # An ambiguous partial word is not a command: Enter completes
                # it like Tab and a second Enter runs it, so `/co` never runs
                # the first alphabetical match by accident. A single match,
                # an exact name or a highlight the person moved still runs.
                self.draft = name + ' '
                self.cursor = len(self.draft)
        elif key == '\x1b':
            if self.draft == '/':
                # A lone slash is only the menu's trigger; leaving it behind
                # turned the next `foo` into the chat message `/foo`.
                self.draft, self.cursor = '', 0
            else:
                self.menu_dismissed = self.draft
        else:
            return False
        return True

    def menu_visible(self, count):
        """Scroll the menu window so the highlighted row is on screen."""
        self.menu_index = min(self.menu_index, count - 1)
        if self.menu_index < self.menu_top:
            self.menu_top = self.menu_index
        elif self.menu_index >= self.menu_top + self.menu_rows:
            self.menu_top = self.menu_index - self.menu_rows + 1
        self.menu_top = max(0, min(self.menu_top, count - self.menu_rows))

    # -- Reverse-i-search history mode -----------------------------------------

    def history_search_key(self, key):
        """A key arrived while reverse-i-search is open."""
        import curses
        # Backspace trims the query without stepping.
        if key in (curses.KEY_BACKSPACE, '\x7f', '\b'):
            if self.history_query:
                self.history_query = self.history_query[:-1]
                self.history_match_index = len(self.history)
                self.history_search_step()
            return
        if key in ('\n', '\r', curses.KEY_ENTER):
            self.history_search_accept()
            return
        if key == '\x12':  # Ctrl-R: step to an older match.
            self.history_search_step(older=True)
            return
        if key == '\x13':  # Ctrl-S: step to a newer match.
            self.history_search_step(older=False)
            return
        if key in ('\x03', '\x1b'):
            self.history_search_cancel()
            return
        if isinstance(key, str) and key.isprintable():
            self.history_query += key
            self.history_match_index = len(self.history)
            self.history_search_step()

    def history_search_step(self, older=True):
        """Move the match pointer to the nearest entry containing the query.

        Matches are searched from the newest end; `older=True` continues moving
        toward older entries so repeated ^R cycles them, matching readline.
        A `None` match leaves the draft unchanged but keeps the query live so
        the user sees the search box without a candidate (Claude Code does
        the same).
        """
        history = self.history
        if not history or not self.history_query:
            self.draft, self.cursor = '', 0
            return
        step = -1 if older else 1
        start = (self.history_match_index - 1) if older else (self.history_match_index + 1)
        # Bound the search to the actual history; clamp the pointer so a ^S
        # past the newest match clears the draft instead of wrapping.
        for index in range(start, -1 if older else len(history), step):
            if self.history_query in history[index]:
                self.history_match_index = index
                self.draft, self.cursor = history[index], len(history[index])
                return
            if not older and index >= len(history):
                break
        # No match. Keep the query but show what we currently hold.
        if self.history_match_index is not None and self.history_match_index < len(history):
            self.draft, self.cursor = history[self.history_match_index], len(history[self.history_match_index])
        else:
            self.draft, self.cursor = '', 0

    def history_search_accept(self):
        """Replace the draft with the current match and close the search."""
        self.history_query = None
        self.history_saved = None
        self.history_match_index = None
        # The chosen line is now the draft - rewind history_index so that the
        # next Up walks from the present rather than from this match.
        self.history_index = len(self.history)
        self.history_draft = self.draft
        self.menu_index = self.menu_top = 0
        self.menu_dismissed = None

    def history_search_cancel(self):
        """Discard the search and restore the draft as it was when ^R arrived."""
        if self.history_saved is not None:
            self.draft, self.cursor = self.history_saved
        self.history_query = None
        self.history_saved = None
        self.history_match_index = None
        self.menu_index = self.menu_top = 0
        self.menu_dismissed = None

    def command_menu(self):
        """The slash commands matching the draft, with their summaries."""
        if (self.completer is None or self.answering or not hasattr(self.completer, 'menu')
                or self.draft == self.menu_dismissed):
            return []
        return self.completer.menu(self.draft)

    def status_text(self):
        if self.answering:
            return 'answer needed'
        if self.busy:
            state = self.motion.state
            if state not in ('thinking', 'streaming', 'tool'):
                state = 'thinking'
            return f'{state} {time.monotonic() - self.turn_started:.0f}s'
        if self.mode:
            return self.mode
        return 'typing' if self.draft else 'ready'

    def setup_colors(self):
        import curses
        if 'NO_COLOR' in os.environ:
            return
        try:
            if not curses.has_colors():
                return
            curses.start_color()
            curses.use_default_colors()
            # Existing terminal palette only; never mutate shared palette entries.
            palette = (109, 103, 144) if curses.COLORS >= 256 else (6, 4, 3)
            for index, color in enumerate(palette, 1):
                curses.init_pair(index, color, -1)
            self.accents = {'ready': curses.color_pair(1),
                            'typing': curses.color_pair(1),
                            'thinking': curses.color_pair(2),
                            'streaming': curses.color_pair(1),
                            'tool': curses.color_pair(3),
                            'answer needed': curses.color_pair(3)}
            roles = {'body': (252, 7), 'user': (117, 6), 'heading': (153, 6),
                     'strong': (255, 7), 'code': (151, 2), 'tool': (180, 3),
                     'output': (250, 7), 'meta': (245, 7), 'error': (210, 1)}
            for index, (role, colors) in enumerate(roles.items(), 4):
                curses.init_pair(index, colors[0 if curses.COLORS >= 256 else 1], -1)
                self.text_styles[role] = curses.color_pair(index)
            if curses.COLORS >= 256 and curses.COLOR_PAIRS >= 232:
                for color in range(16, 232):
                    curses.init_pair(color, color, -1)
                    self.animation_colors[color] = curses.color_pair(color)
        except curses.error:
            self.accents = {}
            self.animation_colors = {}

    def animation_attr(self, rgb):
        if not self.animation_colors:
            if max(rgb) - min(rgb) < 8:
                return 0
            return self.accents.get(self.motion.state, 0)
        levels = (0, 95, 135, 175, 215, 255)
        r, g, b = (min(range(6), key=lambda i: abs(levels[i] - v)) for v in rgb)
        return self.animation_colors.get(16 + 36 * r + 6 * g + b, 0)

    def editor_wrap(self, slot, text, width):
        """wrap() for the editor, remembered while the text is unchanged.

        draw() runs every frame; re-walking a 100KB+ draft each time pinned the
        CPU and starved input. One entry per slot, so the many short per-line
        wraps in a frame cannot evict it.
        """
        hit = self.wrap_memo.get(slot)
        if hit is None or hit[0] != (text, width):
            hit = self.wrap_memo[slot] = ((text, width), wrap(text, width))
        return list(hit[1])

    def draw(self):
        import curses
        with self.lock:
            screen = self.renderer
            height, width = screen.getmaxyx()
            if height < 8 or width < 16:
                screen.erase()
                try:
                    for y, row in enumerate(too_small_notice(height, width)):
                        screen.addnstr(y, 0, row, max(1, width - 1))
                    screen.refresh()
                except curses.error:
                    pass
                return
            width -= 1
            def put(y, text, attr=0):
                if 0 <= y < height:
                    try:
                        screen.addstr(y, 0, wrap(clean(text), width)[0], attr)
                    except curses.error:
                        pass
            screen.erase()
            put(0, fit_title(self.title, width), curses.A_BOLD)
            def put_animation(y, line, colors):
                # Group adjacent equal attributes to keep SSH output compact.
                if not colors or not self.accents:
                    put(y, line, self.accents.get(self.motion.state, 0))
                    return
                start = 0
                while start < min(width, len(line)):
                    attr = self.animation_attr(colors[start])
                    end = start + 1
                    while end < min(width, len(line)) and self.animation_attr(colors[end]) == attr:
                        end += 1
                    try:
                        screen.addstr(y, start, line[start:end], attr)
                    except curses.error:
                        pass
                    start = end
            glyph_height = min(4, max(0, height - 19))
            for i, line in enumerate(self.glyphs[:glyph_height]):
                put_animation(i + 1, line, self.glyph_colors[i] if i < len(self.glyph_colors) else [])
            top = glyph_height + 1
            for card in list(self.motion.cards.values())[:2]:
                if top + 4 >= height - 10:
                    break
                task = card['task']
                status = task.status
                if status == 'running' and task.proc is not None:
                    code = task.proc.poll()
                    if code is not None:
                        status = f'exited {code}'
                put(top, f'[{task.task_id}] {status}: {task.title}')
                for i, row in enumerate(self.motion.frame(min(width, 48), 2, self.reduced, card['reveal'])):
                    top += 1
                    put_animation(top, row, self.motion.colors[i])
                top += 1
            panel_header = top
            top += 1
            editor_width = max(1, width - 2)
            draft_rows = self.editor_wrap('draft', self.draft, editor_width)
            cursor_rows = self.editor_wrap('cursor', self.draft[:self.cursor], editor_width)
            cy = len(cursor_rows) - 1
            cx = cells(cursor_rows[-1])
            if cx == editor_width:
                cy, cx = cy + 1, 0
                if len(draft_rows) <= cy:
                    draft_rows.append('')
            editor_height = min(4, max(2, len(draft_rows)))
            footer = height - editor_height - 3
            state = self.status_text()
            accent = self.accents.get('answer needed' if self.answering else state.split()[0], 0)
            put(footer, '-' * width, curses.A_DIM)
            # This lane remains visible even when the terminal has no banner room.
            # Timer digits and state labels must not rescale moving geometry.
            strip_width = min(26, max(4, width - 20))
            strip_key = (strip_width, self.motion.time, self.reduced)
            if self.strip_cache is None or self.strip_cache[0] != strip_key:
                self.strip_cache = (strip_key, self.motion.frame(strip_width, 1, self.reduced)[0])
            strip = self.strip_cache[1]
            status = f'{strip} {state}'
            queued = self.turns.qsize()
            if queued and len(status) + len(f' | {queued} queued') <= width:
                status += f' | {queued} queued'
            put(footer + 1, status, accent if self.motion.chroma > .03 else 0)
            offset = max(0, cy - editor_height + 1)
            for i, row in enumerate(draft_rows[offset:offset + editor_height]):
                put(footer + 2 + i, ('> ' if i == 0 else '  ') + row, self.text_styles.get('user', 0))
            if self.history_query is not None:
                # The search box overlays the first editor row, as in
                # Claude Code and readline; the matched draft IS the draft
                # so the editor block above already painted it. Drawing the
                # box last means the reverse-video bar wins on the real
                # screen (RowRenderer paints segments in order).
                prompt = f"(reverse-i-search)`{self.history_query}': "
                put(footer + 2, prompt[:max(1, width - 1)], curses.A_REVERSE)
            put(height - 1, footer_text(width, BUSY_HINTS if self.busy or self.answering else IDLE_HINTS), curses.A_DIM)
            # The transcript ends above the command menu, the menu above the rule.
            panel_end = footer
            menu = self.command_menu()
            if menu:
                self.menu_rows = max(1, min(MENU_ROWS, footer - top - 2))
                self.menu_visible(len(menu))
                rows = menu[self.menu_top:self.menu_top + self.menu_rows]
                name_width = max(len(name) for name, _ in rows)
                fuzzy = getattr(self.completer, 'menu_is_fuzzy', None)
                dim = fuzzy(self.draft) if fuzzy else False
                for i, (name, summary) in enumerate(rows):
                    attr = curses.A_REVERSE if self.menu_top + i == self.menu_index else (
                        curses.A_DIM if dim else 0)
                    lead = f' {name.ljust(name_width)}  '
                    put(panel_end - len(rows) + i, lead + ellipsize(summary, width - cells(lead)), attr)
                rows.append(None)
                put(panel_end - len(rows), menu_hint(width, f'{self.menu_index + 1}/{len(menu)}'), curses.A_DIM)
                panel_end -= len(rows)
            transcript = list(self.transcript_rows(width))
            pending = self.pending_messages()
            if pending:
                transcript.append([(' queued | Enter again to steer | /queue lists all', 'heading')])
                for index, value in enumerate(pending[-3:], max(1, len(pending) - 2)):
                    transcript.append([(f' {index}. ' + clean(value).replace('\n', ' ')[:max(1, width - 5)], 'user')])
            welcome = not transcript
            if welcome:
                transcript = [[(line[:width], 'heading' if i == 0 else 'user')] for i, line in enumerate(WELCOME)]
                # Centre the block so a tall window is not left with an empty lower half.
                transcript[:0] = [[] for _ in range(max(0, (panel_end - top - len(transcript)) // 2))]
            count = max(0, panel_end - top)
            self.scroll = min(self.scroll, max(0, len(transcript) - count))
            self.panel_height = count
            self.panel_bounds = (top, panel_end)
            label = ' welcome ' if welcome else f' conversation | {self.scroll} lines above live | ^G live ' if self.scroll else ' conversation '
            put(panel_header, label.ljust(width, '-'), curses.A_DIM)
            end = len(transcript) - self.scroll
            self.panel_rows = {}
            for i, row in enumerate(transcript[max(0, end - count):end]):
                x = 0
                self.panel_rows[top + i] = ''.join(text for text, _ in row)
                for text, role in row:
                    attr = self.text_styles.get(role, 0)
                    if role in ('heading', 'strong'):
                        attr |= curses.A_BOLD
                    try:
                        screen.addstr(top + i, x, text, attr)
                    except curses.error:
                        pass
                    x += cells(text)
            span = self.selection_span()
            if span and span[0] != span[1]:
                (y1, x1), (y2, x2) = span
                for y in range(max(y1, top), min(y2, panel_end - 1) + 1):
                    start = x1 if y == y1 else 0
                    stop = min(width, x2 + 1) if y == y2 else width
                    try:
                        screen.chgat(y, start, max(0, stop - start), curses.A_REVERSE)
                    except curses.error:
                        pass
            if self.overlay:
                self.draw_overlay(width, height, top, count, screen)
            try:
                screen.move(min(height - 2, footer + 2 + cy - offset), min(width - 1, cx + 2))
                screen.refresh()
            except curses.error:
                pass

    def draw_overlay(self, width, height, panel_top, panel_rows, screen):
        """Render the `?` key overlay over the whole transcript panel.

        The box spans the panel's rows (header row included) and clears its
        full width, so on a short terminal it shows as many keys as fit and
        says how many it left out instead of letting transcript text show
        around it."""
        import curses
        top = max(1, panel_top - 1)
        avail = max(4, panel_rows + 1)
        shown = min(len(BINDINGS), avail - 2)
        rows = list(BINDINGS[:shown])
        key_width = max(len(k) for k, _ in rows)
        box_width = width
        y = top
        foot = y + shown + 1
        more = len(BINDINGS) - shown
        hint = f' any key closes | +{more} more ' if more else ' any key closes '

        def put(row, text, attr=0):
            try:
                screen.addstr(row, 0, text[:box_width].ljust(box_width), attr)
            except curses.error:
                pass
        put(y, '+- keys ' + '-' * box_width, curses.A_BOLD)
        for i, (key, desc) in enumerate(rows, 1):
            put(y + i, f'| {key.ljust(key_width)}  {desc}')
        put(foot, '+-' + hint + '-' * box_width, curses.A_BOLD)

    def restore_draft(self, sid):
        from onecli import session as storage
        with self.lock:
            self.draft = storage.load_draft(sid)
            self.cursor = len(self.draft)
            self.draft_sid = sid
            self.persisted_draft = self.draft

    def persist_draft(self):
        from onecli import session as storage
        with self.lock:
            sid = getattr(self, 'draft_sid', None)
            text = self.saved_draft[0] if self.answering and self.saved_draft else self.draft
            if sid and text != self.persisted_draft:
                try:
                    storage.save_draft(sid, text)
                except OSError:
                    return
                self.persisted_draft = text

    def run(self):
        import curses
        last = time.monotonic()
        # The first frame is what paints the prompt; without this the surface
        # starts idle and the screen stays blank until the first keypress.
        self.touch()
        try:
            while not self.stopped.is_set():
                size = os.get_terminal_size(sys.stdin.fileno())
                if self.screen.getmaxyx() != (size.lines, size.columns):
                    curses.resizeterm(size.lines, size.columns)
                    self.screen.clearok(True)
                    self.touch()
                now = time.monotonic()
                self.flush_escape(now)
                if self.flush_paste(now):
                    self.persist_draft()
                if not self.animating(now):
                    # Idle prompt: no motion, no frame redraw. Poll slowly; new
                    # output, keys and resizes call touch() and resume frames.
                    last = now
                    self.screen.timeout(IDLE_POLL_MS)
                elif now - last >= 1 / 30:
                    h, w = self.screen.getmaxyx()
                    with self.lock:
                        self.motion.advance(now - last, self.reduced)
                        if h >= 20:
                            self.glyphs = self.motion.frame(max(12, w - 1), min(4, h - 19), self.reduced)
                            self.glyph_colors = self.motion.colors
                    last = now
                    self.draw()
                if self.animating(now):
                    self.screen.timeout(max(1, min(16, int((last + 1 / 30 - time.monotonic()) * 1000))))
                # Drain everything already queued, then draw once. One draw per
                # key starves input over a slow SSH link and makes typing drop.
                changed = False
                # A steady stream of input must still leave time for redraws
                # and shutdown; drain a bounded batch per frame.
                for _ in range(256):
                    if self.stopped.is_set():
                        break
                    try:
                        key = self.screen.get_wch()
                    except curses.error:
                        changed = self.drain_done() or changed
                        break
                    if key == curses.KEY_RESIZE:
                        self.screen.clearok(True)
                    else:
                        self.feed(key)
                        changed = True
                    self.screen.timeout(0)
                changed = self.settle_arrows() or changed
                # A paste in flight changes nothing the person can see yet;
                # drawing and saving the draft per batch made a 100KB paste
                # cost O(n^2).
                if changed and not self.pasting:
                    self.persist_draft()
                    self.draw()
        except BaseException as exc:
            self.write(f'\nterminal stopped: {exc}\n')
            self.turns.put(EOFError())
            self.answers.put(EOFError())


def tmux_mouse():
    """True inside tmux when its `mouse` option is on; ONE_NO_MOUSE=1 opts out."""
    if not os.environ.get('TMUX') or os.environ.get('ONE_NO_MOUSE') == '1':
        return False
    try:
        shown = subprocess.run(['tmux', 'show-options', '-Av', 'mouse'], capture_output=True,
                               text=True, timeout=1).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return shown.strip() == 'on'


@contextmanager
def session(title='one', theme=None):
    global ACTIVE
    if (not sys.stdin.isatty() or not sys.stdout.isatty()
            or os.environ.get('TERM', '') in ('', 'dumb')
            or os.environ.get('ONE_PLAIN') == '1'):
        yield None
        return
    try:
        import curses
    except ImportError:
        yield None
        return
    # libedit initializes termios when readline is first imported on macOS.
    # Load it before curses owns the terminal, otherwise inputline.install()
    # can re-enable canonical input, echo and signals under the live editor.
    try:
        import readline  # noqa: F401 - imported for its termios side effect
    except ImportError:
        pass
    original_out, original_err, original_input = sys.stdout, sys.stderr, builtins.input
    # curses holds a bare Esc for ESCDELAY (1000 ms by default) waiting for a
    # sequence; Surface.flush_escape already resolves a lone Esc after 80 ms.
    curses.set_escdelay(25)
    screen = curses.initscr()
    surface = Surface(title, screen, theme)
    thread = None
    try:
        surface.setup_colors()
        curses.noecho()
        curses.raw()
        screen.keypad(True)
        # No mouse reporting: the terminal keeps native select, copy and paste.
        screen.timeout(16)
        # Alternate scroll (DEC 1007): terminals that honour it turn the wheel
        # into Up/Down on the alternate screen, which settle_arrows reads as a
        # scroll. It is not mouse reporting, so native selection stays.
        # tmux with `mouse on` swallows the wheel (copy-mode has no history for
        # an alternate-screen app) unless the app asks for mouse reports, so
        # only there do we ask, and then we own drag-select too.
        mouse = tmux_mouse()
        original_out.write('\x1b[?2004h\x1b[?1007h' + ('\x1b[?1002h\x1b[?1006h' if mouse else ''))
        original_out.flush()
        ACTIVE = surface
        surface.raw_out = original_out
        sys.stdout = sys.stderr = Sink(surface)
        builtins.input = surface.input
        thread = threading.Thread(target=surface.run, name='one-terminal', daemon=True)
        thread.start()
        yield surface
    finally:
        surface.stopped.set()
        if thread is not None:
            thread.join()
        ACTIVE = None
        sys.stdout, sys.stderr, builtins.input = original_out, original_err, original_input
        screen.keypad(False)
        curses.noraw()
        curses.echo()
        curses.endwin()
        original_out.write("\x1b[?1006l\x1b[?1002l\x1b[?1007l\x1b[?2004l")
        original_out.flush()
        # Keep the session readable in normal terminal scrollback after exit.
        from onecli.toolview import result
        for line in surface.lines:
            print(surface.tool_commands[line] if line in surface.tool_commands else '\n'.join(result(surface.tool_results[line], surface.details)) if line in surface.tool_results else line)
        if surface.partial:
            print(surface.partial)

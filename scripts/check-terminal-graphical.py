"""Graphical VTE acceptance; run with system Python under xvfb-run.

Requires GTK3/VTE GIR, python3-gi and xdotool. No provider or user data is used.
Use --interrupt for Ctrl-C during work; --capture-mouse is a negative control.
"""

import os
import sys
import json
import time
import subprocess
import threading
import traceback
from pathlib import Path

if "--child" not in sys.argv and "--display-child" not in sys.argv:
    raise SystemExit(
        subprocess.call(
            [
                "xvfb-run",
                "-a",
                sys.executable,
                str(Path(__file__).resolve()),
                "--display-child",
                *sys.argv[1:],
            ]
        )
    )

ROOT = Path(__file__).resolve().parents[1]
mode = (
    "mouse-negative"
    if "--capture-mouse" in sys.argv
    else "interrupt"
    if "--interrupt" in sys.argv
    else "normal"
)
columns = int(sys.argv[sys.argv.index('--columns') + 1]) if '--columns' in sys.argv else 100
preview = '--tool-preview' in sys.argv
OUT = ROOT / "artifacts/terminal-acceptance" / (f"tools-{columns}" if preview else mode)
OUT.mkdir(parents=True, exist_ok=True)
if "--child" in sys.argv:
    import termios
    from onecli import terminal

    before = termios.tcgetattr(0)
    interrupted = False
    draws = [0]
    try:
        with terminal.session("graphical acceptance") as ui:
            assert ui is not None
            draw = ui.draw

            def counted_draw():
                draws[0] += 1
                draw()

            ui.draw = counted_draw
            if "--capture-mouse" in sys.argv:
                import curses

                curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            # Match an active, interruptible agent turn, not the protected
            # gaps before/after it where SIGINT must not escape.
            ui.busy = ui.allow_steer = "--interrupt" in sys.argv
            ui.write("SELECTION_BEACON graphical terminal acceptance\n")
            if preview:
                ui.tool_start('bash', {'command': 'python3 -m render list --format table ' + '--label example ' * 12})
                ui.tool_result('exit 0\n' + '\n'.join(f'recipe_{i}: image generation' for i in range(30)))
                ui.tool_start('grep', {'path': 'onecli/tools', 'pattern': 'search'})
                ui.tool_result('[rg]\nonecli/tools/search.py:31: scoped search')
                ui.tool_start('bash', {'command': 'python3 missing.py'})
                ui.tool_result('exit 2\nerror: missing.py: no such file')
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and not (OUT / "stop").exists():
                p = OUT / "state.tmp"
                p.write_text(
                    json.dumps(
                        {
                            "draft": ui.draft,
                            "draws": draws[0],
                            "raw": not bool(
                                termios.tcgetattr(0)[3]
                                & (termios.ECHO | termios.ICANON | termios.ISIG)
                            ),
                        }
                    )
                )
                p.replace(OUT / "state.json")
                time.sleep(0.02)
    except KeyboardInterrupt:
        interrupted = True
    (OUT / "exit.json").write_text(
        json.dumps(
            {"restored": termios.tcgetattr(0) == before, "interrupted": interrupted}
        )
    )
    raise SystemExit
import gi  # noqa: E402 - only the graphical half below needs gi

gi.require_version("Gtk", "3.0")
gi.require_version("Vte", "2.91")
from gi.repository import Gtk, Vte, GLib, Gdk, Pango  # noqa: E402 - after require_version

win = Gtk.Window(title="one-terminal-acceptance")
term = Vte.Terminal()
child_status = []
term.connect("child-exited", lambda terminal, status: child_status.append(status))
term.set_font(Pango.FontDescription("Monospace 12"))
term.set_size(columns, 36)
win.add(term)
win.show_all()
results = {
    "mode": mode,
    "commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "vte": [Vte.get_major_version(), Vte.get_minor_version()],
}
fail = []


def call(fn):
    event = threading.Event()
    box = []

    def wrapped():
        try:
            box.append(fn())
        except BaseException as e:
            box.append(e)
        event.set()
        return False

    GLib.idle_add(wrapped)
    assert event.wait(10)
    if isinstance(box[0], BaseException):
        raise box[0]
    return box[0]


def wait(fn):
    end = time.monotonic() + 12
    while time.monotonic() < end:
        try:
            v = fn()
            if v:
                return v
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    raise AssertionError("condition timed out")


def xd(*args):
    subprocess.run(["xdotool", *map(str, args)], check=True)


def state():
    return json.loads((OUT / "state.json").read_text())


def screenshot():
    window = win.get_window()
    w, h = win.get_size()
    Gdk.pixbuf_get_from_window(window, 0, 0, w, h).savev(
        str(OUT / "selection.png"), "png", [], []
    )


def run():
    try:
        wait(lambda: state()["raw"])
        time.sleep(1)
        text = call(lambda: term.get_text_format(Vte.Format.TEXT))
        (OUT / "screen.txt").write_text(text)
        if preview:
            assert 'Ctrl-O:' not in text and 'exit 0' not in text
            assert 'missing.py' in text
            results['columns'] = call(term.get_column_count)
            assert results['columns'] == columns, results['columns']
            call(screenshot)
            (OUT / 'selection.png').rename(OUT / 'compact.png')
            call(lambda: term.feed_child(b'\x0f'))
            wait(lambda: 'recipe_29' in call(lambda: term.get_text_format(Vte.Format.TEXT)))
            call(screenshot)
            (OUT / 'selection.png').rename(OUT / 'expanded.png')
            call(lambda: term.feed_child(b'\x0f'))
            wait(lambda: 'recipe_29' not in call(lambda: term.get_text_format(Vte.Format.TEXT)))
        rows = text.splitlines()
        row = next(i for i, line in enumerate(rows) if "SELECTION_BEACON" in line)
        col = rows[row].index("SELECTION_BEACON")
        cw, ch = call(lambda: (term.get_char_width(), term.get_char_height()))
        wid = subprocess.check_output(
            ["xdotool", "search", "--name", "^one-terminal-acceptance$"], text=True
        ).splitlines()[0]
        xd("windowfocus", wid)
        xd("mousemove", "--window", wid, int((col + 0.2) * cw), int((row + 0.5) * ch))
        xd("mousedown", 1)
        xd(
            "mousemove",
            "--window",
            wid,
            int((col + len("SELECTION_BEACON")) * cw),
            int((row + 0.5) * ch),
        )
        xd("mouseup", 1)
        time.sleep(0.3)
        selected = call(
            lambda: Gtk.Clipboard.get(Gdk.SELECTION_PRIMARY).wait_for_text()
        )
        results["selection"] = selected
        assert selected == "SELECTION_BEACON", repr(selected)
        call(screenshot)
        call(lambda: term.copy_clipboard_format(Vte.Format.TEXT))
        copied = call(
            lambda: Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).wait_for_text()
        )
        results["clipboard"] = copied
        assert copied == selected
        paste = "paste first\npaste second"
        call(lambda: Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text(paste, -1))
        call(term.paste_clipboard)
        wait(lambda: state()["draft"] == paste)
        results["multiline_paste"] = state()["draft"]
        suffix = "abcdefghijklmnopqrstuvwxyz" * 20
        before_draws = state()["draws"]
        xd("type", "--clearmodifiers", "--delay", 1, suffix)
        wait(lambda: state()["draft"] == paste + suffix)
        results["typed_characters"] = len(suffix)
        results["draws_during_typing"] = state()["draws"] - before_draws
        assert results["draws_during_typing"] > 1
        if "--interrupt" in sys.argv:
            xd("key", "--clearmodifiers", "ctrl+c")
        else:
            (OUT / "stop").touch()
        wait(lambda: (OUT / "exit.json").exists())
        results["exit"] = json.loads((OUT / "exit.json").read_text())
        assert results["exit"]["restored"]
        assert results["exit"]["interrupted"] == ("--interrupt" in sys.argv)
        wait(lambda: len(child_status))
        results["child_status"] = child_status[0]
        assert child_status[0] == 0
    except BaseException:
        fail.append(traceback.format_exc())
    finally:
        (OUT / "stop").touch()
        results["errors"] = fail
        (OUT / "result.json").write_text(json.dumps(results, indent=2))
        GLib.idle_add(Gtk.main_quit)


for f in ("stop", "state.json", "exit.json"):
    (OUT / f).unlink(missing_ok=True)
env = dict(
    os.environ,
    TERM="xterm-256color",
    ONE_PLAIN="0",
    ONE_REDUCED_MOTION="1",
    PYTHONPATH=str(ROOT),
    HOME=str(OUT / "home"),
)
(OUT / "home").mkdir(exist_ok=True)
term.spawn_sync(
    Vte.PtyFlags.DEFAULT,
    str(ROOT),
    [sys.executable, str(Path(__file__).resolve()), "--child", *sys.argv[1:]],
    [k + "=" + v for k, v in env.items()],
    GLib.SpawnFlags.DEFAULT,
    None,
    None,
    None,
)
threading.Thread(target=run, daemon=True).start()
Gtk.main()
print(json.dumps(results, indent=2))
sys.exit(bool(fail))

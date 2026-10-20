# Development

Licensed under the one license 0.6.3, see [LICENSE.md](LICENSE.md). These are
development instructions, not a promise of issue-response times. Read
[security and reporting](SECURITY.md) before submitting sensitive findings.

Use Linux with Python 3.10 or newer. From an authorized checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests -q
python scripts/update-readme.py --check
python -m build
```

The sandbox tests additionally require bubblewrap and util-linux (`prlimit`) and
a kernel permitting unprivileged namespaces. The glyph renderer tests use Node.js 22.
A skipped prerequisite-dependent test is not proof that its boundary works. Do not weaken host security settings on a shared machine
just to run a test; use a disposable VM or container instead.

Before a packaging change, test the actual wheel, not an editable source install:

```sh
# From the checkout, after python -m build:
python scripts/smoke-wheel.py --help  # no package installation required for help
checkout_dir="$PWD"
smoke_dir="$(mktemp -d)"
python3 -m venv "$smoke_dir/venv"
"$smoke_dir/venv/bin/python" -m pip install --no-deps "$checkout_dir"/dist/*.whl
cd "$smoke_dir"
"$smoke_dir/venv/bin/python" -I "$checkout_dir/scripts/smoke-wheel.py"
```

The `-I` flag ignores ambient PYTHONPATH and user site packages. Running the check
with an interpreter that has no wheel installed is an error with setup guidance;
the script must not add the checkout to its import path to make that error go
away. It also rejects editable/source imports and execution inside the checkout.
It uses a loopback fixture, not paid models. Its bench/arena/stress checks exercise
the installed child launchers, not every end-to-end command scenario.
Tests and examples must not depend on personal credentials or operator data.

For each fix, add a regression test, show it failing before the fix and passing
afterwards, and describe any limitations. Keep generated README sections current
with `python scripts/update-readme.py`. Keep version metadata and
`onecli.__version__` aligned. Report vulnerabilities privately through GitHub
private vulnerability reporting on this repository (the **Security** tab,
**Report a vulnerability**), as [SECURITY.md](SECURITY.md) describes; do not put
secrets, live exploits against shared systems, or private transcripts in public
issues.

## Graphical terminal acceptance (Linux)

With system Python's GTK3/VTE bindings, Xvfb and xdotool already installed:

```sh
/usr/bin/python3 scripts/check-terminal-graphical.py
/usr/bin/python3 scripts/check-terminal-graphical.py --interrupt
# Negative control: MUST exit nonzero because mouse capture blocks selection.
/usr/bin/python3 scripts/check-terminal-graphical.py --capture-mouse
```

The harness creates its own virtual display and isolated child HOME, runs the
real terminal surface without a provider, and saves JSON receipts plus a selection
screenshot under `artifacts/terminal-acceptance/`. It checks mouse drag selection,
PRIMARY and CLIPBOARD readback, VTE clipboard paste with embedded newlines, 520
real key events with redraw progress, process exit and exact termios restoration.
The interrupt case sends Ctrl-C while the surface is busy. Copy/paste uses VTE's
clipboard API; desktop shortcut bindings and macOS are outside this coverage.
Use the OS Python that can import `gi`, not a virtualenv without those bindings.

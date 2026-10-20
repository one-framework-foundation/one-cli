"""`one models`, `one tune`, `one serve`, `one setup`, `one doctor`.

These sit in front of the flat `one <prompt>` CLI. A first word that matches
one of them is a command; everything else is still a prompt, and `one -- ...`
escapes a prompt that happens to start with one of these words.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from .. import global_context, render
from . import catalog, hostfacts, menu, pull, recipes, serve, tune

VERBS = ("models", "tune", "serve", "setup", "doctor", "workgate")


def _err(msg: str) -> int:
    print(render.red(msg), file=sys.stderr)
    return 2


def _log(msg: str) -> None:
    print(render.dim(msg), file=sys.stderr)


def cmd_workgate(argv: list[str]) -> int:
    from ..workgate import main
    return main(argv)


def _gb(n: int | float) -> str:
    return f"{n / 1e9:.1f} GB"


# ---------------------------------------------------------------- models

def cmd_models(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="one models")
    sub = p.add_subparsers(dest="action")
    sub.add_parser("list", help="Curated models and what is on this host.")
    s = sub.add_parser("search", help="Find GGUF repos on HuggingFace.")
    s.add_argument("query", nargs="+")
    s.add_argument("--limit", type=int, default=15)
    g = sub.add_parser("pull", help="Fetch a model, drafter included.")
    g.add_argument("name", help="Catalog name, or a HuggingFace repo id.")
    g.add_argument("--file", default="", help="Which GGUF, for a live repo.")
    g.add_argument("--as", dest="alias", default="", help="Name to file it under.")
    g.add_argument("--no-draft", action="store_true",
                   help="Skip the speculative drafter. It is a tenth of the "
                        "size and it is what the fast recipe needs.")
    v = sub.add_parser("verify", help="Re-check size and sha256 on disk.")
    v.add_argument("name", nargs="?", default="")
    r = sub.add_parser("remove", help="Delete a model's files.")
    r.add_argument("name")
    args = p.parse_args(argv)
    action = args.action or "list"

    if action == "list":
        return _models_list()
    if action == "search":
        return _models_search(" ".join(args.query), args.limit)
    if action == "pull":
        return _models_pull(args)
    if action == "verify":
        return _models_verify(args.name)
    if action == "remove":
        return _models_remove(args.name)
    return _err(f"unknown action {action}")


def _models_list() -> int:
    root = hostfacts.models_dir()
    models = catalog.load()
    print(render.kicker(f"catalog  ({root})"), file=sys.stderr)
    for m in models:
        state = "on disk" if m.present(root) else "not pulled"
        if m.present(root) and m.draft and not m.draft_present(root):
            state = "on disk, no drafter"
        mark = render.green if m.present(root) else render.faint
        print(render.ink(f"{m.name:30}") + render.dim(f"{m.quant:8}")
              + render.dim(f"{_gb(m.total_bytes()):>9}  ") + mark(state))
    known = {m.file for m in models}
    if m2 := [(p, s) for p, s in catalog.local(root) if p.name not in known]:
        print(render.kicker("also on this host (not in the catalog)"),
              file=sys.stderr)
        for p, s in m2:
            print(render.ink(f"{p.name:48}") + render.dim(f"{_gb(s):>9}"))
    print(render.footer([f"{len(models)} catalogued"]), file=sys.stderr)
    return 0


def _models_search(query: str, limit: int) -> int:
    hits = catalog.search(query, limit)
    if not hits:
        return _err("nothing found (or HuggingFace unreachable)")
    print(render.kicker(f"huggingface: {query}"), file=sys.stderr)
    for h in hits:
        print(render.ink(f"{h['repo']:60}")
              + render.dim(f"{h['downloads']:>9} downloads")
              + render.faint(f"  {h['likes']} likes"))
    print(render.footer([f"{len(hits)} repos",
                         "one models pull <repo> to fetch"]), file=sys.stderr)
    return 0


def _resolve(name: str, file: str = "", alias: str = "") -> catalog.Model | None:
    m = catalog.find(name)
    if m:
        return m
    if "/" in name:
        return catalog.from_repo(name, file, alias)
    return None


def _models_pull(args) -> int:
    m = _resolve(args.name, args.file, args.alias)
    if not m:
        return _err(f"no model called {args.name} - `one models list` or "
                    f"`one models search {args.name}`")
    root = hostfacts.models_dir()
    try:
        todo = [(a, d) for a, d, ok in pull.plan(m, root, not args.no_draft) if not ok]
    except pull.PullError as exc:
        return _err(str(exc))
    if not todo:
        print(render.green(f"{m.name} is already on disk and verified"),
              file=sys.stderr)
        return 0
    need = sum(a.size for a, _ in todo)
    print(render.kicker(f"pull {m.name}"), file=sys.stderr)
    for a, _d in todo:
        print(render.ink(f"  {a.file:52}") + render.dim(f"{_gb(a.size):>9}"),
              file=sys.stderr)
    print(render.dim(f"  {_gb(need)} into {root} "
                     f"({pull.free_bytes(root) / 1e9:.0f} GB free)"),
          file=sys.stderr)
    if not menu.confirm("fetch these?", default=True):
        return 1
    line = menu.Line()
    def progress(p: pull.Progress) -> None:
        line.update(f"  {menu.bar(p.share)} {p.share:5.1%}  {p.file[:40]}  "
                    f"{p.rate_mb_s} MB/s")
    try:
        got = pull.pull(m, root, not args.no_draft, on_progress=progress,
                        log=lambda s: line.done(render.dim("  " + s)))
    except pull.PullError as exc:
        line.done()
        return _err(str(exc))
    line.done(render.green(f"  {len(got)} file(s) verified into {root}"))
    if not m.curated:
        path = catalog.add_user_model(m)
        print(render.dim(f"  added to {path}"), file=sys.stderr)
    print(render.footer([f"one tune {m.name}",
                         "measures the fastest way to serve it here"]),
          file=sys.stderr)
    return 0


def _models_verify(name: str) -> int:
    root = hostfacts.models_dir()
    found = catalog.find(name) if name else None
    if name and not found:
        return _err(f"no model called {name}")
    models = [found] if found else catalog.load()
    bad = 0
    for m in models:
        for a in m.artifacts():
            dest = root / a.file
            if not dest.is_file():
                continue
            ok, why = pull.verify(dest, a.size, a.sha256)
            paint = render.green if ok else render.red
            bad += 0 if ok else 1
            print(render.ink(f"{a.file:52}") + paint(why))
    return 1 if bad else 0


def _models_remove(name: str) -> int:
    m = catalog.find(name)
    if not m:
        return _err(f"no model called {name}")
    root = hostfacts.models_dir()
    try:
        files = [p for a in m.artifacts() if (p := pull.safe_join(root, a.file)).is_file()]
    except (pull.PullError, ValueError) as exc:
        return _err(str(exc))
    if not files:
        print(render.dim(f"{m.name} is not on disk"), file=sys.stderr)
        return 0
    for f in files:
        print(render.ink(f"  {f}") + render.dim(f"  {_gb(f.stat().st_size)}"),
              file=sys.stderr)
    if not menu.confirm("delete these?", default=False):
        return 1
    try:
        gone = pull.remove(m, root)
    except (pull.PullError, OSError) as exc:
        return _err(str(exc))
    print(render.green(f"removed {len(gone)} file(s)"), file=sys.stderr)
    return 0


# ---------------------------------------------------------------- tune

def cmd_tune(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="one tune")
    p.add_argument("name", nargs="?", default="",
                   help="Model to sweep. Omit with --show to read the ledger.")
    p.add_argument("--show", action="store_true", help="Print the ledger.")
    p.add_argument("--all-ngram", action="store_true",
                   help="Sweep every ngram variant, not just the default one.")
    p.add_argument("--port", type=int, default=tune.TUNE_PORT)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--measure", metavar="URL",
                   help="Measure a lane that is already running, and print the "
                        "numbers without touching the ledger.")
    p.add_argument("--model-id", default="", help="Model id for --measure.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if args.measure:
        res = tune.measure(args.measure, args.model_id)
        if args.json:
            print(json.dumps({"tok_s": res.tok_s, "acceptance": res.acceptance,
                              "error": res.error,
                              "samples": [s.__dict__ for s in res.samples]},
                             indent=2))
            return 0 if res.ok() else 1
        if res.error and not res.tok_s:
            return _err(res.error)
        print(render.kicker(args.measure), file=sys.stderr)
        for s in res.samples:
            acc = f"  {s.acceptance:.0%} accepted" if s.draft_n else ""
            print(render.ink(f"  {s.task:10}") + render.dim(f"{s.tok_s:7.2f} tok/s")
                  + render.faint(acc))
        print(render.footer([f"{res.tok_s} tok/s overall"]), file=sys.stderr)
        return 0

    if args.show or not args.name:
        return _tune_show(args.json)

    m = catalog.find(args.name)
    if not m:
        return _err(f"no model called {args.name}")
    facts = hostfacts.collect()
    spec = recipes.supported_spec_types(facts)
    cands = recipes.candidates(m, facts, spec_types=spec,
                               all_ngram=args.all_ngram)
    skips = recipes.skipped(m, facts, spec_types=spec)
    print(render.kicker(f"tune {m.name} on {facts.hostname}"), file=sys.stderr)
    for c in cands:
        print(render.ink(f"  {c.name:14}") + render.faint(c.why), file=sys.stderr)
    for n, why in skips:
        print(render.dim(f"  {n:14}skipped: {why}"), file=sys.stderr)
    print(render.dim(f"  ~{len(cands) * 3} minutes, one lane at a time on "
                     f":{args.port}"), file=sys.stderr)
    if not menu.confirm("run the sweep?", default=True):
        return 1
    t0 = time.time()
    try:
        entry = tune.sweep(m, facts, all_ngram=args.all_ngram, port=args.port,
                           context=args.context, log=_log)
    except RuntimeError as exc:
        return _err(str(exc))
    _print_entry(entry)
    print(render.footer([f"{time.time() - t0:.0f}s",
                         f"written to {tune.LEDGER}"]), file=sys.stderr)
    return 0


def _print_entry(entry: dict) -> None:
    print(render.kicker(f"{entry['model']} on {entry['host']}"), file=sys.stderr)
    for c in entry.get("candidates", []):
        if c.get("error"):
            print(render.ink(f"  {c['recipe']:14}") + render.red(c["error"][:60]))
            continue
        acc = f"  {c['acceptance']:.0%} accepted" if c.get("acceptance") else ""
        mark = " <-- fastest" if c["recipe"] == entry["winner"] else ""
        print(render.ink(f"  {c['recipe']:14}")
              + render.dim(f"{c['tok_s']:7.2f} tok/s")
              + render.faint(acc) + render.green(mark))
    if entry.get("speedup"):
        print(render.green(f"  {entry['speedup']}x over baseline"),
              file=sys.stderr)


def _tune_show(as_json: bool) -> int:
    data = tune.load_ledger()
    entries = data.get("entries", {})
    if as_json:
        print(json.dumps(data, indent=2))
        return 0
    if not entries:
        print(render.dim("nothing measured yet - `one tune <model>`"),
              file=sys.stderr)
        return 0
    facts = hostfacts.collect()
    print(render.kicker("verified fastest recipes"), file=sys.stderr)
    for key, e in sorted(entries.items()):
        stale = e.get("runner") and e["runner"] != facts.runner() \
            and e.get("host") == facts.hostname
        note = render.yellow("  stale runner") if stale else ""
        print(render.ink(f"{key:44}") + render.dim(f"{e['winner']:12}")
              + render.dim(f"{e['tok_s']:7.2f} tok/s")
              + render.faint(f"  {e.get('speedup', 0)}x  {e['measured_at'][:10]}")
              + note)
    return 0


# ---------------------------------------------------------------- serve

def cmd_serve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="one serve")
    p.add_argument("name", nargs="?", default="")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--recipe", default="",
                   help="Override the measured winner. Says so in the output.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command instead of running it.")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    facts = hostfacts.collect()
    if args.list or not args.name:
        return _serve_list(facts)
    m = catalog.find(args.name)
    if not m:
        return _err(f"no model called {args.name}")
    if args.stop:
        ok = serve.stop(m, facts)
        print((render.green if ok else render.yellow)(
            f"{'stopped' if ok else 'nothing running for'} {m.name}"),
            file=sys.stderr)
        return 0
    port = args.port or _free_port(facts)
    if args.dry_run:
        cmd, recipe, entry = serve.plan(m, port, facts, args.context,
                                        args.parallel,
                                        force_recipe=args.recipe)
        print(render.dim(f"# recipe {recipe.name}: {recipe.why}"),
              file=sys.stderr)
        print(" ".join(cmd))
        return 0
    try:
        lane = serve.start(m, port, facts, args.context, args.parallel,
                           force_recipe=args.recipe, log=_log)
    except (RuntimeError, subprocess.SubprocessError) as exc:
        return _err(str(exc))
    if lane.verified:
        print(render.green(f"{m.name} serving on :{port} with the measured "
                           f"fastest recipe ({lane.recipe})"), file=sys.stderr)
    else:
        print(render.yellow(
            f"{m.name} serving on :{port} with {lane.recipe} - UNVERIFIED on "
            f"this host. `one tune {m.name}` measures what is actually "
            f"fastest here."), file=sys.stderr)
    print(render.dim(f"registered as a backend: one -m {m.alias or m.name}"),
          file=sys.stderr)
    return 0


def _free_port(facts: hostfacts.HostFacts, start: int = 8911) -> int:
    taken = {lane.port for lane in serve.running(facts)}
    port = start
    while port in taken:
        port += 1
    return port


def _serve_list(facts: hostfacts.HostFacts) -> int:
    lanes = serve.running(facts)
    if not lanes:
        print(render.dim("no lanes running on this host"), file=sys.stderr)
        return 0
    ledger = tune.load_ledger().get("entries", {})
    print(render.kicker(f"lanes on {facts.hostname}"), file=sys.stderr)
    for lane in lanes:
        e = ledger.get(f"{lane.model}@{facts.hostname}")
        if e and e.get("winner") == lane.recipe:
            verdict = render.green(f"  measured fastest ({e['speedup']}x)")
        elif e:
            verdict = render.yellow(f"  ledger says {e['winner']} is faster")
        else:
            verdict = render.faint("  unmeasured")
        print(render.ink(f"{lane.model:34}") + render.dim(f":{lane.port:<6}")
              + render.dim(f"{lane.recipe:14}") + verdict)
    return 0


# ---------------------------------------------------------------- doctor

def cmd_doctor(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="one doctor")
    p.add_argument("--json", action="store_true",
                   help="host facts only, as JSON")
    p.add_argument("--spec-types", action="store_true",
                   help="also probe the llama.cpp runner's speculative "
                        "decoding types (may start a container; slow)")
    args = p.parse_args(argv)
    facts = hostfacts.collect()
    if args.json:
        print(json.dumps(facts.to_json(), indent=2))
        return 0
    from onecli import onboard
    print(render.kicker("backends"), file=sys.stderr)
    lines, usable = onboard.doctor_report()
    for line in lines:
        print(render.ink("  " + line))
    print(render.kicker("instructions"), file=sys.stderr)
    try:
        for line in global_context.doctor_lines():
            print(render.ink("  " + line))
    except (OSError, ValueError) as exc:
        print(render.yellow(f"  global context: {exc}"))
    print(render.kicker("this host"), file=sys.stderr)
    for line in hostfacts.describe(facts):
        print(render.ink("  " + line))
    spec = (recipes.supported_spec_types(facts)
            if facts.can_serve() and args.spec_types else [])
    print(render.kicker("serving"), file=sys.stderr)
    if not facts.can_serve():
        print(render.yellow("  no llama-server and no usable docker - this "
                            "host cannot serve local weights"))
    elif args.spec_types:
        print(render.ink(f"  spec-types   {', '.join(spec) or 'none reported'}"))
    lanes = serve.running(facts)
    print(render.ink(f"  lanes        {len(lanes)} running"))
    root = hostfacts.models_dir()
    on_disk = [m for m in catalog.load() if m.present(root)]
    print(render.ink(f"  models       {len(on_disk)} of "
                     f"{len(catalog.load())} catalogued on disk"))
    measured = [k for k, e in tune.load_ledger().get("entries", {}).items()
                if e.get("host") == facts.hostname]
    print(render.ink(f"  tuned        {len(measured)} model(s) measured here"))
    missing = [m.name for m in on_disk
               if f"{m.name}@{facts.hostname}" not in
               tune.load_ledger().get("entries", {})]
    if missing:
        print(render.yellow(f"  untuned: {', '.join(missing)}"), file=sys.stderr)
        print(render.dim(f"  `one tune {missing[0]}` measures it"),
              file=sys.stderr)
    return 0 if usable else 1


# ---------------------------------------------------------------- setup

def cmd_setup(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="one setup",
        description="Download, measure and serve local model weights. To "
                    "connect a hosted provider or a server you already run, "
                    "run `one` with nothing configured; `one doctor` shows "
                    "what is configured.")
    p.add_argument("--yes", action="store_true",
                   help="Pull the default recommended model once without "
                        "prompting; may download large files. Does not tune or serve.")
    args = p.parse_args(argv)
    if not args.yes and not menu._tty():
        print("one setup requires a terminal; use --yes for one default model "
              "pull, or one models pull NAME for an explicit choice.", file=sys.stderr)
        return 2
    facts = hostfacts.collect()

    print(render.banner("setup", ["one-cli local serving"]), file=sys.stderr)
    print(render.kicker("what this machine is"), file=sys.stderr)
    for line in hostfacts.describe(facts):
        print(render.ink("  " + line))

    configured_context = False
    can_serve = facts.can_serve()
    if not args.yes:
        try:
            selected, explicit = global_context.source()
            if not explicit or not can_serve:
                configured_context = _setup_context(selected)
        except (OSError, ValueError) as exc:
            return _err(f"global context: {exc}")

    if not can_serve:
        print(render.yellow(
            "\n  No llama-server binary and no usable docker daemon here.\n"
            "  This host can still DRIVE remote models (`one --list`), it just\n"
            "  cannot serve local weights. Install docker or llama.cpp and\n"
            "  run `one setup` again."), file=sys.stderr)
        return 0 if configured_context else 1

    if args.yes:
        return _setup_pull(facts, automatic=True)

    while True:
        root = hostfacts.models_dir()
        on_disk = [m for m in catalog.load() if m.present(root)]
        ledger = tune.load_ledger().get("entries", {})
        untuned = [m for m in on_disk
                   if f"{m.name}@{facts.hostname}" not in ledger]
        lanes = serve.running(facts)

        options = [
            ("choose global instructions", "Claude's file or one-cli's own file"),
            ("pull a model", f"{len(on_disk)} on disk in {root}"),
            ("measure the fastest way to serve one",
             f"{len(untuned)} on disk and unmeasured"
             if untuned else "everything on disk is measured"),
            ("start a lane", f"{len(lanes)} running"),
            ("show what is running and how fast", ""),
            ("done", ""),
        ]
        pick = menu.choose("setup", options, default=1)
        if pick in (-1, 5):
            break
        if pick == 0:
            _setup_context(global_context.source()[0])
        elif pick == 1:
            _setup_pull(facts)
        elif pick == 2:
            _setup_tune(untuned or on_disk)
        elif pick == 3:
            _setup_serve(facts, on_disk)
        elif pick == 4:
            _serve_list(facts)
            _tune_show(False)
    print(render.footer(["one doctor", "shows this again at any time"]),
          file=sys.stderr)
    return 0


def _setup_context(current: str) -> bool:
    options = [
        ("Claude global instructions", "~/.claude/CLAUDE.md"),
        ("one-cli global instructions", "~/.one-cli/AGENTS.md"),
    ]
    pick = menu.choose("global instructions", options,
                       default=global_context.SOURCES.index(current))
    if pick < 0:
        return False
    selected = global_context.SOURCES[pick]
    path = global_context.set_source(selected)
    print(render.dim(f"  saved {selected} in {path}"), file=sys.stderr)
    return True


def _recommended(facts: hostfacts.HostFacts) -> list[catalog.Model]:
    """Catalogued models that fit this host's memory, biggest first.

    Fit is checked against the measured pool, not a rule of thumb: a 27B at Q4
    plus its drafter is comfortable on unified 120 GB and impossible on a
    16 GB card, and offering it there wastes an hour of someone's download.
    """
    usable_mb = hostfacts.usable_memory_mb(facts)
    out = []
    for m in catalog.load():
        need_mb = (m.total_bytes() / 1e6) * 1.15
        if not usable_mb or need_mb <= usable_mb:
            out.append(m)
    return sorted(out, key=lambda m: -m.total_bytes())


def _setup_pull(facts: hostfacts.HostFacts, automatic: bool = False) -> int:
    root = hostfacts.models_dir()
    fits = _recommended(facts)
    if not fits:
        print(render.yellow("  nothing in the catalog fits this host's memory"),
              file=sys.stderr)
        return 1
    opts = []
    for m in fits:
        state = "on disk" if m.present(root) else _gb(m.total_bytes())
        opts.append((m.name, f"{m.quant} {m.role}  {state}"))
    opts.append(("search HuggingFace instead", ""))
    pick = 0 if automatic else menu.choose("which model", opts, default=0)
    if pick == -1:
        return 0
    if pick == len(opts) - 1:
        q = menu.ask("search for:")
        if q:
            _models_search(q, 15)
            repo = menu.ask("repo id to pull (blank to skip):")
            if repo:
                return _models_pull(argparse.Namespace(name=repo, file="", alias="",
                                                        no_draft=False))
        return 0
    return _models_pull(argparse.Namespace(name=fits[pick].name, file="", alias="",
                                          no_draft=False))


def _setup_tune(models: list[catalog.Model]) -> None:
    if not models:
        print(render.dim("  no models on disk to measure - pull one first"),
              file=sys.stderr)
        return
    opts = [(m.name, f"{m.quant}  drafter "
             f"{'on disk' if m.draft_present() else 'absent'}") for m in models]
    pick = menu.choose("measure which model", opts, default=0)
    if pick == -1:
        return
    cmd_tune([models[pick].name])


def _setup_serve(facts: hostfacts.HostFacts,
                 models: list[catalog.Model]) -> None:
    if not models:
        print(render.dim("  no models on disk to serve - pull one first"),
              file=sys.stderr)
        return
    ledger = tune.load_ledger().get("entries", {})
    opts = []
    for m in models:
        e = ledger.get(f"{m.name}@{facts.hostname}")
        opts.append((m.name, f"{e['winner']} {e['speedup']}x measured"
                     if e else "unmeasured - will serve baseline"))
    pick = menu.choose("serve which model", opts, default=0)
    if pick == -1:
        return
    port = menu.ask("port", str(_free_port(facts)))
    cmd_serve([models[pick].name, "--port", port])


# ---------------------------------------------------------------- dispatch

def dispatch(argv: list[str]) -> int | None:
    """Run a serving verb, or return None so the caller treats it as a prompt."""
    if not argv or argv[0] not in VERBS:
        return None
    verb, rest = argv[0], argv[1:]
    if verb == "models":
        return cmd_models(rest)
    if verb == "tune":
        return cmd_tune(rest)
    if verb == "serve":
        return cmd_serve(rest)
    if verb == "setup":
        return cmd_setup(rest)
    if verb == "doctor":
        return cmd_doctor(rest)
    if verb == "workgate":
        return cmd_workgate(rest)
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    code = dispatch(argv)
    if code is None:
        print(f"usage: one {{{'|'.join(VERBS)}}} ...", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":  # `python3 -m onecli.serving.cli tune ...`
    raise SystemExit(main())

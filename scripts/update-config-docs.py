#!/usr/bin/env python3
"""Regenerate docs/configuration.md: every `one` flag and every environment
variable one-cli reads.

Flags come from the live parser (onecli.cli.build_parser), so their help text
is the one `one --help` prints. Environment variables are found by scanning
onecli/ for os.environ reads; each one needs a line in ENV below, and --check
fails when the code reads a variable this table does not describe.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "configuration.md"
sys.path.insert(0, str(ROOT))

ENV_READ = re.compile(
    r"""(?:environ\.get|getenv|environ\.setdefault|environ\.pop)\(\s*["']([A-Z][A-Z0-9_]+)["']"""
    r"""|environ\[\s*["']([A-Z][A-Z0-9_]+)["']\s*\]"""
    # A name held in a module constant (`LEDGER_ENV = "ONE_..."`) and read
    # through it.
    r"""|^[A-Z][A-Z0-9_]*_ENV\s*=\s*["']([A-Z][A-Z0-9_]+)["']""", re.M)

# name -> what it does. Grouped by area in the rendered page.
ENV: dict[str, tuple[str, str]] = {
    # Backends and models
    "ONE_CLI_API_KEY": ("Backends", "API key for a backend given with `--base-url` on the command line."),
    "ONE_DEFAULT_BACKEND": ("Backends", "Backend name to use when none is chosen with `-m`."),
    "ONE_DEFAULT_MODEL": ("Backends", "Fallback for `ONE_DEFAULT_BACKEND` when that is unset."),
    "ONE_CLI_LANE_HEALTH": ("Backends", "Lane health state file shared across processes (default `~/.config/one-cli/lane_health.json`)."),
    "ONE_MODELS_DIR": ("Backends", "Directory for GGUF model weights used by local serving; created if missing, and used instead of the default search locations."),
    "HF_TOKEN": ("Backends", "Hugging Face token sent when the serving catalog reads model metadata."),
    # Terminal and editor
    "EDITOR": ("Terminal", "Editor for `/edit` in the chat; `VISUAL`, then `vi`, when unset."),
    "VISUAL": ("Terminal", "Editor for `/edit` when `EDITOR` is unset."),
    "TMUX": ("Terminal", "Set by tmux; read to route clipboard copies and mouse handling through it."),
    "ONE_NO_MOUSE": ("Terminal", "`1` turns off mouse support inside tmux."),
    "ONE_PLAIN": ("Terminal", "`1` turns off the full-screen terminal UI and prints plain lines."),
    "ONE_CLI_CALLER": ("Terminal", "Names the program that launched `one`, for session records; otherwise read from the parent process on Linux."),
    # Media (senses)
    "ONE_PARAKEET_URL": ("Media", "Base URL of a speech-to-text server: any server that answers `GET /health` and takes a multipart `file` upload at `POST /transcribe`, returning JSON `{\"text\": ...}`. No default."),
    "ONE_PARAKEET_REMOTE_URL": ("Media", "Speech-to-text server with the same API, probed only when `ONE_PARAKEET_URL` is unset (a set `ONE_PARAKEET_URL` is used as given); no default."),
    "ONE_PARAKEET_TIMEOUT": ("Media", "Seconds to wait for a transcription (default 120)."),
    "DEEPGRAM_API_KEY": ("Media", "Enables Deepgram as a hosted transcription fallback."),
    "GROQ_API_KEY": ("Media", "Enables Groq as a hosted transcription fallback."),
    "GROQ_STT_MODEL": ("Media", "Groq transcription model (default `whisper-large-v3`)."),
    "ONE_TTS_URL": ("Media", "Pins one text-to-speech endpoint and disables the ladder; `ONE_TTS_SHAPE` names its API."),
    "ONE_TTS_REMOTE_URL": ("Media", "Text-to-speech endpoint used when no other is configured; no default."),
    "ONE_TTS_SHAPE": ("Media", "API shape of `ONE_TTS_URL` (default `elevenlabs`)."),
    "ONE_TTS_ENDPOINTS": ("Media", "Text-to-speech ladder as a comma-separated list of `name=url=shape`."),
    "ONE_TTS_VOICE": ("Media", "Default voice for `--speak` (default `alba`)."),
    "ONE_TTS_TIMEOUT": ("Media", "Seconds to wait for speech synthesis (default 30)."),
    "ONE_CLI_VISION_BACKENDS": ("Media", "Comma-separated backend names to use for describing images, in order."),
    "ONE_SENSES_OLLAMA_ENDPOINTS": ("Media", "Comma-separated Ollama URLs probed for vision models (default loopback only)."),
    "ONE_SENSES_DESCRIBE_TIMEOUT": ("Media", "Seconds per image description request (default 15)."),
    "ONE_SENSES_DESCRIBE_BUDGET": ("Media", "Total seconds for describing one attachment (default 30)."),
    "ONE_SENSES_DESCRIBE_MAX_TOKENS": ("Media", "Token cap for an image description (default 1200)."),
    "ONE_SENSES_MAX_VIDEO_SECONDS": ("Media", "Longest video accepted as an attachment (default 600)."),
    "ONE_SENSES_MAX_VIDEO_BYTES": ("Media", "Largest video file accepted (default 256 MiB)."),
    "ONE_SENSES_FRAME_BUDGET": ("Media", "Most frames described per video (default 6)."),
    "ONE_SENSES_LOG": ("Media", "Media routing log file (default `~/.one-cli/senses_log.jsonl`)."),
    "ONE_SENSES_SECONDS_PER_FRAME": ("Media", "Video seconds per described frame, below the frame budget (default 10)."),
    # Integrations
    "ONECLI_NO_PRESENCE": ("Integrations", "Set to 1 to disable endpoint presence probes and refusal checks; lanes remain unchecked."),
    "ONE_INFER_CONFIG": ("Integrations", "Scoped inference engine config (default `~/.config/one-cli/infer.json`); without it every inference call is refused."),
    "ONE_INFER_BUDGET_DIR": ("Integrations", "Where the scoped inference engine keeps its durable budget ledgers (default `~/.local/state/one-cli/infer-budget`)."),
    "ONE_INFER_LEDGER": ("Integrations", "Scoped inference call ledger, one row per attempt (default `~/.local/state/one-cli/infer_calls.jsonl`)."),
    "ONE_DETACHED_OWNER": ("Integrations", "Set to `caller` when a parent process owns background jobs and route selection: one-cli then builds no fallback chain and starts no local detached workers. See [progress policy](progress-policy.md#parent-processes)."),
    # Evaluation and testing
    "ONE_CLI_ANALYSIS_DIR": ("Evaluation", "Where run analysis records go (default `~/.one-cli/analysis`)."),
    "ONE_CLI_CORPUS": ("Evaluation", "Task corpus directory for the benchmark tools."),
    "KILO_BIN": ("Evaluation", "Path to the Kilo CLI, overriding the one found on PATH."),
    "MIMOCODE_BIN": ("Evaluation", "Path to the MiMo Code CLI, overriding the `mimo` found on PATH."),
    "ONE_STRESS_DIR": ("Evaluation", "Scratch directory for `one-stress` (default `.scratch/stress`)."),
    "ONE_STRESS_PORT": ("Evaluation", "Port of the `one-stress` stub provider (default 18642)."),
    "ONE_STRESS_LOG": ("Evaluation", "File the `one-stress` stub writes every request body to."),
    # Advanced
    "ONE_BINDING_FD": ("Advanced", "Set by one-cli for a code-mode program: the socket its tool binding talks over. Not for users."),
}

AREAS = ["Backends", "Terminal", "Media", "Integrations", "Evaluation", "Advanced"]


def env_reads() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "onecli").rglob("*.py"):
        for groups in ENV_READ.findall(path.read_text(errors="replace")):
            found.add(next(g for g in groups if g))
    return found


def documented_elsewhere() -> set[str]:
    """Variables the hand-written docs already explain."""
    text = "".join(p.read_text() for p in (ROOT / "docs").glob("*.md") if p != DOC)
    text += (ROOT / "README.md").read_text()
    return {name for name in env_reads() if re.search(rf"\b{name}\b", text)}


def flags_table() -> str:
    from onecli.cli import build_parser
    rows = []
    for action in build_parser()._actions:
        if not action.option_strings or action.help == argparse.SUPPRESS:
            continue
        names = ", ".join(f"`{o}`" for o in action.option_strings)
        help_text = " ".join((action.help or "").split()).replace("|", "\\|")
        if "%(default)" in help_text:
            help_text = help_text.replace("%(default)s", str(action.default))
        rows.append(f"| {names} | {help_text} |")
    return "| Flag | Meaning |\n|---|---|\n" + "\n".join(rows)


def env_table(names: set[str]) -> str:
    out = []
    for area in AREAS:
        rows = [f"| `{n}` | {ENV[n][1]} |" for n in sorted(names) if ENV[n][0] == area]
        if rows:
            out.append(f"### {area}\n\n| Variable | Meaning |\n|---|---|\n" + "\n".join(rows))
    return "\n\n".join(out)


def render() -> str:
    reads = env_reads()
    missing = sorted(reads - set(ENV) - documented_elsewhere())
    if missing:
        raise SystemExit("environment variables read by onecli/ with no description "
                         f"in scripts/update-config-docs.py ENV: {', '.join(missing)}")
    stale = sorted(set(ENV) - reads)
    if stale:
        raise SystemExit(f"ENV describes variables onecli/ no longer reads: {', '.join(stale)}")
    elsewhere = sorted(documented_elsewhere() - set(ENV))
    return (
        "# Configuration reference\n\n"
        "<!-- Generated by scripts/update-config-docs.py; do not edit by hand. -->\n\n"
        "Every flag `one` accepts and every environment variable one-cli reads. "
        "The [manual](manual.md) and [permissions](permissions.md) pages explain "
        "how they fit together.\n\n"
        "## Flags\n\n" + flags_table() + "\n\n"
        "## Environment variables\n\n" + env_table(set(ENV)) + "\n\n"
        "### Described in the other pages\n\n"
        + ", ".join(f"`{n}`" for n in elsewhere) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the page is stale")
    args = parser.parse_args()
    text = render()
    if DOC.exists() and DOC.read_text() == text:
        return 0
    if args.check:
        print("docs/configuration.md is stale; run scripts/update-config-docs.py", file=sys.stderr)
        return 1
    DOC.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

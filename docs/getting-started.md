# Getting started

[Back to README](../README.md) · [Permissions](permissions.md) · [Full manual](manual.md)

## Install

Python 3.10 or newer, Linux or macOS (Windows: use WSL). The base CLI has no
third-party runtime dependencies. The optional relay extra installs FastAPI, Uvicorn and
HTTPX; see the [support matrix](support-matrix.md). The package distribution is
`one-core-cli`; the command is `one`. Install it from PyPI:

```sh
python -m pip install one-core-cli
```

or from a checkout:

```sh
git clone https://github.com/one-framework-foundation/one-cli && cd one-cli
python3 -m venv .venv && . .venv/bin/activate
python -m pip install .
```

## First run

Run `one` in a trusted, version-controlled workspace. With nothing configured,
it walks you through setup in the terminal:

1. It looks for a local model server on the usual ports: llama-server `:8080`,
   ollama `:11434`, LM Studio `:1234`. Pick a model it found, or
2. choose a hosted provider: Anthropic, OpenAI, OpenRouter, or any
   OpenAI-compatible URL. Point it at the environment variable that holds your
   key (nothing is stored), or type the key with hidden input; it is saved in a
   `0600` file under `~/.one-providers/`.
3. It sends one short test prompt to the selected endpoint. Hosted providers
   may charge for this request; choosing a local endpoint avoids a hosted test.
   The test does not attach workspace files. If it fails, setup reports the
   error and removes the provider file it just wrote.

Then ask it something:

```sh
one "explain what this repo does"            # one task, in the current directory
one --read-only "where is the config parsed?"
one                                          # an interactive session
one -r                                       # continue the last session
one doctor                                   # what one-cli can see, and what works
```

`one setup` lets you choose the global instruction file: Claude's
`~/.claude/CLAUDE.md` or one-cli's `~/.one-cli/AGENTS.md`. Workspace
`CLAUDE.md`, `AGENTS.md` and `SOUL.md` remain separate. Existing installs keep
reading the Claude file until a choice is saved; `one doctor` shows the active
source and flags an implicit choice or a missing file. `one setup --yes` does
not change the instruction source.

Per workspace, `one` reads `CLAUDE.md`, then `AGENTS.md`, then an optional
`SOUL.md` from the workspace directory and adds them to the system prompt.
`SOUL.md` is a free-form persona or standing-instructions file for that
workspace: who the agent is and how it should work there. Unlike the other two,
it is read only from the workspace itself, never from parent directories up to
the git root. Symlinked context files are refused with a warning.

Not in a terminal (CI, scripts)? `one` prints the drop-in file to write and
exits non-zero. The full format is in [Backends](manual.md#backends).

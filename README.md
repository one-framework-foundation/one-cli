<div align="center">

<img src="https://raw.githubusercontent.com/one-framework-foundation/one-cli/main/assets/banner.svg" alt="one-cli - a terminal coding agent over the raw wire" width="100%">

# one-cli

A terminal coding agent for OpenAI- and Anthropic-compatible models. It reads,
edits, searches and runs commands in your workspace.

[![python](https://img.shields.io/badge/python-3.10+-8b7f72?style=flat-square&labelColor=1a1a1a)](https://github.com/one-framework-foundation/one-cli/blob/main/pyproject.toml)
[![status](https://img.shields.io/badge/status-v0.1%20alpha-facc15?style=flat-square&labelColor=1a1a1a)](https://github.com/one-framework-foundation/one-cli/blob/main/README.md#status)
[![license](https://img.shields.io/badge/license-one%200.6.3-8b7f72?style=flat-square&labelColor=1a1a1a)](https://github.com/one-framework-foundation/one-cli/blob/main/README.md#license)
[![wires](https://img.shields.io/badge/wires-openai%20%C2%B7%20anthropic-8b7f72?style=flat-square&labelColor=1a1a1a)](https://github.com/one-framework-foundation/one-cli/blob/main/docs/manual.md#backends)

<sub>part of the <b>one framework</b></sub>

</div>

## Get started

Requires Python 3.10 or newer on Linux or macOS. On Windows, use WSL.

```sh
python -m pip install one-core-cli
one
```

To work from a checkout instead:

```sh
git clone https://github.com/one-framework-foundation/one-cli.git one-cli && cd one-cli
python3 -m venv .venv && . .venv/bin/activate
python -m pip install .
one
```

On first run, `one` helps you choose a local or hosted model and tests the
connection. Then try:

```sh
one --read-only "explain what this repo does"
one "make a change in this repo"
```

**By default, `one` can edit files and run shell commands as your user.** Work in
a trusted, version-controlled directory. Use `--read-only` to ask a question
without granting write or shell tools. A hosted model may charge for requests.

`one update` upgrades one-cli with whatever installed it (uv tool, pipx or pip)
and `one update --check` only reports. A terminal session prints one line when
a newer release is out, checked at most once a day in the background; set
`ONECLI_NO_UPDATE_CHECK=1` to turn that off.

[First-run setup](https://github.com/one-framework-foundation/one-cli/blob/main/docs/getting-started.md) · [Permissions and cost](https://github.com/one-framework-foundation/one-cli/blob/main/docs/permissions.md)

## Status

<!-- readme:project-facts:start -->
`v0.1.1` · Python `>=3.10` · standard-library runtime.

Installed commands: `one`, `one-arena`, `one-bench`, `one-chat`, `one-parity`, `one-stress`, `one-toolprobe`.
<!-- readme:project-facts:end -->

Alpha (v0.1). Expect flags to change between versions.

## Documentation

- [Getting started](https://github.com/one-framework-foundation/one-cli/blob/main/docs/getting-started.md): setup, first run and common commands
- [Permissions](https://github.com/one-framework-foundation/one-cli/blob/main/docs/permissions.md): workspace access, tools and spending limits
- [Manual](https://github.com/one-framework-foundation/one-cli/blob/main/docs/manual.md): sessions, backends, tools, MCP, serving and benchmarks
- [Contributing](https://github.com/one-framework-foundation/one-cli/blob/main/CONTRIBUTING.md) · [Support matrix](https://github.com/one-framework-foundation/one-cli/blob/main/docs/support-matrix.md) · [Changelog](https://github.com/one-framework-foundation/one-cli/blob/main/CHANGELOG.md)
- [Verifying a release](https://github.com/one-framework-foundation/one-cli/blob/main/docs/reproducible-releases.md) · [Security and reporting](https://github.com/one-framework-foundation/one-cli/blob/main/SECURITY.md)

## License

one-cli is licensed under the one license 0.6.3, included as
[`LICENSE.md`](https://github.com/one-framework-foundation/one-cli/blob/main/LICENSE.md) in this repository and in both release archives.

---

<sub>License: one license 0.6.3. one-cli is a <b>one framework</b> project.</sub>

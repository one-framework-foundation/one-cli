# Support matrix

The public `.github/workflows/ci.yml` runs on pushes and pull requests to
`main`, using GitHub-hosted `ubuntu-latest` and `macos-latest` runners.
A configured job is not evidence that a runner completed it.

| | ubuntu-latest | macos-latest |
|---|---|---|
| Python 3.10 | build, install, `smoke-wheel.py` | build, install, `smoke-wheel.py` |
| Python 3.13 | build, install, `smoke-wheel.py` | build, install, `smoke-wheel.py` |

The `tests` job runs on ubuntu-latest with Python 3.11. The `leak-scan`
job runs on ubuntu-latest with Python 3.12. The wheel matrix runs on
ubuntu-latest and macos-latest for Python 3.10 and 3.13, every leg
unconditionally.

`pyproject.toml` declares `requires-python = ">=3.10"`, with wheel coverage at 3.10 and 3.13, and test coverage at 3.11.
Python 3.12 is used by leak-scan but has no separate wheel or test job. Windows is not supported directly; use WSL, which runs the Linux path.

The base CLI declares no third-party runtime dependencies. The optional
`relay` extra installs FastAPI, Uvicorn and HTTPX; install it with
`pip install "one-core-cli[relay]"`. The test and development extras are
separate from the base install. Their dependency ranges are declared in
`pyproject.toml`.

To add a Python version or OS to this matrix, add it to the `matrix:` block
in `.github/workflows/ci.yml` first - this document follows CI, CI does
not follow this document.

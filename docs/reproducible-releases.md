# Verifying a release

Every release is built from a reviewed commit on `main`. Its tag points at a
commit on the `release/v<version>` branch that descends from that reviewed
commit and differs from it only in release records: the admission records in
`docs/release-stamps/` and, when approvals were recorded after review,
`docs/release-rights.json`. You can check the artifacts you downloaded against
the published hashes, re-run the release admission check, or rebuild the
artifacts and compare.

## Release files

A release publishes a wheel and a source distribution. PyPI lists the sha256 of
each file on the project's release page; compare that with the file you
downloaded:

```bash
sha256sum one_core_cli-<version>-py3-none-any.whl
```

The license expression and the sha256 of `LICENSE.md` are declared in
[`docs/release-rights.json`](release-rights.json), and the release workflow
refuses an artifact whose license material does not match.

## Release admission

The release workflow (`.github/workflows/release.yml`) checks out the tagged
commit, builds the artifacts with `scripts/build-release.py` (which builds the
reviewed commit named in the build record, after checking that the tag differs
from it only in release records), and runs
`scripts/check-release-rights.py --commit <tag commit> dist/*` against
`docs/release-rights.json` and `docs/release-stamps/` of that checkout. A
release that fails this check is not published, so every published release
has these admission records committed in the repository at its tag:

- `build.json`: the build record, naming the built commit and the sha256 of
  each artifact;
- six domain stamps: `leak.json`, `safety.json`, `git.json`, `readme.json`,
  `code.json` and `neatness.json`;
- `notary.json`: the seal over those six stamps.

Every identity in the records is an opaque handle. The check requires each
record to bind the release commit (or a reviewed ancestor that differs from it
only in release records) and the exact sha256 of every artifact being checked,
the six stamps to pass, the seal to match the stamp files byte for byte, and
the builder, stampers and notary to be distinct. It also checks the license
expression and license file bytes inside each artifact against
`docs/release-rights.json`.

Anyone can re-run the check. Make a full clone of the repository (not a
shallow one: the check uses Git history to relate the reviewed commit to the
tag), check out the tag, put the release's wheel and source distribution, and
nothing else, into `dist/`, and run:

```bash
git checkout v<version>
python scripts/check-release-rights.py --commit "$(git rev-parse HEAD)" dist/*
```

It prints a single pass line, or `RELEASE BLOCKED:` followed by each reason.
A pass shows that the records and the files agree; it does not establish legal
ownership.

## Rebuilding the bytes

Plain `python -m build` is **not byte reproducible here**: even with
commit-derived `SOURCE_DATE_EPOCH` and the same tool versions, two builds give
identical wheels but different sdist hashes, because setuptools includes
wall-clock gzip and generated-file tar metadata. Pinning alone is
insufficient (see the
[upstream issue](https://github.com/pypa/setuptools/issues/2133)).

Use Python 3.13.12 and the pinned toolchain:

```bash
python -m pip install -r scripts/release-build-requirements.txt
python scripts/build-release.py --outdir dist
```

The builder exports a clean Git tree, builds the sdist, normalizes tar
ownership and timestamps plus gzip metadata, and builds the wheel from that
normalized sdist with commit-derived `SOURCE_DATE_EPOCH`. No source content is
rewritten. Compare your output with the published files using `sha256sum`.
Differing Python or zlib versions are not expected to reproduce identical
compressed bytes, so note the versions you used.

## Automatic network activity

Automatic network inventory (also in SECURITY.md):

| Trigger | Endpoint and credentials | Cache and disable control |
| --- | --- | --- |
| Interactive startup with all three streams on a terminal and no `CI` | GET `https://pypi.org/pypi/one-core-cli/json`; no credentials, `one-cli/<version>` User-Agent | Shared atomic 24-hour attempt reservation, including failures; `ONECLI_NO_UPDATE_CHECK=1` or `"update_check": false` in `~/.one-cli/config.json` disables it |
| Backend resolution/discovery, including `--list` | GET `http://127.0.0.1:8080/v1/models` and each comma-separated `ONE_LLAMACPP_HOSTS` base plus `/v1/models`; no credentials | No cache; explicit `--base-url` bypasses discovery unless `--fallback-model` or `ONE_CLI_FALLBACK` names a chain, and attachment `ONE_CLI_VISION_BACKENDS` names then resolve from drop-ins without it; unset `ONE_LLAMACPP_HOSTS` to remove extra hosts |
| Local/private backend presence on `--list`, explicit `-m NAME` selection, and chat `/model` | GET endpoint root `/health` without credentials, then OpenAI base `/models` with Bearer backend API key, or Anthropic root `/v1/models` with backend `x-api-key` and `anthropic-version`; only exact localhost/private IP literals qualify | In-process 30 seconds up / 300 seconds for connection/health failures / 30 seconds for unsuccessful OpenAI listings; selection rechecks verdicts older than 10 seconds; `ONECLI_NO_PRESENCE=1` disables probes and presence refusal |
| Role drop-in backend resolution/discovery, including `--list`, when `ONE_PROVIDER_ROLE` is set | GET configured provider base plus `/models` on any configured host; no credentials | In-process 300 seconds on success / 60 seconds on failure; `ONECLI_NO_PRESENCE` does not disable it. Remove the role drop-in, or replace `ONE_PROVIDER_ROLE` with an explicit `ONE_PROVIDER_MODEL`, to disable role discovery |
| Attachment perception | Audio and video attachments probe the speech endpoints configured with `ONE_PARAKEET_URL` / `ONE_PARAKEET_REMOTE_URL`; no credentials | Once per process; omit audio/video attachments or use `--media-route off` to avoid these probes. Text-only and image routing do not probe speech or TTS services |
| Spoken answer with `--speak` | GET each TTS rung in order until one answers - `ONE_TTS_URL` alone, else each `ONE_TTS_ENDPOINTS` entry, else `ONE_TTS_REMOTE_URL` - at `/health` for `synthesize` rungs and the endpoint root for ElevenLabs-shape rungs, then POST the answer text to `/synthesize` or `/v1/text-to-speech/<voice>` there; no credentials | Once per process; no probe when none of these is set; speech-in endpoints are not probed; omit `--speak` to disable |
| Attachment capability metadata | Image attachment inline capability checks: GET selected backend base `/v1/models`; image description routing: GET `<base>/v1/models` for each `ONE_CLI_VISION_BACKENDS` backend (`capabilities.vision_backends()` -> `declares()`); audio under `--inline-audio` or `--media-route inline`: GET selected backend base `/v1/models` (`senses/ear.py` `perceive()` -> `declares(backend, 'audio')`); all automatic, including public HTTPS destinations; no credentials | In-process 60 seconds, including misses; omit attachments or use `--media-route off` to disable |
| Local vision fallback discovery | When image/video perception needs a fallback: GET `/api/tags` and POST `/api/show` with a model name, no credentials; default `http://127.0.0.1:11434`, replaced by comma-separated `ONE_SENSES_OLLAMA_ENDPOINTS` destinations | In-process 60 seconds, including misses; empty `ONE_SENSES_OLLAMA_ENDPOINTS` disables local discovery; `ONE_CLI_VISION_BACKENDS` selects explicit backend names instead, resolved by backend discovery, or from drop-ins only under an explicit `--base-url`; `--media-route off` disables attachment routing |
| Automatic image description | Fallback image perception and sampled video frames POST image bytes and a prompt to `/v1/chat/completions` at discovered local vision backends or explicitly selected `ONE_CLI_VISION_BACKENDS`; configured backend credentials and headers accompany inference | No result cache; transient failures cool down for 60 seconds; omit attachments or use `--media-route off` to disable |
| Relay transcription upload | Audio attachments and extracted video audio: multipart POST audio bytes to `<resolved relay>/transcribe` (the endpoint configured with `ONE_PARAKEET_URL` / `ONE_PARAKEET_REMOTE_URL`); no credentials | No transcript cache; omit audio/video attachments or use `--media-route off` to disable |
| Hosted transcription fallback | Audio attachments and extracted video audio, after local relay transcription is unavailable or fails: POST audio bytes to `https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&detect_language=true` with `Authorization: Token` from `DEEPGRAM_API_KEY`, then multipart audio to `https://api.groq.com/openai/v1/audio/transcriptions` with `Authorization: Bearer` from `GROQ_API_KEY` (model `GROQ_STT_MODEL`, default `whisper-large-v3`) | No transcript cache; only providers with configured keys are attempted; unset both keys, omit audio/video attachments, or use `--media-route off` to disable |
| First-run onboarding local detection | GET localhost server `/v1/models` endpoints `127.0.0.1:8080`, `127.0.0.1:11434`, `127.0.0.1:1234`; no credentials | No cache; configure a backend before first run or use `--base-url` to bypass onboarding |
| Fallback candidate eligibility for OpenRouter, unless already cooling | GET configured OpenRouter API base plus `/key`; Bearer provider API key | Successful verdict cached 900 seconds in lanehealth state; select a non-OpenRouter endpoint and disable fallback with `--no-fallback` to avoid it |
| Anthropic-compatible model request without an explicit output ceiling | GET selected base `/v1/models/<model>` and, if needed, `/v1/models`; configured `x-api-key`, `Authorization: Bearer <key>`, `anthropic-version` and extra provider headers | Maximum cached per wire instance; set `--max-tokens` to disable these metadata probes |
| Context sizing for a selected plain HTTP backend | GET base `/props`; no credentials | No cache; HTTPS endpoints are not probed |
| Server token counting for a selected plain HTTP backend | POST base `/apply-template`, then `/tokenize`, containing conversation text; no credentials | No cache; HTTPS endpoints are not probed |

Bare `import onecli`, `one --help` and `one --version` make no network calls.

Explicit onboarding provider selection may GET the selected provider's model
list with its API key and send a test model request; these are user-triggered,
uncached calls. Model turns, tools and explicitly requested update/serve/media
commands also contact their selected endpoints. `one update` queries PyPI and
prints the installer upgrade command before executing it.

## Continuous integration

The public workflow is `.github/workflows/ci.yml`, running on `ubuntu-latest`
and `macos-latest` with Python 3.10, 3.11 and 3.13; `scripts/leak-scan.py` is
part of the gate. Tag publishing is defined by `release.yml`.

## Published artifacts

A `v<version>` tag runs `release.yml`, which checks the tag against both
version declarations, rebuilds and smoke-tests the artifacts, then retains the
wheel and source distribution (`dist/*`) as CI artifacts for 7 days. Because
the repository is public, anyone with read access, including any signed-in
GitHub user, can download them. Download and review them before that window closes;
the `publish` job uploads those same bytes to PyPI.

The upload uses PyPI Trusted Publishing: the `publish` job runs in the `pypi`
environment and exchanges its GitHub OIDC token for a short-lived PyPI token,
so the repository holds no long-lived upload credential.

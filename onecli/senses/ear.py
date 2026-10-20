"""Senses Ear - audio in.

The ladder is: a configured speech-to-text server -> Deepgram -> Groq, first
non-empty wins. The server is whatever the operator names with ONE_PARAKEET_URL
(any server with the same /health and /transcribe API); the cloud rungs exist
for when it is down or unset.

A missing API key SKIPS a rung. It is not a failure, and reporting it as one
sends whoever reads the log looking for an outage at Deepgram that never
happened.

The default output is a transcript injected as text, not raw audio on the
request. That is deliberate: the transcript path is the proven one across every
backend, while inline audio works on exactly the models that declare it. Inline
stays reachable behind an explicit flag rather than being guessed at.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from onecli.senses import capabilities as caps_mod
from onecli.senses.base import INLINE, TRANSCRIBE, Outcome, Rung

RELAY_TIMEOUT = float(os.environ.get("ONE_PARAKEET_TIMEOUT", "120"))
DEEPGRAM_URL = ("https://api.deepgram.com/v1/listen"
                "?model=nova-2&smart_format=true&detect_language=true")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")


WORK_PREFIX = "senses_ear_"


def _quote(value: str) -> str:
    """Escape a value for a quoted Content-Disposition parameter (WHATWG form
    encoding): quotes, CR and LF become percent escapes so a hostile filename
    cannot end the parameter or inject a header or part."""
    return (str(value).replace("%", "%25").replace('"', "%22")
            .replace("\r", "%0D").replace("\n", "%0A"))


def discard_workdir(path: str, prefix: str) -> None:
    """Remove the private dir a helper made for `path`, only if it is ours."""
    parent = os.path.dirname(path or "")
    if os.path.basename(parent).startswith(prefix):
        shutil.rmtree(parent, ignore_errors=True)


def _multipart(url: str, field: str, filename: str, data: bytes,
               headers: dict | None = None, fields: dict | None = None,
               timeout: float = RELAY_TIMEOUT) -> dict | None:
    """POST one file as multipart/form-data over urllib.

    Hand-rolled because the package has zero dependencies on purpose - one-cli
    talks to every wire itself, and adding requests for one upload would put a
    dependency in front of the thing that is supposed to have none.
    """
    boundary = f"----onecli{uuid.uuid4().hex}"
    body = bytearray()
    for key, value in (fields or {}).items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{_quote(key)}"\r\n\r\n{value}\r\n').encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f'name="{_quote(field)}"; filename="{_quote(filename)}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode()
    body += data
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def extract_audio(video_path: str, caps=None) -> str:
    """Video audio track -> 16kHz mono WAV, the shape every STT rung wants.

    Raises when ffmpeg is absent or the video has no audio track, because a
    silent WAV
    transcribes to an empty string and reads in the log as "STT failed".
    """
    caps = caps or caps_mod.probe()
    if not caps.ffmpeg:
        raise RuntimeError("ffmpeg not on this host")
    out = os.path.join(tempfile.mkdtemp(prefix=WORK_PREFIX), "track.wav")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", video_path, "-vn",
             "-ac", "1", "-ar", "16000", "-f", "wav", out],
            capture_output=True, timeout=600)
        if proc.returncode != 0 or not os.path.isfile(out):
            tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(f"ffmpeg could not extract audio: {tail[-1] if tail else ''}")
    except BaseException:
        discard_workdir(out, WORK_PREFIX)
        raise
    return out


def _parakeet(path: str, caps) -> tuple[str, str]:
    with open(path, "rb") as fh:
        data = fh.read()
    got = _multipart(f"{caps_mod.PARAKEET_RELAY.rstrip('/')}/transcribe",
                     "file", os.path.basename(path), data)
    # The server reports no language. That gap is real and the Tongue records
    # it rather than inventing one.
    return ((got or {}).get("text") or "").strip(), ""


def _deepgram(path: str, caps) -> tuple[str, str]:
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    req = urllib.request.Request(
        DEEPGRAM_URL, data=open(path, "rb").read(), method="POST",
        headers={"Authorization": f"Token {key}",
                 "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=RELAY_TIMEOUT) as resp:
        got = json.loads(resp.read().decode("utf-8", "replace"))
    channels = ((got.get("results") or {}).get("channels") or [{}])
    first = channels[0]
    alts = first.get("alternatives") or [{}]
    return (alts[0].get("transcript") or "").strip(), first.get("detected_language", "")


def _groq(path: str, caps) -> tuple[str, str]:
    key = os.environ.get("GROQ_API_KEY", "")
    got = _multipart(GROQ_URL, "file", os.path.basename(path),
                     open(path, "rb").read(),
                     headers={"Authorization": f"Bearer {key}"},
                     fields={"model": GROQ_MODEL, "response_format": "verbose_json"})
    got = got or {}
    return (got.get("text") or "").strip(), got.get("language", "")


def _ladder(caps):
    """The rungs in order, each with the reason it would be skipped."""
    return (
        ("parakeet", _parakeet,
         "" if caps.parakeet else
         "no speech-in relay configured (ONE_PARAKEET_URL)"),
        ("deepgram", _deepgram,
         "" if caps.deepgram_key else "DEEPGRAM_API_KEY not set"),
        ("groq", _groq,
         "" if caps.groq_key else "GROQ_API_KEY not set"),
    )


def transcribe(path: str, caps=None, fallback: bool = True) -> Outcome:
    """Climb the ladder. First non-empty transcript wins."""
    caps = caps or caps_mod.probe()
    out = Outcome(modality="audio")
    for name, fn, skip_why in _ladder(caps):
        if skip_why:
            out.rungs.append(Rung(name, skipped=skip_why))
            continue
        started = time.monotonic()
        try:
            text, lang = fn(path, caps)
        except Exception as exc:   # every rung is a network call; none may raise out
            out.rungs.append(Rung(name, error=f"{type(exc).__name__}: {exc}",
                                  ms=int((time.monotonic() - started) * 1000)))
            if not fallback:
                break
            continue
        ms = int((time.monotonic() - started) * 1000)
        if text:
            out.rungs.append(Rung(name, ok=True, ms=ms))
            out.route = TRANSCRIBE
            out.text = text
            out.language = lang or ""
            return out
        out.rungs.append(Rung(name, error="empty transcript", ms=ms))
        if not fallback:
            break
    return out


def perceive(attachment, backend=None, caps=None, *, allow_inline: bool = False,
             fallback: bool = True) -> Outcome:
    """Route one audio attachment.

    `allow_inline` is the explicit flag: only then may raw audio ride the
    request, and only to a backend that declares it.
    """
    caps = caps or caps_mod.probe()
    if allow_inline and backend is not None and caps.declares(backend, "audio"):
        out = Outcome(modality="audio", route=INLINE, inline=True)
        out.rungs.append(Rung("inline", ok=True))
        return out
    return transcribe(attachment.path, caps=caps, fallback=fallback)

"""Senses Lip - voice out.

`--speak [VOICE]` sends the final answer to whichever TTS rung answers first
(see `capabilities.tts_ladder`), writes the audio next to the session, then
prints the path.

One rule dominates this module: TTS being down is REPORTED, never silent. A
client that returns None on every failure path makes a dead service and an
empty answer look identical to the caller; this module does not reproduce that
shape.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

from onecli.senses import capabilities as caps_mod
from onecli.senses.base import Outcome, Rung

DEFAULT_VOICE = os.environ.get("ONE_TTS_VOICE", "alba")
TTS_TIMEOUT = float(os.environ.get("ONE_TTS_TIMEOUT", "30"))

# Language -> voice. Only pairs we can stand behind live here; an unknown
# language keeps the default rather than picking a voice that does not exist,
# because pocket-tts answers an unknown voice with an error, not a fallback.
VOICE_BY_LANGUAGE: dict[str, str] = {}


def _request(endpoint, text: str, voice: str) -> urllib.request.Request:
    """Build the call for whichever wire shape this rung speaks.

    Two shapes, because two kinds of TTS are in use and neither is going
    away to suit a single client: an ElevenLabs-compatible proxy (one POST per voice, voice
    in the path) and pocket-tts's own `/synthesize` (voice in the body).
    """
    base = endpoint.url.rstrip("/")
    if endpoint.shape == caps_mod.TTS_ELEVENLABS:
        return urllib.request.Request(
            f"{base}/v1/text-to-speech/{urllib.parse.quote(voice, safe='')}",
            data=json.dumps({"text": text}).encode(), method="POST",
            headers={"Content-Type": "application/json"})
    return urllib.request.Request(
        f"{base}/synthesize",
        data=json.dumps({"text": text, "voice": voice, "format": "ogg"}).encode(),
        method="POST", headers={"Content-Type": "application/json"})


def synthesize(text: str, out_path: str, voice: str = DEFAULT_VOICE,
               endpoint=None) -> bytes:
    endpoint = endpoint or caps_mod.TTS_ENDPOINT
    if endpoint is None:
        raise RuntimeError("no TTS endpoint resolved")
    with urllib.request.urlopen(_request(endpoint, text, voice),
                                timeout=TTS_TIMEOUT) as resp:
        audio = resp.read()
    if not audio:
        raise RuntimeError(f"{endpoint.name} returned an empty body")
    with open(out_path, "wb") as fh:
        fh.write(audio)
    return audio


def speak(text: str, out_path: str, voice: str = "", language: str = "",
          caps=None) -> Outcome:
    """Synthesize `text`. The outcome carries the path in `text` when it worked."""
    caps = caps or caps_mod.probe(modalities={"speech"})
    out = Outcome(modality="speech", language=language or "")
    chosen = voice or VOICE_BY_LANGUAGE.get((language or "").lower(), DEFAULT_VOICE)
    if not text.strip():
        out.rungs.append(Rung("pocket-tts", skipped="nothing to say"))
        return out
    endpoint = caps_mod.TTS_ENDPOINT
    if not caps.tts or endpoint is None:
        tried = ", ".join(f"{r.name} ({r.url})" for r in caps_mod.tts_ladder())
        out.rungs.append(Rung("tts", skipped=f"no endpoint answered - tried {tried}"))
        return out
    started = time.monotonic()
    try:
        synthesize(text, out_path, chosen, endpoint=endpoint)
    except Exception as exc:
        out.rungs.append(Rung(endpoint.name, error=f"{type(exc).__name__}: {exc}",
                              ms=int((time.monotonic() - started) * 1000)))
        return out
    out.rungs.append(Rung(endpoint.name, ok=True,
                          ms=int((time.monotonic() - started) * 1000)))
    out.route = "speak"
    out.text = out_path
    return out


def output_path(session_id: str, step: int = 0) -> str:
    """Next to the session file, so the audio and the transcript that produced
    it are found together and a --resume can still point at both."""
    from onecli import session as session_mod

    directory = session_mod.SESSION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"-{step}" if step else ""
    return str(directory / f"{session_id}{suffix}.ogg")

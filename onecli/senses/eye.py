"""Senses Eye - vision in.

The ladder for an image, in order:

1. inline to the model, when the target backend's own model declares vision.
   Nothing is lossier than describing a picture to a model that could have
   looked at it.
2. OCR with tesseract, when the binary is present and the text is substantial.
   For a screenshot or an infographic the embedded text IS the content.
3. describe with a vision-capable backend from `config.discover()`, and inject
   that description as text.

2 and 3 are complementary, not exclusive - returning early on any OCR text
throws away the visual structure of exactly the images
that needed both. So both run and the results merge.

Video gets a real gate first (duration, size, frame budget) via ffprobe, then
sampled keyframes described, plus the audio track handed to the Ear. Without
this gate a very long video is described frame by frame at unbounded cost; the
gate is what keeps that bounded.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

from onecli.senses import capabilities as caps_mod
from onecli.senses import ear
from onecli.senses.base import DESCRIBE, INLINE, Outcome, Rung

# The gate. Generous enough that a normal clip passes, hard enough that a
# two-hour recording is refused instead of turning into a 400-frame bill.
MAX_VIDEO_SECONDS = float(os.environ.get("ONE_SENSES_MAX_VIDEO_SECONDS", "600"))
MAX_VIDEO_BYTES = int(os.environ.get("ONE_SENSES_MAX_VIDEO_BYTES", str(256 * 1024 * 1024)))
FRAME_BUDGET = int(os.environ.get("ONE_SENSES_FRAME_BUDGET", "6"))
# The budget is a ceiling, not a quota. A short clip described at the full
# budget pays for several near-identical stills, so the frame count scales
# with duration and only long video reaches the cap.
SECONDS_PER_FRAME = float(os.environ.get("ONE_SENSES_SECONDS_PER_FRAME", "10"))


def frames_for(seconds: float, budget: int = FRAME_BUDGET) -> int:
    """How many stills this duration is worth. Always at least one."""
    if seconds <= 0:
        return 1
    return max(1, min(budget, math.ceil(seconds / SECONDS_PER_FRAME)))

OCR_MIN_CHARS = 15          # below this an OCR result is noise, not content
OCR_MAX_CHARS = 2000
DESCRIBE_TIMEOUT = float(os.environ.get("ONE_SENSES_DESCRIBE_TIMEOUT", "15"))
DESCRIBE_BUDGET = float(os.environ.get("ONE_SENSES_DESCRIBE_BUDGET", "30"))
# Process-local: failed endpoints are not retried for every saved image.
_FAILURE_UNTIL: dict[tuple[str, str], float] = {}
FAILURE_COOLDOWN = 60.0

# Generous because a thinking vision model (qwen3-vl declares `thinking`) spends
# most of the budget before it writes a word: reasoning can outweigh the
# answer several times over, and at a small cap the answer is the part that
# gets cut. An empty description at finish_reason=length is that, not a blind model.
DESCRIBE_MAX_TOKENS = int(os.environ.get("ONE_SENSES_DESCRIBE_MAX_TOKENS", "1200"))

DESCRIBE_PROMPT = (
    "Describe this image in 2-3 sentences for someone who cannot see it. "
    "Say what it shows, any visible text or labels, and the subject matter."
)


def ocr(path: str, caps=None) -> str:
    """Embedded text, or "" when there is not enough of it to be worth injecting."""
    caps = caps or caps_mod.probe()
    if not caps.tesseract:
        return ""
    proc = subprocess.run(["tesseract", path, "stdout", "-l", "eng+nld"],
                          capture_output=True, timeout=60)
    if proc.returncode != 0:
        return ""
    text = (proc.stdout or b"").decode("utf-8", "replace").strip()
    text = " ".join(text.split("\n\n\n"))
    return text[:OCR_MAX_CHARS] if len(text) >= OCR_MIN_CHARS else ""


def describe_image(path: str, mime: str, backend, prompt: str = DESCRIBE_PROMPT, *, timeout: float | None = None) -> str:
    """One OpenAI-shaped vision call. Every describing backend we reach speaks it."""
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    body = json.dumps({
        "model": backend.model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        "max_tokens": DESCRIBE_MAX_TOKENS,
        "stream": False,
    }).encode()
    headers = dict(getattr(backend, "headers", {}) or {})
    headers["Content-Type"] = "application/json"
    if getattr(backend, "api_key", ""):
        headers["Authorization"] = f"Bearer {backend.api_key}"
    base = backend.base_url.rstrip("/")
    from urllib.parse import urlsplit
    if not urlsplit(base).path:
        base += "/v1"
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=DESCRIBE_TIMEOUT if timeout is None else timeout) as resp:
            got = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        # Retain the server's actual failure instead of an opaque HTTP 500.
        from onecli.providers.base import http_error_detail
        detail = http_error_detail(exc.read(4096).decode("utf-8", "replace"))
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    choice = (got.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        thought = message.get("reasoning") or message.get("reasoning_content") or ""
        if thought or choice.get("finish_reason") == "length":
            # Name the real cause. "empty description" alone reads as a broken
            # model when the model was cut off mid-thought.
            raise RuntimeError(
                f"the model returned only reasoning ({len(thought)} chars, "
                f"finish_reason={choice.get('finish_reason')!r}) - raise "
                "ONE_SENSES_DESCRIBE_MAX_TOKENS")
    return text


# Phrases a model uses when it answered without receiving the image. Matched on
# a lowercased description, and only near its start: a genuine description of a
# photo OF a blind person could contain "cannot see" halfway through.
UNSEEING_MARKERS = (
    "unable to see", "cannot see", "can't see", "not able to see",
    "no image", "haven't provided", "have not provided", "did not provide",
    "didn't provide", "unable to view", "cannot view", "can't view",
    "i don't have access to", "do not have access to", "as an ai",
    "unable to process images", "cannot process images",
)
# Only catches empty and one-word answers. A terse but true caption ("A scanned
# receipt.") is a real description, and a floor high enough to reject it would
# throw away good perception to catch a bad one.
MIN_DESCRIPTION_CHARS = 12


def unseeing(text: str) -> str:
    """The marker that says this description was written without the image, or ""."""
    body = " ".join(text.split()).lower()
    if len(body) < MIN_DESCRIPTION_CHARS:
        return f"only {len(body)} chars"
    head = body[:240]
    for marker in UNSEEING_MARKERS:
        if marker in head:
            return marker
    return ""


def _describe_rung(out: Outcome, path: str, mime: str, caps, label: str,
                   prompt: str = DESCRIBE_PROMPT, backend=None) -> str:
    """Try each vision backend in discovery order; record one rung per attempt."""
    # An explicit --base-url bypasses local discovery; named vision backends
    # still resolve from drop-ins.
    explicit = getattr(backend, "source", "") == "flag" and bool(getattr(backend, "base_url", ""))
    backends = caps_mod.vision_backends(caps, **({"include_local": False} if explicit else {}))
    if not backends:
        out.rungs.append(Rung(label, skipped="no reachable backend declares vision"))
        return ""
    deadline = time.monotonic() + max(0.0, DESCRIBE_BUDGET)
    for b in backends:
        started = time.monotonic()
        remaining = deadline - started
        if remaining <= 0:
            out.rungs.append(Rung(label, skipped="description attempt budget exhausted"))
            break
        key = (b.base_url.rstrip("/"), b.model)
        if _FAILURE_UNTIL.get(key, 0.0) > started:
            out.rungs.append(Rung(f"{label}:{b.name}", skipped="recent perception failure; cooling down"))
            continue
        try:
            text = describe_image(path, mime, b, prompt,
                                  timeout=min(max(0.001, DESCRIBE_TIMEOUT), remaining))
        except Exception as exc:
            from onecli.providers.base import is_transient
            if isinstance(exc, TimeoutError) or is_transient(exc) or "failed to load" in str(exc).lower():
                _FAILURE_UNTIL[key] = time.monotonic() + FAILURE_COOLDOWN
            out.rungs.append(Rung(f"{label}:{b.name}",
                                  error=f"{type(exc).__name__}: {exc}",
                                  ms=int((time.monotonic() - started) * 1000)))
            continue
        _FAILURE_UNTIL.pop(key, None)
        ms = int((time.monotonic() - started) * 1000)
        if not text:
            out.rungs.append(Rung(f"{label}:{b.name}", error="empty description",
                                  ms=ms))
            continue
        blind = unseeing(text)
        if blind:
            # The paired companion the Eye spec asks for, at its smallest: a
            # model that answers without seeing the image writes fluent prose
            # about not seeing it, and that prose is otherwise recorded as a
            # successful description and injected as fact.
            out.rungs.append(Rung(f"{label}:{b.name}",
                                  error=f"answered without seeing the image ({blind})",
                                  ms=ms))
            continue
        out.rungs.append(Rung(f"{label}:{b.name}", ok=True, ms=ms))
        return text
    return ""


def perceive_image(attachment, backend=None, caps=None, *, mode: str = "auto",
                   fallback: bool = True) -> Outcome:
    caps = caps or caps_mod.probe()
    out = Outcome(modality="image")

    if mode in ("auto", INLINE):
        if backend is not None and caps.declares(backend, "vision"):
            out.rungs.append(Rung("inline", ok=True))
            out.route = INLINE
            out.inline = True
            return out
        out.rungs.append(Rung(
            "inline", skipped=(f"backend {getattr(backend, 'name', '?')} declares no vision "
                     f"(model {getattr(backend, 'model', '?')}: "
                     f"{caps.status(backend, 'vision')})")))
        if mode == INLINE or not fallback:
            return out

    parts: list[str] = []
    text = ocr(attachment.path, caps) if caps.tesseract else ""
    if not caps.tesseract:
        out.rungs.append(Rung("ocr", skipped="tesseract not on this host"))
    elif text:
        out.rungs.append(Rung("ocr", ok=True))
        parts.append(f"Text read from the image:\n{text}")
    else:
        out.rungs.append(Rung("ocr", error="no substantial text"))

    caption = _describe_rung(out, attachment.path, attachment.mime, caps, "describe",
                             backend=backend)
    if caption:
        parts.append(f"Description of the image: {caption}")

    if parts:
        out.route = DESCRIBE
        out.text = (f"Image attachment named {os.path.basename(attachment.path)}, "
                    "perceived for you:\n") + "\n\n".join(parts)
    return out


def probe_video(path: str, caps=None) -> dict:
    """Duration and stream list from ffprobe. Empty dict when it cannot answer."""
    caps = caps or caps_mod.probe()
    if not caps.ffprobe:
        return {}
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, timeout=60)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads((proc.stdout or b"{}").decode("utf-8", "replace"))
    except ValueError:
        return {}


def video_gate(path: str, caps=None) -> str:
    """"" when the video may be processed, otherwise the reason it may not."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"cannot stat the video: {exc}"
    if size > MAX_VIDEO_BYTES:
        return (f"video is {size} bytes, over the {MAX_VIDEO_BYTES} byte gate - "
                "cut it or raise ONE_SENSES_MAX_VIDEO_BYTES")
    info = probe_video(path, caps)
    if not info:
        return "ffprobe could not read the video, so its duration is unknown"
    try:
        seconds = float((info.get("format") or {}).get("duration", 0))
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds > MAX_VIDEO_SECONDS:
        return (f"video is {seconds:.0f}s, over the {MAX_VIDEO_SECONDS:.0f}s gate - "
                "trim it or raise ONE_SENSES_MAX_VIDEO_SECONDS")
    return ""


WORK_PREFIX = "senses_eye_"


def sample_frames(path: str, seconds: float, budget: int = FRAME_BUDGET) -> list[str]:
    """Evenly spaced stills. Even spacing over shot detection on purpose: it is
    deterministic, so repeated runs see the same frames, and a keyframe extractor
    that picks different frames per run makes two descriptions incomparable."""
    workdir = tempfile.mkdtemp(prefix=WORK_PREFIX)
    span = max(seconds, 1.0)
    out: list[str] = []
    try:
        for i in range(budget):
            at = span * (i + 0.5) / budget
            frame = os.path.join(workdir, f"frame{i:02d}.jpg")
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-ss", f"{at:.2f}", "-i", path,
                 "-frames:v", "1", "-q:v", "3", frame],
                capture_output=True, timeout=120)
            if proc.returncode == 0 and os.path.isfile(frame):
                out.append(frame)
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    if not out:
        shutil.rmtree(workdir, ignore_errors=True)
    return out


def perceive_video(attachment, backend=None, caps=None, *, mode: str = "auto",
                   fallback: bool = True) -> Outcome:
    """Gate, then sampled frames described, then the audio track via the Ear.

    No wire we speak carries a video, so there is no inline rung here - only
    perception or refusal.
    """
    caps = caps or caps_mod.probe()
    out = Outcome(modality="video")

    if not caps.ffmpeg or not caps.ffprobe:
        out.rungs.append(Rung("gate", skipped="ffmpeg/ffprobe not on this host"))
        return out
    reason = video_gate(attachment.path, caps)
    if reason:
        out.rungs.append(Rung("gate", error=reason))
        return out
    out.rungs.append(Rung("gate", ok=True))

    info = probe_video(attachment.path, caps)
    try:
        seconds = float((info.get("format") or {}).get("duration", 0))
    except (TypeError, ValueError):
        seconds = 0.0

    parts: list[str] = []
    frames: list[str] = []
    wav = ""
    try:
        frames = sample_frames(attachment.path, seconds,
                               budget=frames_for(seconds))
        if not frames:
            out.rungs.append(Rung("frames", error="ffmpeg extracted no frames"))
        else:
            out.rungs.append(Rung("frames", ok=True))
            shots = []
            for i, frame in enumerate(frames):
                at = seconds * (i + 0.5) / len(frames)
                text = _describe_rung(out, frame, "image/jpeg", caps, f"describe:frame{i}",
                                      backend=backend)
                if text:
                    shots.append(f"at {at:.0f}s: {text}")
            if shots:
                parts.append("Frames sampled from the video:\n" + "\n".join(shots))

        has_audio = any(s.get("codec_type") == "audio"
                        for s in (info.get("streams") or []))
        if not has_audio:
            out.rungs.append(Rung("audio", skipped="the video has no audio stream"))
        else:
            try:
                wav = ear.extract_audio(attachment.path, caps)
            except Exception as exc:
                out.rungs.append(Rung("audio", error=f"{type(exc).__name__}: {exc}"))
            else:
                heard = ear.transcribe(wav, caps=caps, fallback=fallback)
                out.rungs.extend(Rung(f"audio:{r.name}", ok=r.ok, skipped=r.skipped,
                                      error=r.error, ms=r.ms) for r in heard.rungs)
                if heard.text:
                    out.language = out.language or heard.language
                    parts.append(f"Transcript of the video audio:\n{heard.text}")

    finally:
        # Frames and the extracted WAV live in private temp dirs; nothing
        # downstream keeps them once the descriptions and transcript exist.
        for leftover in frames:
            ear.discard_workdir(leftover, WORK_PREFIX)
        if wav:
            ear.discard_workdir(wav, ear.WORK_PREFIX)

    if parts:
        out.route = DESCRIBE
        out.text = (f"Video attachment named {os.path.basename(attachment.path)}, "
                    "perceived for you:\n") + "\n\n".join(parts)
    return out


def perceive(attachment, backend=None, caps=None, *, mode: str = "auto",
             fallback: bool = True) -> Outcome:
    from onecli import media

    if attachment.kind == media.VIDEO:
        return perceive_video(attachment, backend, caps, mode=mode, fallback=fallback)
    return perceive_image(attachment, backend, caps, mode=mode, fallback=fallback)

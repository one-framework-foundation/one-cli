"""The one capability source for every Senses.

One capability probe per process, read by all four members, never re-probed
inside a member. Everything here is best effort and never raises: a probe that
cannot answer reports the capability as absent, because a Senses that crashes on
a health check is worse than one that skips a rung.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TypeGuard

METADATA_CACHE_TTL = 60.0

# No endpoint ships as a default. An operator names a speech-to-text server with
# ONE_PARAKEET_URL (or a fallback with ONE_PARAKEET_REMOTE_URL); with neither
# set the ladder has no speech-in rung and reports the capability absent.
PARAKEET_LOCAL = os.environ.get("ONE_PARAKEET_URL", "")
PARAKEET_REMOTE = os.environ.get("ONE_PARAKEET_REMOTE_URL", "")
PARAKEET_RELAY = PARAKEET_LOCAL

# Speech out is resolved the same way, and for the same reason - but it needs
# one thing the speech-in ladder does not: the rungs do not all speak the same
# protocol. So a TTS rung carries its wire shape, and Lip dispatches on it.
#
# Nothing is pinned to a port by default. A hardcoded port is a claim that
# expires; this ladder re-derives the answer on every process instead.
TTS_ELEVENLABS = "elevenlabs"   # POST /v1/text-to-speech/<voice>, body {text}
TTS_SYNTHESIZE = "synthesize"   # POST /synthesize, body {text, voice, format}
TTS_SHAPES = (TTS_ELEVENLABS, TTS_SYNTHESIZE)


@dataclass(frozen=True)
class Endpoint:
    """One rung of a service ladder: where it is and what it speaks.

    `probe` is the path a GET must answer to prove the service is up. An empty
    probe means the service publishes no health route and ANY HTTP reply - a
    404 included - proves it is listening. That distinction is load-bearing: a
    service that serves one POST route may answer 404 on every other path, so
    demanding a 200 from /health would read a live service as dead.
    """

    name: str
    url: str
    shape: str
    probe: str = "/health"


def _tts_default_ladder() -> tuple[Endpoint, ...]:
    """The configured endpoint, or no rung at all."""
    remote_url = os.environ.get("ONE_TTS_REMOTE_URL", "")
    if not remote_url:
        return ()
    return (Endpoint("tts-remote", remote_url, TTS_ELEVENLABS, probe=""),)


def _parse_endpoint_spec(spec: str) -> Endpoint | None:
    """`name=url=shape` -> Endpoint. A malformed entry is dropped, not raised.

    Operator-supplied configuration must never be able to crash a turn, so a
    typo costs that rung and nothing else.
    """
    parts = [p.strip() for p in spec.split("=", 2)]
    if len(parts) != 3 or not all(parts):
        return None
    name, url, shape = parts
    if shape not in TTS_SHAPES:
        return None
    probe = "" if shape == TTS_ELEVENLABS else "/health"
    return Endpoint(name, url.rstrip("/"), shape, probe=probe)


def tts_ladder() -> tuple[Endpoint, ...]:
    """The rungs to try, in order. Read at probe time, never snapshotted here.

    `ONE_TTS_URL` pins one endpoint and disables the ladder - naming an
    endpoint means that endpoint, and quietly trying another one would hide a
    typo behind working audio. `ONE_TTS_ENDPOINTS` replaces the ladder with a
    comma-separated list of `name=url=shape`.
    """
    pinned = os.environ.get("ONE_TTS_URL")
    if pinned:
        shape = os.environ.get("ONE_TTS_SHAPE", TTS_ELEVENLABS)
        if shape not in TTS_SHAPES:
            shape = TTS_ELEVENLABS
        probe = "" if shape == TTS_ELEVENLABS else "/health"
        return (Endpoint("pinned", pinned.rstrip("/"), shape, probe=probe),)
    raw = os.environ.get("ONE_TTS_ENDPOINTS", "").strip()
    if raw:
        rungs = tuple(e for e in (_parse_endpoint_spec(s) for s in raw.split(",")) if e)
        if rungs:
            return rungs
    return _tts_default_ladder()


# The endpoint the last probe settled on, or None when every rung was down.
# Lip reads this rather than a constant, so the two always agree within a run.
TTS_ENDPOINT: "Endpoint | None" = None

_HEALTH_TIMEOUT = 2.0


def _health(url: str, timeout: float = _HEALTH_TIMEOUT) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as r:
            return 200 <= getattr(r, "status", 200) < 300
    except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError, TimeoutError, socket.error):
        return False


def _alive(url: str, path: str, timeout: float = _HEALTH_TIMEOUT) -> bool:
    """Is this service listening?

    With a `path`, the service publishes a health route and must answer 2xx on
    it. Without one, any HTTP status at all - 404 included - proves something
    is bound and speaking HTTP, which is the only signal a service with a
    single POST route can give us. Connection refused stays the one honest
    "down".
    """
    if path:
        try:
            target = f"{url.rstrip('/')}{path if path.startswith('/') else '/' + path}"
            with urllib.request.urlopen(target, timeout=timeout) as r:
                return 200 <= getattr(r, "status", 200) < 300
        except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError, TimeoutError,
                socket.error):
            return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/", timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError, TimeoutError, socket.error):
        return False


def _resolve_tts() -> bool:
    """Walk the ladder, pin the winner in ``TTS_ENDPOINT``, report reachability."""
    global TTS_ENDPOINT
    for rung in tts_ladder():
        if _alive(rung.url, rung.probe):
            TTS_ENDPOINT = rung
            return True
    TTS_ENDPOINT = None
    return False


def _resolve_parakeet() -> bool:
    """Pick the reachable relay and pin ``PARAKEET_RELAY`` to it.

    Ear reads the module attribute rather than a passed-in URL, so the winner
    is written back here and both agree for the rest of the process. An
    explicit ``ONE_PARAKEET_URL`` is honoured as pinned - a person naming an
    endpoint means that endpoint, and silently trying another one would hide a
    typo behind a working transcript.
    """
    global PARAKEET_RELAY
    if os.environ.get("ONE_PARAKEET_URL"):
        return _health(PARAKEET_RELAY)
    for url in (PARAKEET_LOCAL, PARAKEET_REMOTE):
        if not url:
            continue
        if _health(url):
            PARAKEET_RELAY = url
            return True
    PARAKEET_RELAY = PARAKEET_LOCAL
    return False


def _get_json(url: str, timeout: float = 6.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


@dataclass
class Capabilities:
    """What this host can do for media, as of the first probe of this process."""

    ffmpeg: bool = False
    ffprobe: bool = False
    tesseract: bool = False
    parakeet: bool = False
    tts: bool = False
    deepgram_key: bool = False
    groq_key: bool = False
    # Evidence is scoped to the selected endpoint AND model, never a provider
    # alias or transport. Unknown is distinct from a declared negative.
    model_caps: dict[tuple[str, str], list[str] | None] = field(default_factory=dict)
    _documents: dict[str, dict | None] = field(default_factory=dict, repr=False)
    _probed_at: float = field(default_factory=time.monotonic, repr=False)

    def status(self, backend, what: str) -> str:
        """Return supported, unsupported, or unknown for this exact model.

        Live model metadata takes precedence over a configured declaration.
        A drop-in declaration remains bound to its original model and URL even
        when CLI flags select another model. Provider allowlists and wire
        formats are not evidence of a model's senses.
        """
        base = (getattr(backend, "base_url", "") or "").rstrip("/")
        model = getattr(backend, "model", "") or ""
        if what not in ("vision", "audio") or not base or not model:
            return "unknown"
        caps = self._llamacpp_caps(base, model)
        if caps is None:
            binding = getattr(backend, "capability_binding", None)
            declared = getattr(backend, "declared_capabilities", None)
            if binding == (base, model) and self._valid_caps(declared):
                caps = declared
        if caps is None:
            return "unknown"
        return "supported" if what in caps else "unsupported"

    def declares(self, backend, what: str) -> bool:
        """Only known model support AND a capable wire permit media."""
        from onecli import media

        kind = {"vision": media.IMAGE, "audio": media.AUDIO}.get(what)
        return bool(kind and self.status(backend, what) == "supported"
                    and media.wire_supports(getattr(backend, "wire", ""), kind))

    @staticmethod
    def _valid_caps(value) -> TypeGuard[list[str]]:
        return isinstance(value, list) and all(isinstance(x, str) for x in value)

    def _llamacpp_caps(self, base_url: str, model: str) -> list[str] | None:
        """Read exact model metadata from either models[] or data[].

        Retain the historical method name for callers. Every alias is probed.
        Never borrow a sole unmatched entry: proxies can serve many models.
        Revalidate long-lived CLI snapshots after 60 seconds, including misses.
        """
        now = time.monotonic()
        if now - self._probed_at >= METADATA_CACHE_TTL:
            self.model_caps.clear()
            self._documents.clear()
            self._probed_at = now
        base_url = base_url.rstrip("/")
        key = (base_url, model)
        if key in self.model_caps:
            return self.model_caps[key]
        root = base_url if base_url.endswith("/v1") else base_url + "/v1"
        if root not in self._documents:
            self._documents[root] = _get_json(root + "/models", timeout=2.0)
        data = self._documents[root]
        matches = []
        if isinstance(data, dict):
            for field_name in ("models", "data"):
                entries = data.get(field_name)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if not any(entry.get(k) == model for k in ("id", "model", "name")):
                        continue
                    caps = entry.get("capabilities")
                    if self._valid_caps(caps):
                        matches.append(set(caps))
        # Conflicting explicit lists fail closed for the disputed modality.
        caps = sorted(set.intersection(*matches)) if matches else None
        self.model_caps[key] = caps
        return caps

    def as_row(self) -> dict:
        return {"ffmpeg": self.ffmpeg, "ffprobe": self.ffprobe,
                "tesseract": self.tesseract, "parakeet": self.parakeet,
                "tts": self.tts, "deepgram": self.deepgram_key,
                "groq": self.groq_key}


_CACHE: Capabilities | None = None
_PROBED_SERVICES: set[str] = set()


def probe(refresh: bool = False, *, modalities: set[str] | None = None) -> Capabilities:
    """The process-wide capability snapshot. Probed once unless asked again.

    Cached because the ladder consults it per attachment and per rung: without
    the cache a three-image turn would health-check the relay six times before
    the first request went out.
    """
    global _CACHE
    if refresh:
        _CACHE = None
        _PROBED_SERVICES.clear()
    if _CACHE is None:
        _CACHE = Capabilities(
            ffmpeg=shutil.which("ffmpeg") is not None,
            ffprobe=shutil.which("ffprobe") is not None,
            tesseract=shutil.which("tesseract") is not None,
            deepgram_key=bool(os.environ.get("DEEPGRAM_API_KEY")),
            groq_key=bool(os.environ.get("GROQ_API_KEY")),
        )
    if modalities is None:
        services = {"parakeet", "tts"}
    else:
        # Speech-in serves audio and video; "speech" (Lip's output) needs only TTS.
        services = ({"parakeet"} if {"audio", "video"} & modalities else set()) | (
            {"tts"} if "speech" in modalities else set())
    for service in sorted(services - _PROBED_SERVICES):
        setattr(_CACHE, service, _resolve_parakeet() if service == "parakeet" else _resolve_tts())
        _PROBED_SERVICES.add(service)
    return _CACHE



def reset() -> None:
    """Drop the snapshot. Tests and a long-lived process that saw a service
    come back both need this; nothing on the request path calls it."""
    global _CACHE, TTS_ENDPOINT
    _CACHE = None
    _PROBED_SERVICES.clear()
    TTS_ENDPOINT = None


def _ollama_vision_backends(caps: Capabilities, base: str) -> list:
    """Discover small models through the local admission gate, never bypass it.

    Tags supply exact model IDs and sizes; /api/show supplies model-specific
    modalities. No model download, load, or inference occurs during discovery.
    """
    from onecli.config import Backend

    base = base.rstrip("/")
    key = base + "/api/tags"
    if key not in caps._documents:
        caps._documents[key] = _get_json(key, timeout=2.0)
    tags = caps._documents[key]
    entries = tags.get("models", []) if isinstance(tags, dict) else []
    out = []
    for entry in sorted((e for e in entries if isinstance(e, dict)),
                        key=lambda e: -(e["size"] if isinstance(e.get("size"), int) else 0)):
        name, size = entry.get("name"), entry.get("size")
        # Bound automatic loading; the serving gate retains final authority.
        if not isinstance(name, str) or not isinstance(size, int) or not 0 < size <= 8_000_000_000:
            continue
        show_key = base + "/api/show:" + name
        if show_key not in caps._documents:
            try:
                req = urllib.request.Request(base + "/api/show",
                    data=json.dumps({"model": name}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=2.0) as r:
                    caps._documents[show_key] = json.load(r)
            except (http.client.HTTPException, OSError, ValueError, TimeoutError):
                caps._documents[show_key] = None
        doc = caps._documents[show_key]
        modalities = doc.get("capabilities") if isinstance(doc, dict) else None
        if not caps._valid_caps(modalities) or "vision" not in modalities:
            continue
        b = Backend(name="senses-local/" + base.split("//")[-1] + "/" + name, model=name,
                    base_url=base + "/v1", source="senses-local")
        caps.model_caps[(b.base_url, name)] = modalities
        out.append(b)
    return out


def local_vision_backends(caps: Capabilities) -> list:
    """Use local endpoints, plus any operator-supplied endpoints."""
    # No deployment address ships as a default: an operator opts in with
    # ONE_SENSES_OLLAMA_ENDPOINTS, or only loopback is probed.
    endpoints = os.environ.get("ONE_SENSES_OLLAMA_ENDPOINTS", "http://127.0.0.1:11434")
    out = []
    for base in dict.fromkeys(x.strip() for x in endpoints.split(",") if x.strip()):
        out.extend(_ollama_vision_backends(caps, base))
    return out


def vision_backends(caps: Capabilities | None = None, *, include_local: bool = True) -> list:
    """Local gated perception first; additional routes require explicit opt-in.

    ONE_CLI_VISION_BACKENDS is an ordered list of exact backend names. Naming
    a route authorizes its use, but cannot override model capability evidence.
    `include_local=False` (an explicit --base-url) resolves those names from
    drop-ins only, without probing local llama-servers.
    """
    from onecli import config

    caps = caps or probe()
    # Refresh the shared cache before consulting local tags/show documents.
    if time.monotonic() - caps._probed_at >= METADATA_CACHE_TTL:
        caps.model_caps.clear()
        caps._documents.clear()
        caps._probed_at = time.monotonic()
    names = [x.strip() for x in os.environ.get("ONE_CLI_VISION_BACKENDS", "").split(",") if x.strip()]
    if names:
        available = {b.name: b for b in config.discover(include_local=include_local)}
        return [available[n] for n in dict.fromkeys(names)
                if n in available and caps.declares(available[n], "vision")
                and available[n].wire == config.WIRE_OPENAI]
    return local_vision_backends(caps)

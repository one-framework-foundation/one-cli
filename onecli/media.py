"""Attachments: images, audio and video on the way into a turn.

Three rules this module exists to enforce, in order of how much damage they
prevent:

1. An attachment the wire cannot represent is a hard error, named. A caller
   that accepts `images`, `voice_path` and `video_path` must not then quietly
   ignore one: it produces a confident answer about a picture it never saw.
   Dropping a modality is the defect, not the limitation.
2. Mime comes off the file's own bytes first. An extension is a claim by
   whoever named the file, and a `.png` holding a JPEG is rejected by every
   vision endpoint we have with an error that blames the wrong thing.
3. Canonical history carries the PATH, never the base64. The bytes are read at
   send time by the wire. A session file is replayed on every `--resume`, and
   inlining a few megabytes of base64 per turn makes it unreadable, slow to
   load, and impossible to diff two backends on the same conversation.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

IMAGE = "image"
AUDIO = "audio"
VIDEO = "video"

# The canonical block type. Not an OpenAI type: canonical history stays a
# reference, and each wire turns it into that wire's own block at send time.
MEDIA_BLOCK = "media"

# Leading bytes -> mime. Sniffing beats the extension (rule 2); the table only
# needs the formats an endpoint will actually accept.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"\xff\xf3", "audio/mpeg"),
)

# Containers whose magic sits at a fixed offset rather than byte 0.
_MAGIC_AT: tuple[tuple[int, bytes, str], ...] = (
    (8, b"WEBP", "image/webp"),      # RIFF....WEBP
    (8, b"WAVE", "audio/wav"),       # RIFF....WAVE
    (4, b"ftypqt", "video/quicktime"),
    # ISO-BMFF declares its nature in the `ftyp` major brand at offset 8. An
    # audio-only brand is audio, not video: some containers carry an audio
    # extension while holding M4A bytes, so the brand must be able to win over
    # the extension. These sit BEFORE the generic `ftyp` entry.
    (8, b"M4A ", "audio/mp4"),
    (8, b"M4B ", "audio/mp4"),
    (8, b"M4P ", "audio/mp4"),
    (8, b"F4A ", "audio/mp4"),
    (8, b"F4B ", "audio/mp4"),
    (4, b"ftyp", "video/mp4"),       # after the quicktime brand, so mov wins
)

_KIND_BY_PREFIX = {"image/": IMAGE, "audio/": AUDIO, "video/": VIDEO}


class MediaError(ValueError):
    """An attachment we refuse to send, with the reason a human needs."""


@dataclass(frozen=True)
class Attachment:
    kind: str        # image | audio | video
    path: str        # absolute, so a resumed session in another cwd still finds it
    mime: str

    @property
    def fmt(self) -> str:
        """The bare subtype an endpoint asks for as `format` (wav, mp3, png)."""
        sub = self.mime.split("/", 1)[-1]
        return {"mpeg": "mp3", "x-wav": "wav", "vnd.wave": "wav"}.get(sub, sub)

    def read_bytes(self) -> bytes:
        try:
            return Path(self.path).read_bytes()
        except OSError as exc:
            raise MediaError(f"cannot read {self.kind} {self.path}: {exc}") from exc

    def b64(self) -> str:
        return base64.b64encode(self.read_bytes()).decode("ascii")

    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{self.b64()}"

    def to_block(self) -> dict:
        """The canonical block. Path plus mime, never the bytes (rule 3)."""
        return {"type": MEDIA_BLOCK,
                "media": {"kind": self.kind, "path": self.path, "mime": self.mime}}

    @classmethod
    def from_block(cls, block: dict) -> "Attachment":
        m = block.get("media") or {}
        return cls(kind=m.get("kind", ""), path=m.get("path", ""),
                   mime=m.get("mime", ""))


def sniff_mime(path: str, data: bytes | None = None) -> str:
    """Mime from the bytes, falling back to the extension only if unrecognised."""
    if data is None:
        with open(path, "rb") as fh:
            data = fh.read(32)
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    for offset, magic, mime in _MAGIC_AT:
        if data[offset:offset + len(magic)] == magic:
            return mime
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def load(path: str, expect: str = "") -> Attachment:
    """Validate and read one attachment. `expect` is the modality the flag promised."""
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(full):
        raise MediaError(f"no such file: {full}")
    try:
        with open(full, "rb") as fh:
            head = fh.read(32)
    except OSError as exc:
        raise MediaError(f"cannot read {full}: {exc}") from exc
    if not head:
        raise MediaError(f"empty file: {full}")

    mime = sniff_mime(full, head)
    kind = ""
    for prefix, name in _KIND_BY_PREFIX.items():
        if mime.startswith(prefix):
            kind = name
            break
    if expect and kind != expect:
        raise MediaError(
            f"{full} is {mime}, which is not {expect} - pass it with the "
            f"matching flag or convert it first"
        )
    if not kind:
        raise MediaError(f"{full} is {mime}, which is not image, audio or video")
    return Attachment(kind=kind, path=full, mime=mime)


def user_content(prompt: str, attachments: list[Attachment]) -> str | list[dict]:
    """The canonical user turn. Plain string when there is nothing attached, so
    a text-only history keeps the exact shape every existing test asserts on."""
    if not attachments:
        return prompt
    blocks: list[dict] = []
    if prompt:
        blocks.append({"type": "text", "text": prompt})
    blocks.extend(a.to_block() for a in attachments)
    return blocks


def blocks_of(content: object) -> list[dict]:
    """Media blocks inside a canonical content value, in order."""
    if not isinstance(content, list):
        return []
    return [b for b in content
            if isinstance(b, dict) and b.get("type") == MEDIA_BLOCK]


# What each wire can actually put on the request. Video is on neither: no
# endpoint takes a video frame-stream over either protocol, so a
# video attachment is refused rather than described in prose.
WIRE_MODALITIES: dict[str, set[str]] = {
    "openai": {IMAGE, AUDIO},
    "anthropic": {IMAGE},
}


def wire_supports(wire: str, kind: str) -> bool:
    return kind in WIRE_MODALITIES.get(wire, set())


def check_supported(attachments: list[Attachment], wire: str, backend: str) -> None:
    """Refuse before the request, naming modality, backend and wire.

    Called at the CLI edge so nothing is sent at all, and again inside each
    wire's conversion, because a resumed history can carry a media block that
    this run's backend cannot take.
    """
    for a in attachments:
        if not wire_supports(wire, a.kind):
            raise MediaError(
                f"{a.kind} input is not supported by backend {backend!r} on the "
                f"{wire} wire (attachment {a.path}). one-cli refuses to drop it "
                f"or send it as prose - use a backend whose wire carries "
                f"{a.kind}, or remove the attachment."
            )

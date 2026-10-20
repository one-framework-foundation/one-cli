"""The single decision point every modality goes through.

The brief's first determinism rule is "one routing decision per modality" -
code behind one chokepoint, not scattered per-call. `cli.py` and `agent.py`
call `route()` and nothing else in this package; the members are reached
through here.

What comes back is what the turn should carry: the attachments that stay inline
on the request, and the text that perception produced for the ones that do not.
`onecli/media.py`'s refusal survives underneath for `--no-media-fallback`. With
fallback on, a modality that no route could perceive is never dropped silently:
the turn carries a "NOT READ" note naming every rung tried, so the model answers
the text and tells the person the media was unreadable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from onecli import media
from onecli.senses import capabilities as caps_mod
from onecli.senses import ear, eye, logbook, tongue
from onecli.senses.base import AUTO, DESCRIBE, INLINE, OFF, ROUTES, TRANSCRIBE

# Which member owns which modality. One entry per modality, by construction.
# The MODULE is held, not its function: binding `eye.perceive` at import time
# would freeze the member the router calls, which is wrong for a package whose
# members are meant to be swappable at their own seam.
_MEMBER = {
    media.IMAGE: ("eye", eye),
    media.VIDEO: ("eye", eye),
    media.AUDIO: ("ear", ear),
}

# A mode that asks for a route this modality does not have is a no-op for that
# modality rather than an error: `--media-route transcribe` on a turn holding
# an image and a voice note should still transcribe the voice note.
_MODE_APPLIES = {
    media.IMAGE: (AUTO, INLINE, DESCRIBE, OFF),
    media.VIDEO: (AUTO, DESCRIBE, OFF),
    media.AUDIO: (AUTO, INLINE, TRANSCRIBE, OFF),
}


@dataclass
class Routed:
    inline: list = field(default_factory=list)   # attachments left on the request
    texts: list[str] = field(default_factory=list)
    outcomes: list = field(default_factory=list)

    @property
    def language(self) -> str:
        return tongue.detected(self.outcomes)

    def preamble(self) -> str:
        """The perceptions, as one block prepended to the user's prompt.

        The header says where these came from because without it a model spends
        its first two steps trying to `read` the attachment off disk - observed
        live, an image that had already been described.
        """
        body = "\n\n".join(t for t in self.texts if t)
        if not body:
            return ""
        return (
            "Perceived media, already read for you by the Senses router. The "
            "files are attachments, not paths in the workspace - do not try to "
            "open them.\n\n" + body
        )


def route_history(messages, backend, **options) -> list[dict]:
    """Revalidate saved attachments against this run's selected backend.

    Replace media in its original turn, retaining the question and block order.
    Work on copies so a failed perception leaves the saved history intact. A
    repeated reference is perceived once; successful text persists in the next
    session checkpoint and requires no auxiliary calls on later resumes.
    """
    result = []
    perceived = {}
    for message in messages:
        content = message.get("content")
        if not media.blocks_of(content):
            result.append(message)
            continue
        blocks = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != media.MEDIA_BLOCK:
                blocks.append(block)
                continue
            attachment = media.Attachment.from_block(block)
            if attachment not in perceived:
                perceived[attachment] = route([attachment], backend, **options)
            routed = perceived[attachment]
            if routed.inline:
                blocks.append(block)
            elif routed.texts:
                blocks.append({"type": "text", "text": routed.preamble()})
        result.append({**message, "content": blocks})
    return result


def route(attachments, backend, *, mode: str = AUTO, fallback: bool = True,
          allow_inline_audio: bool = False, caps=None) -> Routed:
    """Perceive every attachment. Raises `media.MediaError` when one cannot be.

    `fallback=False` (`--no-media-fallback`) means: take the first rung or
    fail. It exists because a benchmark comparing two backends on the same
    picture must know the picture reached the model, not a description of it
    written by a third model.
    """
    if mode not in ROUTES:
        raise media.MediaError(
            f"unknown media route {mode!r} - one of {', '.join(ROUTES)}")
    result = Routed()
    if not attachments or mode == OFF:
        return result
    caps = caps or caps_mod.probe(modalities={att.kind for att in attachments})

    for att in attachments:
        member, module = _MEMBER.get(att.kind, ("", None))
        if module is None:
            raise media.MediaError(f"no Senses owns {att.kind!r} ({att.path})")

        effective = mode if mode in _MODE_APPLIES[att.kind] else AUTO
        if effective == OFF:
            continue

        started = time.monotonic()
        if att.kind == media.AUDIO:
            out = module.perceive(att, backend, caps,
                                  allow_inline=allow_inline_audio or effective == INLINE,
                                  fallback=fallback)
        else:
            out = module.perceive(att, backend, caps, mode=effective,
                                  fallback=fallback)
        ms = int((time.monotonic() - started) * 1000)
        logbook.record(out, member=member, path=att.path, ms=ms,
                       extra={"backend": getattr(backend, "name", ""),
                              "mode": mode,
                              "language": tongue.label(out.language)})
        result.outcomes.append(out)

        if out.inline:
            # Inline still has to survive the wire check - a resumed history can
            # carry a block this run's backend cannot take.
            media.check_supported([att], backend.wire, backend.name)
            result.inline.append(att)
        elif out.text:
            result.texts.append(out.text)
        else:
            tried = (f"{att.kind} input {att.path} could not be routed to backend "
                     f"{getattr(backend, 'name', '?')!r}: no rung worked. Tried - "
                     f"{out.ladder() or 'nothing'}.")
            if not fallback:
                # A benchmark must know the picture reached the model.
                raise media.MediaError(tried + " one-cli refuses to drop it.")
            # The person's text question is still answerable; failing the whole
            # turn on a photo nobody can read cost a live agent the answer.
            # Not silent: the model is told and must say so.
            result.texts.append(
                f"[{att.kind} NOT READ] {tried} The {att.kind} content is "
                f"unknown to you. Answer the written part of the request and "
                f"tell the person plainly that you could not read the {att.kind}.")

    return result

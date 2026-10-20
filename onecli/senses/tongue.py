"""Senses Tongue - language.

Its whole discipline is one rule: never claim a language nothing detected.

The detection coverage is genuinely uneven by backend and the gap is
documented rather than papered over:

- Deepgram runs with `detect_language=true` and reports `detected_language`.
- Groq's verbose_json carries `language`.
- The speech-to-text server rung reports NO language at all. So a turn
  transcribed by that rung has no language, and that is recorded as `unknown` - not guessed
  from the text, not defaulted to English.

Which means the Tongue's steering is only as wide as its detection. It steers
the Lip's voice when a rung reported a language, and otherwise leaves the
default alone.
"""

from __future__ import annotations

from onecli.senses import lip

UNKNOWN = "unknown"


def normalize(language: str) -> str:
    """`en-US` -> `en`. "" stays "" - absence is not a value."""
    lang = (language or "").strip().lower()
    if not lang:
        return ""
    return lang.split("-")[0].split("_")[0]


def detected(outcomes) -> str:
    """The first language any rung actually reported, or "" when none did."""
    for out in outcomes:
        lang = normalize(getattr(out, "language", ""))
        if lang:
            return lang
    return ""


def voice_for(language: str, default: str = "") -> str:
    """Steer the Lip. An undetected language keeps the default voice."""
    lang = normalize(language)
    if not lang:
        return default or lip.DEFAULT_VOICE
    return lip.VOICE_BY_LANGUAGE.get(lang, default or lip.DEFAULT_VOICE)


def label(language: str) -> str:
    """What the log row says. `unknown` is a real answer here."""
    return normalize(language) or UNKNOWN

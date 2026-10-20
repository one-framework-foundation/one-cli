"""The Senses: the routing table per modality, and every refusal.

Nothing here touches a live service. Each rung is a network call or a
subprocess, and both are stubbed - a suite that needs the parakeet relay up is
a suite that goes red for a reason that has nothing to do with the code. The
live smoke against the real services is run separately and reported by hand.
"""

from __future__ import annotations

import base64
import json
import struct

import pytest

from onecli import config, media
from onecli.senses import capabilities as caps_mod
from onecli.senses import ear, eye, lip, logbook, router, tongue
from onecli.senses.base import Outcome, Rung

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def wav_bytes() -> bytes:
    return (b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + struct.pack("<I", 16)
            + struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8)
            + b"data" + struct.pack("<I", 0))


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """No probe cache and no real log file leaks between tests."""
    caps_mod.reset()
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k: None)
    monkeypatch.setattr(logbook, "LOG_PATH", tmp_path / "senses_log.jsonl")
    yield
    caps_mod.reset()


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(PNG_BYTES)
    return media.load(str(p))


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "note.wav"
    p.write_bytes(wav_bytes())
    return media.load(str(p))


@pytest.fixture
def mp4(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16)
    return media.load(str(p))


def caps(**kw):
    """A capability snapshot with everything off unless the test turns it on."""
    return caps_mod.Capabilities(**kw)


def backend(name="llamacpp/x", model="x", base_url="http://127.0.0.1:8080/v1",
            wire=config.WIRE_OPENAI):
    return config.Backend(name=name, model=model, base_url=base_url, wire=wire)


# --- the one capability source -----------------------------------------------


def test_capability_probe_runs_once_per_process(monkeypatch):
    calls = {"which": 0, "health": 0}

    def which(name):
        calls["which"] += 1
        return "/usr/bin/" + name

    monkeypatch.setattr(caps_mod.shutil, "which", which)
    monkeypatch.setattr(caps_mod, "PARAKEET_LOCAL", "http://speech.test")
    monkeypatch.setenv("ONE_TTS_URL", "http://tts.test")
    monkeypatch.setattr(caps_mod, "_health", lambda url, timeout=2.0:
                        calls.__setitem__("health", calls["health"] + 1) or True)
    monkeypatch.setattr(caps_mod, "_alive", lambda url, path, timeout=2.0:
                        calls.__setitem__("health", calls["health"] + 1) or True)

    first = caps_mod.probe()
    for _ in range(5):
        assert caps_mod.probe() is first
    assert calls["which"] == 3      # ffmpeg, ffprobe, tesseract - once each
    assert calls["health"] == 2     # parakeet and the first TTS rung - once each


def test_llamacpp_capability_list_is_asked_not_assumed(monkeypatch):
    asked = []

    def get(url, timeout=6.0):
        asked.append(url)
        return {"models": [{"model": "qwen3-vl:8b",
                            "capabilities": ["completion", "vision"]}]}

    monkeypatch.setattr(caps_mod, "_get_json", get)
    c = caps()
    b = backend(model="qwen3-vl:8b")
    assert c.declares(b, "vision") is True
    assert c.declares(b, "audio") is False
    # Asked once for the model, then cached: a three-image turn must not
    # re-interrogate the server per attachment.
    assert asked == ["http://127.0.0.1:8080/v1/models"]


def test_a_capability_list_for_another_model_is_not_ours(monkeypatch):
    """Two models listed, neither ours - a miss must not borrow the first one."""
    monkeypatch.setattr(caps_mod, "_get_json", lambda url, timeout=6.0: {
        "models": [{"model": "a", "capabilities": ["completion", "vision"]},
                   {"model": "b", "capabilities": ["completion", "vision"]}]})
    assert caps().declares(backend(model="c"), "vision") is False


def test_a_model_that_declares_no_vision_gets_no_image(monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda url, timeout=6.0: {
        "models": [{"model": "qwen3.5:27b", "capabilities": ["completion"]}]})
    assert caps().declares(backend(model="qwen3.5:27b"), "vision") is False


def test_a_backend_with_no_capability_list_cannot_get_inline_images():
    c = caps()
    anth = backend(name="claude", base_url="https://api.anthropic.com",
                   wire=config.WIRE_ANTHROPIC)
    assert c.declares(anth, "vision") is False    # model support is unknown
    assert c.declares(anth, "audio") is False     # it does not carry audio


# --- Senses Eye -------------------------------------------------------------


def test_eye_sends_inline_when_the_backend_has_the_sense(png, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": ["vision"]}]})
    out = eye.perceive(png, backend(), caps())
    assert out.route == "inline" and out.inline is True
    assert [r.name for r in out.rungs] == ["inline"]


def test_eye_falls_to_ocr_and_describe_when_the_backend_is_blind(png, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(eye, "ocr", lambda path, c=None: "INVOICE total 40")
    monkeypatch.setattr(caps_mod, "vision_backends", lambda c=None: [backend("llamacpp/vl")])
    monkeypatch.setattr(eye, "describe_image", lambda *a, **k: "A scanned receipt.")

    out = eye.perceive(png, backend(), caps(tesseract=True))
    assert out.route == "describe" and out.inline is False
    assert "INVOICE total 40" in out.text and "A scanned receipt." in out.text
    names = [r.name for r in out.rungs]
    assert names == ["inline", "ocr", "describe:llamacpp/vl"]


def test_eye_skips_ocr_when_tesseract_is_absent(png, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(caps_mod, "vision_backends", lambda c=None: [backend("llamacpp/vl")])
    monkeypatch.setattr(eye, "describe_image", lambda *a, **k: "A cat.")
    out = eye.perceive(png, backend(), caps(tesseract=False))
    ocr_rung = next(r for r in out.rungs if r.name == "ocr")
    assert ocr_rung.skipped == "tesseract not on this host"
    assert not ocr_rung.error          # absent is not failed


def test_eye_refusal_names_every_rung_it_tried(png, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(caps_mod, "vision_backends", lambda c=None: [])
    monkeypatch.setattr(eye, "ocr", lambda path, c=None: "")

    message = router.route([png], backend(), caps=caps(tesseract=True)).texts[0]
    assert "declares no vision" in message          # rung 1
    assert "ocr: failed" in message                 # rung 2
    assert "no reachable backend declares vision" in message   # rung 3
    with pytest.raises(media.MediaError, match="refuses to drop it"):
        router.route([png], backend(), fallback=False, caps=caps(tesseract=True))


def test_no_media_fallback_stops_at_the_first_rung(png, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(eye, "ocr", lambda path, c=None: "plenty of text here")
    with pytest.raises(media.MediaError):
        router.route([png], backend(), fallback=False, caps=caps(tesseract=True))


# --- the video gate -----------------------------------------------------------


def test_video_gate_refuses_a_video_longer_than_the_cap(mp4, monkeypatch):
    monkeypatch.setattr(eye, "probe_video",
                        lambda path, c=None: {"format": {"duration": "99999"}})
    reason = eye.video_gate(mp4.path, caps(ffprobe=True))
    assert "over the 600s gate" in reason


def test_video_gate_refuses_an_oversized_file(mp4, monkeypatch):
    monkeypatch.setattr(eye, "MAX_VIDEO_BYTES", 4)
    assert "over the 4 byte gate" in eye.video_gate(mp4.path, caps(ffprobe=True))


def test_video_gate_blocks_the_whole_route_and_says_why(mp4, monkeypatch):
    monkeypatch.setattr(eye, "probe_video",
                        lambda path, c=None: {"format": {"duration": "99999"}})
    out = eye.perceive(mp4, backend(), caps(ffmpeg=True, ffprobe=True))
    assert out.route == "off" and not out.ok
    assert "gate: failed" in out.ladder()


def test_video_frames_are_described_and_the_audio_goes_to_the_ear(mp4, monkeypatch):
    monkeypatch.setattr(eye, "video_gate", lambda path, c=None: "")
    monkeypatch.setattr(eye, "probe_video", lambda path, c=None: {
        "format": {"duration": "12"},
        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}]})
    monkeypatch.setattr(eye, "sample_frames", lambda path, seconds, budget=6: ["/tmp/f0.jpg"])
    monkeypatch.setattr(caps_mod, "vision_backends", lambda c=None: [backend("llamacpp/vl")])
    monkeypatch.setattr(eye, "describe_image", lambda *a, **k: "A person waving.")
    monkeypatch.setattr(ear, "extract_audio", lambda path, c=None: "/tmp/track.wav")
    monkeypatch.setattr(ear, "_parakeet", lambda path, c: ("hello there", ""))

    out = eye.perceive(mp4, backend(), caps(ffmpeg=True, ffprobe=True, parakeet=True))
    assert out.route == "describe"
    assert "A person waving." in out.text and "hello there" in out.text
    assert "audio:parakeet" in out.ladder()


# --- Senses Ear -------------------------------------------------------------


def test_ear_ladder_order_is_parakeet_then_deepgram_then_groq(wav, monkeypatch):
    order = []
    for name in ("_parakeet", "_deepgram", "_groq"):
        def fn(path, c, _n=name):
            order.append(_n)
            return "", ""
        monkeypatch.setattr(ear, name, fn)
    out = ear.transcribe(wav.path, caps(parakeet=True, deepgram_key=True, groq_key=True))
    assert order == ["_parakeet", "_deepgram", "_groq"]
    assert not out.ok


def test_ear_first_non_empty_transcript_wins(wav, monkeypatch):
    monkeypatch.setattr(ear, "_parakeet", lambda p, c: ("the local one", ""))
    monkeypatch.setattr(ear, "_deepgram", lambda p, c: pytest.fail("must not be reached"))
    out = ear.transcribe(wav.path, caps(parakeet=True, deepgram_key=True))
    assert out.route == "transcribe" and out.text == "the local one"
    assert [r.name for r in out.rungs] == ["parakeet"]


def test_a_missing_key_skips_a_rung_it_does_not_fail_it(wav, monkeypatch):
    monkeypatch.setattr(ear, "_parakeet", lambda p, c: ("", ""))
    out = ear.transcribe(wav.path, caps(parakeet=True))   # no keys at all
    by_name = {r.name: r for r in out.rungs}
    assert by_name["deepgram"].skipped == "DEEPGRAM_API_KEY not set"
    assert by_name["groq"].skipped == "GROQ_API_KEY not set"
    assert not by_name["deepgram"].error and not by_name["groq"].error


def test_a_rung_that_raises_climbs_to_the_next_one(wav, monkeypatch):
    def boom(path, c):
        raise OSError("relay refused the connection")
    monkeypatch.setattr(ear, "_parakeet", boom)
    monkeypatch.setattr(ear, "_deepgram", lambda p, c: ("cloud caught it", "nl"))
    out = ear.transcribe(wav.path, caps(parakeet=True, deepgram_key=True))
    assert out.text == "cloud caught it" and out.language == "nl"
    assert "relay refused the connection" in out.rungs[0].error


def test_audio_is_transcribed_not_sent_inline_by_default(wav, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": ["audio"]}]})
    monkeypatch.setattr(ear, "_parakeet", lambda p, c: ("spoken words", ""))
    out = ear.perceive(wav, backend(), caps(parakeet=True))
    assert out.route == "transcribe" and out.inline is False


def test_inline_audio_needs_both_the_flag_and_a_backend_that_declares_it(wav, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": ["audio"]}]})
    out = ear.perceive(wav, backend(), caps(), allow_inline=True)
    assert out.route == "inline" and out.inline is True


# --- Senses Tongue ----------------------------------------------------------


def test_tongue_never_claims_a_language_nothing_detected():
    parakeet_only = Outcome(modality="audio", language="")
    assert tongue.detected([parakeet_only]) == ""
    assert tongue.label("") == "unknown"
    assert tongue.voice_for("") == lip.DEFAULT_VOICE


def test_tongue_normalizes_and_steers_the_lip_voice(monkeypatch):
    monkeypatch.setitem(lip.VOICE_BY_LANGUAGE, "nl", "daan")
    assert tongue.detected([Outcome(modality="audio", language="nl-NL")]) == "nl"
    assert tongue.voice_for("nl-NL") == "daan"


# --- Senses Lip -------------------------------------------------------------


EL = caps_mod.Endpoint("tts-local", "http://tts.test", caps_mod.TTS_ELEVENLABS,
                       probe="")


def test_lip_reports_a_dead_tts_it_does_not_go_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_TTS_REMOTE_URL", "http://remote-host.test:9889")
    monkeypatch.setattr(caps_mod, "TTS_ENDPOINT", None)
    out = lip.speak("hello", str(tmp_path / "a.ogg"), caps=caps(tts=False))
    assert not out.ok
    # The refusal has to name every endpoint tried, or an operator cannot tell
    # a dead service from a ladder pointed at the wrong host.
    assert "no endpoint answered" in out.ladder()
    assert "tts-remote" in out.ladder()


def test_lip_refuses_when_the_probe_passed_but_no_endpoint_was_pinned(tmp_path,
                                                                     monkeypatch):
    """`tts=True` with no resolved rung must not reach synthesize().

    The two travel together, and a snapshot restored from elsewhere could
    disagree; reaching synthesize() there would raise instead of laddering.
    """
    monkeypatch.setattr(caps_mod, "TTS_ENDPOINT", None)
    monkeypatch.setattr(lip, "synthesize", lambda *a, **k: pytest.fail("called"))
    out = lip.speak("hello", str(tmp_path / "a.ogg"), caps=caps(tts=True))
    assert not out.ok


def test_lip_reports_the_synthesis_error_itself(tmp_path, monkeypatch):
    def boom(text, path, voice, endpoint=None):
        raise OSError("connection reset")
    monkeypatch.setattr(caps_mod, "TTS_ENDPOINT", EL)
    monkeypatch.setattr(lip, "synthesize", boom)
    out = lip.speak("hello", str(tmp_path / "a.ogg"), caps=caps(tts=True))
    assert not out.ok and "connection reset" in out.ladder()
    assert "tts-local" in out.ladder()


def test_lip_writes_the_file_and_returns_its_path(tmp_path, monkeypatch):
    target = tmp_path / "answer.ogg"
    monkeypatch.setattr(caps_mod, "TTS_ENDPOINT", EL)
    monkeypatch.setattr(lip, "synthesize",
                        lambda text, path, voice, endpoint=None:
                        open(path, "wb").write(b"OggS"))
    out = lip.speak("hello", str(target), caps=caps(tts=True))
    assert out.ok and out.text == str(target)


def test_lip_speaks_elevenlabs_on_an_elevenlabs_rung():
    """Voice in the PATH, body carries only the text."""
    req = lip._request(EL, "hello", "aiden")
    assert req.full_url == "http://tts.test/v1/text-to-speech/aiden"
    assert json.loads(req.data.decode()) == {"text": "hello"}


def test_lip_speaks_synthesize_on_a_pocket_tts_rung():
    """Voice in the BODY. Two shapes on one ladder is the whole point."""
    pocket = caps_mod.Endpoint("pocket", "http://tts.test",
                               caps_mod.TTS_SYNTHESIZE)
    req = lip._request(pocket, "hello", "alba")
    assert req.full_url == "http://tts.test/synthesize"
    assert json.loads(req.data.decode()) == {"text": "hello", "voice": "alba",
                                             "format": "ogg"}


def test_a_voice_with_a_slash_cannot_escape_the_path():
    req = lip._request(EL, "hi", "../../admin")
    assert req.full_url == "http://tts.test/v1/text-to-speech/..%2F..%2Fadmin"


# --- the TTS ladder -----------------------------------------------------------


def test_the_default_tts_ladder_is_local_only(monkeypatch):
    monkeypatch.delenv("ONE_TTS_URL", raising=False)
    monkeypatch.delenv("ONE_TTS_REMOTE_URL", raising=False)
    monkeypatch.delenv("ONE_TTS_ENDPOINTS", raising=False)
    assert caps_mod.tts_ladder() == ()


def test_a_configured_remote_url_is_the_only_rung(monkeypatch):
    monkeypatch.delenv("ONE_TTS_URL", raising=False)
    monkeypatch.delenv("ONE_TTS_ENDPOINTS", raising=False)
    monkeypatch.setenv("ONE_TTS_REMOTE_URL", "http://remote-host.test:9889")
    rungs = caps_mod.tts_ladder()
    assert [r.url for r in rungs] == ["http://remote-host.test:9889"]


def test_no_rung_pins_a_port_by_default():
    assert caps_mod.tts_ladder() == ()


def test_a_pinned_endpoint_disables_the_ladder(monkeypatch):
    monkeypatch.setenv("ONE_TTS_URL", "http://named.test/")
    rungs = caps_mod.tts_ladder()
    assert len(rungs) == 1
    assert rungs[0].url == "http://named.test" and rungs[0].name == "pinned"


def test_an_endpoint_list_replaces_the_ladder(monkeypatch):
    monkeypatch.delenv("ONE_TTS_URL", raising=False)
    monkeypatch.setenv("ONE_TTS_ENDPOINTS",
                       "a=http://a.test=synthesize,b=http://b.test=elevenlabs")
    rungs = caps_mod.tts_ladder()
    assert [(r.name, r.shape) for r in rungs] == [("a", "synthesize"),
                                                  ("b", "elevenlabs")]


def test_a_malformed_rung_is_dropped_not_raised(monkeypatch):
    """A typo in operator config costs that rung, never the turn."""
    monkeypatch.delenv("ONE_TTS_URL", raising=False)
    monkeypatch.setenv("ONE_TTS_ENDPOINTS",
                       "broken,x=http://x.test=telepathy,ok=http://ok.test=elevenlabs")
    assert [r.name for r in caps_mod.tts_ladder()] == ["ok"]


def test_the_resolver_takes_the_first_rung_that_answers(monkeypatch):
    monkeypatch.delenv("ONE_TTS_URL", raising=False)
    monkeypatch.setenv("ONE_TTS_ENDPOINTS",
                       "dead=http://dead.test=elevenlabs,live=http://live.test=elevenlabs")
    monkeypatch.setattr(caps_mod, "_alive",
                        lambda url, path, timeout=2.0: "live" in url)
    assert caps_mod._resolve_tts() is True
    assert caps_mod.TTS_ENDPOINT.name == "live"


def test_the_resolver_reports_absent_when_every_rung_is_down(monkeypatch):
    monkeypatch.delenv("ONE_TTS_URL", raising=False)
    monkeypatch.delenv("ONE_TTS_ENDPOINTS", raising=False)
    monkeypatch.setattr(caps_mod, "_alive", lambda url, path, timeout=2.0: False)
    assert caps_mod._resolve_tts() is False
    assert caps_mod.TTS_ENDPOINT is None


def test_a_404_proves_a_probeless_service_is_listening(monkeypatch):
    """A service that serves one POST route may 404 every other path.

    Demanding a 200 from /health reads that live service as dead - which is
    exactly how Lip stayed dark while the proxy was up.
    """
    import urllib.error

    def urlopen(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(caps_mod.urllib.request, "urlopen", urlopen)
    assert caps_mod._alive("http://tts.test", "") is True


def test_connection_refused_is_still_down(monkeypatch):
    def urlopen(url, timeout=None):
        raise OSError("refused")
    monkeypatch.setattr(caps_mod.urllib.request, "urlopen", urlopen)
    assert caps_mod._alive("http://tts.test", "") is False


# --- the router and the measure rung -----------------------------------------


def test_route_off_attaches_nothing_at_all(png):
    routed = router.route([png], backend(), mode="off", caps=caps())
    assert routed.inline == [] and routed.preamble() == ""


def test_a_mode_that_does_not_apply_to_a_modality_falls_back_to_auto(wav, monkeypatch):
    """`--media-route describe` on a voice note must still transcribe it."""
    monkeypatch.setattr(ear, "_parakeet", lambda p, c: ("still heard", ""))
    routed = router.route([wav], backend(), mode="describe",
                          caps=caps(parakeet=True))
    assert "still heard" in routed.preamble()


def test_an_unknown_route_is_refused_by_name(png):
    with pytest.raises(media.MediaError) as exc:
        router.route([png], backend(), mode="hear", caps=caps())
    assert "unknown media route" in str(exc.value)


def test_every_decision_appends_one_log_row(wav, monkeypatch):
    monkeypatch.setattr(ear, "_parakeet", lambda p, c: ("", ""))
    monkeypatch.setattr(ear, "_deepgram", lambda p, c: ("heard it", "en-US"))
    router.route([wav], backend(name="llamacpp/x"), caps=caps(parakeet=True,
                                                            deepgram_key=True))
    rows = [json.loads(line) for line in logbook.LOG_PATH.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["member"] == "ear"
    assert row["modality"] == "audio"
    assert row["route"] == "transcribe"
    assert row["ok"] is True
    assert row["backend"] == "llamacpp/x" and row["mode"] == "auto"
    assert row["language"] == "en"
    assert row["path"] == wav.path
    assert isinstance(row["ts"], float) and isinstance(row["ms"], int)
    # groq is absent because deepgram already won - the ladder stops climbing,
    # it does not record rungs it never reached.
    assert [r["rung"] for r in row["rungs"]] == ["parakeet", "deepgram"]
    assert row["rungs"][0]["error"] == "empty transcript"
    assert row["rungs"][1]["ok"] is True


def test_a_refused_route_is_logged_too(png, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(caps_mod, "vision_backends", lambda c=None: [])
    routed = router.route([png], backend(), caps=caps())
    assert "[image NOT READ]" in routed.texts[0]
    rows = [json.loads(line) for line in logbook.LOG_PATH.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["ok"] is False
    assert rows[0]["language"] == "unknown"


def test_the_log_never_raises_into_the_caller(monkeypatch, tmp_path):
    """A full disk must not turn a working perception into a failed turn."""
    monkeypatch.setattr(logbook, "LOG_PATH", tmp_path / "nope" / "x.jsonl")
    monkeypatch.setattr(logbook.Path, "mkdir",
                        lambda self, **kw: (_ for _ in ()).throw(OSError("read-only")))
    row = logbook.record(Outcome(modality="audio"), member="ear")
    assert row["member"] == "ear"


def test_the_medias_own_refusal_is_still_the_last_rung_for_inline(png, monkeypatch):
    """An inline route to a wire that cannot carry the modality still refuses."""
    out = Outcome(modality="image", route="inline", inline=True,
                  rungs=[Rung("inline", ok=True)])
    monkeypatch.setattr(eye, "perceive", lambda *a, **k: out)
    weird = config.Backend(name="odd", model="m", base_url="http://x", wire="carrier-pigeon")
    with pytest.raises(media.MediaError) as exc:
        router.route([png], weird, caps=caps())
    assert "not supported by backend" in str(exc.value)


# --- who may be chosen to perceive, and whether it really saw anything -------


def test_a_wire_that_can_carry_an_image_does_not_nominate_a_describer(monkeypatch):
    """The defect this closes: an image sent to a text-only openai endpoint.

    `declares` answers "can this request be sent", which for a backend with no
    endpoint is a fact about the wire. Auto-selection needs a claim about the
    model, so a plain drop-in is not offered as a describer.
    """
    drop_in = backend(name="dashscope", model="qwen-max",
                      base_url="https://example.invalid/v1")
    monkeypatch.setattr(config, "discover", lambda: [drop_in])
    c = caps()
    assert c.declares(drop_in, "vision") is False      # model support is unknown
    assert caps_mod.vision_backends(c) == []


def test_a_drop_in_can_declare_its_own_modalities(monkeypatch):
    seeing = backend(name="qwen-vl", model="qwen-vl-max",
                     base_url="https://example.invalid/v1")
    monkeypatch.setenv("ONE_CLI_VISION_BACKENDS", "qwen-vl")
    seeing.capability_binding = (seeing.base_url, seeing.model)
    seeing.declared_capabilities = ["vision"]
    monkeypatch.setattr(config, "discover", lambda **k: [seeing])
    assert caps_mod.vision_backends(caps()) == [seeing]


def test_a_named_backend_cannot_be_opted_in_without_model_evidence(monkeypatch):
    b = backend(name="gpt5", model="gpt-5", base_url="https://example.invalid/v1")
    monkeypatch.setattr(config, "discover", lambda **k: [b])
    monkeypatch.setenv("ONE_CLI_VISION_BACKENDS", "other,gpt5")
    assert caps_mod.vision_backends(caps()) == []


def test_explicit_base_url_resolves_named_vision_backends_without_local_discovery(
        png, tmp_path, monkeypatch):
    """SECURITY.md: an explicit --base-url bypasses local discovery."""
    (tmp_path / "seer.env").write_text(
        "ONE_PROVIDER_MODEL=seer-1\n"
        "OPENAI_BASE_URL=https://example.invalid/v1\n"
        "ONE_PROVIDER_MODALITIES=vision\n")
    monkeypatch.setenv(config.PROVIDER_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(config.LLAMACPP_HOSTS_ENV, "http://gpu-box:8911")
    monkeypatch.setenv("ONE_CLI_VISION_BACKENDS", "seer")
    asked = []
    monkeypatch.setattr(config, "_http_json", lambda url, timeout=4.0: asked.append(url))
    monkeypatch.setattr(eye, "describe_image", lambda path, mime, b, *a, **k:
                        f"A card, described by {b.name}.")
    explicit = config.resolve(base_url="http://explicit.test/v1", model="text-only")
    routed = router.route([png], explicit, caps=caps())
    assert "described by seer" in routed.preamble()
    assert asked == []


def test_modalities_come_off_the_drop_in_file(tmp_path, monkeypatch):
    (tmp_path / "seer.env").write_text(
        "ONE_PROVIDER_MODEL=seer-1\n"
        "OPENAI_BASE_URL=https://example.invalid/v1\n"
        "ONE_PROVIDER_MODALITIES=vision, audio\n")
    monkeypatch.setenv(config.PROVIDER_DIR_ENV, str(tmp_path))
    found = [b for b in config.discover(include_local=False) if b.name == "seer"]
    assert found and found[0].declared_capabilities == ["vision", "audio"]
    assert found[0].capability_binding == (found[0].base_url, found[0].model)


@pytest.mark.parametrize("answer", [
    "I'm unable to see the image you're referring to because you haven't provided it.",
    "I cannot see any image in this conversation.",
    "As an AI, I do not have access to images.",
    "ok",
])
def test_a_model_that_answered_without_the_image_fails_its_rung(png, monkeypatch, answer):
    """A fluent sentence about not seeing the picture is not a description."""
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(eye, "ocr", lambda path, c=None: "")
    monkeypatch.setattr(caps_mod, "vision_backends",
                        lambda c=None: [backend("blind"), backend("llamacpp/vl")])
    seen = []

    def describe(path, mime, b, prompt=eye.DESCRIBE_PROMPT, **kwargs):
        seen.append(b.name)
        return answer if b.name == "blind" else "A white card with black lettering."

    monkeypatch.setattr(eye, "describe_image", describe)
    out = eye.perceive(png, backend(), caps(tesseract=False))

    assert seen == ["blind", "llamacpp/vl"]        # it climbed rather than accepting
    assert "A white card with black lettering." in out.text
    blind_rung = [r for r in out.rungs if r.name == "describe:blind"][0]
    assert blind_rung.ok is False and "without seeing" in blind_rung.error


def test_every_describer_answering_blind_is_a_failure_not_a_description(png, monkeypatch):
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(eye, "ocr", lambda path, c=None: "")
    monkeypatch.setattr(caps_mod, "vision_backends", lambda c=None: [backend("blind")])
    monkeypatch.setattr(eye, "describe_image",
                        lambda *a, **k: "I am unable to see the image you sent.")
    out = eye.perceive(png, backend(), caps())
    assert out.ok is False
    assert "unable to see" not in out.text


def test_the_frame_budget_is_a_ceiling_not_a_quota():
    """A 3-second clip is not worth six stills - it cost 124s to prove it."""
    assert eye.frames_for(3) == 1
    assert eye.frames_for(0) == 1
    assert eye.frames_for(25) == 3
    assert eye.frames_for(600) == eye.FRAME_BUDGET
    assert eye.frames_for(600, budget=2) == 2


# --- a remote rung is reachable when the loopback probe is not --------------

def test_parakeet_falls_over_to_a_configured_remote_when_loopback_is_dead(monkeypatch):
    seen = []

    def health(url, timeout=2.0):
        seen.append(url)
        return url == caps_mod.PARAKEET_REMOTE

    monkeypatch.setattr(caps_mod, "PARAKEET_LOCAL", "http://local.test:9872")
    monkeypatch.setattr(caps_mod, "PARAKEET_REMOTE", "http://remote-host.test:9872")
    monkeypatch.setattr(caps_mod, "PARAKEET_RELAY", "http://local.test:9872")
    monkeypatch.setattr(caps_mod, "_health", health)
    monkeypatch.delenv("ONE_PARAKEET_URL", raising=False)
    caps_mod.reset()

    assert caps_mod._resolve_parakeet() is True
    assert seen == ["http://local.test:9872", caps_mod.PARAKEET_REMOTE]
    # Ear reads the module attribute, so the winner has to be written back.
    assert caps_mod.PARAKEET_RELAY == caps_mod.PARAKEET_REMOTE


def test_loopback_still_wins_when_it_answers(monkeypatch):
    seen = []

    def health(url, timeout=2.0):
        seen.append(url)
        return True

    monkeypatch.setattr(caps_mod, "PARAKEET_LOCAL", "http://local.test:9872")
    monkeypatch.setattr(caps_mod, "PARAKEET_REMOTE", "http://remote-host.test:9872")
    monkeypatch.setattr(caps_mod, "PARAKEET_RELAY", "http://stale")
    monkeypatch.setattr(caps_mod, "_health", health)
    monkeypatch.delenv("ONE_PARAKEET_URL", raising=False)
    caps_mod.reset()

    assert caps_mod._resolve_parakeet() is True
    assert seen == ["http://local.test:9872"]
    assert caps_mod.PARAKEET_RELAY == "http://local.test:9872"


def test_a_pinned_relay_url_is_not_second_guessed(monkeypatch):
    seen = []
    monkeypatch.setenv("ONE_PARAKEET_URL", "http://pinned:9999")
    monkeypatch.setattr(caps_mod, "PARAKEET_RELAY", "http://pinned:9999")
    monkeypatch.setattr(caps_mod, "_health",
                        lambda url, timeout=2.0: seen.append(url) or False)
    caps_mod.reset()

    # A person naming an endpoint means that endpoint. Trying another one
    # behind their back hides a typo behind a working transcript.
    assert caps_mod._resolve_parakeet() is False
    assert seen == ["http://pinned:9999"]
    assert caps_mod.PARAKEET_RELAY == "http://pinned:9999"


def test_the_parakeet_skip_reason_names_both_endpoints_it_tried(wav, monkeypatch):
    monkeypatch.setattr(ear, "_deepgram", lambda p, c: ("", ""))
    out = ear.transcribe(wav.path, caps(parakeet=False, deepgram_key=True))
    skipped = [r for r in out.rungs if r.name == "parakeet"]
    assert skipped and caps_mod.PARAKEET_LOCAL in skipped[0].skipped
    assert caps_mod.PARAKEET_REMOTE in skipped[0].skipped


def test_unreadable_image_becomes_a_loud_note_not_a_dead_turn(png, monkeypatch):
    """A text question died with its photo: no rung could
    read the image, so the whole turn was refused. The question is answerable;
    the model must be told the image was not read, naming the rungs tried."""
    monkeypatch.setattr(caps_mod, "_get_json", lambda *a, **k:
                        {"models": [{"model": "x", "capabilities": []}]})
    monkeypatch.setattr(caps_mod, "vision_backends", lambda c=None: [])
    monkeypatch.setattr(eye, "ocr", lambda path, c=None: "")
    routed = router.route([png], backend(), caps=caps(tesseract=True))
    note = routed.texts[0]
    assert note.startswith("[image NOT READ]")
    assert "declares no vision" in note and "ocr: failed" in note
    assert "could not read the image" in note
    assert not routed.inline

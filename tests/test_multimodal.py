"""Attachments: mime sniffing, both wire conversions, and the refusals.

Fixture bytes are generated here rather than committed: a 1x1 PNG and a
44-byte silent WAV are small enough to write inline, and a repo with binaries
in it cannot be reviewed in a diff.
"""

from __future__ import annotations

import base64
import json
import struct

import pytest

from onecli import media, session
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.openai_wire import OpenAIWire

# A real 1x1 PNG. Byte-for-byte valid, so the magic table is exercised rather
# than a stand-in that only happens to start with the right eight bytes.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def wav_bytes() -> bytes:
    """A header-only RIFF/WAVE file: enough to sniff, no samples needed."""
    return (b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + struct.pack("<I", 16)
            + struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8)
            + b"data" + struct.pack("<I", 0))


def mp4_bytes() -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16


def m4a_bytes() -> bytes:
    """An ISO-BMFF audio container: `ftyp` with the M4A major brand at offset 8.

    The head of a voice recording whose container is M4A while the file is
    named `.ogg`, as some messaging apps export long recordings.
    """
    return (b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00M4A isommp42"
            b"\x00\x00\x00\x01mdat")


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(PNG_BYTES)
    return str(p)


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "note.wav"
    p.write_bytes(wav_bytes())
    return str(p)


@pytest.fixture
def mp4(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(mp4_bytes())
    return str(p)


def test_mime_comes_from_the_bytes_not_the_extension(tmp_path):
    lying = tmp_path / "actually-a-png.jpg"
    lying.write_bytes(PNG_BYTES)
    att = media.load(str(lying))
    assert att.mime == "image/png"
    assert att.kind == media.IMAGE


def test_wav_and_mp4_are_sniffed_at_their_offset(wav, mp4):
    assert media.load(wav).mime == "audio/wav"
    assert media.load(mp4).mime == "video/mp4"


def test_an_m4a_voice_note_named_ogg_is_audio_not_video(tmp_path):
    """A voice recording named `.ogg` can hold M4A bytes.

    Refusing it as `video/mp4` costs the turn: a pipeline that has already
    failed to transcribe it hands the file on, and it is bounced on its
    container instead of being read as audio.
    """
    p = tmp_path / "memo.ogg"
    p.write_bytes(m4a_bytes())
    att = media.load(str(p), expect=media.AUDIO)
    assert att.kind == media.AUDIO
    assert att.mime == "audio/mp4"
    assert att.fmt == "mp4"


def test_an_m4a_voice_note_is_not_refused_by_the_audio_flag(tmp_path):
    """`--audio <file>.ogg` holding M4A bytes must not raise."""
    p = tmp_path / "recording.ogg"
    p.write_bytes(m4a_bytes())
    media.load(str(p), expect=media.AUDIO)   # raises MediaError before the fix


def test_a_video_brand_still_reads_as_video(tmp_path):
    """The audio brands must not swallow a real video container."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16)
    assert media.load(str(p)).kind == media.VIDEO


def test_unreadable_path_names_the_file(tmp_path):
    missing = str(tmp_path / "gone.png")
    with pytest.raises(media.MediaError) as exc:
        media.load(missing)
    assert missing in str(exc.value)


def test_empty_file_is_refused(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(media.MediaError) as exc:
        media.load(str(p))
    assert "empty file" in str(exc.value)


def test_flag_and_content_must_agree(wav):
    with pytest.raises(media.MediaError) as exc:
        media.load(wav, expect=media.IMAGE)
    assert "not image" in str(exc.value)


def test_encoding_is_base64_of_the_actual_file(png):
    att = media.load(png)
    assert base64.b64decode(att.b64()) == PNG_BYTES
    assert att.data_uri().startswith("data:image/png;base64,")


def test_canonical_block_carries_the_path_not_the_bytes(png):
    att = media.load(png)
    block = att.to_block()
    assert block == {"type": "media",
                     "media": {"kind": "image", "path": png, "mime": "image/png"}}
    assert "base64" not in json.dumps(block)
    assert media.Attachment.from_block(block) == att


def test_text_only_turn_keeps_the_plain_string_shape():
    assert media.user_content("hi", []) == "hi"


def test_openai_wire_converts_image_and_audio(png, wav):
    wire = OpenAIWire("http://x", "some-model")
    content = media.user_content("what is this",
                                 [media.load(png), media.load(wav)])
    out = wire.convert_messages([{"role": "user", "content": content}])
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "what is this"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks[2]["type"] == "input_audio"
    assert blocks[2]["input_audio"]["format"] == "wav"
    assert base64.b64decode(blocks[2]["input_audio"]["data"]) == wav_bytes()


def test_openai_wire_leaves_a_text_only_history_untouched():
    wire = OpenAIWire("http://x", "some-model")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert wire.convert_messages(msgs) == msgs


def test_openai_wire_coalesces_leading_system_messages():
    wire = OpenAIWire("http://x", "some-model")
    msgs = [
        {"role": "system", "content": "base"},
        {"role": "system", "content": "request-local"},
        {"role": "user", "content": "hello"},
    ]
    assert wire.convert_messages(msgs) == [
        {"role": "system", "content": "base\n\nrequest-local"},
        {"role": "user", "content": "hello"},
    ]


def test_openai_wire_strips_name_from_tool_messages():
    wire = OpenAIWire("http://x", "some-model")
    msgs = [
        {"role": "user", "content": "run bash"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "name": "bash", "content": "output"},
    ]
    out = wire.convert_messages(msgs)
    assert out[2] == {"role": "tool", "tool_call_id": "tc1", "content": "output"}
    assert "name" not in out[2]


def test_anthropic_wire_converts_an_image_to_a_base64_source(png):
    content = media.user_content("look", [media.load(png)])
    _, convo = AnthropicWire._split_system(
        [{"role": "user", "content": content}], backend="llama-server")
    blocks = convo[0]["content"]
    assert blocks[0] == {"type": "text", "text": "look"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["type"] == "base64"
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(blocks[1]["source"]["data"]) == PNG_BYTES


def test_anthropic_wire_refuses_audio_naming_modality_backend_and_wire(wav):
    content = media.user_content("transcribe", [media.load(wav)])
    with pytest.raises(media.MediaError) as exc:
        AnthropicWire._split_system([{"role": "user", "content": content}],
                                    backend="llama-server")
    msg = str(exc.value)
    assert "audio" in msg and "llama-server" in msg and "anthropic" in msg


def test_video_is_refused_on_both_wires(mp4):
    att = media.load(mp4)
    for wire in ("openai", "anthropic"):
        with pytest.raises(media.MediaError) as exc:
            media.check_supported([att], wire, "some-backend")
        assert "video" in str(exc.value) and wire in str(exc.value)


def test_session_round_trip_keeps_the_reference_and_re_reads_the_file(
    png, tmp_path, monkeypatch
):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    sid = session.new_id()
    content = media.user_content("what is this", [media.load(png)])
    session.append(sid, {"kind": "messages",
                         "messages": [{"role": "user", "content": content}]})

    raw = session.path_for(sid).read_text()
    assert "base64" not in raw
    assert png in raw

    restored = session.load_messages(sid)
    assert restored[0]["content"] == content
    # The bytes are only materialised on the way out, from the file on disk.
    wire = OpenAIWire("http://x", "some-model")
    blocks = wire.convert_messages(restored)[0]["content"]
    assert base64.b64decode(
        blocks[1]["image_url"]["url"].split(",", 1)[1]) == PNG_BYTES


def test_a_resume_whose_attachment_moved_says_which_file(png, tmp_path):
    att = media.load(png)
    (tmp_path / "shot.png").unlink()
    wire = OpenAIWire("http://x", "some-model")
    with pytest.raises(media.MediaError) as exc:
        wire.convert_messages([{"role": "user", "content": [att.to_block()]}])
    assert png in str(exc.value)

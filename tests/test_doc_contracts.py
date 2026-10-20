"""Shipped descriptions must match guarded search and perception dispatch."""
from pathlib import Path
from types import SimpleNamespace

from onecli import permissions, tools
from onecli.senses import eye

ROOT = Path(__file__).resolve().parents[1]


def test_agent_search_docs_match_guarded_python_dispatch(tmp_path, monkeypatch):
    guards = permissions.path_guards(permissions.chain(str(tmp_path)))
    assert any(isinstance(guard, permissions.RootsGuard) for guard in guards)
    (tmp_path / "note.txt").write_text("needle\n")

    def unguarded(*args, **kwargs):
        raise AssertionError("agent search must use the guarded Python path")

    monkeypatch.setattr(tools.REGISTRY["grep"], "fn", unguarded)
    monkeypatch.setattr(tools.REGISTRY["glob"], "fn", unguarded)
    for name, args in (("grep", {"pattern": "needle"}),
                       ("glob", {"pattern": "*.txt"})):
        out = tools.run(name, args, str(tmp_path), path_guards=guards)
        if name == "grep":
            assert "[python: guarded workspace]" in out
        assert "note.txt" in out
    description = tools.REGISTRY["grep"].description
    assert "bounded Python" in description
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert "agent `grep`/`glob` use bounded Python" in changelog


def test_capability_inventory_matches_image_only_inline_check(monkeypatch):
    calls = []
    caps = SimpleNamespace(ffmpeg=True, ffprobe=True,
                           declares=lambda backend, kind: calls.append(kind) or True)
    attachment = SimpleNamespace(path="clip.mp4", mime="video/mp4")
    monkeypatch.setattr(eye, "video_gate", lambda *args: "")
    monkeypatch.setattr(eye, "probe_video", lambda *args: {"format": {"duration": 1}})
    monkeypatch.setattr(eye, "sample_frames", lambda *args, **kwargs: ["frame.jpg"])
    monkeypatch.setattr(eye, "_describe_rung", lambda *args, **kwargs: "A sampled video frame.")
    monkeypatch.setattr(eye.ear, "discard_workdir", lambda *args: None)
    video = eye.perceive_video(attachment, backend=object(), caps=caps)
    assert video.text
    assert calls == []
    assert not any(rung.name == "inline" for rung in video.rungs)
    assert eye.perceive_image(attachment, backend=object(), caps=caps).inline
    assert calls == ["vision"]
    rows = []
    for path in ("SECURITY.md", "docs/reproducible-releases.md"):
        row = next(line for line in (ROOT / path).read_text().splitlines()
                   if line.startswith("| Attachment capability metadata |"))
        assert "Image attachment inline capability checks: GET" in row
        assert "video perception" not in row
        rows.append(row)
    assert rows[0] == rows[1]

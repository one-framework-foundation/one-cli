"""Temp-file leaks: gate files, staging dirs and senses work dirs."""
import os
import subprocess
from types import SimpleNamespace

import pytest

from onecli.sandbox import bounded as bounded_sandbox
from onecli.sandbox import quota as sandbox_quota
from onecli.sandbox import shell as tool_sandbox
from onecli.senses import ear, eye


@pytest.fixture
def tmpdir_(tmp_path, monkeypatch):
    t = tmp_path / "tmp"
    t.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(t))
    return t


def _roots(tmp_path):
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    (source / "f").write_text("x")
    return source, stage


def _boom(*a, **k):
    raise OSError("launch refused")


# --- sandbox_quota -------------------------------------------------------

needs_systemd = pytest.mark.skipif(
    not all(os.access(b, os.X_OK) for b in
            ("/usr/bin/systemd-run", "/usr/bin/systemctl", "/usr/bin/python3")),
    reason="quota executables unavailable")


@needs_systemd
def test_quota_cleanup_removes_gate(tmp_path, monkeypatch):
    control = tmp_path / "controls"
    launch = sandbox_quota.prepare(["/usr/bin/true"], control_root=control)
    assert len(list(control.iterdir())) == 1
    monkeypatch.setattr(sandbox_quota.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b""))
    launch.cleanup()
    assert list(control.iterdir()) == []


@needs_systemd
def test_quota_cleanup_removes_gate_even_when_stop_fails(tmp_path, monkeypatch):
    control = tmp_path / "controls"
    launch = sandbox_quota.prepare(["/usr/bin/true"], control_root=control)
    monkeypatch.setattr(sandbox_quota.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"bus"))
    with pytest.raises(sandbox_quota.QuotaError):
        launch.cleanup()
    assert list(control.iterdir()) == []


# --- tool_sandbox --------------------------------------------------------

def test_run_shell_launch_failure_leaves_no_staging(tmp_path, monkeypatch):
    source, stage = _roots(tmp_path)
    monkeypatch.setattr(tool_sandbox, "unavailable", lambda: "")
    monkeypatch.setattr(tool_sandbox.subprocess, "Popen", _boom)
    with pytest.raises(tool_sandbox.SandboxError):
        tool_sandbox.run_shell(source, stage, "true")
    assert list(stage.iterdir()) == []


def test_run_shell_result_can_be_released(tmp_path, monkeypatch):
    source, stage = _roots(tmp_path)
    monkeypatch.setattr(tool_sandbox, "unavailable", lambda: "")

    class Fake:
        pid = 2 ** 22 + 1
        returncode = 0

        def __init__(self):
            r, w = os.pipe()
            os.close(w)
            self.stdout = os.fdopen(r, "rb")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(tool_sandbox.subprocess, "Popen", lambda *a, **k: Fake())
    result = tool_sandbox.run_shell(source, stage, "true")
    assert result.workspace.is_dir()  # contract: retained for inspection
    tool_sandbox.release(result)
    assert list(stage.iterdir()) == []
    tool_sandbox.release(result)  # idempotent


def test_release_refuses_foreign_directory(tmp_path):
    victim = tmp_path / "workspace"
    victim.mkdir()
    with pytest.raises(tool_sandbox.SandboxError):
        tool_sandbox.release(tool_sandbox.SandboxResult(0, "", victim))
    assert victim.is_dir()


# --- bounded_sandbox -----------------------------------------------------

@needs_systemd
def test_bounded_launch_failure_leaves_nothing(tmp_path, monkeypatch):
    source, stage = _roots(tmp_path)
    monkeypatch.setattr(sandbox_quota.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b""))
    monkeypatch.setattr(bounded_sandbox.subprocess, "Popen", _boom)
    with pytest.raises(OSError):
        bounded_sandbox.run_bounded_shell(source, stage, "true")
    assert list(stage.rglob("*")) == []


# --- senses ------------------------------------------------------------

def _fake_ffmpeg(rc=0, write=True):
    def run(argv, **k):
        if write:
            open(argv[-1], "wb").write(b"x")
        return subprocess.CompletedProcess(argv, rc, b"", b"bad")
    return run


CAPS = SimpleNamespace(ffmpeg=True, ffprobe=True)


def test_extract_audio_failure_leaves_no_dir(tmpdir_, monkeypatch):
    monkeypatch.setattr(ear.subprocess, "run", _fake_ffmpeg(rc=1, write=False))
    with pytest.raises(RuntimeError):
        ear.extract_audio("v.mp4", CAPS)
    assert list(tmpdir_.iterdir()) == []


def test_extract_audio_timeout_leaves_no_dir(tmpdir_, monkeypatch):
    def hang(argv, **k):
        raise subprocess.TimeoutExpired(argv, 1)
    monkeypatch.setattr(ear.subprocess, "run", hang)
    with pytest.raises(subprocess.TimeoutExpired):
        ear.extract_audio("v.mp4", CAPS)
    assert list(tmpdir_.iterdir()) == []


def test_sample_frames_without_output_leaves_no_dir(tmpdir_, monkeypatch):
    monkeypatch.setattr(eye.subprocess, "run", _fake_ffmpeg(rc=1, write=False))
    assert eye.sample_frames("v.mp4", 10, budget=2) == []
    assert list(tmpdir_.iterdir()) == []


@pytest.mark.parametrize("describe_fails", [False, True])
def test_perceive_video_cleans_frames_and_wav(tmpdir_, monkeypatch, describe_fails):
    monkeypatch.setattr(eye.subprocess, "run", _fake_ffmpeg())
    monkeypatch.setattr(ear.subprocess, "run", _fake_ffmpeg())
    monkeypatch.setattr(eye, "video_gate", lambda p, c=None: "")
    monkeypatch.setattr(eye, "probe_video", lambda p, c=None: {
        "format": {"duration": "10"}, "streams": [{"codec_type": "audio"}]})

    def describe(out, frame, mime, caps, label, backend=None):
        assert os.path.isfile(frame)
        if describe_fails:
            raise RuntimeError("backend down")
        return "a cat"
    monkeypatch.setattr(eye, "_describe_rung", describe)
    monkeypatch.setattr(ear, "transcribe", lambda wav, caps=None, fallback=True: (
        os.path.isfile(wav) and ear.Outcome(modality="audio", text="hi")))
    att = SimpleNamespace(path="/x/v.mp4")
    if describe_fails:
        with pytest.raises(RuntimeError):
            eye.perceive_video(att, caps=CAPS)
    else:
        assert "a cat" in eye.perceive_video(att, caps=CAPS).text
    assert list(tmpdir_.iterdir()) == []


def test_multipart_filename_cannot_inject_headers(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def urlopen(req, timeout=None):
        seen["body"] = req.data
        seen["boundary"] = req.get_header("Content-type").split("boundary=")[1]
        return Resp()
    monkeypatch.setattr(ear.urllib.request, "urlopen", urlopen)
    evil = 'a"\r\nX-Evil: 1\r\n\r\nz.wav'
    ear._multipart("http://x", "file", evil, b"DATA", fields={'k"\r\nX-K: 1': "v"})
    body, boundary = seen["body"], seen["boundary"].encode()
    assert b"\r\nX-Evil" not in body and b"\r\nX-K" not in body
    # one field part + one file part + terminator, nothing smuggled in
    assert body.count(b"--" + boundary) == 3
    head = body.split(b"DATA")[0]
    assert head.count(b"\r\n") == 4 + 4  # field part (4) + file part (4)

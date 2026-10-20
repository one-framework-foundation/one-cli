"""Regression tests for the rc3 code audit. Each one failed before its fix."""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

from onecli import guard, hooks, jobs, permissions, render, terminal
from onecli.providers import ProviderError
from onecli.providers.anthropic_wire import AnthropicWire
from onecli.providers.openai_wire import OpenAIWire
from onecli.shellrun import run_shell
from onecli.tools import fs, shell

DENY_ENVELOPE = json.dumps({"hookSpecificOutput": {
    "hookEventName": "PreToolUse", "permissionDecision": "deny",
    "permissionDecisionReason": "no writes today"}})


def _printer(tmp_path, text: str) -> str:
    script = tmp_path / "hook.py"
    script.write_text(f"import sys; sys.stdin.read(); print({text!r})")
    return f"{sys.executable} {script}"


def test_pretooluse_hook_json_deny_on_exit_zero_blocks(tmp_path):
    spec = hooks.HookSpec(event=hooks.PRE_TOOL_USE, command=_printer(tmp_path, DENY_ENVELOPE))
    outcome = hooks.HookSet(specs=[spec], workspace=str(tmp_path)).pre_tool_use(
        "write", {"path": "a"}, mutating=True)
    assert outcome.blocked
    assert outcome.reason == "no writes today"


def test_subprocess_guard_json_deny_on_exit_zero_blocks(tmp_path):
    g = guard.SubprocessGuard([_printer(tmp_path, DENY_ENVELOPE)], workspace=str(tmp_path))
    verdict = g.check("write", {"path": "a"}, True)
    assert not verdict.allowed
    assert "no writes today" in verdict.reason


def test_path_rule_matches_through_symlinked_workspace(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    rule = permissions.Rule.parse("Write(.env)")
    assert rule.matches("write", {"path": ".env"}, str(link))


def test_edit_keeps_crlf_and_non_utf8_bytes(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"caf\xe9\r\nold\r\nend\r\n")
    assert fs.edit(str(tmp_path), "a.txt", "old", "new").startswith("edited")
    assert p.read_bytes() == b"caf\xe9\r\nnew\r\nend\r\n"


def test_multiline_lf_edit_matches_a_crlf_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    assert fs.edit(str(tmp_path), "a.txt", "one\ntwo", "uno\ndos").startswith("edited")
    assert p.read_bytes() == b"uno\r\ndos\r\nthree\r\n"


def test_edit_with_unencodable_text_leaves_file_intact(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("keep me")
    assert fs.edit(str(tmp_path), "a.txt", "keep", "\udc80").startswith("error")
    assert p.read_text() == "keep me"


def test_edit_refuses_an_empty_old_string(tmp_path):
    # "" occurs between every character: counted, it reported 5 replacements
    # on "abc\n" and spliced new_string through the whole file.
    p = tmp_path / "a.txt"
    p.write_text("abc\n")
    out = fs.edit(str(tmp_path), "a.txt", "", "X", replace_all=True)
    assert out.startswith("error") and "empty" in out
    assert p.read_text() == "abc\n"


def test_write_with_unencodable_content_leaves_file_intact(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("keep me")
    out = fs.write(str(tmp_path), "a.txt", "bad \ud800")
    assert out.startswith("error")
    assert p.read_text() == "keep me"


def test_bash_tool_survives_non_utf8_output(tmp_path):
    out = shell.bash(str(tmp_path), "printf 'a\\xffb'")
    assert out.startswith("exit 0")
    assert "a" in out and "b" in out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_run_shell_timeout_kills_background_children(tmp_path):
    pidfile = tmp_path / "pid"
    with pytest.raises(subprocess.TimeoutExpired):
        run_shell(f"sleep 30 & echo $! > {pidfile}; sleep 30", timeout=1,
                  cwd=str(tmp_path))
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 3
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid), "background child outlived the timeout"


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: dict = {}

    def do_GET(self):
        _Recorder.seen = dict(self.headers)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


def _redirect_to(target: str):
    class Redirector(_Recorder):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
    return Redirector


def _serve(handler) -> http.server.HTTPServer:
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("cross_origin", [True, False])
def test_redirect_drops_credentials_only_across_origins(cross_origin):
    end = _serve(_Recorder)
    # Another port is another origin; the same server redirecting to itself is not.
    start = _serve(_redirect_to(f"http://127.0.0.1:{end.server_port}/end")) if cross_origin else end
    _Recorder.seen = {}
    try:
        path = "/start" if cross_origin else "/end"
        req = urllib.request.Request(
            f"http://127.0.0.1:{start.server_port}{path}",
            headers={"Authorization": "Bearer secret", "x-api-key": "secret", "X-Keep": "1"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.read() == b"ok"
    finally:
        for server in {start, end}:
            server.shutdown()
    names = {k.lower() for k in _Recorder.seen}
    if cross_origin:
        assert not {"authorization", "x-api-key", "x-keep"} & names
    else:
        assert {"authorization", "x-api-key", "x-keep"} <= names


def test_default_port_is_the_same_origin():
    from onecli import netio
    assert netio._origin("https://h/x") == netio._origin("https://H:443/y")
    assert netio._origin("http://h") != netio._origin("http://h:8080")


@pytest.mark.parametrize("wire", [OpenAIWire, AnthropicWire])
@pytest.mark.parametrize("data", [None, [], "x"])
def test_non_object_reply_is_a_provider_error(wire, data):
    with pytest.raises(ProviderError):
        wire._to_turn(data)


def test_openai_null_choice_is_a_provider_error():
    with pytest.raises(ProviderError):
        OpenAIWire._to_turn({"choices": [None]})


def test_c1_controls_are_stripped_from_model_text():
    assert render.untrusted("a\u009b2Jb\u009d0;t\u0007c") == "a2Jb0;tc"
    assert terminal.clean("a\u009b2Jb") == "a2Jb"


def test_damaged_job_record_does_not_break_the_store(tmp_path):
    root, tmp_path = tmp_path, tmp_path / "s1"
    tmp_path.mkdir()
    (tmp_path / "j1.json").write_text("[]")
    (tmp_path / "j2.json").write_text(json.dumps({"command": "x", "started": None}))
    (tmp_path / "j3.json").write_text(json.dumps({"command": "ok", "exit_code": 0}))
    store = jobs.JobStore("s1", root=root)
    assert list(store.jobs) == ["j3"]


def test_relay_rejects_malformed_bodies_with_400(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from onecli.relay import server
    monkeypatch.setitem(server._TOKENS, "t", "tester")
    client = TestClient(server.app, raise_server_exceptions=False)
    auth = {"Authorization": "Bearer t", "Content-Type": "application/json"}
    for path in ("/v1/chat/completions", "/v1/messages"):
        assert client.post(path, content=b"{not json", headers=auth).status_code == 400
        assert client.post(path, content=b"[]", headers=auth).status_code == 400
        for tokens in (b'"abc"', b"1e999"):
            bad = b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": ' + tokens + b"}"
            assert client.post(path, content=bad, headers=auth).status_code == 400

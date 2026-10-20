"""Regression tests for the sixth v0.1.1 release-review round."""
import argparse
import time
import types

import pytest

from onecli import cli, commands, permissions
from onecli import jobs as joblib
from onecli import tools as toolkit
from onecli.agent import Agent
from onecli.providers.base import ProviderError, ToolCall, Turn
from onecli.tasks import TaskManager, build_runner_cmd, session_options

# OSC 52 writes the clipboard, CSI 2J clears the screen.
EVIL = "echo hi \x1b]52;c;ZXZpbA==\x07\x1b[2J"


def _clean(text: str) -> bool:
    # Our own colour codes may remain; the model's control sequences may not.
    return "\x1b]52" not in text and "\x1b[2J" not in text and "\x07" not in text


# --- 1. detached workers inherit the session hooks ---------------------------

def _args(**kw):
    base = dict(guard=[], allowed_tools=[], disallowed_tools=[], add_dir=[],
                guard_timeout=None, hooks=[], trust_project_config=False,
                no_hooks=False, workspace="", allow_detached=False)
    return argparse.Namespace(**{**base, **kw})


def test_worker_args_carry_hooks_trust_and_no_hooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hooks.json").write_text('{"hooks": {}}')
    opts = session_options(_args(hooks=["hooks.json"], trust_project_config=True,
                                 workspace=str(tmp_path)), "default")
    worker = opts["worker_args"]
    assert worker[worker.index("--hooks") + 1] == str(tmp_path / "hooks.json")
    assert "--trust-project-config" in worker and not opts["refusal"]
    assert "--no-hooks" in session_options(_args(no_hooks=True), "default")["worker_args"]
    # The worker's own parser accepts exactly what was forwarded.
    cmd, _ = build_runner_cmd("one", "x", workspace=str(tmp_path), worker_args=worker)
    parsed = cli.build_parser().parse_args(cmd[cmd.index("x"):])
    assert parsed.hooks == [str(tmp_path / "hooks.json")] and parsed.trust_project_config


def test_a_hooks_source_the_worker_cannot_reopen_refuses_the_spawn(tmp_path):
    opts = session_options(_args(hooks=[str(tmp_path / "gone.json")],
                                 workspace=str(tmp_path)), "default")
    manager = TaskManager(execute=True, workspace=str(tmp_path), root=tmp_path / "tasks", **opts)
    with pytest.raises(RuntimeError, match="cannot forward --hooks"):
        manager.spawn(title="t", prompt="t")
    assert not (tmp_path / "tasks").exists()


# --- 2. job output past the first MiB ----------------------------------------

def _settle(job, store, limit=10.0):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        store.poll()
        if not job.running:
            return True
        time.sleep(0.02)
    return False


def test_job_output_reaches_lines_past_the_scan_cap(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start("true", str(tmp_path))
    assert _settle(job, store)
    lines = 30000
    job.log.write_bytes(b"".join(b"line %06d %s\n" % (i, b"x" * 40) for i in range(1, lines + 1)))
    assert job.log.stat().st_size > joblib.OUTPUT_CAP
    tail = store.output(job.id, offset=lines - 3, limit=3)
    assert "line 029998" in tail and "line 030000" in tail
    assert "no output yet" not in tail and "truncated" not in tail
    assert "past the end" in store.output(job.id, offset=lines + 5)
    # The cap bounds the returned window and says so.
    window = store.output(job.id, offset=10, limit=lines)
    assert "line 000011" in window and "scan cap" in window


# --- 3. '**/' is whole directories only -------------------------------------

PATHS = {"src/config.py": True, "src/a/b/config.py": True,
         "src/notconfig.py": False, "src/a/evilconfig.py": False}


def test_double_star_slash_deny_matches_whole_directories_only(tmp_path):
    guard = permissions.chain(str(tmp_path), "default", disallowed=["Read(src/**/config.py)"])
    for path, hit in PATHS.items():
        assert guard.check("read", {"path": path}, False).allowed is not hit, path


def test_double_star_slash_allow_matches_whole_directories_only(tmp_path):
    guard = permissions.chain(str(tmp_path), "default", allowed=["Write(src/**/config.py)"])
    for path, hit in PATHS.items():
        assert guard.check("write", {"path": path, "content": ""}, False).allowed is hit, path


# --- 4. a symlinked workspace -----------------------------------------------

class ScriptedProvider:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        if not self.turns:
            raise ProviderError("script exhausted")
        return self.turns.pop(0)


def test_cli_resolves_a_symlinked_workspace(tmp_path, monkeypatch):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    seen = []

    def stop(args):
        seen.append(args.workspace)
        raise ValueError("stop")

    monkeypatch.setattr(cli, "_mask_options", stop)
    assert cli.main(["hi", "-w", str(tmp_path / "link")]) == 2
    assert seen == [str((tmp_path / "real").resolve())]


def test_chat_resolves_a_symlinked_workspace(tmp_path):
    from onecli.chat import Chat
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    backend = types.SimpleNamespace(name="fake", model="m", display="fake",
                                    base_url="", wire="openai")
    chat = Chat(backend, workspace=str(tmp_path / "link"))
    assert chat.workspace == str((tmp_path / "real").resolve())


def test_tools_work_in_a_symlinked_workspace(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("needle here\n")
    link = tmp_path / "link"
    link.symlink_to(real)
    calls = [("write", {"path": "b.txt", "content": "needle two\n"}),
             ("read", {"path": "a.txt"}), ("ls", {"path": "."}),
             ("grep", {"pattern": "needle"}), ("glob", {"pattern": "*.txt"})]
    provider = ScriptedProvider([Turn(tool_calls=[ToolCall(id=f"t{i}", name=n, arguments=a)])
                                 for i, (n, a) in enumerate(calls)] + [Turn(text="done")])
    # A deny rule makes grep and glob take the guarded in-process walk.
    guard = permissions.chain(str(link), "default", disallowed=["Read(secret.env)"])
    agent = Agent(provider, str(link), guard=guard)
    assert agent.workspace == str(real.resolve())
    result = agent.run("go")
    outputs = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert len(outputs) == 5
    write, read, ls, grep, glob = outputs
    assert write.startswith("created") and "needle here" in read
    assert "a.txt" in ls and "b.txt" in ls
    assert "a.txt" in grep and "b.txt" in grep
    assert "a.txt" in glob and "b.txt" in glob


def test_a_search_root_the_walk_cannot_open_is_an_error(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("needle\n")
    (tmp_path / "link").symlink_to(real)
    guards = permissions.path_guards(permissions.chain(str(tmp_path), "default",
                                                       disallowed=["Read(secret.env)"]))
    for name, arguments in (("grep", {"pattern": "needle", "path": "link"}),
                            ("glob", {"pattern": "*.txt", "path": "link"})):
        out = toolkit.run(name, arguments, str(tmp_path), path_guards=guards)
        assert out.startswith("error: cannot open search path"), out


# --- 5. survivors printed at exit -------------------------------------------

class _Store:
    def close(self):
        return [f"j1 {EVIL}"]


def test_close_jobs_strips_terminal_control(monkeypatch, capsys):
    monkeypatch.setattr(cli.shelltool, "reap", lambda: [EVIL])
    cli._close_jobs(_Store())
    err = capsys.readouterr().err
    assert "stopped at exit" in err and _clean(err)


def test_one_shot_exit_strips_terminal_control(monkeypatch, tmp_path, capsys):
    class FakeAgent:
        def __init__(self, *_, **kw):
            self.chain = types.SimpleNamespace(current=types.SimpleNamespace(name="fake"))

        def run(self, *a, **k):
            return types.SimpleNamespace(text="ok", messages=[], steps=[], tool_calls=0,
                                         cost={}, stopped="answered", error=None)

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    monkeypatch.setattr(cli, "_context_settings", lambda a, b: {})
    monkeypatch.setattr(cli, "_hooks", lambda *a, **k: None)
    monkeypatch.setattr(cli.session, "append", lambda *a, **k: None)
    monkeypatch.setattr(cli.joblib, "JobStore", lambda sid: _Store())
    monkeypatch.setattr(cli.shelltool, "reap", lambda: [EVIL])
    monkeypatch.setattr(cli.spilllib, "SPILL_DIR", tmp_path / "spill")
    args = cli.build_parser().parse_args(["hi", "-w", str(tmp_path), "--quiet"])
    backend = types.SimpleNamespace(name="fake", model="m", display="fake",
                                    base_url="", wire="openai")
    try:
        cli.run_once(args, backend, None, "sid-r6")
    except Exception:  # noqa: BLE001 - only the exit prints matter here
        pass
    err = capsys.readouterr().err
    assert "job stopped at exit" in err and "background process stopped at exit" in err
    assert _clean(err)


# --- 6. slash commands that show job and worker text ------------------------

def test_ps_and_task_log_strip_terminal_control(tmp_path):
    store = joblib.JobStore("s", tmp_path)
    job = store.start(EVIL, str(tmp_path))
    assert _settle(job, store)
    job.log.write_text(EVIL + "\n")
    task = types.SimpleNamespace(output=lambda: EVIL)
    tasks = types.SimpleNamespace(poll=lambda: None, get=lambda tid: task,
                                  list_all=lambda: [types.SimpleNamespace(task_id="1")])
    repl = commands.Repl.__new__(commands.Repl)
    repl.jobs, repl.tasks = store, tasks
    repl.args = argparse.Namespace(workspace=str(tmp_path))
    for line in ("/ps", f"/ps {job.id}", "/task log 1"):
        out = commands.dispatch(line, repl).text
        assert "echo hi" in out and _clean(out), line


def test_mcp_failure_reason_strips_terminal_control():
    client = types.SimpleNamespace(servers=[], failures=[("srv", EVIL)])
    repl = commands.Repl.__new__(commands.Repl)
    repl.mcp_client = client
    out = commands.dispatch("/mcp", repl).text
    assert "echo hi" in out and _clean(out)


# --- 7. the not-found hint hides denied names -------------------------------

def test_not_found_hint_omits_names_a_read_rule_denies(tmp_path):
    (tmp_path / "credentials.env").write_text("SECRET=1\n")
    (tmp_path / "credentials.txt").write_text("public\n")
    guards = permissions.path_guards(permissions.chain(
        str(tmp_path), "default", disallowed=["Read(credentials.env)"]))
    for name in ("read", "edit"):
        arguments = {"path": "credential.env"}
        if name == "edit":
            arguments.update(old_string="a", new_string="b")
        out = toolkit.run(name, arguments, str(tmp_path), path_guards=guards)
        assert "no such file" in out and "credentials.txt" in out, out
        assert "credentials.env" not in out, out
    ls = toolkit.run("ls", {"path": "credential.env"}, str(tmp_path), path_guards=guards)
    assert "credentials.env" not in ls and "credentials.txt" in ls

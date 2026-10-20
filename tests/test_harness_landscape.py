"""Offline tests for the harnesses added by the landscape survey.

Same rules as test_arena.py: no network, no binaries. What is asserted is that
each adapter points its harness at THIS endpoint with THIS key, writes its
config inside the cell's private dir, and reads back what the harness prints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onecli import harnesses  # noqa: E402
from onecli.config import WIRE_ANTHROPIC, WIRE_OPENAI, Backend  # noqa: E402


def _cell(tmp_path: Path, wire: str = WIRE_OPENAI, **kw) -> harnesses.Cell:
    backend = Backend(name="lane", model="m-1", base_url="https://example.test/v1",
                      wire=wire, api_key="sk-test")
    workspace = tmp_path / "ws"
    cfg = tmp_path / "cfg"
    workspace.mkdir(parents=True, exist_ok=True)
    cfg.mkdir(parents=True, exist_ok=True)
    return harnesses.Cell(backend=backend, model_id=kw.pop("model_id", ""),
                          workspace=workspace, prompt="do the thing",
                          cfg_dir=cfg, **kw)


# --------------------------------------------------------------------------
# every new adapter points at your configured endpoint and nowhere else
# --------------------------------------------------------------------------

def test_every_adapter_builds_a_launch_for_our_endpoint(tmp_path):
    """The one property that makes a row comparable: our base_url, our key."""
    for name in ("aider", "goose", "crush", "gptme", "mini", "droid"):
        cell = _cell(tmp_path / name)
        launch = harnesses.HARNESSES[name].build(cell)
        blob = " ".join(launch.cmd) + json.dumps(launch.env or {})
        for path in sorted((tmp_path / name / "cfg").rglob("*")):
            if path.is_file():
                blob += path.read_text(errors="ignore")
        # goose is the one that splits the host from the path; every other
        # adapter carries the base_url whole.
        assert "https://example.test" in blob, name
        assert "sk-test" in blob, name
        assert launch.env["HOME"] == str(tmp_path / name / "cfg"), name


def test_aider_prefixes_the_model_per_wire(tmp_path):
    openai = harnesses._aider_build(_cell(tmp_path / "o"))
    assert "openai/m-1" in openai.cmd
    assert openai.env["OPENAI_API_BASE"] == "https://example.test/v1"
    anthropic = harnesses._aider_build(_cell(tmp_path / "a", wire=WIRE_ANTHROPIC))
    assert "anthropic/m-1" in anthropic.cmd
    assert anthropic.env["ANTHROPIC_API_BASE"] == "https://example.test/v1"
    assert "--yes-always" in openai.cmd and "--no-auto-commits" in openai.cmd


def test_goose_splits_host_from_path(tmp_path):
    """goose joins OPENAI_HOST and OPENAI_BASE_PATH itself, so /v1 must come off
    the host or the request lands on /v1/v1/chat/completions."""
    launch = harnesses._goose_build(_cell(tmp_path))
    assert launch.env["OPENAI_HOST"] == "https://example.test"
    assert launch.env["OPENAI_BASE_PATH"] == "v1/chat/completions"
    assert launch.env["GOOSE_MODEL"] == "m-1"
    assert "--no-session" in launch.cmd and "json" in launch.cmd


def test_goose_uses_the_anthropic_host_on_the_anthropic_wire(tmp_path):
    launch = harnesses._goose_build(_cell(tmp_path, wire=WIRE_ANTHROPIC))
    assert launch.env["GOOSE_PROVIDER"] == "anthropic"
    assert launch.env["ANTHROPIC_HOST"] == "https://example.test/v1"


def test_crush_config_is_a_directory_with_a_pinned_model(tmp_path):
    launch = harnesses._crush_build(_cell(tmp_path))
    cfg_home = Path(launch.env["CRUSH_GLOBAL_CONFIG"])
    assert cfg_home.is_dir()
    cfg = json.loads((cfg_home / "crush.json").read_text())
    assert cfg["providers"]["onelane"]["base_url"] == "https://example.test/v1"
    assert cfg["providers"]["onelane"]["type"] == "openai-compat"
    # Without this map crush silently prefers a catwalk provider over ours.
    assert cfg["models"]["large"] == {"model": "m-1", "provider": "onelane"}
    # ... and its own state must not land in the workspace, where the diff
    # report would count it as the agent touching files.
    assert "--data-dir" in launch.cmd


def test_crush_switches_provider_type_on_the_anthropic_wire(tmp_path):
    launch = harnesses._crush_build(_cell(tmp_path, wire=WIRE_ANTHROPIC))
    cfg = json.loads((Path(launch.env["CRUSH_GLOBAL_CONFIG"]) / "crush.json").read_text())
    assert cfg["providers"]["onelane"]["type"] == "anthropic"


def test_gptme_is_non_interactive_and_json(tmp_path):
    launch = harnesses._gptme_build(_cell(tmp_path))
    assert "-n" in launch.cmd and "json" in launch.cmd
    assert launch.env["OPENAI_BASE_URL"] == "https://example.test/v1"
    assert "openai/m-1" in launch.cmd


def test_gptme_read_only_drops_the_writing_tools(tmp_path):
    launch = harnesses._gptme_build(_cell(tmp_path, read_only=True))
    assert "read,shell" in launch.cmd


def test_mini_keeps_the_packaged_config_and_ignores_pricing(tmp_path):
    launch = harnesses._mini_build(_cell(tmp_path))
    assert "mini.yaml" in launch.cmd, "-c replaces the default config file"
    assert "model.model_kwargs.api_base=https://example.test/v1" in launch.cmd
    # custom endpoints are not in litellm's price table, a hard error by default
    assert launch.env["MSWEA_COST_TRACKING"] == "ignore_errors"


def test_droid_writes_a_byok_model_into_a_private_factory_home(tmp_path):
    launch = harnesses._droid_build(_cell(tmp_path))
    settings = json.loads((tmp_path / "cfg/.factory/settings.json").read_text())
    model = settings["customModels"][0]
    assert model["baseUrl"] == "https://example.test/v1"
    assert model["provider"] == "generic-chat-completion-api"
    # the index in customModels is part of the id droid exec expects
    assert "custom:onelane-0" in launch.cmd


def test_model_id_override_reaches_the_new_adapters(tmp_path):
    for name in ("aider", "goose", "crush", "gptme", "mini", "droid"):
        cell = _cell(tmp_path / name, model_id="other-model")
        launch = harnesses.HARNESSES[name].build(cell)
        blob = " ".join(launch.cmd) + json.dumps(launch.env or {})
        for path in sorted((tmp_path / name / "cfg").rglob("*")):
            if path.is_file():
                blob += path.read_text(errors="ignore")
        assert "other-model" in blob, name


def test_no_adapter_inherits_the_operators_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leaked")
    monkeypatch.setenv("GROQ_API_KEY", "leaked")
    for name in ("aider", "goose", "crush", "gptme", "mini", "droid"):
        launch = harnesses.HARNESSES[name].build(_cell(tmp_path / name))
        assert launch.env.get("GROQ_API_KEY") is None, name
        assert launch.env.get("ANTHROPIC_API_KEY") in (None, "sk-test"), name


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------

def test_goose_parse_reads_the_assistant_answer_and_tools():
    blob = json.dumps({"messages": [
        {"role": "assistant", "content": [{"type": "toolRequest"}]},
        {"role": "user", "content": [{"type": "toolResponse"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "svc/pricing.py:apply_vat"}]},
    ]})
    result = harnesses._goose_parse(blob, "")
    assert result.text == "svc/pricing.py:apply_vat"
    assert result.steps == 2 and result.tool_calls == 1


def test_goose_parse_falls_back_to_prose():
    result = harnesses._goose_parse("just words", "")
    assert result.text == "just words"


def test_gptme_parse_takes_the_last_assistant_message():
    stream = "\n".join([
        json.dumps({"role": "assistant", "content": "thinking"}),
        json.dumps({"role": "system", "content": "tool output"}),
        json.dumps({"role": "assistant", "content": "svc/pricing.py:apply_vat"}),
    ])
    result = harnesses._gptme_parse(stream, "")
    assert result.text == "svc/pricing.py:apply_vat"
    assert result.steps == 2 and result.tool_calls == 1


def test_droid_parse_reads_the_result_event():
    stream = "\n".join([
        json.dumps({"type": "assistant"}),
        json.dumps({"type": "tool_use"}),
        json.dumps({"type": "result", "result": "svc/pricing.py:apply_vat"}),
    ])
    result = harnesses._droid_parse(stream, "")
    assert result.text == "svc/pricing.py:apply_vat"
    assert result.steps == 1 and result.tool_calls == 1


def test_a_crash_is_reported_as_a_crash():
    result = harnesses._plain_parse("", "boom")
    assert result.stopped == "crash" and result.error == "boom"


# --------------------------------------------------------------------------
# the skip table is a fact about the harness, never a zero score
# --------------------------------------------------------------------------

def test_skipped_harnesses_each_carry_a_reason():
    assert set(harnesses.SKIPPED) == {"amp", "auggie", "cursor-agent", "gemini",
                                      "grok", "forge", "octofriend", "plandex",
                                      "swe-agent"}
    for name, reason in harnesses.SKIPPED.items():
        assert len(reason) > 40, name
    assert set(harnesses.SKIPPED) & set(harnesses.HARNESSES) == set()


def test_the_private_prefix_is_searched_before_the_operators_path():
    assert harnesses.EXTRA_BINS[0].name == "bin"
    assert harnesses.EXTRA_BINS[0].parent.name == ".harnesses"


# --- hermes (NousResearch/hermes-agent) ------------------------------------


def test_hermes_writes_a_private_home_with_our_provider(tmp_path):
    launch = harnesses.HARNESSES["hermes"].build(_cell(tmp_path))
    home = Path(launch.env["HERMES_HOME"])
    assert home.is_relative_to(tmp_path)
    cfg = (home / "config.yaml").read_text()
    assert "api: https://example.test/v1" in cfg
    assert "transport: openai_chat" in cfg
    assert "key_env: ONELANE_KEY" in cfg
    # Memory and the user profile are per-HOME state: left on, a cell would
    # depend on which cells ran before it in the same dir.
    assert "memory_enabled: false" in cfg


def test_hermes_switches_transport_on_the_anthropic_wire(tmp_path):
    launch = harnesses.HARNESSES["hermes"].build(_cell(tmp_path, wire=WIRE_ANTHROPIC))
    cfg = (Path(launch.env["HERMES_HOME"]) / "config.yaml").read_text()
    assert "transport: anthropic_messages" in cfg


def test_hermes_is_one_shot_and_unattended(tmp_path):
    launch = harnesses.HARNESSES["hermes"].build(_cell(tmp_path))
    assert "-z" in launch.cmd
    # `-z` prints the final response and nothing else: no banner, no spinner,
    # no session line. Without --yolo it stops at the first approval prompt
    # with no tty to answer it.
    assert "--yolo" in launch.cmd
    assert launch.cmd[launch.cmd.index("-z") + 1] == "do the thing"


def test_hermes_read_only_blocks_writes_without_changing_modes(tmp_path, monkeypatch):
    import shutil
    import subprocess
    import pytest

    if not shutil.which("bwrap"):
        pytest.skip("requires bubblewrap")
    cell = _cell(tmp_path, read_only=True)
    existing = cell.workspace / "existing.txt"
    existing.write_text("before")
    existing.chmod(0o640)
    original_mode = cell.workspace.stat().st_mode
    stub = tmp_path / "hermes"
    stub.write_text("#!/bin/sh\n"
                    "cat existing.txt\n"
                    "chmod u+w . existing.txt 2>/dev/null\n"
                    "echo changed > existing.txt\n"
                    "echo created > newfile.txt\n"
                    "echo config-ok > \"$HOME/state\"\n"
                    "mktemp > \"$HOME/temp-path\"\n")
    stub.chmod(0o755)
    real_which = harnesses._which
    monkeypatch.setattr(harnesses, "_which", lambda name:
                        str(stub) if name == "hermes" else real_which(name))
    launch = harnesses.HARNESSES["hermes"].build(cell)
    proc = subprocess.run(launch.cmd, cwd=launch.cwd, env=launch.env,
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert "before" in proc.stdout
    assert (cell.cfg_dir / "state").read_text() == "config-ok\n"
    temp = Path((cell.cfg_dir / "temp-path").read_text().strip())
    assert temp.is_relative_to(cell.cfg_dir) and temp.is_file()
    assert existing.read_text() == "before"
    assert not (cell.workspace / "newfile.txt").exists()
    assert existing.stat().st_mode & 0o777 == 0o640
    assert cell.workspace.stat().st_mode == original_mode



def test_hermes_takes_the_system_prompt_as_its_identity(tmp_path):
    launch = harnesses.HARNESSES["hermes"].build(
        _cell(tmp_path, system="you are terse"))
    assert (Path(launch.env["HERMES_HOME"]) / "SOUL.md").read_text() == "you are terse"


# --- claude (Claude Code itself, pointed at a non-Anthropic endpoint) -----


def test_claude_points_at_our_endpoint_on_the_anthropic_wire(tmp_path):
    launch = harnesses.HARNESSES["claude"].build(_cell(tmp_path, wire=WIRE_ANTHROPIC))
    assert launch.env["ANTHROPIC_BASE_URL"] == "https://example.test/v1"
    assert launch.env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
    assert launch.env["ANTHROPIC_MODEL"] == "m-1"
    assert launch.cmd[-1] == "do the thing"


def test_claude_takes_the_system_prompt_via_its_own_flag(tmp_path):
    launch = harnesses.HARNESSES["claude"].build(
        _cell(tmp_path, wire=WIRE_ANTHROPIC, system="you are terse"))
    assert launch.cmd[launch.cmd.index("--system-prompt") + 1] == "you are terse"


def test_claude_read_only_restricts_the_tool_set_and_still_finds_the_prompt(tmp_path):
    """`--tools`, like `--allowedTools`, is a variadic flag: with no `--`
    terminator it swallows the trailing prompt argument and claude exits 1
    with "no prompt provided" instead of running the cell."""
    launch = harnesses.HARNESSES["claude"].build(
        _cell(tmp_path, wire=WIRE_ANTHROPIC, read_only=True))
    tools_idx = launch.cmd.index("--tools")
    assert launch.cmd[tools_idx + 1] == "Read,Grep,Glob"
    assert launch.cmd[tools_idx + 2] == "--"
    assert launch.cmd[-1] == "do the thing"


def test_claude_gets_the_task_step_budget_and_no_ambient_mcp_tools(tmp_path):
    """Without --strict-mcp-config every MCP server registered on the host
    loaded into the cell, and without --max-turns
    claude ran uncapped while `one` stopped at the task budget."""
    launch = harnesses.HARNESSES["claude"].build(_cell(tmp_path, wire=WIRE_ANTHROPIC))
    assert "--strict-mcp-config" in launch.cmd
    turns = launch.cmd.index("--max-turns")
    assert launch.cmd[turns + 1] == "30"
    assert launch.cmd[-1] == "do the thing"


def test_claude_streams_json_so_the_parser_sees_every_turn(tmp_path):
    """`--output-format json` only reports the final result: tool_calls was
    always 0 and tokens covered only the last turn. stream-json exposes every
    assistant event along the way."""
    launch = harnesses.HARNESSES["claude"].build(_cell(tmp_path, wire=WIRE_ANTHROPIC))
    assert "--output-format" in launch.cmd
    fmt = launch.cmd.index("--output-format")
    assert launch.cmd[fmt + 1] == "stream-json"
    assert "--verbose" in launch.cmd


def _stream_json(*events):
    return "\n".join(json.dumps(e) for e in events)


def test_claude_parse_counts_tool_calls_and_usage_across_every_turn():
    stdout = _stream_json(
        {"type": "assistant", "message": {
            "content": [{"type": "tool_use", "name": "Read"},
                        {"type": "text", "text": "..."}],
            "usage": {"input_tokens": 10, "output_tokens": 5,
                      "cache_read_input_tokens": 100}}},
        {"type": "assistant", "message": {
            "content": [{"type": "tool_use", "name": "Grep"}],
            "usage": {"input_tokens": 8, "output_tokens": 3,
                      "cache_read_input_tokens": 110}}},
        {"type": "result", "result": "ok", "num_turns": 3, "is_error": False},
    )
    res = harnesses._claude_parse(stdout, "")
    assert res.text == "ok"
    assert res.steps == 3
    assert res.tool_calls == 2
    assert res.stopped == "answered"
    assert res.cost == {"input_tokens": 18, "output_tokens": 8,
                        "cache_read_input_tokens": 210}


def test_claude_parse_takes_tokens_from_the_result_event_of_real_output():
    """Real Claude Code stream-json from a fair 3-way bench
    bugfix cell). Every `assistant` event carries output_tokens 0, and one
    message repeats its usage once per content block, so summing them gave
    output 0 and input 7,358. The `result` event holds the true totals."""
    stdout = (Path(__file__).parent / "fixtures" / "claude_stream_json_bugfix.jsonl").read_text()
    res = harnesses._claude_parse(stdout, "")
    assert res.stopped == "answered"
    assert res.steps == 4
    assert res.cost == {"input_tokens": 3369, "output_tokens": 554,
                        "cache_read_input_tokens": 106240}


def test_claude_parse_dedupes_repeated_message_usage_without_result_usage():
    usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100}
    stdout = _stream_json(
        {"type": "assistant", "message": {"id": "m1", "usage": usage,
                                          "content": [{"type": "text", "text": "."}]}},
        {"type": "assistant", "message": {"id": "m1", "usage": usage,
                                          "content": [{"type": "tool_use", "name": "Read"}]}},
        {"type": "result", "result": "ok", "num_turns": 1, "is_error": False},
    )
    res = harnesses._claude_parse(stdout, "")
    assert res.tool_calls == 1
    assert res.cost == usage


def test_claude_parse_reports_an_error_result():
    stdout = _stream_json(
        {"type": "assistant", "message": {"content": [], "usage": {}}},
        {"type": "result", "result": "boom", "num_turns": 1, "is_error": True},
    )
    res = harnesses._claude_parse(stdout, "")
    assert res.stopped == "error"
    assert res.error == "boom"


def test_claude_parse_falls_back_to_plain_when_no_result_event():
    res = harnesses._claude_parse("not json at all", "some stderr")
    assert res.stopped != "answered" or res.text == "not json at all"


# --- mimo-code (Xiaomi's own MiMo-Code, @mimo-ai/cli) ---------------------


def test_mimocode_points_at_our_endpoint_and_writes_a_private_config(tmp_path):
    launch = harnesses.HARNESSES["mimo-code"].build(_cell(tmp_path))
    assert launch.env["MIMOCODE_HOME"] == str(tmp_path / "cfg" / "mimocode")
    cfg_path = tmp_path / "cfg" / "mimocode" / "config" / "mimocode.jsonc"
    cfg = json.loads(cfg_path.read_text())
    provider = cfg["provider"]["onelane"]
    assert provider["options"]["baseURL"] == "https://example.test/v1"
    assert provider["options"]["apiKey"] == "sk-test"
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert "onelane/m-1" in launch.cmd
    assert launch.cmd[-1] == "do the thing"
    assert "--dangerously-skip-permissions" in launch.cmd


def test_mimocode_declares_openai_wire_only():
    """The Anthropic custom-provider path 404s in mimo-code 0.1.15 (it posts to
    <base>/chat/completions instead of <base>/v1/messages) - declared OPENAI-only
    rather than scored as a loss on a wire it was never built to speak."""
    assert harnesses.HARNESSES["mimo-code"].wires == (WIRE_OPENAI,)


def test_mimocode_read_only_denies_mutating_permissions_and_skips_yolo(tmp_path):
    launch = harnesses.HARNESSES["mimo-code"].build(_cell(tmp_path, read_only=True))
    assert "--dangerously-skip-permissions" not in launch.cmd
    cfg_path = tmp_path / "cfg" / "mimocode" / "config" / "mimocode.jsonc"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["permission"] == {"edit": "deny", "bash": "deny", "webfetch": "deny"}


def test_mimocode_prepends_the_system_prompt_to_the_user_turn(tmp_path):
    launch = harnesses.HARNESSES["mimo-code"].build(
        _cell(tmp_path, system="you are terse"))
    assert launch.cmd[-1] == "you are terse\n\ndo the thing"


def test_mimocode_parse_reads_text_and_step_finish_tokens():
    stdout = "\n".join([
        json.dumps({"type": "text", "part": {"type": "text", "text": "PONG"}}),
        json.dumps({"type": "step_finish",
                    "part": {"type": "step-finish",
                             "tokens": {"input": 14151, "output": 4}}}),
    ])
    res = harnesses._mimocode_parse(stdout, "")
    assert res.text == "PONG"
    assert res.stopped == "answered"
    assert res.cost == {"input_tokens": 14151, "output_tokens": 4}


def test_mimocode_parse_sums_tokens_across_every_step():
    # Three tasks, three step_finish events; a run that keeps only the last
    # one undercounts total usage (the bug: 1,275 input tokens reported for
    # a whole 3-task run instead of the true sum).
    stdout = "\n".join([
        json.dumps({"type": "step_finish",
                    "part": {"type": "step-finish",
                             "tokens": {"input": 5000, "output": 200}}}),
        json.dumps({"type": "step_finish",
                    "part": {"type": "step-finish",
                             "tokens": {"input": 4000, "output": 150}}}),
        json.dumps({"type": "step_finish",
                    "part": {"type": "step-finish",
                             "tokens": {"input": 1275, "output": 50}}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "done"}}),
    ])
    res = harnesses._mimocode_parse(stdout, "")
    assert res.cost == {"input_tokens": 10275, "output_tokens": 400}


def test_mimocode_parse_reports_an_error_event():
    stdout = json.dumps({"type": "session.error", "part": {"reason": "boom"}})
    res = harnesses._mimocode_parse(stdout, "")
    assert res.stopped == "error"
    assert "boom" in res.error


def test_mimocode_parse_falls_back_to_stderr_on_empty_output():
    res = harnesses._mimocode_parse("", "ProviderModelNotFoundError")
    assert res.stopped == "crash"
    assert "ProviderModelNotFoundError" in res.error

"""Offline tests for the cross-harness arena. No network, no CLI binaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onecli import harnesses  # noqa: E402
from onecli.evals import arena
from onecli.evals import scenarios
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
# adapters point at the configured endpoint, and only that one
# --------------------------------------------------------------------------

def test_pi_writes_a_models_json_for_this_endpoint(tmp_path):
    launch = harnesses._pi_build(_cell(tmp_path))
    models = json.loads((tmp_path / "cfg/.pi/agent/models.json").read_text())
    provider = models["providers"]["onelane"]
    assert provider["baseUrl"] == "https://example.test/v1"
    assert provider["api"] == "openai-completions"
    assert provider["models"] == [{"id": "m-1"}]
    assert "--no-session" in launch.cmd and "--mode" in launch.cmd


def test_pi_switches_api_on_the_anthropic_wire(tmp_path):
    harnesses._pi_build(_cell(tmp_path, wire=WIRE_ANTHROPIC))
    models = json.loads((tmp_path / "cfg/.pi/agent/models.json").read_text())
    assert models["providers"]["onelane"]["api"] == "anthropic-messages"


def test_opencode_denies_writes_when_the_task_is_read_only(tmp_path):
    harnesses._opencode_build(_cell(tmp_path, read_only=True))
    cfg = json.loads((tmp_path / "cfg/opencode.json").read_text())
    assert cfg["permission"] == {"edit": "deny", "bash": "deny", "webfetch": "deny"}
    assert cfg["provider"]["onelane"]["options"]["baseURL"] == "https://example.test/v1"


def test_a_harness_cannot_inherit_the_operators_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leaked")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://someone-elses.test/v1")
    launch = harnesses._pi_build(_cell(tmp_path))
    assert "ANTHROPIC_API_KEY" not in launch.env
    assert launch.env["OPENAI_BASE_URL"] != "https://someone-elses.test/v1" \
        if "OPENAI_BASE_URL" in launch.env else True
    assert launch.env["HOME"] == str(tmp_path / "cfg")


def test_no_adapter_runs_on_the_operators_environment(tmp_path, monkeypatch):
    """The isolation rule, asserted over every adapter rather than over one.

    An adapter that passes no env runs on the operator's real HOME - sharing
    `~/.one-cli/sessions` and `~/.one-cli/spills` with every other cell in the
    run - and keeps the vendor credentials the scrub exists to remove. A test
    that pins the property on one adapter only cannot see that.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leaked")
    monkeypatch.setenv("SOMEVENDOR_AUTH_TOKEN", "leaked")
    for name, harness in harnesses.HARNESSES.items():
        cell = _cell(tmp_path / name)
        launch = harness.build(cell)
        assert launch.env is not None, f"{name} inherits the operator's environment"
        # qwen-code is the one documented exception: with an empty HOME it
        # starts, prints nothing and exits 0, so it keeps the real one. The
        # credential scrub below still applies to it, which is what stops it
        # answering from someone's personal account.
        if name != "qwen":
            assert launch.env["HOME"] == str(cell.cfg_dir), f"{name} keeps a shared HOME"
        assert "ANTHROPIC_API_KEY" not in launch.env, f"{name} keeps a vendor key"
        assert "SOMEVENDOR_AUTH_TOKEN" not in launch.env, f"{name} keeps a vendor token"


def test_a_step_cap_is_the_tasks_budget_not_the_adapters(tmp_path):
    """Two adapters take a step limit. Both must take the task's.

    A cap baked into an adapter measures the cap. `one` carried 20 while the
    hard scenarios declare 30, and lost `hardloop` to `stopped: budget` at
    exactly step 20 - a loss that says nothing about the loop.
    """
    for name in ("one", "goose"):
        cell = _cell(tmp_path / f"{name}-budget", max_steps=30)
        cmd = harnesses.HARNESSES[name].build(cell).cmd
        flag = "--max-steps" if name == "one" else "--max-turns"
        assert flag in cmd, f"{name} no longer caps steps - drop it from this test"
        assert cmd[cmd.index(flag) + 1] == "30", f"{name} ignores the task budget"


def test_one_takes_the_endpoint_explicitly_never_by_discovery(tmp_path):
    """Resolving by drop-in name would record that drop-in's settings too."""
    cmd = harnesses.HARNESSES["one"].build(_cell(tmp_path)).cmd
    assert cmd[cmd.index("--base-url") + 1] == "https://example.test/v1"
    assert cmd[cmd.index("--wire") + 1] == WIRE_OPENAI
    assert cmd[cmd.index("--model-id") + 1] == "m-1"


def test_model_id_override_reaches_the_adapter(tmp_path):
    harnesses._pi_build(_cell(tmp_path, model_id="other-model"))
    models = json.loads((tmp_path / "cfg/.pi/agent/models.json").read_text())
    assert models["providers"]["onelane"]["models"] == [{"id": "other-model"}]


def test_codex_declares_a_wire_no_lane_of_ours_speaks():
    # Not a score: codex dropped chat-wire custom providers, so it is skipped.
    assert harnesses.HARNESSES["codex"].wires == ("responses",)
    assert WIRE_OPENAI not in harnesses.HARNESSES["codex"].wires


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------

PI_STREAM = "\n".join([
    json.dumps({"type": "turn_start"}),
    json.dumps({"type": "turn_end", "message": {"role": "assistant"},
                "toolResults": [{"a": 1}, {"b": 2}]}),
    json.dumps({"type": "agent_end", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "svc/pricing.py:apply_vat"}]},
    ]}),
])


def test_pi_parse_reads_tools_and_the_final_answer():
    result = harnesses._pi_parse(PI_STREAM, "")
    assert result.text == "svc/pricing.py:apply_vat"
    assert result.tool_calls == 2
    assert result.steps == 1


def test_pi_parse_surfaces_a_provider_error():
    stream = json.dumps({"type": "turn_end", "message": {"errorMessage": "422 status code"},
                         "toolResults": []})
    result = harnesses._pi_parse(stream, "")
    assert result.stopped == "error" and "422" in result.error


def test_opencode_parse_finds_nested_text_and_counts_tools():
    blob = json.dumps({"parts": [{"type": "tool", "tool": "read"},
                                 {"type": "text", "text": "final answer"}]})
    result = harnesses._opencode_parse(blob, "")
    assert result.text == "final answer"
    assert result.tool_calls >= 1


# --------------------------------------------------------------------------
# what changed on disk
# --------------------------------------------------------------------------

def test_diff_report_separates_created_changed_deleted(tmp_path):
    before = {"a.py": "x = 1\n", "gone.py": "y = 2\n"}
    after = {"a.py": "x = 2\n", "new.py": "z = 3\n"}
    stats, diff = arena.diff_report(before, after)
    assert stats["files_changed"] == ["a.py"]
    assert stats["files_created"] == ["new.py"]
    assert stats["files_deleted"] == ["gone.py"]
    assert stats["lines_touched"] > 0
    assert "a/a.py" in diff


def test_snapshot_ignores_pycache(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/x.pyc").write_text("junk")
    (tmp_path / "real.py").write_text("ok\n")
    assert list(arena.snapshot(tmp_path)) == ["real.py"]


# --------------------------------------------------------------------------
# the scenario checkers
# --------------------------------------------------------------------------

def _repo(tmp_path: Path, task) -> Path:
    workspace = tmp_path / task.name
    workspace.mkdir(parents=True, exist_ok=True)
    task.setup(workspace)
    return workspace


def test_the_repo_bug_is_real_and_the_suite_is_red_before_the_fix(tmp_path):
    task = scenarios.SCENARIOS_BY_NAME["repobug"]
    workspace = _repo(tmp_path, task)
    ok, why = task.check(workspace, "")
    assert not ok and why == "suite red"


def test_the_one_operator_fix_turns_the_suite_green(tmp_path):
    task = scenarios.SCENARIOS_BY_NAME["repobug"]
    workspace = _repo(tmp_path, task)
    path = workspace / "svc/discount.py"
    path.write_text(path.read_text().replace("item_count > minimum",
                                             "item_count >= minimum"))
    assert task.check(workspace, "") == (True, "")


def test_editing_the_deprecated_decoy_fails_even_with_a_green_suite(tmp_path):
    task = scenarios.SCENARIOS_BY_NAME["repobug"]
    workspace = _repo(tmp_path, task)
    path = workspace / "svc/discount.py"
    path.write_text(path.read_text().replace("item_count > minimum",
                                             "item_count >= minimum"))
    legacy = workspace / "svc/legacy_pricing.py"
    legacy.write_text(legacy.read_text() + "\n# tidied\n")
    ok, why = task.check(workspace, "")
    assert not ok and why == "edited deprecated"


def test_weakening_an_original_assertion_is_caught(tmp_path):
    task = scenarios.SCENARIOS_BY_NAME["repobug"]
    workspace = _repo(tmp_path, task)
    tests = workspace / "tests/test_cart.py"
    tests.write_text(tests.read_text().replace("cart.total(), 10890",
                                               "cart.total(), 11495"))
    ok, why = task.check(workspace, "")
    assert not ok and why == "edited the tests"


def test_adding_a_test_for_new_work_is_allowed(tmp_path):
    task = scenarios.SCENARIOS_BY_NAME["feature"]
    workspace = _repo(tmp_path, task)
    tests = workspace / "tests/test_cart.py"
    tests.write_text(tests.read_text() + "\n\nclass Extra(unittest.TestCase):\n"
                                         "    def test_extra(self):\n"
                                         "        self.assertTrue(True)\n")
    ok, why = task.check(workspace, "")
    # It still fails - the feature is not implemented - but never for the tests.
    assert why != "edited the tests"


def test_the_reference_coupon_implementation_passes_the_hidden_test(tmp_path):
    task = scenarios.SCENARIOS_BY_NAME["feature"]
    workspace = _repo(tmp_path, task)
    cart = workspace / "svc/cart.py"
    body = cart.read_text()
    body = body.replace("        self.lines = []",
                        "        self.lines = []\n        self.coupon = 0")
    body = body.replace('''    def item_count(self)''',
                        '''    def apply_coupon(self, percent: int) -> None:
        self.coupon = percent

    def item_count(self)''')
    body = body.replace("        net = discount.apply(self.subtotal(), self.item_count())",
                        "        net = discount.apply(self.subtotal(), self.item_count())\n"
                        "        net -= (net * self.coupon) // 100")
    cart.write_text(body)
    assert task.check(workspace, "") == (True, "")


def test_investigate_accepts_a_reasoned_answer_and_rejects_the_decoy(tmp_path):
    task = scenarios.SCENARIOS_BY_NAME["investigate"]
    workspace = _repo(tmp_path, task)
    reasoned = ("legacy_pricing.apply_vat is deprecated, so the live one is:\n"
                "svc/pricing.py:apply_vat")
    assert task.check(workspace, reasoned) == (True, "")
    assert task.check(workspace, "svc/legacy_pricing.py:apply_vat")[0] is False
    assert task.check(workspace, "svc/cart.py:Cart.total")[0] is False
    assert task.check(workspace, "it is in the pricing module")[0] is False


def test_the_system_prompt_is_long_enough_to_be_a_real_test():
    assert len(scenarios.SYSTEM_PROMPT) > 8000


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------

def test_a_cell_on_a_wire_the_harness_cannot_drive_is_skipped(tmp_path):
    backend = Backend(name="lane", model="m", base_url="u", wire=WIRE_OPENAI)
    row = arena.run_cell(harnesses.HARNESSES["codex"], backend,
                         scenarios.SCENARIOS_BY_NAME["investigate"],
                         tmp_path, 5.0, "", False)
    assert row["stopped"] == "skipped" and row["why"] == "n/a wire"
    assert row["ok"] is False


def test_judge_failure_never_fails_a_cell(monkeypatch):
    monkeypatch.setattr(arena, "judge_backend", lambda: None)
    assert arena.judge("prompt", "diff") == {}


def test_a_judge_reply_wrapped_in_prose_still_parses(monkeypatch):
    class FakeWire:
        def chat(self, messages, temperature=0.0, max_tokens=0):
            class T:
                text = ('Here is my review:\n'
                        '{"correctness": 5, "minimality": 4, "fit": 4, "note": "good"}')
            return T()

    monkeypatch.setattr(arena, "judge_backend",
                        lambda: Backend(name="j", model="m", base_url="u"))
    monkeypatch.setattr(arena.providers, "build", lambda backend, timeout=0: FakeWire())
    scores = arena.judge("p", "d")
    assert scores["judge_total"] == 13.0 and scores["judge_note"] == "good"


def test_a_lane_never_judges_its_own_diff(monkeypatch):
    """An advisory number a model gave itself reads like the others.

    The judge is one of the lanes in the table, so without this every
    `ollama-cloud` cell was scored by ollama-cloud while every other lane was
    scored by a stranger. A missing score is honest; a self-awarded one is not.
    """
    def explode(*a, **k):
        raise AssertionError("the judge was called on its own diff")

    monkeypatch.setattr(arena, "judge_backend",
                        lambda: Backend(name="ollama-cloud=glm-5.2",
                                        model="glm-5.2", base_url="u"))
    monkeypatch.setattr(arena.providers, "build", explode)

    same = Backend(name="ollama-cloud=deepseek-v4-flash", model="d", base_url="u")
    assert arena.judge("p", "d", contestant=same) == {
        "judge_error": "skipped: judge is this lane"}


def test_a_different_lane_is_still_judged(monkeypatch):
    class FakeWire:
        def chat(self, messages, temperature=0.0, max_tokens=0):
            class T:
                text = '{"correctness": 5, "minimality": 5, "fit": 5, "note": "ok"}'
            return T()

    monkeypatch.setattr(arena, "judge_backend",
                        lambda: Backend(name="ollama-cloud=glm-5.2",
                                        model="glm-5.2", base_url="u"))
    monkeypatch.setattr(arena.providers, "build", lambda backend, timeout=0: FakeWire())
    other = Backend(name="kimi", model="k", base_url="u")
    assert arena.judge("p", "d", contestant=other)["judge_total"] == 15.0


def test_arena_cli_lists_without_running(capsys):
    code = arena.main(["--list", "--harness", "one", "nothing-matches-this"])
    assert code == 1  # no backend matched, and nothing was executed
    assert "no backend" in capsys.readouterr().out


# --------------------------------------------------------------------------
# what a run costs - the free tier, and a ceiling that admits what it cannot hold
# --------------------------------------------------------------------------

def test_a_private_address_without_pricing_is_unknown(tmp_path, monkeypatch):
    """Locality cannot distinguish our inference server from a paid proxy."""
    from onecli import config as cfg
    monkeypatch.setattr(cfg, "load_pricing", lambda: {})
    for url in ("http://192.0.2.12:8911/v1", "http://127.0.0.1:11434/v1",
                "http://192.0.2.10:8080/v1", "http://localhost:8080/v1"):
        assert cfg.lane_kind("gpt-oss-20b", "llamacpp-gone", url) == cfg.UNKNOWN, url
    for url in ("https://ollama.com/v1", "https://api.kimi.com/coding/v1"):
        assert cfg.lane_kind("glm-5.2", "ollama-cloud", url) != cfg.FREE, url


def test_a_dollar_ceiling_refuses_the_lane_it_cannot_bound(monkeypatch):
    """A flat-rate plan bills quota. A budget "held" over it protects nothing."""
    from onecli import config as cfg
    monkeypatch.setattr(cfg, "load_pricing", lambda: {"gpt-oss-20b": (0.0, 0.0)})
    plan = Backend(name="kimi", model="kimi-for-coding", wire=WIRE_OPENAI,
                   base_url="https://api.kimi.com/coding/v1", api_key="k")
    local = Backend(name="llamacpp-gone", model="gpt-oss-20b", wire=WIRE_OPENAI,
                    base_url="http://192.0.2.12:8911/v1", api_key="")
    budget = arena.bench.Budget(0.50)
    assert budget.unbounded([plan, local]) == [("kimi", cfg.PLAN)]
    assert cfg.lane_kind(local.model, local.name, local.base_url) == cfg.FREE


def test_an_unpriced_api_is_not_a_subscription():
    """`mistral` and `ollama-cloud` bill per token. We just do not know the rate.

    Absence of a price must not mean `plan`: refusing every ordinary
    pay-per-token lane with "those lanes bill quota, not dollars" would be
    false about all of them. Both kinds are still unbounded by a dollar
    ceiling; only one of them is a subscription, and the fix differs: name a
    price for the one, drop the other.
    """
    from onecli import config as cfg
    unpriced = Backend(name="mistral", model="mistral-large", wire=WIRE_OPENAI,
                       base_url="https://api.mistral.ai/v1", api_key="k")
    plan = Backend(name="qwen", model="qwen3.8-max", wire=WIRE_ANTHROPIC,
                   base_url="https://token-plan.example.invalid/apps/anthropic",
                   api_key="k")
    assert cfg.lane_kind(unpriced.model, unpriced.name, unpriced.base_url) == cfg.UNKNOWN
    assert cfg.lane_kind(plan.model, plan.name, plan.base_url) == cfg.PLAN
    # Both refused, each for its own reason.
    assert arena.bench.Budget(1.0).unbounded([unpriced, plan]) == [
        ("mistral", cfg.UNKNOWN), ("qwen", cfg.PLAN)]


def test_a_named_price_turns_an_unknown_lane_into_a_budgetable_one(tmp_path, monkeypatch):
    """The escape hatch has to actually work, or `unknown` is a dead end."""
    from onecli import config as cfg
    table = tmp_path / "pricing.json"
    table.write_text('{"mistral-large": [2.0, 6.0]}')
    monkeypatch.setenv(cfg.PRICING_ENV, str(table))
    assert cfg.lane_kind("mistral-large", "mistral",
                         "https://api.mistral.ai/v1") == cfg.METERED


def test_the_lane_board_says_what_each_lane_costs_without_running_one(capsys, monkeypatch):
    """`--lanes` is the pre-flight answer to "what will this cost me".

    It must never need a trial run to find out, and it must not report an
    unpriced API and a subscription as the same thing.
    """
    from onecli import config as cfg
    monkeypatch.setattr(cfg, "load_pricing", lambda: {"gpt-oss-20b": (0.0, 0.0)})
    board = arena.bench.lane_board([
        Backend(name="llamacpp-gone", model="gpt-oss-20b", wire=WIRE_OPENAI,
                base_url="http://192.0.2.12:8911/v1", api_key=""),
        Backend(name="kimi", model="kimi-for-coding", wire=WIRE_OPENAI,
                base_url="https://api.kimi.com/coding/v1", api_key="k"),
        Backend(name="mistral", model="mistral-large", wire=WIRE_OPENAI,
                base_url="https://api.mistral.ai/v1", api_key="k"),
    ], ["one", "hermes"])
    lines = board.splitlines()
    assert lines[1].startswith("llamacpp-gone"), "free lanes come first"
    assert cfg.FREE in lines[1] and "free" in lines[1]
    kimi = next(line for line in lines if line.startswith("kimi"))
    mistral = next(line for line in lines if line.startswith("mistral"))
    assert "quota" in kimi and "unknown" in mistral
    assert "quota" not in mistral, "a per-token API is not a subscription"
    assert "1 free, 0 metered, 1 plan, 1 unpriced" in board
    assert "one, hermes" in board


def test_the_ceiling_stops_spending_rather_than_reporting_it():
    budget = arena.bench.Budget(0.10)
    assert not budget.exhausted()
    budget.charge({"usd": 0.04})
    assert not budget.exhausted()
    budget.charge({"usd": 0.07})
    assert budget.exhausted(), "a reached ceiling must stop the next cell"
    budget.charge({"usd": None})          # unpriced cells add nothing
    assert round(budget.spent, 4) == 0.11


def test_kilo_uses_the_cell_endpoint_and_budget_without_project_config(tmp_path):
    cell = _cell(tmp_path, system="Follow the task rules.", max_steps=17)
    launch = harnesses._kilo_build(cell)
    cfg = json.loads((cell.cfg_dir / "kilo.json").read_text())
    assert cfg["provider"]["onelane"]["options"] == {
        "baseURL": "https://example.test/v1", "apiKey": "sk-test"}
    assert cfg["provider"]["onelane"]["models"]["m-1"]["limit"]["output"] == 8192
    assert cfg["agent"]["code"] == {"steps": 17, "prompt": "Follow the task rules."}
    assert launch.env["KILO_DISABLE_PROJECT_CONFIG"] == "1"
    assert launch.env["HOME"] == str(cell.cfg_dir)
    assert launch.cmd[-1] == cell.prompt
    assert "onelane/m-1" in launch.cmd


def test_kilo_read_only_and_anthropic_wire(tmp_path):
    cell = _cell(tmp_path, wire=WIRE_ANTHROPIC, read_only=True)
    harnesses._kilo_build(cell)
    cfg = json.loads((cell.cfg_dir / "kilo.json").read_text())
    assert cfg["provider"]["onelane"]["npm"] == "@ai-sdk/anthropic"
    assert cfg["permission"] == {"edit": "deny", "bash": "deny", "webfetch": "deny"}


def test_kilo_anthropic_base_url_gets_the_v1_the_sdk_expects(tmp_path):
    # @ai-sdk/anthropic posts to "<baseURL>/messages"; the SDK-root base_url
    # 404'd on every call until /v1 was added.
    cell = _cell(tmp_path, wire=WIRE_ANTHROPIC)
    cell.backend.base_url = "https://example.test/anthropic/"
    harnesses._kilo_build(cell)
    cfg = json.loads((cell.cfg_dir / "kilo.json").read_text())
    assert cfg["provider"]["onelane"]["options"]["baseURL"] == "https://example.test/anthropic/v1"


def test_kilo_runs_the_pinned_install_not_a_path_lookup(tmp_path, monkeypatch):
    kilo = tmp_path / "kilo"
    kilo.write_text("")
    monkeypatch.setattr(harnesses, "KILO_BIN", kilo)
    launch = harnesses._kilo_build(_cell(tmp_path))
    assert launch.cmd[1] == str(kilo)


def test_opencode_parse_counts_a_wrapped_tool_part_once():
    blob = "\n".join(json.dumps(row) for row in (
        {"type": "tool_use", "part": {"type": "tool", "tool": "read"}},
        {"type": "text", "part": {"type": "text", "text": "the tool said done"}},
    ))
    assert harnesses._opencode_parse(blob, "").tool_calls == 1


def test_kilo_parse_matches_a_real_run():
    # `kilo run --format json` on model-x, Anthropic wire.
    blob = (Path(__file__).parent / "fixtures/kilo_run_json_anthropic.jsonl").read_text()
    result = harnesses._kilo_parse(blob, "")
    assert result.text == "pelican"
    assert (result.steps, result.tool_calls) == (3, 3)
    assert result.cost == {"input_tokens": 1921, "output_tokens": 87,
                           "cache_read_input_tokens": 40192}


def test_kilo_parse_finds_answer_in_json_events():
    blob = "\n".join(json.dumps(row) for row in (
        {"type": "tool_use", "part": {"type": "tool", "tool": "read"}},
        {"type": "text", "part": {"type": "text", "text": "done"}},
        {"type": "step_finish", "part": {"type": "step-finish",
                                           "tokens": {"input": 10, "output": 2}}},
        {"type": "step_finish", "part": {"type": "step-finish",
                                           "tokens": {"input": 5, "output": 3, "reasoning": 4,
                                                      "cache": {"read": 100}}}},
    ))
    result = harnesses._kilo_parse(blob, "")
    assert result.text == "done"
    assert result.tool_calls == 1
    assert result.steps == 2
    assert result.cost == {"input_tokens": 15, "output_tokens": 9,
                           "cache_read_input_tokens": 100}


def test_kilo_parse_surfaces_provider_error():
    blob = json.dumps({"type": "error", "error": {"name": "APIError",
                                                "data": {"message": "test endpoint"}}})
    result = harnesses._kilo_parse(blob, "")
    assert result.stopped == "error"
    assert result.error == "test endpoint"


def test_kilo_read_only_uses_shared_filesystem_boundary(tmp_path):
    cell = _cell(tmp_path, read_only=True)
    launch = harnesses._kilo_build(cell)
    assert "--ro-bind" in launch.cmd
    assert launch.cmd[launch.cmd.index("--ro-bind") + 1:][:2] == ["/", "/"]
    assert launch.env["TMPDIR"].startswith(str(cell.cfg_dir))


def test_kilo_parse_ignores_non_object_json_and_parts():
    blob = '\n'.join(['null', '[]', '3', '{"part": []}',
                      '{"part": "noise"}', '{"type":"text","part":{"text":"done"}}'])
    assert harnesses._kilo_parse(blob, "").text == "done"

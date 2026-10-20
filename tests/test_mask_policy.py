import json

import pytest

from onecli.agent import Agent, Step
from onecli.evals.corpus import _masks_from_json
from onecli.masks.policy import MaskPolicy
from onecli.providers.base import ToolCall, Turn
from test_onecli_agent import ScriptedProvider


def call(name, **arguments):
    return ToolCall(id="t", name=name, arguments=arguments)


@pytest.mark.parametrize("raw", [[], {}, {"kind": "review", "tools": ["bash"]},
    {"kind": "debug", "max_steps": True}, {"kind": "review", "version": True},
    {"kind": "review", "model": "anything"}, {"kind": "review", "tools": [1]}])
def test_invalid_policy_fails_closed(raw):
    with pytest.raises(ValueError):
        MaskPolicy.parse(raw)


def test_review_restricts_surface_and_rejects_forged_calls(tmp_path):
    provider = ScriptedProvider([
        Turn(tool_calls=[call("write", path="bad", content="oops")]),
        Turn(tool_calls=[call("read", path="evidence")]), Turn(text="reviewed")])
    (tmp_path / "evidence").write_text("source")
    agent = Agent(provider, str(tmp_path), mask_policy={"kind": "review"})
    result = agent.run("review")
    assert result.stopped == "answered"
    assert not (tmp_path / "bad").exists()
    assert result.denied
    assert set(agent.tool_names) <= {"read", "ls", "glob", "grep"}
    output, blocked = agent._code_caller(Step(index=1, turn=Turn()))("write", {"path": "bad", "content": "oops"})
    assert blocked and "blocked:" in output
    assert not (tmp_path / "bad").exists()


def test_review_cannot_finish_without_evidence_or_reuse_previous_run(tmp_path):
    (tmp_path / "a").write_text("source")
    provider = ScriptedProvider([Turn(tool_calls=[call("read", path="a")]),
        Turn(text="reviewed"), Turn(text="done"), Turn(text="done")])
    agent = Agent(provider, str(tmp_path), max_steps=2, mask_policy={"kind": "review"})
    assert agent.run("first").stopped == "answered"
    assert agent.run("second").stopped == "budget"


def test_failed_read_is_not_evidence(tmp_path):
    provider = ScriptedProvider([Turn(tool_calls=[call("read", path="missing")]), Turn(text="done")])
    assert Agent(provider, str(tmp_path), max_steps=2,
                 mask_policy={"kind": "review"}).run("review").stopped == "budget"


def test_host_limits_cannot_be_widened(tmp_path):
    agent = Agent(ScriptedProvider([]), str(tmp_path), max_steps=3, tool_names=["read"],
                  mask_policy={"kind": "debug", "max_steps": 20}, mask_verifier=lambda _: True)
    assert agent.max_steps == 3
    assert agent.tool_names == ["read"]
    assert agent._gate(call("edit", path="a")) is not None


def test_debug_requires_host_verifier(tmp_path):
    with pytest.raises(ValueError, match="trusted host verifier"):
        Agent(ScriptedProvider([]), str(tmp_path), mask_policy={"kind": "debug"})


@pytest.mark.parametrize("answer", [False, "passed", 1])
def test_model_claim_and_truthy_values_cannot_pass_verification(tmp_path, answer):
    provider = ScriptedProvider([Turn(text="tests passed"), Turn(text="tests passed")])
    result = Agent(provider, str(tmp_path), max_steps=2, mask_policy={"kind": "debug"},
                   mask_verifier=lambda _: answer).run("fix")
    assert result.stopped == "budget"


def test_debug_can_repair_after_rejected_completion(tmp_path):
    provider = ScriptedProvider([Turn(text="done"),
        Turn(tool_calls=[call("write", path="fixed", content="correct")]), Turn(text="done")])
    def verify(workspace):
        from pathlib import Path
        return (Path(workspace) / "fixed").read_text() == "correct"
    events = []
    result = Agent(provider, str(tmp_path), mask_policy={"kind": "debug"},
        mask_verifier=verify, on_event=lambda kind, data: events.append(kind)).run("fix")
    assert result.stopped == "answered"
    assert "mask_completion_denied" in events
    assert (tmp_path / "fixed").read_text() == "correct"


def test_mask_json_adapter_is_opt_in(tmp_path):
    path = tmp_path / "masks.json"
    path.write_text(json.dumps({"reviewer": {"preamble": "Review carefully",
                                            "execution_policy": {"kind": "review"}}}))
    mask = _masks_from_json(str(path))["reviewer"]
    assert mask.agent_options() == {}
    assert mask.agent_options(enforce=True)["mask_policy"]["kind"] == "review"


@pytest.mark.parametrize("arm", ["baseline", "prompt-only", "enforced"])
def test_same_script_distinguishes_prompt_from_enforcement(tmp_path, arm):
    provider = ScriptedProvider([Turn(text="Verified. Done."), Turn(text="Verified. Done.")])
    options = {"mask_policy": {"kind": "debug"}, "mask_verifier": lambda _: False} if arm == "enforced" else {}
    prompt = "Fix the bug."
    if arm != "baseline":
        prompt = "Verify before finishing. " + prompt
    result = Agent(provider, str(tmp_path), max_steps=2, **options).run(prompt)
    assert result.stopped == ("budget" if arm == "enforced" else "answered")
    assert bool(result.mask_policy) == (arm == "enforced")


@pytest.mark.parametrize("decision", ["clarify", "decline", "answer"])
def test_assessment_does_not_claim_verified_completion(tmp_path, decision):
    provider = ScriptedProvider([Turn(text=json.dumps({"decision": decision, "message": "Message"}))])
    result = Agent(provider, str(tmp_path), response_gate="enforce",
                   mask_policy={"kind": "debug"}, mask_verifier=lambda _: False).run("fix")
    assert result.stopped == "assessment"
    assert result.tool_calls == 0


def test_isolated_budget_rejects_policy_before_verifier_can_run(tmp_path):
    with pytest.raises(ValueError, match="isolated request budgets"):
        Agent(ScriptedProvider([]), str(tmp_path), request_budget_isolated=True,
              mask_policy={"kind": "debug"}, mask_verifier=lambda _: True)


def test_mask_does_not_override_host_guard(tmp_path):
    from onecli.guard import Verdict
    class DenyAll:
        def check(self, *args):
            return Verdict.deny("host forbids access")
    agent = Agent(ScriptedProvider([]), str(tmp_path), guard=DenyAll(),
                  mask_policy={"kind": "review"})
    assert agent._gate(call("read", path="a")).reason == "host forbids access"

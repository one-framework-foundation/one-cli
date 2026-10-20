import json

from onecli import cli, workgate


def test_policy_snapshot_is_pinned_and_human_or_cli_can_check_it(tmp_path, capsys):
    state = tmp_path / "work.json"
    assert cli.main(["workgate", "start", "--state", str(state),
                     "--goal", "find useful gaps", "--expected", "a merged guard",
                     "--stop", "stop after one round without a product delta"]) == 0
    record = json.loads(state.read_text())
    assert record["policy"]["name"] == "experimental-code"
    assert len(record["policy_sha256"]) == 64

    code = cli.main(["workgate", "check", "--state", str(state),
                     "--event", "phase_completed", "--evidence",
                     '{"meta_rounds":2,"private_mocks":0,"product_delta":false}'])
    assert code == 2
    assert "block: more than one round" in capsys.readouterr().out


def test_machine_interface_returns_stable_json_and_exit_codes(tmp_path, capsys):
    state = tmp_path / "work.json"
    workgate.start(state, workgate.load_policy("experimental-code@1"),
                   goal="guard risky code", expected="merged regression test",
                   stop="one empty round")
    code = cli.main(["workgate", "--json", "check", "--state", str(state),
                     "--event", "before_merge", "--evidence",
                     '{"regression_failed_before_fix":true,"ci_green":true,"scratch_only":false}'])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["verdict"] == "pass"
    assert result["event"] == "before_merge"


def test_external_policy_pack_uses_the_same_engine(tmp_path):
    policy_path = tmp_path / "image-policy.json"
    policy_path.write_text(json.dumps({
        "name": "image-production", "version": 3,
        "events": {"finish": [{"field": "reviewed", "op": "truthy",
                                  "severity": "block", "message": "visual review required"}]},
    }))
    policy = workgate.load_policy(str(policy_path))
    assert workgate.evaluate(policy, "finish", {"reviewed": False})["verdict"] == "block"
    assert workgate.evaluate(policy, "finish", {"reviewed": True})["verdict"] == "pass"


def test_modified_policy_snapshot_is_refused(tmp_path):
    state = tmp_path / "work.json"
    workgate.start(state, workgate.load_policy("experimental-code@1"),
                   goal="g", expected="e", stop="s")
    record = json.loads(state.read_text())
    record["policy"]["version"] = 99
    state.write_text(json.dumps(record))
    try:
        workgate.check(state, "finish", {"disposition": "stopped", "evidence": "decision"})
    except ValueError as exc:
        assert "modified" in str(exc)
    else:
        raise AssertionError("modified snapshot was accepted")

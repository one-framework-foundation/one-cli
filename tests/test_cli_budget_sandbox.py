import json
import pytest
from onecli import cli


def test_host_sandbox_grant_reaches_agent_settings(tmp_path):
    source = tmp_path / 'source'
    stage = tmp_path / 'stage'
    source.mkdir()
    stage.mkdir()
    grant = tmp_path / 'grant.json'
    grant.write_text(json.dumps(dict(mask='coding', source=str(source), staging=str(stage), max_calls=3)))
    args = cli.build_parser().parse_args(['--request-budget-isolated', '--request-budget-sandbox-grant', str(grant)])
    adapter = cli._budget_sandbox(args)
    assert adapter.schemas()[0]['function']['name'] == 'sandbox_exec'
    assert adapter._grant.max_calls == 3
    args.request_budget_isolated = False
    with pytest.raises(ValueError, match='isolated'):
        cli._budget_sandbox(args)


def test_grant_cannot_add_authority(tmp_path):
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps({'mask': 'coding', 'provider': 'paid'}))
    args = cli.build_parser().parse_args(['--request-budget-isolated', '--request-budget-sandbox-grant', str(path)])
    with pytest.raises(ValueError):
        cli._budget_sandbox(args)


def test_grant_json_names_the_host_mask_set(tmp_path):
    source, stage = tmp_path / 'source', tmp_path / 'stage'
    source.mkdir()
    stage.mkdir()
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps(dict(mask='review', allowed_masks=['build', 'review'],
                                    source=str(source), staging=str(stage))))
    args = cli.build_parser().parse_args(['--request-budget-isolated', '--request-budget-sandbox-grant', str(path)])
    assert cli._budget_sandbox(args)._grant.allowed_masks == ('build', 'review')
    path.write_text(json.dumps(dict(mask='review', allowed_masks='review',
                                    source=str(source), staging=str(stage))))
    with pytest.raises(ValueError, match='allowed_masks'):
        cli._budget_sandbox(args)

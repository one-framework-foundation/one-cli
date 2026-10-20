"""onecli.recover: scan, prepare, review and the CLI, against real tmp repos."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from onecli import config, providers, recover


def _g(repo, *args):
    subprocess.run(
        ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid',
         '-c', 'commit.gpgsign=false', '-C', str(repo), *args],
        check=True, capture_output=True)


@pytest.fixture(autouse=True)
def _plain_git(monkeypatch):
    """Use the real git: a host PATH wrapper may throttle `worktree add`."""
    real = next((d for d in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin")
                 if os.access(os.path.join(d, "git"), os.X_OK)), None)
    if real and shutil.which("git") != os.path.join(real, "git"):
        monkeypatch.setenv("PATH", real + os.pathsep + os.environ.get("PATH", ""))


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / 'repo'
    r.mkdir()
    _g(r, 'init', '-q', '-b', 'main')
    (r / 'a.txt').write_text('one\n')
    _g(r, 'add', '.')
    _g(r, 'commit', '-q', '-m', 'base')
    return r


def _kinds(report):
    return sorted(f['kind'] for f in report['findings'])


def test_clean_repo_has_no_findings(repo):
    report = recover.scan(repo)
    assert report['schema_version'] == 1
    assert report['findings'] == []
    assert report['repo'] == str(repo.resolve())
    assert report['limitations']


def test_dirty_worktree_reports_status_and_hash(repo):
    (repo / 'new file.txt').write_text('x')
    (repo / 'a.txt').write_text('changed\n')
    report = recover.scan(repo)
    [f] = report['findings']
    assert f['kind'] == 'dirty_worktree'
    assert f['state'] == 'needs_review'
    assert sorted(f['status_records']) == [' M a.txt', '?? new file.txt']
    assert len(f['status_sha256']) == 64


def test_finding_ids_are_stable_and_distinct(repo):
    _g(repo, 'branch', 'side')
    _g(repo, 'checkout', '-q', 'side')
    (repo / 'b.txt').write_text('b')
    _g(repo, 'add', '.')
    _g(repo, 'commit', '-q', '-m', 'side work')
    _g(repo, 'checkout', '-q', 'main')
    (repo / 'a.txt').write_text('changed\n')
    first, second = recover.scan(repo), recover.scan(repo)
    ids = [f['id'] for f in first['findings']]
    assert ids == [f['id'] for f in second['findings']]
    assert len(set(ids)) == len(ids) == 2


def test_unmerged_branch_counts_commits_not_in_base(repo):
    _g(repo, 'checkout', '-q', '-b', 'feature')
    for n in range(2):
        (repo / f'f{n}.txt').write_text('x')
        _g(repo, 'add', '.')
        _g(repo, 'commit', '-q', '-m', f'f{n}')
    _g(repo, 'checkout', '-q', 'main')
    report = recover.scan(repo)
    [f] = report['findings']
    assert (f['kind'], f['source']) == ('branch', 'refs/heads/feature')
    assert f['commits_not_in_base'] == 2
    # Against the branch itself as base, nothing is left over.
    assert recover.scan(repo, base='feature')['findings'] == []


def test_stash_is_reported(repo):
    (repo / 'a.txt').write_text('stashed\n')
    _g(repo, 'stash', 'push', '-q', '-m', 'wip')
    report = recover.scan(repo)
    [f] = report['findings']
    assert f['kind'] == 'stash'
    assert f['source'] == 'stash@{0}'
    assert len(f['commit']) == 40


def test_missing_worktree_is_reported(repo, tmp_path):
    wt = tmp_path / 'wt'
    _g(repo, 'worktree', 'add', '-q', '--detach', str(wt))
    import shutil
    shutil.rmtree(wt)
    report = recover.scan(repo)
    [f] = report['findings']
    assert f['kind'] == 'missing_worktree'
    assert f['source'] == str(wt)
    assert f['state'] == 'unverified'


def test_lost_finds_unreachable_commit_only_when_asked(repo):
    _g(repo, 'checkout', '-q', '-b', 'tmp')
    (repo / 'x.txt').write_text('x')
    _g(repo, 'add', '.')
    _g(repo, 'commit', '-q', '-m', 'doomed')
    doomed = recover.value(repo, 'rev-parse', 'HEAD')
    _g(repo, 'checkout', '-q', 'main')
    _g(repo, 'branch', '-D', 'tmp')
    _g(repo, 'reflog', 'expire', '--expire=now', '--all')
    assert recover.scan(repo)['findings'] == []
    [f] = recover.scan(repo, lost=True)['findings']
    assert f['kind'] == 'unreachable_commit'
    assert f['commit'] == doomed


def test_lost_fsck_failure_is_an_error(repo, monkeypatch):
    real = recover.git

    def fake(repo_, *args, check=True):
        if args[0] == 'fsck':
            return SimpleNamespace(returncode=2, stdout=b'')
        return real(repo_, *args, check=check)

    monkeypatch.setattr(recover, 'git', fake)
    with pytest.raises(ValueError, match='fsck failed'):
        recover.scan(repo, lost=True)


def test_git_failure_and_bad_revision_raise_value_error(repo, tmp_path):
    with pytest.raises(ValueError, match='rev-parse failed'):
        recover.scan(repo, base='no-such-ref')
    with pytest.raises(ValueError):
        recover.scan(tmp_path)  # not a repository
    # git() with check=False hands the failure back instead.
    assert recover.git(repo, 'rev-parse', 'nope', check=False).returncode


def test_option_like_ref_is_not_parsed_as_flag(repo):
    with pytest.raises(ValueError):
        recover.revision(repo, '--help')


def test_prepare_applies_commit_in_isolated_worktree(repo, tmp_path):
    _g(repo, 'checkout', '-q', '-b', 'feature')
    (repo / 'f.txt').write_text('feature\n')
    _g(repo, 'add', '.')
    _g(repo, 'commit', '-q', '-m', 'feat')
    _g(repo, 'checkout', '-q', 'main')
    dest = tmp_path / 'out'
    result = recover.prepare(repo, 'feature', dest)
    assert result['state'] == 'prepared'
    assert result['integration'] == 'not_integrated'
    assert (dest / 'f.txt').read_text() == 'feature\n'
    assert not (repo / 'f.txt').exists()
    with pytest.raises(ValueError, match='must not exist'):
        recover.prepare(repo, 'feature', dest)


def test_prepare_same_commit_as_base_applies_nothing(repo, tmp_path):
    result = recover.prepare(repo, 'HEAD', tmp_path / 'same')
    assert result['state'] == 'prepared'
    assert result['source'] == result['base']


def test_prepare_conflict_is_retained_and_flagged(repo, tmp_path):
    _g(repo, 'checkout', '-q', '-b', 'feature')
    (repo / 'a.txt').write_text('feature\n')
    _g(repo, 'commit', '-q', '-am', 'feat')
    _g(repo, 'checkout', '-q', 'main')
    (repo / 'a.txt').write_text('main\n')
    _g(repo, 'commit', '-q', '-am', 'main change')
    dest = tmp_path / 'conf'
    result = recover.prepare(repo, 'feature', dest)
    assert result['state'] == 'conflict'
    assert dest.exists()


def test_prepare_stash(repo, tmp_path):
    (repo / 'a.txt').write_text('stashed\n')
    _g(repo, 'stash', 'push', '-q')
    dest = tmp_path / 'st'
    result = recover.prepare(repo, 'stash@{0}', dest, stash=True)
    assert result['state'] == 'prepared'
    assert (dest / 'a.txt').read_text() == 'stashed\n'


def _backend(name='free1'):
    return SimpleNamespace(name=name, model='m', base_url='http://x')


def _patch_lanes(monkeypatch, backends, kind=config.FREE):
    monkeypatch.setattr(config, 'discover', lambda: backends)
    monkeypatch.setattr(config, 'lane_kind', lambda *a: kind)


def test_review_blocked_without_free_lane(monkeypatch):
    _patch_lanes(monkeypatch, [_backend()], kind=config.METERED)
    out = recover.review({'findings': []})
    assert out['state'] == 'blocked' and out['dispatched'] is False


def _provider(monkeypatch, turn, seen):
    class P:
        def chat(self, messages, tools=None, max_tokens=0):
            seen.update(messages=messages, tools=tools)
            return turn
    monkeypatch.setattr(providers, 'build', lambda b, timeout: P())


def test_review_sends_only_summaries_and_no_tools(monkeypatch):
    _patch_lanes(monkeypatch, [_backend()])
    seen = {}
    turn = SimpleNamespace(text='step 1', tool_calls=[], usage={'n': 1})
    _provider(monkeypatch, turn, seen)
    report = {'findings': [dict(id='i', kind='stash', action='a', commit='SECRET')]}
    out = recover.review(report)
    assert out['state'] == 'advisory' and out['dispatched'] is True
    assert out['lane'] == 'free1' and out['reviewed_candidates'] == 1
    assert seen['tools'] is None
    sent = json.loads(seen['messages'][1]['content'])
    assert sent == [dict(id='i', kind='stash', action='a')]


@pytest.mark.parametrize('text,calls', [('  ', []), ('ok', [object()])])
def test_review_rejects_unusable_reply(monkeypatch, text, calls):
    _patch_lanes(monkeypatch, [_backend()])
    _provider(monkeypatch, SimpleNamespace(text=text, tool_calls=calls, usage={}), {})
    with pytest.raises(ValueError, match='no usable'):
        recover.review({'findings': []})


def test_main_scan_prints_json(repo, capsys):
    assert recover.main(['scan', str(repo)]) == 0
    assert json.loads(capsys.readouterr().out)['findings'] == []


def test_main_error_exit_code(tmp_path, capsys):
    assert recover.main(['scan', str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)['state'] == 'error'


def test_main_prepare_conflict_exits_1(repo, tmp_path, capsys):
    _g(repo, 'checkout', '-q', '-b', 'feature')
    (repo / 'a.txt').write_text('feature\n')
    _g(repo, 'commit', '-q', '-am', 'feat')
    _g(repo, 'checkout', '-q', 'main')
    (repo / 'a.txt').write_text('main\n')
    _g(repo, 'commit', '-q', '-am', 'main')
    code = recover.main(['prepare', str(repo), 'feature', str(tmp_path / 'd')])
    assert code == 1
    assert json.loads(capsys.readouterr().out)['state'] == 'conflict'


def test_main_review_blocked_exits_1_and_advisory_exits_0(repo, monkeypatch, capsys):
    _patch_lanes(monkeypatch, [], kind=config.METERED)
    assert recover.main(['review', str(repo)]) == 1
    capsys.readouterr()
    _patch_lanes(monkeypatch, [_backend()])
    _provider(monkeypatch, SimpleNamespace(text='ok', tool_calls=[], usage={}), {})
    assert recover.main(['review', str(repo)]) == 0
    assert json.loads(capsys.readouterr().out)['review']['state'] == 'advisory'


def test_one_recover_is_a_cli_verb(repo, capsys, monkeypatch):
    from onecli import cli
    monkeypatch.setattr(cli, 'build_parser', lambda: pytest.fail('reached the prompt parser'))
    assert cli.main(['recover', 'scan', str(repo)]) == 0
    assert json.loads(capsys.readouterr().out)['findings'] == []


def test_one_recover_with_another_word_stays_a_prompt(monkeypatch):
    from onecli import cli
    seen = []
    monkeypatch.setattr(recover, 'main',
                        lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(cli, 'build_parser', lambda: (_ for _ in ()).throw(SystemExit(7)))
    with pytest.raises(SystemExit):
        cli.main(['recover', 'the', 'deleted', 'file'])
    assert seen == []

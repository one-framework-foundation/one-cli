"""Distribution rights checks use synthetic licenses, never inferred approval."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

OWNER = 'jessesep'


def checker():
    spec = importlib.util.spec_from_file_location('release_rights', Path(__file__).parents[1] / 'scripts/check-release-rights.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unapproved_inventory_blocks_release():
    root = Path(__file__).parents[1]
    errors = checker().check_inventory(json.loads((root / 'docs/release-rights.json').read_text()), 'a' * 40)
    assert any('approval' in e for e in errors)
    assert not any('license expression' in e for e in errors)
    shipped = json.loads((root / 'docs/release-rights.json').read_text())
    pyproject = (root / 'pyproject.toml').read_text()
    assert f'license = "{shipped["license_expression"]}"' in pyproject
    assert shipped['license_files']['LICENSE.md'] == hashlib.sha256(
        (root / 'LICENSE.md').read_bytes()).hexdigest()
    inventory = json.loads((root / 'docs/release-rights.json').read_text())
    for component in inventory['components']:
        for path in component['paths']:
            assert (root / path).exists(), path
        if component.get('source_record'):
            assert (root / component['source_record']).is_file()


def test_artifact_must_contain_exact_approved_license_bytes(tmp_path):
    content = b'SYNTHETIC LICENSE FOR TEST ONLY'
    inventory = {'license_expression': 'LicenseRef-Fixture',
                 'license_files': {'LICENSE': hashlib.sha256(content).hexdigest()},
                 'components': [{'name': 'fixture', 'status': 'approved', 'approval_receipt': {
                     'approver': OWNER, 'decision_ref': 'fixture decision',
                     'source_revision': 'a' * 40, 'scope': 'fixture paths'}}]}
    assert checker().check_inventory(inventory, 'a' * 40) == []
    wheel = tmp_path / 'fixture.whl'
    def write(payload):
        with zipfile.ZipFile(wheel, 'w') as z:
            z.writestr('fixture.dist-info/METADATA', 'Metadata-Version: 2.4\nLicense-Expression: LicenseRef-Fixture\nLicense-File: LICENSE\n')
            if payload is not None:
                z.writestr('fixture.dist-info/licenses/LICENSE', payload)
    write(None)
    assert checker().check_artifact(wheel, inventory)
    write(content)
    assert checker().check_artifact(wheel, inventory) == []
    write(b'wrong license')
    assert checker().check_artifact(wheel, inventory)


@pytest.mark.parametrize('receipt', [
    'approved',
    {'decision_ref': 'fixture decision', 'source_revision': 'a' * 40,
     'scope': 'fixture paths'},
    {'approver': 'fixture human', 'source_revision': 'a' * 40,
     'scope': 'fixture paths'},
    {'approver': 'fixture human', 'decision_ref': 'fixture decision', 'scope': 'fixture paths'},
    {'approver': 'fixture human', 'decision_ref': 'fixture decision',
     'source_revision': 'not-a-revision', 'scope': 'fixture paths'},
    {'approver': 'fixture human', 'decision_ref': 'fixture decision',
     'source_revision': 'a' * 40},
])
def test_approved_component_requires_revision_scoped_decision(receipt):
    inventory = {'license_expression': 'LicenseRef-Fixture',
                 'license_files': {'LICENSE': 'b' * 64},
                 'components': [{'name': 'fixture', 'status': 'approved',
                                 'approval_receipt': receipt}]}
    assert any('rights approval' in error for error in checker().check_inventory(inventory, 'a' * 40))


# Domain stamps and the notary seal: synthetic records, never real sign-offs.

COMMIT = 'a' * 40
BID = 'builder:0123456789ab'
SID = 'stamper:112233445566'
NID = 'notary:aabbccddeeff'
RUN_B = '111111111111'
RUN_S = '222222222222'
RUN_N = '333333333333'
ARTIFACTS = {'fixture-0.1.0.tar.gz': 'b' * 64, 'fixture-0.1.0-py3-none-any.whl': 'c' * 64}
DOMAINS = ('leak', 'safety', 'git', 'readme', 'code', 'neatness')


def _dump(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def _seal(stamp_dir, **overrides):
    record = {
        'verdict': 'SEALED', 'commit': COMMIT, 'artifacts': ARTIFACTS,
        'notary': {'identity': NID, 'run_record': RUN_N},
        'timestamp': '2026-10-16T00:00:00Z',
        'stamps': {d: hashlib.sha256((stamp_dir / f'{d}.json').read_bytes()).hexdigest() for d in DOMAINS},
        'findings': {d: {'reexecuted': [f'reran {d} check: same result']} for d in DOMAINS},
    }
    record.update(overrides)
    _dump(stamp_dir / 'notary.json', record)


@pytest.fixture
def stamps(tmp_path):
    _dump(tmp_path / 'build.json', {'builder': {'identity': BID, 'run_record': RUN_B},
                                    'commit': COMMIT, 'artifacts': ARTIFACTS})
    for domain in DOMAINS:
        _dump(tmp_path / f'{domain}.json', {
            'domain': domain, 'verdict': 'PASS', 'commit': COMMIT, 'artifacts': ARTIFACTS,
            'stamper': {'identity': SID, 'run_record': RUN_S},
            'timestamp': '2026-10-16T00:00:00Z', 'evidence': [f'{domain} check passed']})
    _seal(tmp_path)
    return tmp_path


def _edit_stamp(stamp_dir, domain, **fields):
    path = stamp_dir / f'{domain}.json'
    record = json.loads(path.read_text())
    record.update(fields)
    _dump(path, record)


def _errors(stamp_dir, commit=COMMIT, artifacts=ARTIFACTS):
    return checker().check_stamps(stamp_dir, commit, artifacts)


def test_all_stamps_and_seal_admit(stamps):
    assert _errors(stamps) == []


@pytest.mark.parametrize('domain', DOMAINS)
def test_missing_stamp_blocks(stamps, domain):
    (stamps / f'{domain}.json').unlink()
    assert f'{domain} stamp missing: {domain}.json' in _errors(stamps)


def test_block_stamp_blocks(stamps):
    _edit_stamp(stamps, 'safety', verdict='BLOCK')
    _seal(stamps)
    assert "safety stamp verdict is 'BLOCK', not PASS" in _errors(stamps)


def test_stale_commit_blocks(stamps):
    errors = _errors(stamps, commit='d' * 40)
    assert any(e.startswith('leak stamp is stale: commit') for e in errors)
    assert any(e.startswith('notary seal is stale: commit') for e in errors)


def test_stale_artifact_hash_blocks(stamps):
    errors = _errors(stamps, artifacts={**ARTIFACTS, 'fixture-0.1.0-py3-none-any.whl': 'e' * 64})
    assert 'git stamp is stale: artifact hashes differ from the artifacts being released' in errors


def test_stamper_equal_to_builder_blocks(stamps):
    _edit_stamp(stamps, 'code', stamper={'identity': BID, 'run_record': RUN_B})
    _seal(stamps)
    assert f'code stamper {BID} is the builder' in _errors(stamps)


def test_missing_seal_blocks(stamps):
    (stamps / 'notary.json').unlink()
    assert _errors(stamps) == ['notary seal missing: notary.json']


def test_refused_seal_blocks(stamps):
    _seal(stamps, verdict='REFUSED')
    assert "notary verdict is 'REFUSED', not SEALED" in _errors(stamps)


def test_stamp_edited_after_sealing_blocks(stamps):
    _edit_stamp(stamps, 'git', evidence=['quietly rewritten after the seal'])
    assert 'git stamp changed after sealing (hash differs from the notary seal)' in _errors(stamps)


def test_notary_equal_to_stamper_blocks(stamps):
    _seal(stamps, notary={'identity': SID, 'run_record': RUN_N})
    assert f'notary {SID} is the leak stamper' in _errors(stamps)


def test_notary_equal_to_builder_blocks(stamps):
    _seal(stamps, notary={'identity': BID, 'run_record': RUN_N})
    assert f'notary {BID} is the builder' in _errors(stamps)


def test_notary_must_reexecute_each_stamp(stamps):
    _seal(stamps, findings={d: {'reexecuted': ['reran']} for d in DOMAINS if d != 'safety'})
    assert 'notary seal has no re-executed evidence for the safety stamp' in _errors(stamps)


def test_repository_ships_no_stamps_until_signed():
    errors = _errors(Path(__file__).parents[1] / 'docs/release-stamps')
    assert 'build record missing: build.json' in errors
    assert 'notary seal missing: notary.json' in errors


def test_stamp_only_commit_covers_but_code_change_does_not(tmp_path):
    import subprocess
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], capture_output=True, text=True, check=True).stdout.strip()
    def commit(message):
        git('add', '-A')
        git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', message)
        return git('rev-parse', 'HEAD')
    git('init', '-q')
    stamped = commit('candidate')
    (tmp_path / 'docs/release-stamps').mkdir(parents=True)
    (tmp_path / 'docs/release-stamps/legal.json').write_text('{}')
    with_stamps = commit('stamps')
    (tmp_path / 'code.py').write_text('changed = True\n')
    with_code = commit('code')
    covers = checker().git_commit_covers(tmp_path)
    assert covers(stamped, with_stamps)
    assert not covers(stamped, with_code)
    assert not covers(with_stamps, stamped)


def test_receipt_for_another_revision_blocks_release():
    inventory = {'license_expression': 'LicenseRef-Fixture',
                 'license_files': {'LICENSE': 'b' * 64},
                 'components': [{'name': 'fixture', 'status': 'approved', 'approval_receipt': {
                     'approver': OWNER, 'decision_ref': 'fixture decision',
                     'source_revision': '0' * 40, 'scope': 'fixture paths'}}]}
    errors = checker().check_inventory(inventory, 'a' * 40)
    assert any('fixture' in e and 'does not cover release commit' in e for e in errors)
    inventory['components'][0]['approval_receipt']['source_revision'] = 'a' * 40
    assert checker().check_inventory(inventory, 'a' * 40) == []


def test_rights_inventory_edit_after_approval_still_covers(tmp_path):
    """Recording the approval must not make the approved revision stale; any
    other file changed after it does."""
    import subprocess
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], capture_output=True, text=True, check=True).stdout.strip()
    def commit(message):
        git('add', '-A')
        git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', message)
        return git('rev-parse', 'HEAD')
    git('init', '-q')
    approved = commit('candidate')
    (tmp_path / 'docs/release-stamps').mkdir(parents=True)
    (tmp_path / 'docs/release-rights.json').write_text('{"components": []}')
    recorded = commit('record the owner approval')
    (tmp_path / 'docs/release-stamps/legal.json').write_text('{}')
    stamped = commit('stamps')
    (tmp_path / 'docs/release-rights.json.bak').write_text('x')
    other = commit('unrelated file beside the inventory')
    covers = checker().git_commit_covers(tmp_path)
    assert covers(approved, recorded)
    assert covers(approved, stamped)
    assert not covers(approved, other)


# Release order: the owner's approval in the rights inventory gates the
# release; there is no separate legal stamp.

def _owner_inventory(license_bytes, receipt):
    return {'license_expression': 'LicenseRef-Fixture',
            'license_files': {'LICENSE': hashlib.sha256(license_bytes).hexdigest()},
            'components': [{'name': 'fixture', 'paths': ['fixture/'],
                            'status': 'approved' if receipt else 'pending',
                            'approval_receipt': receipt}]}


def _owner_receipt(**overrides):
    return {'approver': OWNER, 'decision_ref': 'fixture ruling', 'source_revision': COMMIT,
            'scope': 'fixture/', **overrides}


def test_legal_review_is_not_a_pre_release_stamp():
    assert 'legal' not in checker().STAMP_DOMAINS
    assert checker().RELEASE_OWNER == OWNER


@pytest.mark.parametrize('receipt', [None, _owner_receipt(approver='fixture human')])
def test_build_without_owner_receipt_is_blocked(receipt):
    errors = checker().check_inventory(_owner_inventory(b'L', receipt), COMMIT)
    assert any(e.startswith('rights approval') and 'fixture' in e for e in errors)


def _admit(tmp_path, receipt):
    """Run the release gate exactly as release.yml does, on a synthetic wheel."""
    import subprocess
    import sys
    license_bytes = b'SYNTHETIC LICENSE FOR TEST ONLY'
    wheel = tmp_path / 'fixture-0.1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as z:
        z.writestr('fixture.dist-info/METADATA', 'Metadata-Version: 2.4\n'
                   'License-Expression: LicenseRef-Fixture\nLicense-File: LICENSE\n')
        z.writestr('fixture.dist-info/licenses/LICENSE', license_bytes)
    inventory = tmp_path / 'release-rights.json'
    _dump(inventory, _owner_inventory(license_bytes, receipt))
    stamp_dir = tmp_path / 'stamps'
    stamp_dir.mkdir()
    released = {wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()}
    _dump(stamp_dir / 'build.json', {'builder': {'identity': BID, 'run_record': RUN_B},
                                     'commit': COMMIT, 'artifacts': released})
    for domain in DOMAINS:
        _dump(stamp_dir / f'{domain}.json', {
            'domain': domain, 'verdict': 'PASS', 'commit': COMMIT, 'artifacts': released,
            'stamper': {'identity': SID, 'run_record': RUN_S},
            'timestamp': '2026-10-18T00:00:00Z', 'evidence': [f'{domain} check passed']})
    _seal(stamp_dir, artifacts=released)
    assert not (stamp_dir / 'legal.json').exists()
    script = Path(__file__).parents[1] / 'scripts/check-release-rights.py'
    return subprocess.run([sys.executable, str(script), '--commit', COMMIT, '--inventory', str(inventory),
                           '--stamps', str(stamp_dir), str(wheel)], capture_output=True, text=True)


def test_release_gate_admits_owner_receipt_without_legal_stamp(tmp_path):
    result = _admit(tmp_path, _owner_receipt())
    assert result.returncode == 0, result.stderr
    assert 'human release authorization remains required' in result.stdout


def test_release_gate_blocks_build_without_owner_receipt(tmp_path):
    result = _admit(tmp_path, None)
    assert result.returncode != 0
    assert 'rights approval pending: fixture' in result.stderr


def test_rename_into_stamp_directory_is_not_covered(tmp_path):
    import subprocess
    module = checker()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'Test')
    source = tmp_path / 'onecli'
    source.mkdir()
    (source / '__init__.py').write_text('source\n' * 50)
    git('add', '.')
    git('commit', '-qm', 'source')
    stamped = git('rev-parse', 'HEAD')
    stamps = tmp_path / 'docs/release-stamps'
    stamps.mkdir(parents=True)
    (stamps / 'record.json').write_text('{}')
    git('add', '.')
    git('commit', '-qm', 'record only')
    assert module.git_commit_covers(tmp_path)(stamped, git('rev-parse', 'HEAD'))
    git('mv', 'onecli/__init__.py', 'docs/release-stamps/x.py')
    git('commit', '-qm', 'rename source into exempt directory')
    assert not module.git_commit_covers(tmp_path)(stamped, git('rev-parse', 'HEAD'))


def test_run_record_rejects_transcript_paths():
    assert checker()._identity({'builder': {'identity': 'fixture', 'run_record': '/tmp/private/transcript'}}, 'builder') is None


def test_stamp_admission_scans_evidence(tmp_path):
    # A generic leak, constructed so this regression fixture itself is safe.
    (tmp_path / 'build.json').write_text(json.dumps({'evidence': ''.join(['/home/', 'operator-secret', '/job'])}))
    errors = checker().check_stamps(tmp_path, 'a' * 40, {})
    assert any('release record leak:' in error for error in errors)


def _fixture_repo(tmp_path):
    """A small repo carrying the real release scripts and a synthetic project."""
    import shutil
    source = Path(__file__).parents[1]
    root = tmp_path / 'repo'
    (root / 'scripts').mkdir(parents=True)
    (root / 'fixturepkg').mkdir()
    for name in ('build-release.py', 'check-release-rights.py', 'leak-scan.py'):
        shutil.copy2(source / 'scripts' / name, root / 'scripts' / name)
    shutil.copy2(source / 'LICENSE.md', root / 'LICENSE.md')
    (root / 'pyproject.toml').write_text(
        '[build-system]\n'
        'requires = ["setuptools>=77"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\n'
        'name = "fixture-cli"\n'
        'version = "0.0.1"\n'
        'description = "fixture"\n'
        'readme = "README.md"\n'
        'requires-python = ">=3.10"\n'
        'license = "LicenseRef-fixture-1.0"\n'
        'license-files = ["LICENSE.md"]\n\n'
        '[tool.setuptools.packages.find]\n'
        'include = ["fixturepkg", "fixturepkg.*"]\n')
    (root / 'MANIFEST.in').write_text('include LICENSE.md\n')
    (root / 'README.md').write_text('# fixture\n')
    (root / '.gitignore').write_text('dist*/\nbuild/\n*.egg-info/\n__pycache__/\n')
    (root / 'fixturepkg' / '__init__.py').write_text('__version__ = "0.0.1"\n')
    (root / 'fixturepkg' / 'x.py').write_text('X = 1\n')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

    def commit(message):
        git('add', '-A')
        git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '-q', '-m', message)
        return git('rev-parse', 'HEAD')

    git('init', '-q')
    return root, git, commit


def _hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(directory).iterdir()}


def _write_records(root, revision, artifacts):
    stamp_dir = root / 'docs/release-stamps'
    stamp_dir.mkdir(parents=True, exist_ok=True)
    _dump(stamp_dir / 'build.json', {
        'builder': {'identity': BID, 'run_record': RUN_B},
        'commit': revision, 'artifacts': artifacts})
    for index, domain in enumerate(DOMAINS):
        _dump(stamp_dir / f'{domain}.json', {
            'domain': domain, 'verdict': 'PASS', 'commit': revision,
            'artifacts': artifacts,
            'stamper': {'identity': f'stamper:{index:012x}', 'run_record': f'{index:012x}'},
            'timestamp': '2026-10-19T00:00:00Z',
            'evidence': [f'{domain} check passed on fixture']})
    _dump(stamp_dir / 'notary.json', {
        'verdict': 'SEALED', 'commit': revision, 'artifacts': artifacts,
        'notary': {'identity': NID, 'run_record': RUN_N},
        'timestamp': '2026-10-19T00:00:00Z',
        'stamps': {d: hashlib.sha256((stamp_dir / f'{d}.json').read_bytes()).hexdigest()
                   for d in DOMAINS},
        'findings': {d: {'reexecuted': ['reran fixture check: same result']} for d in DOMAINS}})
    _dump(root / 'docs/release-rights.json', {
        'license_expression': 'LicenseRef-fixture-1.0',
        'license_files': {'LICENSE.md': hashlib.sha256((root / 'LICENSE.md').read_bytes()).hexdigest()},
        'components': [{'name': 'fixture', 'paths': ['fixturepkg/'], 'status': 'approved',
                        'approval_receipt': {'approver': OWNER, 'decision_ref': 'fixture decision',
                                             'source_revision': revision, 'scope': 'fixture paths'}}]})


def test_build_and_admission_scripts_work_as_a_pair(tmp_path):
    """The real build script and the real admission script agree on stamps.

    Synthetic records with opaque handles, written after the reviewed commit:
    the build record covers the release, the rebuilt bytes are identical, and
    admission passes. A wrong artifact hash in a stamp blocks admission.
    """
    import subprocess
    pytest.importorskip('build')
    root, git, commit = _fixture_repo(tmp_path)
    reviewed = commit('reviewed source')
    build = [sys.executable, str(root / 'scripts/build-release.py'), '--outdir']
    first = subprocess.run([*build, str(root / 'dist')], cwd=root, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    artifacts = _hashes(root / 'dist')
    _write_records(root, reviewed, artifacts)
    commit('release records')
    second = subprocess.run([*build, str(root / 'dist2')], cwd=root, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert _hashes(root / 'dist2') == artifacts, 'the normalised build is not reproducible'
    release = sorted(str(path) for path in (root / 'dist2').iterdir())
    check = subprocess.run(
        [sys.executable, str(root / 'scripts/check-release-rights.py'),
         '--commit', git('rev-parse', 'HEAD'), *release],
        cwd=root, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    assert 'human release authorization remains required' in check.stdout

    stamp = json.loads((root / 'docs/release-stamps/git.json').read_text())
    name = next(iter(stamp['artifacts']))
    stamp['artifacts'] = {**stamp['artifacts'], name: '0' * 64}
    _dump(root / 'docs/release-stamps/git.json', stamp)
    commit('a stamp records the wrong artifact hash')
    blocked = subprocess.run(
        [sys.executable, str(root / 'scripts/check-release-rights.py'),
         '--commit', git('rev-parse', 'HEAD'), *release],
        cwd=root, capture_output=True, text=True)
    assert blocked.returncode != 0
    assert 'artifact hashes differ' in blocked.stderr or 'artifact hashes do not match' in blocked.stderr

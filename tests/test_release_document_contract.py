"""Release procedure and package metadata must survive public projection."""
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_no_relative_links():
    text = (ROOT / 'README.md').read_text()
    targets = re.findall(r'\]\(([^\s)]+)', text)
    targets += re.findall(r'^\s*\[[^\]]+\]:\s*(\S+)', text, re.M)
    assert targets
    assert all(re.match(r'https?://', target) for target in targets), targets


def test_security_advisory_channel_is_documented():
    security = (ROOT / 'SECURITY.md').read_text()
    assert 'https://github.com/one-framework-foundation/one-cli/security/advisories/new' in security
    assert 'not yet been established' not in security


def test_lint_extra_includes_jsonschema_stubs():
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'types-jsonschema' in metadata['project']['optional-dependencies']['lint']


def test_skipped_harness_reasons_are_inline():
    source = (ROOT / 'onecli/harnesses.py').read_text()
    cited = re.findall(r'docs/[\w./-]+\.md', source)
    assert [path for path in cited if not (ROOT / path).exists()] == []
    assert 'endpoint constraint' in source


def test_release_network_inventory_matches_security():
    def inventory(name):
        text = (ROOT / name).read_text()
        return text.split('Automatic network inventory (also in SECURITY.md):')[1].split('Assistant `')[0].strip()
    assert inventory('docs/reproducible-releases.md').split('## Continuous integration')[0].strip() == inventory('SECURITY.md')


def test_release_admission_doc_states_where_the_records_are():
    """The verification doc must say the admission records ship at the tag and
    how to re-run the check; it once said they existed only at release time."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'release_rights', ROOT / 'scripts/check-release-rights.py')
    rights = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rights)
    text = (ROOT / 'docs/reproducible-releases.md').read_text()
    section = text.split('## Release admission')[1].split('\n## ')[0]
    for name in ['build.json', 'notary.json', *(f'{d}.json' for d in rights.STAMP_DOMAINS)]:
        assert f'`{name}`' in section, name
    assert 'committed in the repository at its tag' in section
    assert 'full clone' in section
    assert 'python scripts/check-release-rights.py --commit "$(git rev-parse HEAD)" dist/*' in section
    for stale in ('not in the repository', 'cannot be re-run', 'only at release time'):
        assert stale not in text
    workflow = ROOT / '.github/workflows/release.yml'
    public = ROOT / '.github/public-workflows/release.yml'
    shipped = (public if public.exists() else workflow).read_text()
    assert 'python scripts/check-release-rights.py --commit "$GITHUB_SHA" dist/*' in shipped
    assert 'python scripts/build-release.py' in shipped

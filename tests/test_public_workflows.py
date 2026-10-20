"""Public PR CI is hosted; only the tag publisher uses the release runner."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / ".github" / "workflows"
WORKFLOWS = sorted(DIR.glob("*.yml"))


def test_public_workflow_set_exists():
    assert WORKFLOWS, f"no public workflows in {DIR}"


def test_public_workflow_policy():
    import runpy
    runpy.run_path(str(ROOT / "scripts/check-public-workflows.py"))["check"](DIR)


@pytest.mark.parametrize("before,after", [
    ("needs: build", "needs: missing"),
    ("runs-on: ubuntu-latest", "runs-on: [self-hosted, one-release]"),
    ("labels: [self-hosted, one-release]", "labels: ubuntu-latest"),
    ("group: one-release", "group: Default"),
    ("    environment:", "    runs-on: ubuntu-latest\n    environment:"),
    ("      id-token: write", "      contents: write"),
    ("    runs-on: ubuntu-latest", "    permissions: {id-token: write}\n    runs-on: ubuntu-latest"),
    ('tags: ["v*"]', 'branches: [main]'),
    ('--commit "$GITHUB_SHA" dist/*', '--commit "$GITHUB_SHA" other/*'),
    ("actions/checkout@11d5960a326750d5838078e36cf38b85af677262", "actions/checkout@v4"),
    ("actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093", "actions/download-artifact@v4"),
])
def test_public_release_policy_rejects_regressions(tmp_path, before, after):
    import runpy
    import shutil
    for path in WORKFLOWS:
        shutil.copy(path, tmp_path / path.name)
    release = tmp_path / "release.yml"
    original = release.read_text()
    assert before in original
    release.write_text(original.replace(before, after))
    with pytest.raises(AssertionError):
        runpy.run_path(str(ROOT / "scripts/check-public-workflows.py"))["check"](tmp_path)


def test_public_release_filename_is_required(tmp_path):
    import runpy
    import shutil
    for path in WORKFLOWS:
        shutil.copy(path, tmp_path / path.name)
    (tmp_path / "release.yml").rename(tmp_path / "publish.yml")
    with pytest.raises(AssertionError):
        runpy.run_path(str(ROOT / "scripts/check-public-workflows.py"))["check"](tmp_path)


def test_public_ci_runs_the_leak_scan_and_tests():
    text = "\n".join(p.read_text() for p in WORKFLOWS)
    assert "python scripts/leak-scan.py ." in text
    assert "gitleaks" in text and "sha256sum -c" in text
    assert re.search(r'pip install -e "\.\[test\]"', text)
    assert "python -m pytest tests -q" in text

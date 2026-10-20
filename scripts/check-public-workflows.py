#!/usr/bin/env python3
"""Fail closed on runner/credential changes in the public workflow set."""
from pathlib import Path
import re
import sys


def check(directory):
    paths = sorted(directory.glob('*.yml'))
    assert {p.name for p in paths} == {'ci.yml', 'release.yml'}, 'unexpected workflow set'
    for path in paths:
        text = path.read_text()
        assert 'pull_request_target' not in text and 'workflow_run' not in text
        assert 'secrets.' not in text
        assert re.search(r'^permissions:\n  contents: read\n', text, re.M)
        assert text.count('persist-credentials: false') == text.count('uses: actions/checkout')
        for ref in re.findall(r'uses:\s*(\S+)', text):
            assert re.fullmatch(r'[\w.-]+/[\w.-]+@[0-9a-f]{40}', ref), f'action not pinned to a commit: {ref}'
        if path.name == 'release.yml':
            assert re.search(r'^on:\n  push:\n    tags: \["v\*"\]\n\npermissions:', text, re.M)
            assert re.findall(r'^  ([\w-]+):$', text.split('jobs:\n', 1)[1], re.M) == ['build', 'publish']
            build, publish = text.split('\n  publish:\n')
            assert build.count('runs-on:') == publish.count('runs-on:') == 1
            assert 'runs-on: ubuntu-latest' in build
            assert 'self-hosted' not in build and 'id-token' not in build and 'write' not in build
            assert 'needs: build' in publish
            assert "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')" in publish
            assert 'runs-on:\n      group: one-release\n      labels: [self-hosted, one-release]' in publish
            assert 'environment:\n      name: pypi' in publish
            assert 'permissions:\n      id-token: write\n' in publish
            assert text.count('id-token:') == 1 and text.count(': write') == 1
            assert 'actions/checkout' not in publish
            assert 'docker' not in publish.lower()
            assert 'pypa/gh-action-pypi-publish' not in publish
            assert 'actions/download-artifact@' in publish
            assert text.count('name: one-cli-${{ github.ref_name }}') == 2
            assert 'https://pypi.org/_/oidc/mint-token' in publish
            assert '"twine", "upload"' in publish
            required_steps = (
                'scripts/build-release.py', 'scripts/check-release-rights.py --commit "$GITHUB_SHA" dist/*',
                'scripts/smoke-wheel.py', 'actions/upload-artifact@')
            assert all(step in build for step in required_steps)
            order = [build.index(step) for step in required_steps]
            assert order == sorted(order)
        else:
            assert 'self-hosted' not in text and 'write' not in text and 'id-token' not in text
            for value in re.findall(r'runs-on:\s*(.+)', text):
                assert re.fullmatch(r'(ubuntu|macos|windows)-latest|\$\{\{ matrix\.os \}\}', value.strip())
            for names in re.findall(r'os:\s*\[([^\]]*)\]', text):
                assert all(re.fullmatch(r'(ubuntu|macos|windows)-latest', name.strip()) for name in names.split(','))


if __name__ == '__main__':
    check(Path(sys.argv[1]))
    print('Public workflow policy: PASS')

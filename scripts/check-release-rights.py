"""Fail release admission on unresolved rights, missing notices or missing sign-offs.

This verifies the release owner's per-component approvals, the exact license
bytes inside each artifact, six domain stamps and the notary seal over them.
It does not establish legal ownership.
Run after building: python scripts/check-release-rights.py --commit <sha> dist/*
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile


# The only approver whose approvals admit a release: the repository owner's
# GitHub login, a public identity. Kept here, not in the stamp-exempt
# inventory, so changing it invalidates the stamps.
RELEASE_OWNER = 'jessesep'


def check_inventory(inventory: dict, commit: str,
                    commit_ok=lambda approved, released: approved == released) -> list[str]:
    """``commit_ok(approved, released)`` decides whether an approval's
    ``source_revision`` covers the release commit, as for stamps."""
    errors = []
    if not inventory.get('license_expression'):
        errors.append('approved license expression missing')
    if not inventory.get('license_files'):
        errors.append('approved license file hashes missing')
    components = inventory.get('components', [])
    if not components:
        errors.append('component rights approval inventory missing')
    for component in components:
        receipt = component.get('approval_receipt')
        fields = ('approver', 'decision_ref', 'source_revision', 'scope')
        complete = (isinstance(receipt, dict)
                    and all(isinstance(receipt.get(field), str) and receipt[field].strip()
                            for field in fields)
                    and re.fullmatch(r'[0-9a-f]{40}', receipt['source_revision']) is not None)
        name = component.get('name', '(unnamed)')
        if component.get('status') != 'approved' or not complete:
            errors.append(f"rights approval pending: {name}")
        elif receipt['approver'] != RELEASE_OWNER:
            errors.append(f"rights approval for {name} is not the release owner's: "
                          f"approver {receipt['approver']!r}")
        elif not commit_ok(receipt['source_revision'], commit):
            errors.append(f"rights approval for {name} is stale: source_revision "
                          f"{receipt['source_revision']} does not cover release commit {commit}")
    return errors


def check_artifact(path: Path, inventory: dict) -> list[str]:
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith('/')}
        metadata = [data for name, data in files.items() if name.endswith('.dist-info/METADATA')]
        license_prefix = next((name[:-len('METADATA')] + 'licenses/' for name in files
                               if name.endswith('.dist-info/METADATA')), '')
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path) as archive:
            files = {member.name.split('/', 1)[-1]: archive.extractfile(member).read()
                     for member in archive.getmembers() if member.isfile()}
        metadata = [data for name, data in files.items() if name == 'PKG-INFO']
        license_prefix = ''
    else:
        return [f'{path.name}: unsupported release artifact']
    errors = []
    if len(metadata) != 1:
        return [f'{path.name}: expected one distribution metadata file']
    headers = BytesParser().parsebytes(metadata[0])
    if headers.get('License-Expression') != inventory.get('license_expression') or not inventory.get('license_expression'):
        errors.append(f'{path.name}: approved License-Expression missing or mismatched')
    declared = headers.get_all('License-File', [])
    for name, digest in inventory.get('license_files', {}).items():
        payload = files.get(license_prefix + name)
        if name not in declared or payload is None or hashlib.sha256(payload).hexdigest() != digest:
            errors.append(f'{path.name}: approved license/notice missing or mismatched: {name}')
    return errors


# Rights are gated by the owner's approvals in the rights inventory, not by a
# domain stamp, so there is no 'legal' stamp.
STAMP_DOMAINS = ('leak', 'safety', 'git', 'readme', 'code', 'neatness')
# Release records written after the reviewed revision: the stamps, and the
# rights inventory that records the owner's approvals.
STAMP_EXEMPT_PREFIX = 'docs/release-stamps/'
RIGHTS_INVENTORY = 'docs/release-rights.json'


def _load(path: Path, label: str, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f'{label} missing: {path.name}')
        return None
    try:
        record = json.loads(path.read_bytes())
    except ValueError:
        errors.append(f'{label} unreadable: {path.name}')
        return None
    if not isinstance(record, dict):
        errors.append(f'{label} unreadable: {path.name}')
        return None
    return record


# Identities and run records are opaque handles, never names: a shipped
# record must not reveal an agent, model, person or run.
IDENTITY_PATTERN = re.compile(r'^(?:builder|stamper|notary):[0-9a-f]{12}$')
RUN_RECORD_PATTERN = re.compile(r'^[0-9a-f]{12}$')


def _identity(record: dict, role: str) -> str | None:
    party = record.get(role)
    if (isinstance(party, dict)
            and isinstance(party.get('identity'), str)
            and IDENTITY_PATTERN.fullmatch(party['identity'])
            and isinstance(party.get('run_record'), str)
            and RUN_RECORD_PATTERN.fullmatch(party['run_record'])):
        return party['identity']
    return None


def check_stamps(stamp_dir: Path, commit: str, artifacts: dict[str, str],
                 commit_ok=lambda stamped, released: stamped == released) -> list[str]:
    """Six domain stamps from non-builder stampers, sealed by a distinct notary.

    Every identity is an opaque ``builder:``/``stamper:``/``notary:`` handle
    followed by 12 hex digits; every run record is 12 hex digits.

    ``artifacts`` maps artifact file name to the sha256 of the bytes being released.
    ``commit_ok(stamped, released)`` decides whether a stamped commit covers the release.
    """
    errors: list[str] = []
    if stamp_dir.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("release_leaks", Path(__file__).with_name("leak-scan.py"))
        scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner)
        patterns = dict(scanner.GENERIC)
        if scanner.PRIVATE_LIST.exists():
            patterns.update(scanner.load_private())
        for rel, line, category in scanner.scan(stamp_dir.resolve(), patterns):
            errors.append(f"release record leak: {rel}:{line}: {category}")
    build = _load(stamp_dir / 'build.json', 'build record', errors)
    builder = _identity(build, 'builder') if build else None
    if build and not builder:
        errors.append('build record has no builder identity and run record')
    if build and not commit_ok(build.get('commit'), commit):
        errors.append(f"build record commit {build.get('commit')} does not cover release commit {commit}")
    if build and build.get('artifacts') != artifacts:
        errors.append('build record artifact hashes do not match the artifacts being released')

    stamp_hashes: dict[str, str] = {}
    stampers: dict[str, str] = {}
    for domain in STAMP_DOMAINS:
        path = stamp_dir / f'{domain}.json'
        stamp = _load(path, f'{domain} stamp', errors)
        if stamp is None:
            continue
        stamp_hashes[domain] = hashlib.sha256(path.read_bytes()).hexdigest()
        if stamp.get('domain') != domain:
            errors.append(f'{domain} stamp names domain {stamp.get("domain")!r}')
        if stamp.get('verdict') != 'PASS':
            errors.append(f'{domain} stamp verdict is {stamp.get("verdict")!r}, not PASS')
        if not commit_ok(stamp.get('commit'), commit):
            errors.append(f"{domain} stamp is stale: commit {stamp.get('commit')} does not cover {commit}")
        if stamp.get('artifacts') != artifacts:
            errors.append(f'{domain} stamp is stale: artifact hashes differ from the artifacts being released')
        if not stamp.get('timestamp') or not stamp.get('evidence'):
            errors.append(f'{domain} stamp lacks timestamp or evidence lines')
        stamper = _identity(stamp, 'stamper')
        if not stamper:
            errors.append(f'{domain} stamp has no stamper identity and run record')
            continue
        stampers[domain] = stamper
        if builder and stamper == builder:
            errors.append(f'{domain} stamper {stamper} is the builder')

    seal = _load(stamp_dir / 'notary.json', 'notary seal', errors)
    if seal is None:
        return errors
    if seal.get('verdict') != 'SEALED':
        errors.append(f"notary verdict is {seal.get('verdict')!r}, not SEALED")
    notary = _identity(seal, 'notary')
    if not notary:
        errors.append('notary seal has no notary identity and run record')
    elif builder and notary == builder:
        errors.append(f'notary {notary} is the builder')
    for domain, stamper in stampers.items():
        if notary and notary == stamper:
            errors.append(f'notary {notary} is the {domain} stamper')
    if not commit_ok(seal.get('commit'), commit):
        errors.append(f"notary seal is stale: commit {seal.get('commit')} does not cover {commit}")
    if seal.get('artifacts') != artifacts:
        errors.append('notary seal is stale: artifact hashes differ from the artifacts being released')
    if not seal.get('timestamp'):
        errors.append('notary seal lacks a timestamp')
    sealed = seal.get('stamps') if isinstance(seal.get('stamps'), dict) else {}
    findings = seal.get('findings') if isinstance(seal.get('findings'), dict) else {}
    for domain in STAMP_DOMAINS:
        if domain in stamp_hashes and sealed.get(domain) != stamp_hashes[domain]:
            errors.append(f'{domain} stamp changed after sealing (hash differs from the notary seal)')
        finding = findings.get(domain)
        if not isinstance(finding, dict) or not finding.get('reexecuted'):
            errors.append(f'notary seal has no re-executed evidence for the {domain} stamp')
    return errors


def git_commit_covers(repo: Path):
    """A reviewed commit covers a release commit when only release records
    (stamps and the rights inventory) changed between them."""
    def covers(stamped, released) -> bool:
        if not stamped or not released:
            return False
        if stamped == released:
            return True
        ancestor = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', stamped, released],
                                  capture_output=True)
        if ancestor.returncode != 0:
            return False
        changed = subprocess.run(['git', '-C', str(repo), 'diff', '--no-renames', '--name-only', '-z', stamped, released],
                                 capture_output=True, text=True, check=True).stdout.split("\0")[:-1]
        return all(name == RIGHTS_INVENTORY or name.startswith(STAMP_EXEMPT_PREFIX)
                   for name in changed)
    return covers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--inventory', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs/release-rights.json')
    parser.add_argument('--stamps', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs/release-stamps')
    parser.add_argument('--commit', required=True, help='source commit the artifacts are released from')
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    covers = git_commit_covers(Path(__file__).resolve().parents[1])
    errors = check_inventory(inventory, args.commit, covers)
    for path in args.artifacts:
        errors.extend(check_artifact(path, inventory))
    released = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.artifacts}
    errors.extend(check_stamps(args.stamps, args.commit, released, covers))
    if errors:
        raise SystemExit('RELEASE BLOCKED:\n' + '\n'.join(errors))
    print('Owner approvals, license bytes, six stamps and the notary seal match; '
          'human release authorization remains required.')


if __name__ == '__main__':
    main()

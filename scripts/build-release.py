#!/usr/bin/env python3
"""Build reproducible release bytes from a clean, reviewed Git revision.

Install scripts/release-build-requirements.txt first. Output must be empty.
The admission checker still validates the resulting bytes against every stamp.
"""
import argparse
import gzip
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='dist', type=Path)
    args = parser.parse_args()
    output = args.outdir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('output directory must be empty')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    record = ROOT / 'docs/release-stamps/build.json'
    revision = json.loads(record.read_text())['commit'] if record.exists() else head
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        parser.error('build record commit must be a full lowercase Git SHA')
    spec = importlib.util.spec_from_file_location('rights', ROOT / 'scripts/check-release-rights.py')
    rights = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rights)
    if not rights.git_commit_covers(ROOT)(revision, head):
        parser.error('build record does not cover HEAD')
    epoch = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', revision], text=True))
    env = {**os.environ, 'SOURCE_DATE_EPOCH': str(epoch)}
    print(f'Rebuilding reviewed source {revision}; SOURCE_DATE_EPOCH={epoch}', flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        source = work / 'source'
        source.mkdir()
        archive = work / 'source.tar'
        subprocess.run(['git', 'archive', '--output', str(archive), revision], check=True)
        with tarfile.open(archive) as tree:
            tree.extractall(source, filter='data')
        subprocess.run([sys.executable, '-m', 'build', '--no-isolation', '--sdist',
                        '--outdir', str(output), str(source)], env=env, check=True)
        sdist, = output.glob('*.tar.gz')
        normalized = work / 'normalized.tar.gz'
        # setuptools' sdist uses wall-clock gzip and generated-file timestamps.
        # Keep content/modes, canonicalize archive metadata with stdlib tar/gzip.
        with tarfile.open(sdist) as original, normalized.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as target:
                    for member in sorted(original.getmembers(), key=lambda item: item.name):
                        member.mtime = epoch
                        member.uid = member.gid = 0
                        member.uname = member.gname = ''
                        member.pax_headers = {}
                        target.addfile(member, original.extractfile(member) if member.isfile() else None)
        sdist.write_bytes(normalized.read_bytes())
        unpacked = work / 'unpacked'
        unpacked.mkdir()
        with tarfile.open(sdist) as tree:
            tree.extractall(unpacked, filter='data')
        package, = unpacked.iterdir()
        subprocess.run([sys.executable, '-m', 'build', '--no-isolation', '--wheel',
                        '--outdir', str(output), str(package)], env=env, check=True)


if __name__ == '__main__':
    main()

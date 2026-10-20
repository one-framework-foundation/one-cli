import pytest
from onecli import terminal, toolview
from onecli.tools import search


@pytest.mark.parametrize('width', [40, 100])
def test_compact_tools_keep_details_and_failures(width):
    ui = terminal.Surface()
    command = 'python3 -m render list ' + 'long-argument ' * 20
    ui.tool_start('bash', {'command': command})
    ui.tool_result('exit 0\n' + '\n'.join(f'item {i}' for i in range(50)))
    rows = ui.transcript_rows(width)
    text = '\n'.join(str(row) for row in rows)
    assert 'Ctrl-O:' not in text
    assert 'exit 0' not in text
    assert len(rows) <= 3
    ui.key('\x0f')
    assert 'item 49' in str(ui.transcript_rows(width))
    assert 'long-argument' in str(ui.transcript_rows(width))
    ui.tool_result('exit 1\npermission denied')
    ui.key('\x0f')
    assert 'permission denied' in str(ui.transcript_rows(width))


@pytest.mark.parametrize('name', ['grep', 'glob'])
def test_home_search_requires_narrow_path(tmp_path, monkeypatch, name):
    monkeypatch.setattr(search.Path, 'home', classmethod(lambda cls: tmp_path))
    (tmp_path / 'needle.txt').write_text('needle')
    out = search.grep(str(tmp_path), 'needle') if name == 'grep' else search.glob_files(str(tmp_path), '**/*')
    assert out.startswith('error: search scope too broad')


def test_glob_prunes_archives_and_worktrees(tmp_path):
    for directory in ['src', 'archive', 'backups', 'scratch/wt-old']:
        path = tmp_path / directory
        path.mkdir(parents=True)
        (path / 'probe.py').write_text('needle')
    out = search.glob_files(str(tmp_path), '**/*.py')
    assert '/src/probe.py' in out
    assert '/archive/' not in out and '/backups/' not in out and '/wt-old/' not in out


def test_nonzero_exit_stays_visible():
    assert 'exit 7' in '\n'.join(toolview.result_rows('exit 7\nno match', 40))


@pytest.mark.parametrize('command', ['find ~ -iname probe', 'ls -d ~/*/', 'rg probe ~'])
def test_shell_home_sweep_refused(tmp_path, command, monkeypatch):
    from onecli.tools import shell
    from onecli.tools.shell import bash
    monkeypatch.setattr(shell.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('broad search executed')))
    assert bash(str(tmp_path), command).startswith('error: search scope too broad')


def test_shell_rg_pattern_is_not_a_search_root(tmp_path):
    from onecli.tools.shell import _broad_search
    assert not _broad_search("rg / src", str(tmp_path))
    assert not _broad_search("rg -g '*.py' / src", str(tmp_path))
    assert not _broad_search("find src -name /", str(tmp_path))
    assert _broad_search("rg --files ~", str(tmp_path))


def test_python_search_prunes_and_caps_results(tmp_path, monkeypatch):
    monkeypatch.setattr(search.shutil, 'which', lambda _: None)
    for directory in ['src', 'archive', 'scratch/wt-old']:
        path = tmp_path / directory
        path.mkdir(parents=True)
        for i in range(150):
            (path / f'probe{i}.txt').write_text('needle\n')
    out = search.grep(str(tmp_path), 'needle', max_results=1000)
    assert sum(':1:needle' in line for line in out.splitlines()) == 100
    assert '150 matching lines in 150 files' in out
    assert '/archive/' not in out and '/wt-old/' not in out


@pytest.mark.parametrize('command', ["rg --regexp=needle ~", "rg --file=patterns.txt ~"])
def test_shell_rg_inline_pattern_still_checks_root(tmp_path, command):
    from onecli.tools.shell import _broad_search
    assert _broad_search(command, str(tmp_path))


@pytest.mark.parametrize('command', ["ls src\necho /", "ls src\ncd ..", "ls src\\\n  docs"])
def test_shell_newline_ends_the_discovery_command(tmp_path, command):
    from onecli.tools.shell import _broad_search
    assert not _broad_search(command, str(tmp_path))


def test_shell_sweep_after_a_newline_is_refused(tmp_path):
    from onecli.tools.shell import _broad_search
    assert _broad_search('python3 -c "print(1)"\nls /', str(tmp_path))


@pytest.mark.parametrize('command', ['ls ..', 'ls -la ../', 'rg -n foo ..', 'find .. -name x'])
def test_shell_workspace_ancestor_listing_is_allowed(tmp_path, command):
    from onecli.tools.shell import _broad_search
    workspace = tmp_path / 'a' / 'b'
    workspace.mkdir(parents=True)
    assert not _broad_search(command, str(workspace))
    assert not _broad_search(f'ls {tmp_path}', str(workspace))


@pytest.mark.parametrize('command', ['ls /', 'ls ~', 'find / -name x', 'rg foo ~', 'ls -d ~/*/'])
def test_shell_home_and_root_sweeps_stay_refused(tmp_path, command):
    from onecli.tools.shell import _broad_search
    assert _broad_search(command, str(tmp_path / 'a'))


def test_search_tools_accept_a_workspace_ancestor(tmp_path):
    workspace = tmp_path / 'a' / 'b'
    workspace.mkdir(parents=True)
    (tmp_path / 'a' / 'note.txt').write_text('needle\n')
    assert 'note.txt:1:needle' in search.grep(str(workspace), 'needle', path='..')
    assert 'note.txt' in search.glob_files(str(workspace), '*.txt', path='..')


def _pruning_corpus(root):
    for name in ['src/ok.py', 'src/wt-notes.md', 'src/a.bakery.py', 'src/.bakrc',
                 'wt-old/x.py', 'cfg.bak.d/x.py', '.bak/x.py', 'archive/x.py']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('needle\n')


KEPT = {'src/ok.py', 'src/wt-notes.md', 'src/a.bakery.py', 'src/.bakrc'}


def _hits(out, root):
    return {line.split(':', 1)[0].removeprefix(f'{root}/') for line in out.splitlines()
            if line.startswith(f'{root}/') and ':1:' in line}


class _AllowAll:
    def check(self, *_args):
        return type('Verdict', (), {'allowed': True})()


@pytest.mark.parametrize('backend', ['rg', 'python', 'python-tracked', 'restricted'])
def test_pruning_skips_directories_not_files_in_every_backend(tmp_path, backend):
    import shutil
    import subprocess
    root = tmp_path / 'w'
    _pruning_corpus(root)
    if backend == 'python-tracked':
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'add', '-f', '.'], check=True)
    name = {'python-tracked': 'python'}.get(backend, backend)
    if name == 'rg' and not shutil.which(name):
        pytest.skip(f'{name} not on PATH on this host: {name} pruning not measured')
    if backend == 'restricted':
        out = search.restricted('grep', str(root), {'pattern': 'needle'}, [_AllowAll()])
    else:
        out = search.search(str(root), 'needle', backend=name)
    assert _hits(out, root) == KEPT, out
    assert 'wt-*' in out and '.bak*' in out


def test_glob_prunes_directories_not_files(tmp_path):
    _pruning_corpus(tmp_path)
    for out in (search.glob_files(str(tmp_path), '**/*'),
                search._restricted_glob(tmp_path, '**/*', {}, [])):
        found = {line.removeprefix(f'{tmp_path}/') for line in out.splitlines()
                 if line.startswith(str(tmp_path))}
        assert found == KEPT, out


def test_cap_footer_does_not_suggest_raising_max_results(tmp_path, monkeypatch):
    monkeypatch.setattr(search.shutil, 'which', lambda _: None)
    for i in range(150):
        (tmp_path / f'probe{i}.txt').write_text('needle\n')
    assert 'raise max_results' not in search.grep(str(tmp_path), 'needle', max_results=1000)
    assert 'raise max_results' in search.grep(str(tmp_path), 'needle', max_results=10)


def test_grep_schema_states_the_real_cap():
    from pathlib import Path
    from onecli import tools
    text = Path(tools.__file__).read_text()
    assert 'max 300' not in text and 'Default 100, max 100' in text

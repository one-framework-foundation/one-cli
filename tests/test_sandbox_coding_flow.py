"""Host-owned source -> offline repair/test -> explicit artifact inspection."""
from onecli.sandbox.shell import run_shell, read_artifact


def test_repair_test_and_inspect_without_writeback(tmp_path):
    source = tmp_path / 'approved-project'
    stage = tmp_path / 'staging'
    source.mkdir()
    stage.mkdir()
    original = b'def add(a, b):\n    return a - b\n'
    (source / 'app.py').write_bytes(original)
    (source / 'test_app.py').write_text(
        'import unittest\nfrom app import add\n'
        'class Check(unittest.TestCase):\n'
        '    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n')
    result = run_shell(source, stage,
        "python3 -c \"from pathlib import Path; p=Path('app.py'); "
        "p.write_text(p.read_text().replace('a - b', 'a + b'))\" && "
        "python3 -B -m unittest -v")
    assert result.returncode == 0, result.output
    assert 'OK' in result.output
    assert read_artifact(result, 'app.py') == b'def add(a, b):\n    return a + b\n'
    assert (source / 'app.py').read_bytes() == original

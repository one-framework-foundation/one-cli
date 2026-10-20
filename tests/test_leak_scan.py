"""scripts/leak-scan.py refuses infrastructure leaks and accepts placeholders.

Fixture values are assembled at runtime so this file does not trip the scanner
it tests.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leak-scan.py"
spec = importlib.util.spec_from_file_location("leak_scan", SCRIPT)
leak_scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leak_scan)

HOME = "/" + "home"


def _fixture(*parts):
    """Construct synthetic leaks at runtime, outside static path expressions."""
    return "".join(parts)


@pytest.mark.parametrize("line,category", [
    ("url = 'http://" + "100.101" + ".7.9:8911/v1'", "tailnet_ip"),
    (_fixture("lan = '", "192.168", ".1.20'"), "private_lan_ip"),
    ("path = '" + HOME + "/someone/project'", "home_path"),
    ("see `" + HOME + "/someone`", "home_path"),
    (_fixture("chat = ", "-100", "1234567890"), "telegram_chat_id"),
    ("host = 'box." + "corp-net" + ".ts.net'", "tailnet_domain"),
    (_fixture("key = '", "-----BEGIN RSA ", "PRIVATE KEY-----'"), "private_key"),
])
def test_leaks_fail(tmp_path, line, category):
    (tmp_path / "leak.py").write_text(line + "\n")
    findings = leak_scan.scan(tmp_path, leak_scan.GENERIC)
    assert [(f, n, c) for f, n, c in findings] == [("leak.py", 1, category)]
    assert leak_scan.main([str(tmp_path)]) == 1


@pytest.mark.parametrize("line", [
    "url = 'http://" + "100.64.0.12" + ":8911/v1'",
    "lan = '" + "192.0.2.1" + "'",
    "path = '" + HOME + "/user/project'",
    "host = 'box.example.ts.net'",
    "cidr = '100.64.0.0/10'",
])
def test_placeholders_pass(tmp_path, line):
    (tmp_path / "ok.py").write_text(line + "\n")
    assert leak_scan.scan(tmp_path, leak_scan.GENERIC) == []


def test_finding_never_echoes_the_value(tmp_path, capsys):
    value = "100.101" + ".7.9"
    (tmp_path / "leak.txt").write_text(f"x {value}\n")
    assert leak_scan.main([str(tmp_path)]) == 1
    out = capsys.readouterr()
    assert value not in out.out + out.err


def test_symlink_target_is_scanned(tmp_path):
    (tmp_path / "link").symlink_to(HOME + "/someone/.harnesses")
    assert leak_scan.scan(tmp_path, leak_scan.GENERIC) == [("link", 1, "home_path")]


def test_file_names_and_binary_files_are_scanned(tmp_path):
    (tmp_path / ("notes-" + "100.101" + ".7.9.md")).write_text("clean\n")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\x00\xff" + (HOME + "/someone/x").encode())
    assert sorted(leak_scan.scan(tmp_path, leak_scan.GENERIC)) == [
        ("notes-" + "100.101" + ".7.9.md", 0, "tailnet_ip"), ("pic.png", 1, "home_path")]


def test_private_list_format(tmp_path):
    rules = tmp_path / "list.txt"
    rules.write_text("# comment\n\nhost\t(?i)\\bbox-a\\b\n")
    assert leak_scan.load_private(rules) == {"host#3": "(?i)\\bbox-a\\b"}
    (tmp_path / "a.md").write_text("ran on Box-A\n")
    assert leak_scan.scan(tmp_path, leak_scan.load_private(rules)) == [("a.md", 1, "host")]


def test_this_tree_is_clean():
    """The tree this test runs in must pass its own scan.

    The generic scan runs only when no local pattern list is present; with
    one, the caller scans with the full list instead.
    """
    if leak_scan.PRIVATE_LIST.exists():
        pytest.skip("skipped when a local pattern list is present")
    root = SCRIPT.parents[1]
    assert leak_scan.scan(root, leak_scan.GENERIC) == []


@pytest.mark.parametrize("source", [
    'p = os.path.join("hidden", "deployment", "worker.py")',
    'p = os.path.join(ROOT, "hidden", "deployment", "worker.py")',
    'p = Path(ROOT) / "hidden" / "deployment" / "worker.py"',
    'p = Path("hidden") / "deployment" / "worker.py"',
    'p = Path("hidden", "deployment", "worker.py")',
    'p = "hidden/" "deployment/" "worker.py"',
    'p = "hidden/" + "deployment/" + "worker.py"',
])
def test_split_private_paths(tmp_path, source):
    (tmp_path / "split.py").write_text(source)
    findings = leak_scan.scan(tmp_path, {"synthetic_private": r"hidden/deployment/worker\.py"})
    assert findings == [("split.py", 0, "synthetic_private")]


def test_unrelated_literals_are_not_joined(tmp_path):
    (tmp_path / "separate.py").write_text('a = "hidden"\nb = "deployment"\nc = "worker.py"\n')
    assert leak_scan.scan(tmp_path, {"synthetic_private": r"hidden/deployment/worker\.py"}) == []


def test_non_git_fallback_skips_generated_caches(tmp_path):
    """An archive copy (no .git) holds caches that were never shipped."""
    leak = ("p = '" + HOME + "/someone/project'\n").encode()
    for rel in ("__pycache__/x.cpython-312.pyc", ".pytest_cache/v/cache/nodeids",
                ".mypy_cache/x.json", ".ruff_cache/x", "stray.pyc"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(leak)
    assert leak_scan.scan(tmp_path, leak_scan.GENERIC) == []
    (tmp_path / "real.bin").write_bytes(b"\0" + leak)
    assert [(f, c) for f, _, c in leak_scan.scan(tmp_path, leak_scan.GENERIC)] == [
        ("real.bin", "home_path")]

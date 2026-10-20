import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "update-readme.py"
SPEC = importlib.util.spec_from_file_location("update_readme", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_replace_block_only_changes_named_block():
    source = "before\n<!-- readme:x:start -->\nold\n<!-- readme:x:end -->\nafter\n"
    assert MODULE.replace_block(source, "x", "new") == (
        "before\n<!-- readme:x:start -->\nnew\n<!-- readme:x:end -->\nafter\n"
    )


def test_benchmark_index_contains_only_public_guidance():
    rendered = MODULE.benchmark_index()
    assert "dated snapshots" in rendered
    assert "scripts/one-bench" in rendered
    assert "scripts/one-arena" in rendered
    assert "docs/" not in rendered


def test_project_facts_are_derived_without_importing_the_application():
    config = MODULE.project_config()
    assert config["version"] == "0.1.1"
    assert "one" in config["scripts"]


def test_project_facts_carry_no_test_count():
    """A count changed on every PR and conflicted on every merge."""
    facts = MODULE.project_facts(MODULE.project_config())
    assert "test functions" not in facts


def test_check_mode_returns_one_for_stale_readme(tmp_path, monkeypatch, capsys):
    readme = tmp_path / "README.md"
    readme.write_text("old\n")
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "render", lambda: {readme: "new\n"})
    monkeypatch.setattr(sys, "argv", ["update-readme.py", "--check"])
    assert MODULE.main() == 1
    assert "README.md is stale" in capsys.readouterr().err

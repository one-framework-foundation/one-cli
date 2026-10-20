"""Native skills: listed in the system prompt, loaded whole in one call."""

from __future__ import annotations

from onecli import permissions, skills
from onecli import tools as toolkit

BODY = "---\nname: schematic-drafting\ndescription: \"KiCad layout rules\"\n---\n\n" + \
       "\n".join(f"rule {i}: keep wires on grid" for i in range(3000))


def _skill(root, folder, body=BODY):
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body)
    return d


def test_skills_in_the_home_library_are_disclosed_in_the_system_prompt(tmp_path):
    home, ws = tmp_path / "home", tmp_path / "ws"
    ws.mkdir()
    _skill(home / ".claude" / "skills", "schematic-drafting")
    system = permissions.compose_system("base", str(ws), home=str(home))
    assert "- schematic-drafting: KiCad layout rules" in system


def test_the_skill_tool_returns_the_whole_skill_in_one_call(tmp_path, monkeypatch):
    home, ws = tmp_path / "home", tmp_path / "ws"
    ws.mkdir()
    folder = _skill(home / ".claude" / "skills", "schematic-drafting")
    monkeypatch.setenv("HOME", str(home))
    out = toolkit.run("skill", {"name": "schematic-drafting"}, str(ws))
    assert "rule 2999:" in out
    assert str(folder) in out


def test_a_workspace_skill_overrides_the_home_one_by_name(tmp_path):
    home, ws = tmp_path / "home", tmp_path / "ws"
    _skill(home / ".claude" / "skills", "schematic-drafting")
    _skill(ws / ".claude" / "skills", "local", BODY.replace("KiCad layout rules", "project rules"))
    found = skills.discover(str(ws), str(home), trust_project=True)
    assert found["schematic-drafting"].description == "project rules"


def test_an_unknown_skill_names_the_ones_that_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _skill(tmp_path / ".claude" / "skills", "schematic-drafting")
    out = toolkit.run("skill", {"name": "nope"}, str(tmp_path))
    assert "no skill named 'nope'" in out and "schematic-drafting" in out

from pathlib import Path
from shutil import copytree

from core.middleware.skills import create_read_skill_tool, discover_skills


def test_profile_skill_is_discoverable_from_an_unrelated_cwd(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "skills" / "profile"
    home = tmp_path / "home"
    installed = home / ".quasipilot" / "skills" / "profile"
    copytree(source, installed)
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "project"
    cwd.mkdir()

    skills = discover_skills(cwd)
    assert len(skills) == 1
    assert skills[0].name == "profile"
    assert skills[0].path == (installed / "SKILL.md").resolve()
    assert skills[0].description

    result = create_read_skill_tool(cwd=cwd).invoke({"name": "profile"})
    body = (source / "SKILL.md").read_text().split("---", 2)[2].strip()
    assert body in result
    assert f"Skill directory: {installed.resolve()}" in result

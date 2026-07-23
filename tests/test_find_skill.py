import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "head_pod" / "tools"))
from find_skill import EMBED_FILENAME, find_skills, format_matches, parse_frontmatter, search


def _write_skill(directory: Path, name: str, description: str, body: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    skill_md = directory / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    return skill_md


def _unit_vec(index: int, size: int = 8) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index % size] = 1.0
    return vector


class TestParseFrontmatter:
    def test_valid_frontmatter(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: Calc\ndescription: Does math.\n---\n# Body")
        assert parse_frontmatter(path) == {"name": "Calc", "description": "Does math."}

    def test_missing_description(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: Calc\n---\n")
        assert parse_frontmatter(path) == {"name": "Calc"}


class TestFindSkills:
    def test_finds_valid_skills(self, tmp_path):
        _write_skill(tmp_path / "skills" / "calc", "Calculator", "Does arithmetic.")
        skills = find_skills(tmp_path / "skills")
        assert skills[0]["path"] == "/workspace/skills/calc/SKILL.md"

    def test_skips_skill_missing_required_frontmatter(self, tmp_path):
        _write_skill(tmp_path / "ok", "Good Skill", "Has both fields.")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("---\nname: Incomplete\n---\n")
        skills = find_skills(tmp_path)
        assert [skill["name"] for skill in skills] == ["Good Skill"]


class TestSearch:
    def test_returns_most_relevant_skill_first(self, tmp_path):
        calc_md = _write_skill(tmp_path / "calc", "Calculator", "Performs arithmetic.")
        weather_md = _write_skill(tmp_path / "weather", "Weather", "Fetches forecasts.")
        np.save(calc_md.parent / EMBED_FILENAME, _unit_vec(0))
        np.save(weather_md.parent / EMBED_FILENAME, _unit_vec(1))

        results = search(_unit_vec(0), find_skills(tmp_path), min_similarity=0.0)
        assert [item["name"] for item in results][:2] == ["Calculator", "Weather"]

    def test_skills_below_threshold_are_excluded(self, tmp_path):
        skill_md = _write_skill(tmp_path / "calc", "Calculator", "Performs arithmetic.")
        np.save(skill_md.parent / EMBED_FILENAME, _unit_vec(1))
        results = search(_unit_vec(0), find_skills(tmp_path), min_similarity=0.5)
        assert results == []

    def test_zero_results_are_allowed(self, tmp_path):
        assert search(_unit_vec(0), [], min_similarity=0.0) == []

    def test_plain_query_text_is_not_accepted(self, tmp_path):
        _write_skill(tmp_path / "calc", "Calculator", "Performs arithmetic.")
        with pytest.raises(ValueError):
            search("arithmetic", find_skills(tmp_path), min_similarity=0.0)

    def test_missing_sidecar_is_not_generated_by_container(self, tmp_path):
        skill_md = _write_skill(tmp_path / "calc", "Calculator", "Performs arithmetic.")
        results = search(_unit_vec(0), find_skills(tmp_path), min_similarity=0.0)
        assert results == []
        assert not (skill_md.parent / EMBED_FILENAME).exists()

    def test_invalid_sidecar_is_deleted_and_skipped_without_regeneration(self, tmp_path):
        skill_md = _write_skill(tmp_path / "calc", "Calculator", "Performs arithmetic.")
        embed_path = skill_md.parent / EMBED_FILENAME
        embed_path.write_bytes(b"bad-sidecar")
        results = search(_unit_vec(0), find_skills(tmp_path), min_similarity=0.0)
        assert results == []
        assert not embed_path.exists()

    def test_incompatible_dimensions_delete_sidecar_and_skip_skill(self, tmp_path):
        skill_md = _write_skill(tmp_path / "calc", "Calculator", "Performs arithmetic.")
        embed_path = skill_md.parent / EMBED_FILENAME
        np.save(embed_path, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        results = search(np.array([1.0, 0.0], dtype=np.float32), find_skills(tmp_path), min_similarity=0.0)
        assert results == []
        assert not embed_path.exists()


class TestFormatting:
    def test_up_to_twenty_matches_include_name_description_and_path(self):
        formatted = format_matches([
            {"name": "Alpha", "description": "Does alpha.", "path": "/workspace/skills/alpha/SKILL.md"}
        ])
        assert "name: Alpha" in formatted
        assert "description: Does alpha." in formatted
        assert "path: /workspace/skills/alpha/SKILL.md" in formatted

    def test_between_twenty_one_and_one_hundred_matches_omit_description(self):
        formatted = format_matches([
            {"name": f"Skill {index}", "description": f"Desc {index}", "path": f"/workspace/skills/{index}/SKILL.md"}
            for index in range(21)
        ])
        assert "description:" not in formatted
        assert "path:" in formatted

    def test_more_than_one_hundred_matches_include_only_names(self):
        formatted = format_matches([
            {"name": f"Skill {index}", "description": f"Desc {index}", "path": f"/workspace/skills/{index}/SKILL.md"}
            for index in range(101)
        ])
        assert "description:" not in formatted
        assert "path:" not in formatted
        assert formatted.count("name:") == 101

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from backhead.skills import (
    EMBED_FILENAME,
    _find_skills,
    _load_or_create_embedding,
    _parse_skill_meta,
    find_skill_matches,
    format_skill_matches,
    generate_skill_header,
)


def _write_skill(directory: Path, name: str, description: str, body: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    skill_md = directory / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    return skill_md


def _unit_vec(index: int, size: int = 8) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index % size] = 1.0
    return vector


def _make_embed(keyword_to_index: dict[str, int], size: int = 8):
    def embed_fn(texts):
        result = []
        for text in texts:
            text_lower = text.lower()
            index = next((value for key, value in keyword_to_index.items() if key in text_lower), size - 1)
            result.append(_unit_vec(index, size))
        return np.array(result, dtype=np.float32)

    return embed_fn


class TestParseSkillMeta:
    def test_valid_frontmatter(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: Calc\ndescription: Does math.\n---\n# Body")
        assert _parse_skill_meta(path) == {"name": "Calc", "description": "Does math."}

    def test_missing_name_returns_none(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\ndescription: Does math.\n---\n")
        assert _parse_skill_meta(path) is None

    def test_missing_description_returns_none(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: Calc\n---\n")
        assert _parse_skill_meta(path) is None


class TestFindSkills:
    def test_finds_valid_skills(self, tmp_path):
        _write_skill(tmp_path / "alpha", "Alpha", "Alpha task.")
        _write_skill(tmp_path / "beta", "Beta", "Beta task.")
        skills = _find_skills(tmp_path)
        assert {skill["name"] for skill in skills} == {"Alpha", "Beta"}

    def test_skips_skills_missing_required_frontmatter(self, tmp_path):
        _write_skill(tmp_path / "ok", "Good Skill", "Has both fields.")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("---\nname: Incomplete\n---\n")
        skills = _find_skills(tmp_path)
        assert [skill["name"] for skill in skills] == ["Good Skill"]


class TestLoadOrCreateEmbedding:
    def _simple_embed(self, texts):
        return np.array([_unit_vec(0)] * len(texts), dtype=np.float32)

    def test_creates_cache_file_on_first_call(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME
        result = _load_or_create_embedding(skill, self._simple_embed)
        assert embed_path.exists()
        assert np.allclose(result, _unit_vec(0))

    def test_cached_embedding_is_reused_on_second_call(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        call_count = [0]

        def counting_embed(texts):
            call_count[0] += 1
            return np.array([_unit_vec(0)] * len(texts), dtype=np.float32)

        _load_or_create_embedding(skill, counting_embed)
        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 1

    def test_embedding_is_regenerated_when_skill_md_is_newer(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        call_count = [0]

        def counting_embed(texts):
            call_count[0] += 1
            return np.array([_unit_vec(call_count[0] - 1)] * len(texts), dtype=np.float32)

        _load_or_create_embedding(skill, counting_embed)
        time.sleep(0.01)
        skill_md.write_text(skill_md.read_text())
        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 2

    def test_unreadable_sidecar_is_deleted_and_regenerated(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME
        embed_path.write_bytes(b"not-a-valid-npy")
        fresh = _unit_vec(2)

        def fresh_embed(texts):
            return np.array([fresh] * len(texts), dtype=np.float32)

        result = _load_or_create_embedding(skill, fresh_embed)
        assert np.allclose(result, fresh)
        assert np.allclose(np.load(embed_path, allow_pickle=False), fresh)

    def test_malformed_sidecar_is_deleted_and_regenerated(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME
        np.save(embed_path, np.array([[1.0, 0.0]], dtype=np.float32))
        fresh = _unit_vec(3)

        def fresh_embed(texts):
            return np.array([fresh] * len(texts), dtype=np.float32)

        result = _load_or_create_embedding(skill, fresh_embed)
        assert np.allclose(result, fresh)
        assert np.allclose(np.load(embed_path, allow_pickle=False), fresh)

    def test_incompatible_dimensions_trigger_deletion_and_regeneration(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME
        np.save(embed_path, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        fresh = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        def fresh_embed(texts):
            return np.array([fresh] * len(texts), dtype=np.float32)

        result = _load_or_create_embedding(skill, fresh_embed, expected_dimensions=4)
        assert np.allclose(result, fresh)
        assert np.allclose(np.load(embed_path, allow_pickle=False), fresh)


class TestFindSkillMatches:
    def test_missing_skills_dir_returns_empty(self, tmp_path):
        matches = find_skill_matches(_unit_vec(0), tmp_path, embed_fn=_make_embed({}))
        assert matches == []

    def test_below_threshold_skills_are_excluded(self, tmp_path):
        _write_skill(tmp_path / "skills" / "calc", "Calculator", "Does arithmetic.")
        matches = find_skill_matches(
            _unit_vec(0),
            tmp_path,
            min_similarity=0.5,
            embed_fn=_make_embed({"calculator": 1, "arithmetic": 1}),
        )
        assert matches == []

    def test_matches_are_sorted_and_use_workspace_paths(self, tmp_path):
        _write_skill(tmp_path / "skills" / "calc", "Calculator", "Does arithmetic.")
        _write_skill(tmp_path / "skills" / "weather", "Weather", "Gets forecasts.")
        matches = find_skill_matches(
            _unit_vec(0),
            tmp_path,
            min_similarity=0.0,
            embed_fn=_make_embed({"calculator": 0, "arithmetic": 0, "weather": 1, "forecast": 1}),
        )
        assert [match["name"] for match in matches][:2] == ["Calculator", "Weather"]
        assert all(match["path"].startswith("/workspace/") for match in matches)

    def test_skill_body_is_not_embedded_or_injected(self, tmp_path):
        _write_skill(
            tmp_path / "skills" / "calc",
            "Calculator",
            "Does arithmetic.",
            body="# Secret body\nDo not inject me.",
        )
        seen_texts = []

        def embed_fn(texts):
            seen_texts.extend(texts)
            return np.array([_unit_vec(0)] * len(texts), dtype=np.float32)

        header = generate_skill_header("arithmetic", tmp_path, min_similarity=0.0, embed_fn=embed_fn)
        assert seen_texts == ["arithmetic", "Calculator. Does arithmetic."]
        assert "Do not inject me." not in header

    def test_generate_skill_header_returns_none_when_no_match(self, tmp_path):
        _write_skill(tmp_path / "skills" / "calc", "Calculator", "Does arithmetic.")
        header = generate_skill_header(
            "weather",
            tmp_path,
            min_similarity=0.1,
            embed_fn=_make_embed({"weather": 0, "calculator": 1, "arithmetic": 1}),
        )
        assert header is None

    def test_generate_skill_header_formats_small_result_sets(self, tmp_path):
        _write_skill(tmp_path / "skills" / "calc", "Calculator", "Does arithmetic.")
        header = generate_skill_header(
            "arithmetic",
            tmp_path,
            min_similarity=0.0,
            embed_fn=_make_embed({"arithmetic": 0, "calculator": 0}),
        )
        assert header is not None
        assert header.startswith("[Relevant skills]")
        assert "name: Calculator" in header
        assert "description: Does arithmetic." in header
        assert "path: /workspace/skills/calc/SKILL.md" in header


class TestFormatSkillMatches:
    def test_up_to_twenty_matches_include_name_description_and_path(self):
        formatted = format_skill_matches([
            {"name": "Alpha", "description": "Does alpha.", "path": "/workspace/skills/alpha/SKILL.md"}
        ])
        assert "name: Alpha" in formatted
        assert "description: Does alpha." in formatted
        assert "path: /workspace/skills/alpha/SKILL.md" in formatted

    def test_between_twenty_one_and_one_hundred_matches_omit_description(self):
        matches = [
            {"name": f"Skill {index}", "description": f"Desc {index}", "path": f"/workspace/skills/{index}/SKILL.md"}
            for index in range(21)
        ]
        formatted = format_skill_matches(matches)
        assert "description:" not in formatted
        assert "path:" in formatted

    def test_more_than_one_hundred_matches_include_only_names(self):
        matches = [
            {"name": f"Skill {index}", "description": f"Desc {index}", "path": f"/workspace/skills/{index}/SKILL.md"}
            for index in range(101)
        ]
        formatted = format_skill_matches(matches)
        assert "description:" not in formatted
        assert "path:" not in formatted
        assert formatted.count("name:") == 101

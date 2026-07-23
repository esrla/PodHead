# Tests for backhead.skills: metadata parsing, embedding cache, and skill-header generation.

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from backhead.skills import (
    EMBED_FILENAME,
    _find_skills,
    _load_or_create_embedding,
    _parse_skill_meta,
    generate_skill_header,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _write_skill(directory: Path, name: str, description: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    skill_md = directory / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: {description}\n---\n")
    return skill_md


def _unit_vec(index: int, size: int = 8) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    v[index % size] = 1.0
    return v


def _make_embed(keyword_to_index: dict, size: int = 8):
    """Return an embed_fn that maps text to a deterministic unit vector."""

    def embed_fn(texts):
        result = []
        for t in texts:
            tl = t.lower()
            idx = next((v for k, v in keyword_to_index.items() if k in tl), 7)
            result.append(_unit_vec(idx, size))
        return np.array(result)

    return embed_fn


# ── _parse_skill_meta ──────────────────────────────────────────────────────────


class TestParseSkillMeta:
    def test_valid_frontmatter(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\ndescription: Does math.\n---\n# Body")
        assert _parse_skill_meta(p) == {"name": "Calc", "description": "Does math."}

    def test_missing_name_returns_none(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\ndescription: Does math.\n---\n")
        assert _parse_skill_meta(p) is None

    def test_missing_description_returns_none(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\n---\n")
        assert _parse_skill_meta(p) is None

    def test_no_frontmatter_returns_none(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("# Just a heading\n")
        assert _parse_skill_meta(p) is None

    def test_unclosed_frontmatter_returns_none(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\n")
        assert _parse_skill_meta(p) is None

    def test_invalid_yaml_returns_none(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\n: [\n---\n")
        assert _parse_skill_meta(p) is None

    def test_extra_fields_are_ignored(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\ndescription: Does math.\nauthor: Alice\n---\n")
        result = _parse_skill_meta(p)
        assert result == {"name": "Calc", "description": "Does math."}
        assert "author" not in result


# ── _find_skills ───────────────────────────────────────────────────────────────


class TestFindSkills:
    def test_finds_valid_skills(self, tmp_path):
        _write_skill(tmp_path / "alpha", "Alpha", "Alpha task.")
        _write_skill(tmp_path / "beta", "Beta", "Beta task.")
        skills = _find_skills(tmp_path)
        names = {s["name"] for s in skills}
        assert names == {"Alpha", "Beta"}

    def test_skips_skills_missing_name_or_description(self, tmp_path):
        _write_skill(tmp_path / "ok", "Good Skill", "Has both fields.")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("---\nname: Incomplete\n---\n")
        skills = _find_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0]["name"] == "Good Skill"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert _find_skills(tmp_path) == []

    def test_skill_dict_has_required_keys(self, tmp_path):
        _write_skill(tmp_path / "s", "S", "Does S.")
        skill = _find_skills(tmp_path)[0]
        assert {"name", "description", "skill_md"} <= set(skill.keys())


# ── _load_or_create_embedding ──────────────────────────────────────────────────


class TestLoadOrCreateEmbedding:
    def _simple_embed(self, texts):
        return np.array([_unit_vec(0)] * len(texts))

    def test_creates_cache_file_on_first_call(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME
        assert not embed_path.exists()
        _load_or_create_embedding(skill, self._simple_embed)
        assert embed_path.exists()

    def test_cached_embedding_is_reused_on_second_call(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        call_count = [0]

        def counting_embed(texts):
            call_count[0] += 1
            return np.array([_unit_vec(0)] * len(texts))

        _load_or_create_embedding(skill, counting_embed)
        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 1

    def test_embedding_is_regenerated_when_skill_md_is_newer(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        call_count = [0]

        def counting_embed(texts):
            call_count[0] += 1
            return np.array([_unit_vec(call_count[0] - 1)] * len(texts))

        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 1
        time.sleep(0.01)
        skill_md.write_text(skill_md.read_text())  # bump mtime
        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 2

    def test_stale_cache_is_not_used_after_skill_md_updated(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Original.")
        skill = {"name": "S", "description": "Original.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME

        old_vec = _unit_vec(6)
        np.save(embed_path, old_vec)
        old_mtime = embed_path.stat().st_mtime - 1
        os.utime(embed_path, (old_mtime, old_mtime))

        time.sleep(0.01)
        skill_md.write_text("---\nname: S\ndescription: Updated.\n---\n")
        skill["description"] = "Updated."

        new_vec = _unit_vec(3)

        def fresh_embed(texts):
            return np.array([new_vec] * len(texts))

        result = _load_or_create_embedding(skill, fresh_embed)
        assert np.allclose(result, new_vec)


# ── generate_skill_header ──────────────────────────────────────────────────────


class TestGenerateSkillHeader:
    def test_returns_none_when_skills_dir_missing(self, tmp_path):
        result = generate_skill_header("hello", tmp_path, embed_fn=_make_embed({}))
        assert result is None

    def test_returns_none_when_skills_dir_is_empty(self, tmp_path):
        (tmp_path / "skills").mkdir()
        result = generate_skill_header("hello", tmp_path, embed_fn=_make_embed({}))
        assert result is None

    def test_returns_formatted_header_with_matching_skill(self, tmp_path):
        _write_skill(tmp_path / "skills" / "calc", "Calculator", "Does arithmetic.")
        embed_fn = _make_embed({"calc": 0, "arithmetic": 0})
        result = generate_skill_header("arithmetic", tmp_path, top=1, embed_fn=embed_fn)
        assert result is not None
        assert "[Skill context" in result
        assert "Calculator" in result
        assert "Does arithmetic." in result

    def test_top_limits_number_of_returned_skills(self, tmp_path):
        for i in range(5):
            _write_skill(tmp_path / "skills" / f"skill{i}", f"Skill{i}", f"Does thing {i}.")
        result = generate_skill_header("anything", tmp_path, top=2, embed_fn=_make_embed({}))
        assert result is not None
        skill_lines = [ln for ln in result.splitlines() if ln.startswith("- ")]
        assert len(skill_lines) == 2

    def test_header_starts_with_context_marker(self, tmp_path):
        _write_skill(tmp_path / "skills" / "s", "S", "Does S.")
        result = generate_skill_header("s", tmp_path, embed_fn=_make_embed({"s": 0}))
        assert result is not None
        assert result.startswith("[Skill context")

    def test_most_relevant_skill_listed_first(self, tmp_path):
        _write_skill(tmp_path / "skills" / "calc", "Calculator", "Does arithmetic.")
        _write_skill(tmp_path / "skills" / "weather", "Weather", "Fetches forecast.")
        embed_fn = _make_embed({"calc": 0, "arithmetic": 0, "weather": 1, "forecast": 1})
        result = generate_skill_header("arithmetic calculation", tmp_path, top=2, embed_fn=embed_fn)
        assert result is not None
        lines = [ln for ln in result.splitlines() if ln.startswith("- ")]
        assert "Calculator" in lines[0]

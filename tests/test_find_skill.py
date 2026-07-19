# Tests for find_skill.py: metadata parsing, embedding invalidation, and search results.

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "head_pod" / "tools"))
from find_skill import (
    EMBED_FILENAME,
    _load_or_create_embedding,
    find_skills,
    parse_frontmatter,
    search,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_skill(directory: Path, name: str, description: str, body: str = "") -> Path:
    """Write a SKILL.md into directory and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    skill_md = directory / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    return skill_md


def _unit_vec(index: int, size: int = 8) -> np.ndarray:
    """Return a unit vector with a 1.0 at position index % size."""
    v = np.zeros(size, dtype=np.float32)
    v[index % size] = 1.0
    return v


def _make_embed(keyword_to_index: dict, size: int = 8):
    """Return a mock embed_fn that maps text to a deterministic unit vector.

    The first keyword found in the lowercased text determines the index.
    Texts with no matching keyword map to index 7.
    """
    def embed_fn(texts):
        result = []
        for t in texts:
            tl = t.lower()
            idx = next((v for k, v in keyword_to_index.items() if k in tl), 7)
            result.append(_unit_vec(idx, size))
        return np.array(result)
    return embed_fn


# ── metadata parsing ───────────────────────────────────────────────────────────

class TestParseFrontmatter:
    def test_valid_frontmatter(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\ndescription: Does math.\n---\n# Body")
        assert parse_frontmatter(p) == {"name": "Calc", "description": "Does math."}

    def test_missing_description(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\n---\n")
        assert parse_frontmatter(p) == {"name": "Calc"}

    def test_missing_name(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\ndescription: Does math.\n---\n")
        assert parse_frontmatter(p) == {"description": "Does math."}

    def test_no_frontmatter(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("# Just a heading\n")
        assert parse_frontmatter(p) == {}

    def test_unclosed_frontmatter(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\n")
        assert parse_frontmatter(p) == {}

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\n: [\n---\n")
        assert parse_frontmatter(p) == {}

    def test_extra_fields_ignored(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: Calc\ndescription: Does math.\nauthor: Alice\n---\n")
        result = parse_frontmatter(p)
        assert result == {"name": "Calc", "description": "Does math."}
        assert "author" not in result

    def test_empty_frontmatter(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\n---\n")
        assert parse_frontmatter(p) == {}


# ── find_skills ────────────────────────────────────────────────────────────────

class TestFindSkills:
    def test_finds_valid_skills(self, tmp_path):
        _write_skill(tmp_path / "calc", "Calculator", "Does arithmetic.")
        _write_skill(tmp_path / "weather", "Weather", "Gets the forecast.")
        skills = find_skills(tmp_path)
        names = [s["name"] for s in skills]
        assert "Calculator" in names
        assert "Weather" in names

    def test_skips_skill_missing_name_or_description(self, tmp_path):
        _write_skill(tmp_path / "ok", "Good Skill", "Has both fields.")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("---\nname: Incomplete\n---\n")
        skills = find_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0]["name"] == "Good Skill"

    def test_empty_directory(self, tmp_path):
        assert find_skills(tmp_path) == []

    def test_skill_dict_keys(self, tmp_path):
        _write_skill(tmp_path / "s", "S", "Does S.")
        skill = find_skills(tmp_path)[0]
        assert {"name", "description", "skill_md", "path"} <= set(skill.keys())


# ── embedding invalidation ─────────────────────────────────────────────────────

class TestEmbeddingInvalidation:
    def _simple_embed(self, texts):
        return np.array([_unit_vec(0)] * len(texts))

    def test_embedding_created_on_first_call(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME
        assert not embed_path.exists()
        _load_or_create_embedding(skill, self._simple_embed)
        assert embed_path.exists()

    def test_cached_embedding_reused(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}

        call_count = [0]
        def counting_embed(texts):
            call_count[0] += 1
            return np.array([_unit_vec(0)] * len(texts))

        _load_or_create_embedding(skill, counting_embed)
        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 1  # second call used the cache

    def test_embedding_regenerated_when_skill_md_is_newer(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Does S.")
        skill = {"name": "S", "description": "Does S.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME

        call_count = [0]
        def counting_embed(texts):
            call_count[0] += 1
            return np.array([_unit_vec(call_count[0] - 1)] * len(texts))

        # Generate initial embedding
        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 1

        # Touch SKILL.md to make it newer than the embedding
        time.sleep(0.01)
        skill_md.write_text(skill_md.read_text())  # update mtime

        _load_or_create_embedding(skill, counting_embed)
        assert call_count[0] == 2  # embedding was regenerated

    def test_stale_cache_not_used_after_skill_md_updated(self, tmp_path):
        skill_md = _write_skill(tmp_path / "s", "S", "Original description.")
        skill = {"name": "S", "description": "Original description.", "skill_md": skill_md}
        embed_path = skill_md.parent / EMBED_FILENAME

        # Write a "wrong" cached embedding manually, with an older mtime
        old_vec = _unit_vec(6)
        np.save(embed_path, old_vec)
        old_mtime = embed_path.stat().st_mtime - 1
        os.utime(embed_path, (old_mtime, old_mtime))

        # Update SKILL.md (newer mtime)
        time.sleep(0.01)
        skill_md.write_text("---\nname: S\ndescription: Updated description.\n---\n")
        skill["description"] = "Updated description."

        new_vec = _unit_vec(3)
        def fresh_embed(texts):
            return np.array([new_vec] * len(texts))

        result = _load_or_create_embedding(skill, fresh_embed)
        assert np.allclose(result, new_vec)


# ── search results ─────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_most_relevant_skill_first(self, tmp_path):
        _write_skill(tmp_path / "calc", "Calculator", "Performs arithmetic and math.")
        _write_skill(tmp_path / "weather", "Weather", "Fetches current weather forecast.")
        skills = find_skills(tmp_path)

        # "calc" keyword → index 0; "weather" keyword → index 1; query "calc" → index 0
        embed_fn = _make_embed({"calc": 0, "arithmetic": 0, "math": 0,
                                 "weather": 1, "forecast": 1})
        results = search("arithmetic calculation", skills, top=5, embed_fn=embed_fn)
        assert results[0]["name"] == "Calculator"

    def test_top_limits_results(self, tmp_path):
        for i in range(6):
            _write_skill(tmp_path / f"skill{i}", f"Skill {i}", f"Does thing {i}.")
        skills = find_skills(tmp_path)
        embed_fn = _make_embed({})  # all map to same index → equal scores
        results = search("anything", skills, top=3, embed_fn=embed_fn)
        assert len(results) <= 3

    def test_empty_skills_returns_empty(self, tmp_path):
        assert search("query", [], embed_fn=_make_embed({})) == []

    def test_result_contains_required_keys(self, tmp_path):
        _write_skill(tmp_path / "s", "S", "Does S.")
        skills = find_skills(tmp_path)
        results = search("s", skills, embed_fn=_make_embed({"s": 0}))
        assert results
        for key in ("name", "description", "path", "score"):
            assert key in results[0]

    def test_scores_descending(self, tmp_path):
        _write_skill(tmp_path / "a", "Alpha", "Alpha description.")
        _write_skill(tmp_path / "b", "Beta", "Beta description.")
        _write_skill(tmp_path / "c", "Gamma", "Gamma description.")
        skills = find_skills(tmp_path)
        embed_fn = _make_embed({"alpha": 0, "beta": 1, "gamma": 2})
        results = search("alpha", skills, top=10, embed_fn=embed_fn)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

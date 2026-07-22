"""Skill-header sidecar generation for the trusted backend.

The backend calls ``generate_skill_header`` before dispatching each incoming
user message to the main LLM. It returns a compact block of the most relevant
skill names and descriptions so the agent is oriented before it replies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import yaml

SKILL_FILENAME = "SKILL.md"
EMBED_FILENAME = ".embed.npy"
_EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def _get_model():
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return normalized embedding vectors for a list of texts."""
    return _get_model().encode(texts, normalize_embeddings=True)


def _parse_skill_meta(skill_md: Path) -> dict | None:
    """Return {name, description} from YAML frontmatter of a SKILL.md, or None."""
    text = skill_md.read_text()
    if not text.startswith("---"):
        return None
    try:
        end = text.index("---", 3)
    except ValueError:
        return None
    try:
        fm = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        return None
    if "name" in fm and "description" in fm:
        return {"name": str(fm["name"]), "description": str(fm["description"])}
    return None


def _find_skills(skills_dir: Path) -> list[dict]:
    """Return list of skill metadata dicts found under skills_dir."""
    skills = []
    for skill_md in sorted(skills_dir.rglob(SKILL_FILENAME)):
        meta = _parse_skill_meta(skill_md)
        if meta:
            skills.append({**meta, "skill_md": skill_md})
    return skills


def _load_or_create_embedding(skill: dict, embed_fn: Callable) -> np.ndarray:
    """Load cached embedding or compute and cache a fresh one.

    The cache file (.embed.npy) lives beside SKILL.md and is invalidated
    whenever SKILL.md has a newer mtime.
    """
    skill_md: Path = skill["skill_md"]
    embed_path = skill_md.parent / EMBED_FILENAME
    skill_mtime = skill_md.stat().st_mtime
    if embed_path.exists() and embed_path.stat().st_mtime >= skill_mtime:
        return np.load(embed_path)
    text = f"{skill['name']}. {skill['description']}"
    vec = embed_fn([text])[0]
    np.save(embed_path, vec)
    return vec


def generate_skill_header(
    message_text: str,
    workspace_path: Path,
    top: int = 3,
    embed_fn: Callable | None = None,
) -> str | None:
    """Return a skill-header sidecar string, or None if no skills are available.

    Reads SKILL.md files from ``workspace_path/skills/``, ranks them by
    cosine similarity against ``message_text``, and formats the top matches
    as a block suitable for injection into the LLM system prompt.
    """
    if embed_fn is None:
        embed_fn = embed

    skills_dir = workspace_path / "skills"
    if not skills_dir.is_dir():
        return None
    skills = _find_skills(skills_dir)
    if not skills:
        return None

    query_vec = embed_fn([message_text])[0]
    scored = []
    for skill in skills:
        vec = _load_or_create_embedding(skill, embed_fn)
        scored.append({**skill, "score": float(np.dot(query_vec, vec))})
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_skills = scored[:top]

    lines = ["[Skill context — relevant skills from workspace]"]
    for s in top_skills:
        lines.append(f"- {s['name']}: {s['description']}")
    return "\n".join(lines)

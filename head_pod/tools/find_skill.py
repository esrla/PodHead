#!/usr/bin/env python3
"""Find skills using a backend-generated embedding vector and shared sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

EMBED_FILENAME = ".embed.npy"
DEFAULT_SKILLS_DIR = "/workspace/skills"
DEFAULT_MIN_SIMILARITY = 0.35


def parse_frontmatter(path):
    """Return {name, description} from YAML frontmatter of a SKILL.md file."""
    text = path.read_text()
    if not text.startswith("---"):
        return {}
    try:
        end = text.index("---", 3)
    except ValueError:
        return {}
    try:
        fm = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        return {}
    return {k: fm[k] for k in ("name", "description") if k in fm}


def find_skills(skills_dir):
    """Return a list of skill dicts found under skills_dir."""
    skills = []
    for skill_md in sorted(Path(skills_dir).rglob("SKILL.md")):
        meta = parse_frontmatter(skill_md)
        if "name" in meta and "description" in meta:
            skills.append(
                {
                    **meta,
                    "skill_md": skill_md,
                    "path": f"/workspace/{skill_md.relative_to(Path(skills_dir).parent)}",
                }
            )
    return skills


def _delete_sidecar(embed_path):
    try:
        embed_path.unlink()
    except FileNotFoundError:
        pass


def _coerce_embedding(value, expected_dimensions=None):
    vec = np.asarray(value, dtype=np.float32)
    if vec.ndim != 1:
        raise ValueError("Embedding must be a one-dimensional vector.")
    if vec.size == 0:
        raise ValueError("Embedding must not be empty.")
    if not np.isfinite(vec).all():
        raise ValueError("Embedding must contain only finite values.")
    if expected_dimensions is not None and vec.shape[0] != expected_dimensions:
        raise ValueError("Embedding dimensions are incompatible.")
    return vec


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Embedding must have a non-zero length.")
    return vec / norm


def _cosine_similarity(left, right):
    if left.shape != right.shape:
        raise ValueError("Embedding dimensions are incompatible.")
    return float(np.dot(_normalize(left), _normalize(right)))


def _load_embedding(skill, expected_dimensions=None):
    embed_path = skill["skill_md"].parent / EMBED_FILENAME
    try:
        return _coerce_embedding(
            np.load(embed_path, allow_pickle=False),
            expected_dimensions=expected_dimensions,
        )
    except Exception:
        _delete_sidecar(embed_path)
        raise


def _result_fields(match_count):
    if match_count <= 20:
        return ("name", "description", "path")
    if match_count <= 100:
        return ("name", "path")
    return ("name",)


def format_matches(matches):
    if not matches:
        return ""
    fields = _result_fields(len(matches))
    lines = []
    for index, match in enumerate(matches):
        if index and fields != ("name",):
            lines.append("")
        for field in fields:
            lines.append(f"{field}: {match[field]}")
    return "\n".join(lines)


def search(query_embedding, skills, min_similarity=DEFAULT_MIN_SIMILARITY):
    """Return all skill dicts whose similarity meets the configured threshold."""
    if not skills:
        return []
    query_vec = _coerce_embedding(query_embedding)
    scored = []
    for skill in skills:
        try:
            vec = _load_embedding(skill, expected_dimensions=query_vec.shape[0])
        except Exception:
            continue
        score = _cosine_similarity(query_vec, vec)
        if score >= min_similarity:
            scored.append({**skill, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _parse_embedding_arg(raw_value):
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("Expected a JSON embedding vector.") from exc
    return _coerce_embedding(value)


def main():
    parser = argparse.ArgumentParser(description="Find relevant skills using an embedding vector.")
    parser.add_argument("embedding", help="Embedding vector as JSON, for example: '[0.1, 0.2, 0.3]'")
    parser.add_argument(
        "--skills-dir", default=DEFAULT_SKILLS_DIR, metavar="DIR",
        help=f"Skills root directory (default: {DEFAULT_SKILLS_DIR})",
    )
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)
    if not skills_dir.is_dir():
        print(f"Skills directory not found: {skills_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        query_embedding = _parse_embedding_arg(args.embedding)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    skills = find_skills(skills_dir)
    results = search(query_embedding, skills)
    if not results:
        print("No matching skills found.")
        return
    print(format_matches(results))


if __name__ == "__main__":
    main()

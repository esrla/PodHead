#!/usr/bin/env python3
# Finds SKILL.md files, builds multilingual embeddings, and returns the most relevant skills.

"""
Usage:
  python find_skill.py <query> [--top N] [--skills-dir DIR]

Reads only the standard `name` and `description` YAML frontmatter fields from each
SKILL.md. Embeddings are cached as .embed.npy beside the corresponding SKILL.md and
regenerated only when SKILL.md is newer than the cache file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

EMBED_FILENAME = ".embed.npy"
DEFAULT_SKILLS_DIR = "/workspace/skills"
DEFAULT_TOP = 5


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
    """Return a list of skill dicts found under skills_dir.

    Each dict has keys: name, description, skill_md (Path), path (str).
    """
    skills = []
    for skill_md in sorted(Path(skills_dir).rglob("SKILL.md")):
        meta = parse_frontmatter(skill_md)
        if "name" in meta and "description" in meta:
            skills.append({**meta, "skill_md": skill_md, "path": str(skill_md.parent)})
    return skills


def _default_embed(texts):
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model.encode(texts, normalize_embeddings=True)


_model = None


def _load_or_create_embedding(skill, embed_fn):
    """Load the cached embedding or generate a new one if SKILL.md is newer."""
    skill_md = skill["skill_md"]
    embed_path = skill_md.parent / EMBED_FILENAME
    skill_mtime = skill_md.stat().st_mtime
    if embed_path.exists() and embed_path.stat().st_mtime >= skill_mtime:
        return np.load(embed_path)
    text = f"{skill['name']}. {skill['description']}"
    vec = embed_fn([text])[0]
    np.save(embed_path, vec)
    return vec


def search(query, skills, top=DEFAULT_TOP, embed_fn=None):
    """Return up to top most relevant skill dicts for query, sorted by score."""
    if embed_fn is None:
        embed_fn = _default_embed
    if not skills:
        return []
    query_vec = embed_fn([query])[0]
    scored = []
    for skill in skills:
        vec = _load_or_create_embedding(skill, embed_fn)
        scored.append({**skill, "score": float(np.dot(query_vec, vec))})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top]


def main():
    parser = argparse.ArgumentParser(description="Find relevant skills by semantic query.")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP, metavar="N",
        help=f"Number of results (default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--skills-dir", default=DEFAULT_SKILLS_DIR, metavar="DIR",
        help=f"Skills root directory (default: {DEFAULT_SKILLS_DIR})",
    )
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)
    if not skills_dir.is_dir():
        print(f"Skills directory not found: {skills_dir}", file=sys.stderr)
        sys.exit(1)

    skills = find_skills(skills_dir)
    if not skills:
        print("No skills found.")
        return

    results = search(args.query, skills, top=args.top)
    for r in results:
        print(f"name: {r['name']}")
        print(f"description: {r['description']}")
        print(f"path: {r['path']}")
        print()


if __name__ == "__main__":
    main()

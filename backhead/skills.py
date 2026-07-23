"""Backend skill sidecar loading, matching, and injection formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

SKILL_FILENAME = "SKILL.md"
EMBED_FILENAME = ".embed.npy"
DEFAULT_SKILL_SIMILARITY_THRESHOLD = 0.35


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


def _delete_sidecar(embed_path: Path) -> None:
    try:
        embed_path.unlink()
    except FileNotFoundError:
        pass


def _coerce_embedding_vector(value: Any, *, expected_dimensions: int | None = None) -> np.ndarray:
    try:
        vec = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding must be a one-dimensional numeric vector.") from exc
    if vec.ndim != 1:
        raise ValueError("Embedding must be a one-dimensional vector.")
    if vec.size == 0:
        raise ValueError("Embedding must not be empty.")
    if not np.isfinite(vec).all():
        raise ValueError("Embedding must contain only finite values.")
    if expected_dimensions is not None and vec.shape[0] != expected_dimensions:
        raise ValueError(
            f"Embedding dimensions are incompatible: expected {expected_dimensions}, got {vec.shape[0]}."
        )
    return vec


def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Embedding must have a non-zero length.")
    return vec / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Embedding dimensions are incompatible.")
    return float(np.dot(_normalize_embedding(left), _normalize_embedding(right)))


def _container_skill_path(skill_md: Path, workspace_path: Path) -> str:
    return str(Path("/workspace") / skill_md.relative_to(workspace_path))


def _load_sidecar_embedding(embed_path: Path, *, expected_dimensions: int | None = None) -> np.ndarray:
    return _coerce_embedding_vector(np.load(embed_path, allow_pickle=False), expected_dimensions=expected_dimensions)


def _create_sidecar_embedding(
    skill: dict,
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    expected_dimensions: int | None = None,
) -> np.ndarray:
    text = f"{skill['name']}. {skill['description']}"
    vecs = embed_fn([text])
    if len(vecs) != 1:
        raise ValueError("Embedding API returned an unexpected number of vectors.")
    vec = _coerce_embedding_vector(vecs[0], expected_dimensions=expected_dimensions)
    np.save(skill["skill_md"].parent / EMBED_FILENAME, vec)
    return vec


def _load_or_create_embedding(
    skill: dict,
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    expected_dimensions: int | None = None,
) -> np.ndarray:
    """Load cached embedding or compute and cache a fresh one."""
    skill_md: Path = skill["skill_md"]
    embed_path = skill_md.parent / EMBED_FILENAME
    skill_mtime = skill_md.stat().st_mtime

    if embed_path.exists() and embed_path.stat().st_mtime >= skill_mtime:
        try:
            return _load_sidecar_embedding(embed_path, expected_dimensions=expected_dimensions)
        except Exception as exc:  # noqa: BLE001
            print(f"Deleting invalid skill sidecar {embed_path}: {exc}")
            _delete_sidecar(embed_path)

    return _create_sidecar_embedding(skill, embed_fn, expected_dimensions=expected_dimensions)


def _result_fields(match_count: int) -> tuple[str, ...]:
    if match_count <= 20:
        return ("name", "description", "path")
    if match_count <= 100:
        return ("name", "path")
    return ("name",)


def format_skill_matches(matches: list[dict], *, section_title: str | None = None) -> str:
    """Format the complete ordered result set for agent-visible output."""
    if not matches:
        return ""

    fields = _result_fields(len(matches))
    lines: list[str] = []
    if section_title:
        lines.append(section_title)

    for index, match in enumerate(matches):
        if index and fields != ("name",):
            lines.append("")
        for field in fields:
            lines.append(f"{field}: {match[field]}")

    return "\n".join(lines)


def find_skill_matches(
    query_embedding: np.ndarray,
    workspace_path: Path,
    *,
    min_similarity: float = DEFAULT_SKILL_SIMILARITY_THRESHOLD,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> list[dict]:
    """Return sorted skill matches whose cosine similarity meets the threshold."""
    skills_dir = workspace_path / "skills"
    if not skills_dir.is_dir():
        return []

    query_vec = _coerce_embedding_vector(query_embedding)
    skills = _find_skills(skills_dir)
    if not skills:
        return []

    scored = []
    for skill in skills:
        vec = _load_or_create_embedding(skill, embed_fn, expected_dimensions=query_vec.shape[0])
        score = _cosine_similarity(query_vec, vec)
        if score >= min_similarity:
            scored.append(
                {
                    **skill,
                    "path": _container_skill_path(skill["skill_md"], workspace_path),
                    "score": score,
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def generate_skill_header(
    message_text: str,
    workspace_path: Path,
    *,
    min_similarity: float = DEFAULT_SKILL_SIMILARITY_THRESHOLD,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> str | None:
    """Return a relevant-skills block, or None when nothing matches."""
    query_vecs = embed_fn([message_text])
    if len(query_vecs) != 1:
        raise ValueError("Embedding API returned an unexpected number of vectors.")
    matches = find_skill_matches(
        query_vecs[0],
        workspace_path,
        min_similarity=min_similarity,
        embed_fn=embed_fn,
    )
    if not matches:
        return None
    return format_skill_matches(matches, section_title="[Relevant skills]")

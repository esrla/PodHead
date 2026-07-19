"""Image storage and normalization helpers."""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path


# Supported image signatures (magic bytes)
_IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",  # need to check bytes 8-12 == b'WEBP'
    b"BM": "bmp",
}


def detect_image_type(data: bytes) -> str | None:
    """Return image extension ('jpg', 'png', etc.) or None if not a known image."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for sig, ext in _IMAGE_SIGNATURES.items():
        if sig != b"RIFF" and data[: len(sig)] == sig:
            return ext
    return None


def save_image(data: bytes, media_root: Path) -> str | None:
    """Validate, normalize (no-op for now), and save image data under media_root.

    Files are written into a ``media`` subdirectory of *media_root* and the
    returned relative path (like 'media/<uuid>.<ext>') is resolvable by
    :func:`load_image_as_base64` given the same *media_root*.
    Returns None if invalid. Prevents path traversal and generates
    backend-controlled filenames.
    """
    if not data:
        return None
    ext = detect_image_type(data)
    if ext is None:
        return None
    media_dir = media_root / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.{ext}"
    if os.sep in filename or "/" in filename:
        return None
    dest = media_dir / filename
    dest.write_bytes(data)
    return f"media/{filename}"


def load_image_as_base64(relative_path: str, media_root: Path) -> str | None:
    """Resolve relative path inside media_root and return base64-encoded image data.

    Returns None if file not found or path is unsafe.
    """
    normalized = os.path.normpath(relative_path)
    if normalized.startswith("..") or os.path.isabs(normalized):
        return None
    full_path = media_root / normalized
    try:
        full_path.resolve().relative_to(media_root.resolve())
    except ValueError:
        return None
    if not full_path.exists():
        return None
    return base64.b64encode(full_path.read_bytes()).decode("ascii")


def get_image_mime_type(relative_path: str) -> str:
    """Return MIME type based on file extension."""
    ext = relative_path.rsplit(".", 1)[-1].lower() if "." in relative_path else ""
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(ext, "image/jpeg")

from pathlib import Path

from backhead import media

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16
JPG = b"\xff\xd8\xff\xe0" + b"0" * 16
GIF = b"GIF89a" + b"0" * 16
BMP = b"BM" + b"0" * 16
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"0" * 8


def test_detect_image_type():
    assert media.detect_image_type(PNG) == "png"
    assert media.detect_image_type(JPG) == "jpg"
    assert media.detect_image_type(GIF) == "gif"
    assert media.detect_image_type(BMP) == "bmp"
    assert media.detect_image_type(WEBP) == "webp"
    assert media.detect_image_type(b"not an image") is None
    assert media.detect_image_type(b"") is None


def test_save_and_load_round_trip(tmp_path: Path):
    rel = media.save_image(PNG, tmp_path)
    assert rel is not None
    assert rel.startswith("media/")
    assert rel.endswith(".png")
    assert (tmp_path / rel).exists()

    b64 = media.load_image_as_base64(rel, tmp_path)
    assert b64 is not None
    import base64
    assert base64.b64decode(b64) == PNG


def test_save_image_rejects_invalid(tmp_path: Path):
    assert media.save_image(b"", tmp_path) is None
    assert media.save_image(b"garbage", tmp_path) is None


def test_load_image_rejects_traversal(tmp_path: Path):
    assert media.load_image_as_base64("../secret.png", tmp_path) is None
    assert media.load_image_as_base64("/etc/passwd", tmp_path) is None


def test_load_missing_returns_none(tmp_path: Path):
    assert media.load_image_as_base64("media/does-not-exist.png", tmp_path) is None


def test_get_image_mime_type():
    assert media.get_image_mime_type("media/x.png") == "image/png"
    assert media.get_image_mime_type("media/x.jpg") == "image/jpeg"
    assert media.get_image_mime_type("media/x.jpeg") == "image/jpeg"
    assert media.get_image_mime_type("media/x.gif") == "image/gif"
    assert media.get_image_mime_type("media/x.webp") == "image/webp"
    assert media.get_image_mime_type("media/x.bmp") == "image/bmp"
    assert media.get_image_mime_type("noext") == "image/jpeg"

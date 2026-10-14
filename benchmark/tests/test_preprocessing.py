from __future__ import annotations

from pathlib import Path

from PIL import Image

from benchmark.src.prepare_data import preprocess_image, save_jpeg, sha256_bytes, sha256_file

def test_resize_preserves_aspect_ratio_and_does_not_upscale() -> None:
    wide = Image.new("RGB", (3000, 1000), (10, 20, 30))
    processed, info = preprocess_image(wide, max_edge=2048)
    assert processed.size[0] == 2048
    assert abs((processed.size[0] / processed.size[1]) - 3) < 0.02
    assert info["resized"] is True
    assert info["upscaled"] is False

    small = Image.new("RGB", (40, 20), (1, 2, 3))
    unchanged, info = preprocess_image(small, max_edge=2048)
    assert unchanged.size == (40, 20)
    assert info["resized"] is False

def test_preprocessing_strips_exif_orientation(tmp_path: Path) -> None:
    image = Image.new("RGB", (80, 40), (0, 0, 0))
    for x in range(80):
        image.putpixel((x, 0), (255, 0, 0))
    exif = image.getexif()
    exif[274] = 6
    source = tmp_path / "oriented.jpg"
    image.save(source, format="JPEG", exif=exif.tobytes())
    with Image.open(source) as reopened:
        assert reopened.getexif().get(274) == 6
        processed, info = preprocess_image(reopened, max_edge=2048)
    assert info["exif_orientation"] == 6
    assert processed.size == (40, 80)
    dest = tmp_path / "clean.jpg"
    save_jpeg(processed, dest, quality=95)
    with Image.open(dest) as cleaned:
        assert cleaned.getexif().get(274) in (None, 1)
        assert "exif" not in cleaned.info or not cleaned.info.get("exif")
        assert cleaned.mode == "RGB"
        assert cleaned.size == (40, 80)

def test_hashing_is_stable_and_sensitive(tmp_path: Path) -> None:
    payload = b"benchmark-bytes"
    assert sha256_bytes(payload) == sha256_bytes(payload)
    assert sha256_bytes(payload) != sha256_bytes(payload + b"x")
    path = tmp_path / "file.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)
    image = Image.new("RGB", (30, 10), (5, 6, 7))
    source = tmp_path / "source.jpg"
    image.save(source, format="JPEG", quality=95)
    processed, _info = preprocess_image(image, max_edge=2048)
    dest = tmp_path / "processed.jpg"
    save_jpeg(processed, dest, quality=95)
    assert sha256_file(source) != sha256_file(dest)
    assert sha256_file(dest) == sha256_file(dest)

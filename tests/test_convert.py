"""Tests for ``mosaic.convert``."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mosaic.convert import convert_directory_to_jpg


def _write_image(
    path: Path,
    color: tuple[int, ...],
    mode: str = "RGB",
    size: tuple[int, int] = (8, 8),
    fmt: str | None = None,
) -> Path:
    """Write an image of the given mode and color to ``path``.

    The format is inferred from the path extension unless ``fmt`` is given.
    """
    image = Image.new(mode, size, color)
    if fmt is None:
        image.save(path)
    else:
        image.save(path, format=fmt)
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_convert_rejects_non_directory(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("hi")
    with pytest.raises(NotADirectoryError):
        convert_directory_to_jpg(not_a_dir)


def test_convert_rejects_bad_quality(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="quality"):
        convert_directory_to_jpg(tmp_path, quality=0)
    with pytest.raises(ValueError, match="quality"):
        convert_directory_to_jpg(tmp_path, quality=100)


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------


def test_convert_png_to_jpeg(tmp_path: Path) -> None:
    src = _write_image(tmp_path / "photo.png", (10, 20, 30))

    written = convert_directory_to_jpg(tmp_path)

    # The PNG source must be gone, replaced by a real JPEG with the same stem.
    assert not src.exists()
    out = tmp_path / "photo.jpg"
    assert out in written
    assert out.is_file()
    with Image.open(out) as img:
        img.load()
        assert img.format == "JPEG"


def test_convert_leaves_existing_jpeg_alone(tmp_path: Path) -> None:
    src = _write_image(tmp_path / "already.jpg", (50, 60, 70), fmt="JPEG")
    original_bytes = src.read_bytes()

    written = convert_directory_to_jpg(tmp_path)

    # Source path is unchanged and the bytes are byte-for-byte identical.
    assert src.exists()
    assert src in written
    assert src.read_bytes() == original_bytes


def test_convert_renames_jpeg_when_sequential(tmp_path: Path) -> None:
    # In rename_sequentially mode, even existing JPEGs get renumbered.
    src = _write_image(tmp_path / "already.jpg", (50, 60, 70), fmt="JPEG")

    written = convert_directory_to_jpg(tmp_path, rename_sequentially=True)

    # 0.jpg should exist; the original name should be gone.
    out = tmp_path / "0.jpg"
    assert out in written
    assert out.is_file()
    assert not src.exists()


def test_convert_sequential_renames_in_order(tmp_path: Path) -> None:
    # Create a few sources with names that sort in a known order.
    _write_image(tmp_path / "a.png", (10, 0, 0))
    _write_image(tmp_path / "b.png", (0, 10, 0))
    _write_image(tmp_path / "c.png", (0, 0, 10))

    written = convert_directory_to_jpg(tmp_path, rename_sequentially=True)

    # Three files: 0.jpg, 1.jpg, 2.jpg.
    names = sorted(p.name for p in written)
    assert names == ["0.jpg", "1.jpg", "2.jpg"]
    for name in names:
        assert (tmp_path / name).is_file()


# ---------------------------------------------------------------------------
# Tricky inputs
# ---------------------------------------------------------------------------


def test_convert_flattens_rgba_onto_white(tmp_path: Path) -> None:
    # A fully-transparent RGBA pixel should composite to white in JPEG.
    src = tmp_path / "trans.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(src)

    convert_directory_to_jpg(tmp_path)

    out = tmp_path / "trans.jpg"
    with Image.open(out) as img:
        img.load()
        assert img.mode == "RGB"
        # Far corner pixel should be white-ish (JPEG is lossy, allow a margin).
        r, g, b = img.getpixel((0, 0))
        assert r > 240 and g > 240 and b > 240


def test_convert_skips_non_image_files(tmp_path: Path) -> None:
    _write_image(tmp_path / "ok.png", (1, 2, 3))
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")

    convert_directory_to_jpg(tmp_path)

    # Notes file is left intact; it isn't deleted or renamed.
    assert notes.exists()
    assert notes.read_text() == "hello"


def test_convert_skips_unreadable_files(tmp_path: Path) -> None:
    _write_image(tmp_path / "good.png", (5, 5, 5))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")

    written = convert_directory_to_jpg(tmp_path)

    # Bad file is left in place (so the user can inspect it) and is not in
    # the returned list of successfully-written files.
    assert bad.exists()
    assert bad not in written
    assert (tmp_path / "good.jpg") in written


def test_convert_handles_collision_with_unique_names(tmp_path: Path) -> None:
    # Two sources whose stems collide after conversion (photo.png + photo.bmp
    # would both want to become photo.jpg). The second one should fall back
    # to photo_1.jpg rather than overwriting the first.
    _write_image(tmp_path / "photo.png", (10, 10, 10))
    _write_image(tmp_path / "photo.bmp", (200, 200, 200), fmt="BMP")

    written = convert_directory_to_jpg(tmp_path)

    names = sorted(p.name for p in written)
    # Both outputs exist with distinct names.
    assert "photo.jpg" in names
    assert "photo_1.jpg" in names
    assert all((tmp_path / n).is_file() for n in names)


def test_convert_skips_hidden_files(tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(hidden)
    visible = _write_image(tmp_path / "visible.png", (4, 5, 6))

    convert_directory_to_jpg(tmp_path)

    # Hidden file is untouched; visible one is converted.
    assert hidden.exists()
    assert not visible.exists()
    assert (tmp_path / "visible.jpg").is_file()


def test_convert_preserves_grayscale_content(tmp_path: Path) -> None:
    # Grayscale ('L') sources should round-trip through JPEG without raising
    # and yield a readable JPEG. JPEG is lossy, so we only check the mean.
    src = tmp_path / "gray.png"
    Image.new("L", (8, 8), 200).save(src)

    convert_directory_to_jpg(tmp_path)

    out = tmp_path / "gray.jpg"
    with Image.open(out) as img:
        img.load()
        assert img.format == "JPEG"
        # After conversion the JPEG is RGB; channels should all be near 200.
        rgb = img.convert("RGB")
        r, g, b = rgb.getpixel((4, 4))
        assert abs(r - 200) < 10
        assert abs(g - 200) < 10
        assert abs(b - 200) < 10


def test_convert_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert convert_directory_to_jpg(tmp_path) == []

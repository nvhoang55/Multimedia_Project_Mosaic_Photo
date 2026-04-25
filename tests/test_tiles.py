"""Tests for ``mosaic.tiles``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mosaic.tiles import (
    SUPPORTED_EXTENSIONS,
    average_rgb,
    average_rgb_batch,
    iter_image_paths,
    load_tile,
    load_tiles,
)


def _make_solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Path:
    """Write a solid-color RGB image to ``path``."""
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------
# iter_image_paths
# ---------------------------------------------------------------------------


def test_iter_image_paths_filters_extensions(tmp_path: Path) -> None:
    _make_solid(tmp_path / "a.jpg", (255, 0, 0))
    _make_solid(tmp_path / "b.png", (0, 255, 0))
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / ".hidden.jpg").write_bytes(b"")  # hidden, should be skipped

    paths = iter_image_paths(tmp_path)
    names = [p.name for p in paths]

    # Only the two real image files, sorted by name, no hidden, no .txt.
    assert names == ["a.jpg", "b.png"]


def test_iter_image_paths_sorted_deterministically(tmp_path: Path) -> None:
    # Create files in non-alphabetical order to ensure sorting actually runs.
    for name in ["zeta.jpg", "alpha.jpg", "mu.jpg"]:
        _make_solid(tmp_path / name, (1, 2, 3))
    paths = iter_image_paths(tmp_path)
    assert [p.name for p in paths] == ["alpha.jpg", "mu.jpg", "zeta.jpg"]


def test_iter_image_paths_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "not_a_dir.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        iter_image_paths(f)


def test_supported_extensions_lowercase() -> None:
    # Sanity check: callers normalize via .lower(), so the set must be
    # lowercase too. A regression here would silently drop tiles.
    assert all(ext == ext.lower() for ext in SUPPORTED_EXTENSIONS)
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# load_tile / load_tiles
# ---------------------------------------------------------------------------


def test_load_tile_returns_rgb(tmp_path: Path) -> None:
    path = _make_solid(tmp_path / "red.png", (255, 0, 0), size=(4, 4))
    tile = load_tile(path)
    assert tile.mode == "RGB"
    assert tile.size == (4, 4)


def test_load_tile_thumbnails_when_target_size_given(tmp_path: Path) -> None:
    # 100x50 source; target 20x20 should preserve aspect ratio (PIL.thumbnail).
    path = tmp_path / "wide.png"
    Image.new("RGB", (100, 50), (10, 20, 30)).save(path)

    tile = load_tile(path, target_size=(20, 20))

    # thumbnail() never enlarges and keeps aspect; expected ~ (20, 10).
    assert tile.size[0] <= 20
    assert tile.size[1] <= 20
    # Aspect preserved (within rounding).
    assert tile.size[0] / tile.size[1] == pytest.approx(2.0, rel=0.1)


def test_load_tile_grayscale_source_is_converted_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (4, 4), 128).save(path)

    tile = load_tile(path)

    assert tile.mode == "RGB"
    # All channels should equal the source gray level.
    arr = np.asarray(tile)
    assert arr.shape == (4, 4, 3)
    assert (arr == 128).all()


def test_load_tiles_skips_unreadable_files(tmp_path: Path) -> None:
    good = _make_solid(tmp_path / "good.jpg", (10, 20, 30))
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not an image")

    tiles = load_tiles([good, bad])

    # Bad file is skipped, good one comes through.
    assert len(tiles) == 1
    assert tiles[0].mode == "RGB"


def test_load_tiles_empty_input_returns_empty_list() -> None:
    assert load_tiles([]) == []


# ---------------------------------------------------------------------------
# average_rgb
# ---------------------------------------------------------------------------


def test_average_rgb_solid_color() -> None:
    image = Image.new("RGB", (16, 8), (100, 150, 200))
    avg = average_rgb(image)
    np.testing.assert_allclose(avg, [100, 150, 200])


def test_average_rgb_two_halves() -> None:
    # Left half white, right half black -> mean is mid-gray.
    image = Image.new("RGB", (10, 10), (0, 0, 0))
    white = Image.new("RGB", (5, 10), (255, 255, 255))
    image.paste(white, (0, 0))

    avg = average_rgb(image)
    np.testing.assert_allclose(avg, [127.5, 127.5, 127.5])


def test_average_rgb_converts_non_rgb_input() -> None:
    # 'L'-mode (grayscale) input must not raise; it should be converted first.
    image = Image.new("L", (4, 4), 200)
    avg = average_rgb(image)
    np.testing.assert_allclose(avg, [200, 200, 200])


def test_average_rgb_returns_float_array() -> None:
    image = Image.new("RGB", (2, 2), (1, 2, 3))
    avg = average_rgb(image)
    # numpy float64 array of length 3.
    assert isinstance(avg, np.ndarray)
    assert avg.dtype == np.float64
    assert avg.shape == (3,)


# ---------------------------------------------------------------------------
# average_rgb_batch
# ---------------------------------------------------------------------------


def test_average_rgb_batch_stacks_rows() -> None:
    a = Image.new("RGB", (3, 3), (10, 20, 30))
    b = Image.new("RGB", (3, 3), (40, 50, 60))
    c = Image.new("RGB", (3, 3), (70, 80, 90))

    result = average_rgb_batch([a, b, c])

    assert result.shape == (3, 3)
    np.testing.assert_allclose(
        result,
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
    )


def test_average_rgb_batch_empty() -> None:
    # Empty input must return a well-shaped (0, 3) array, not raise, so
    # downstream numpy code can stack/concat unconditionally.
    result = average_rgb_batch([])
    assert result.shape == (0, 3)
    assert result.dtype == np.float64

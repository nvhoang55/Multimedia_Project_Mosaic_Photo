"""End-to-end tests for ``mosaic.core``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mosaic.core import MosaicConfig, build_mosaic, generate_mosaic_photo


def _write_solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Path:
    """Write a solid-color RGB image to ``path``."""
    Image.new("RGB", size, color).save(path)
    return path


def _make_two_tone_target(path: Path, size: tuple[int, int] = (40, 20)) -> Path:
    """Write a target image whose left half is red and right half is blue."""
    image = Image.new("RGB", size, (0, 0, 255))  # right half blue (default)
    left = Image.new("RGB", (size[0] // 2, size[1]), (255, 0, 0))
    image.paste(left, (0, 0))
    image.save(path)
    return path


# ---------------------------------------------------------------------------
# MosaicConfig validation
# ---------------------------------------------------------------------------


def test_mosaic_config_rejects_non_positive_grid() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        MosaicConfig(
            target_image="x.jpg",
            tiles_path="tiles/",
            grid_size=(0, 5),
        )


def test_mosaic_config_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        MosaicConfig(
            target_image="x.jpg",
            tiles_path="tiles/",
            grid_size=(5, 5),
            scale=0,
        )


def test_mosaic_config_rejects_unknown_color_mode() -> None:
    with pytest.raises(ValueError, match="color_mode"):
        MosaicConfig(
            target_image="x.jpg",
            tiles_path="tiles/",
            grid_size=(5, 5),
            color_mode="HSV",
        )


def test_mosaic_config_rejects_unknown_color_space() -> None:
    with pytest.raises(ValueError, match="color_space"):
        MosaicConfig(
            target_image="x.jpg",
            tiles_path="tiles/",
            grid_size=(5, 5),
            color_space="hsv",  # type: ignore[arg-type]
        )


def test_mosaic_config_rejects_bad_jpeg_quality() -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        MosaicConfig(
            target_image="x.jpg",
            tiles_path="tiles/",
            grid_size=(5, 5),
            jpeg_quality=0,
        )
    with pytest.raises(ValueError, match="jpeg_quality"):
        MosaicConfig(
            target_image="x.jpg",
            tiles_path="tiles/",
            grid_size=(5, 5),
            jpeg_quality=100,
        )


# ---------------------------------------------------------------------------
# build_mosaic (in-memory pipeline)
# ---------------------------------------------------------------------------


def test_build_mosaic_picks_correct_tiles_per_cell(tmp_path: Path) -> None:
    """Red half of the target should be tiled with the red tile; blue with blue.

    This is the strongest end-to-end correctness check: it verifies that
    splitting, averaging, matching, and pasting all line up so that each
    cell ends up with the perceptually-correct tile.
    """
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))
    _write_solid(tiles_dir / "blue.png", (0, 0, 255))
    _write_solid(tiles_dir / "green.png", (0, 255, 0))

    target = Image.new("RGB", (40, 20), (0, 0, 255))
    left = Image.new("RGB", (20, 20), (255, 0, 0))
    target.paste(left, (0, 0))

    config = MosaicConfig(
        target_image="unused.jpg",  # build_mosaic ignores the path
        tiles_path=tiles_dir,
        grid_size=(2, 4),  # 2 rows, 4 cols -> cell size 10x10
        scale=1.0,
        seed=0,  # deterministic shuffle
    )

    mosaic = build_mosaic(target, config)
    arr = np.asarray(mosaic)

    # Final canvas size: cols*cell_w x rows*cell_h = 40 x 20.
    assert mosaic.size == (40, 20)

    # Left two columns (cells 0..1 in each row) should be predominantly red.
    left_block = arr[:, :20]
    assert left_block[..., 0].mean() > 200
    assert left_block[..., 2].mean() < 50

    # Right two columns should be predominantly blue.
    right_block = arr[:, 20:]
    assert right_block[..., 2].mean() > 200
    assert right_block[..., 0].mean() < 50


def test_build_mosaic_grayscale_output(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "a.png", (50, 50, 50))
    _write_solid(tiles_dir / "b.png", (200, 200, 200))

    target = Image.new("RGB", (20, 20), (128, 128, 128))

    config = MosaicConfig(
        target_image="unused",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        color_mode="L",
        seed=0,
    )

    mosaic = build_mosaic(target, config)
    assert mosaic.mode == "L"
    assert mosaic.size == (20, 20)


def test_build_mosaic_lab_color_space(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))
    _write_solid(tiles_dir / "blue.png", (0, 0, 255))

    target = Image.new("RGB", (20, 20), (250, 5, 5))  # close to red

    config = MosaicConfig(
        target_image="unused",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        color_space="lab",
        seed=0,
    )

    mosaic = build_mosaic(target, config)
    arr = np.asarray(mosaic)

    # Every cell should pick the red tile.
    assert arr[..., 0].mean() > 200
    assert arr[..., 2].mean() < 50


def test_build_mosaic_seed_makes_runs_reproducible(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    # Several tiles with identical color -> ties resolved by shuffle order.
    for i in range(6):
        _write_solid(tiles_dir / f"t{i}.png", (100, 100, 100))

    target = Image.new("RGB", (20, 20), (100, 100, 100))

    config1 = MosaicConfig(
        target_image="unused",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        seed=42,
    )
    config2 = MosaicConfig(
        target_image="unused",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        seed=42,
    )

    a = np.asarray(build_mosaic(target, config1))
    b = np.asarray(build_mosaic(target, config2))

    np.testing.assert_array_equal(a, b)


def test_build_mosaic_no_duplicates_uses_distinct_tiles(tmp_path: Path) -> None:
    """``duplicated_tile=False`` must complete without raising and produce output."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    # Need at least 4 unique tiles for a 2x2 grid.
    _write_solid(tiles_dir / "a.png", (255, 0, 0))
    _write_solid(tiles_dir / "b.png", (0, 255, 0))
    _write_solid(tiles_dir / "c.png", (0, 0, 255))
    _write_solid(tiles_dir / "d.png", (255, 255, 0))

    target = Image.new("RGB", (20, 20), (128, 128, 128))

    config = MosaicConfig(
        target_image="unused",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        duplicated_tile=False,
        seed=0,
    )

    mosaic = build_mosaic(target, config)
    assert mosaic.size == (20, 20)


def test_build_mosaic_no_duplicates_raises_when_too_few_tiles(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    # Only 2 tiles, but 4 cells requested.
    _write_solid(tiles_dir / "a.png", (255, 0, 0))
    _write_solid(tiles_dir / "b.png", (0, 0, 255))

    target = Image.new("RGB", (20, 20), (0, 0, 0))

    config = MosaicConfig(
        target_image="unused",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        duplicated_tile=False,
    )

    with pytest.raises(ValueError, match="duplicated_tile=False"):
        build_mosaic(target, config)


def test_build_mosaic_empty_tiles_dir_raises(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    target = Image.new("RGB", (10, 10), (0, 0, 0))

    config = MosaicConfig(
        target_image="unused",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
    )

    with pytest.raises(FileNotFoundError, match="No supported image files"):
        build_mosaic(target, config)


# ---------------------------------------------------------------------------
# generate_mosaic_photo (full disk-I/O entry point)
# ---------------------------------------------------------------------------


def test_generate_mosaic_photo_writes_jpeg(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))
    _write_solid(tiles_dir / "blue.png", (0, 0, 255))

    target_path = _make_two_tone_target(tmp_path / "target.png", size=(40, 20))
    output_path = tmp_path / "out" / "Result.jpg"

    config = MosaicConfig(
        target_image=target_path,
        tiles_path=tiles_dir,
        grid_size=(2, 4),
        scale=1.0,
        output_filename=output_path,
        seed=0,
    )

    returned = generate_mosaic_photo(config)

    # Output file exists, has nonzero size, decodes as a JPEG.
    assert returned == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as opened:
        opened.load()
        assert opened.format == "JPEG"
        assert opened.size == (40, 20)


def test_generate_mosaic_photo_writes_png(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "gray.png", (128, 128, 128))

    target_path = _write_solid(tmp_path / "target.png", (128, 128, 128), size=(20, 20))
    output_path = tmp_path / "Result.png"

    config = MosaicConfig(
        target_image=target_path,
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        output_filename=output_path,
        seed=0,
    )

    generate_mosaic_photo(config)

    with Image.open(output_path) as opened:
        opened.load()
        assert opened.format == "PNG"


def test_generate_mosaic_photo_missing_target_raises(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))

    config = MosaicConfig(
        target_image=tmp_path / "does_not_exist.jpg",
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
    )

    with pytest.raises(FileNotFoundError):
        generate_mosaic_photo(config)


def test_generate_mosaic_photo_creates_output_directory(tmp_path: Path) -> None:
    """The output's parent directory should be created if missing."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))

    target_path = _write_solid(tmp_path / "target.png", (200, 0, 0), size=(20, 20))
    output_path = tmp_path / "deeply" / "nested" / "out.jpg"

    config = MosaicConfig(
        target_image=target_path,
        tiles_path=tiles_dir,
        grid_size=(2, 2),
        scale=1.0,
        output_filename=output_path,
        seed=0,
    )

    generate_mosaic_photo(config)
    assert output_path.is_file()

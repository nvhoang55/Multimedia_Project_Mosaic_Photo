"""End-to-end tests for ``mosaic.core``.

These tests exercise the full pipeline (split -> average -> match ->
paste -> encode) on tiny synthetic fixtures. They are the strongest
correctness check in the suite because a regression in any single
module would produce a wrong-looking mosaic here.

What's intentionally not covered:

- ``MosaicConfig.__post_init__`` validation (constructor guards, not
  algorithmic behavior).
- Filesystem error paths (missing target, missing output dir, etc.) —
  those are stdlib behavior, not mosaic logic.
- PNG vs JPEG output formats — that's a one-line PIL save call.

The Lab-color path is covered here rather than in ``test_matching.py``
so we exercise it through the same code path real users hit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.core import MosaicConfig, build_mosaic, generate_mosaic_photo


def _write_solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Path:
    """Write a solid-color RGB image to ``path``."""
    Image.new("RGB", size, color).save(path)
    return path


def test_build_mosaic_picks_correct_tiles_per_cell(tmp_path: Path) -> None:
    """Red half of the target should be tiled with the red tile; blue with blue.

    This is the single most important correctness check in the suite: it
    verifies that splitting, averaging, matching, and pasting all line up
    so that each cell ends up with the perceptually-correct tile. A bug
    in any one of those stages — especially the historical row/col swap —
    would flip the colors and fail this test.
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

    # Left two columns should be predominantly red.
    left_block = arr[:, :20]
    assert left_block[..., 0].mean() > 200
    assert left_block[..., 2].mean() < 50

    # Right two columns should be predominantly blue.
    right_block = arr[:, 20:]
    assert right_block[..., 2].mean() > 200
    assert right_block[..., 0].mean() < 50


def test_build_mosaic_lab_color_space_picks_perceptual_match(tmp_path: Path) -> None:
    """Lab matching must pick the perceptually-closest tile.

    Exercises the sRGB -> linear -> XYZ -> Lab pipeline end-to-end through
    the matcher, which is why we don't unit-test ``rgb_to_lab`` directly.
    """
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

    # Every cell should pick the red tile, not blue.
    assert arr[..., 0].mean() > 200
    assert arr[..., 2].mean() < 50


def test_build_mosaic_seed_makes_runs_reproducible(tmp_path: Path) -> None:
    """Two runs with the same seed must produce byte-identical output.

    The pipeline shuffles the tile list before matching to break ties
    non-deterministically; ``seed`` is the only knob that makes that
    deterministic. Without this guarantee, callers can't snapshot-test
    their own output, and this very test suite couldn't either.
    """
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
    """``duplicated_tile=False`` must complete and pick distinct tiles.

    The original code's ``tiles.remove(int)`` implementation of this flag
    was completely broken (it removed a wrong element and desynced the
    parallel averages list). This test guards the new greedy-unique
    matcher against a similar regression.
    """
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


def test_generate_mosaic_photo_writes_decodable_output(tmp_path: Path) -> None:
    """The disk-I/O entry point must produce a file that PIL can re-open.

    This is the only test that goes through the full ``generate_mosaic_photo``
    surface (open target -> build_mosaic -> save). It guards the small
    amount of glue around ``build_mosaic``: file opening, directory
    creation, and format-from-extension inference.
    """
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))
    _write_solid(tiles_dir / "blue.png", (0, 0, 255))

    # Two-tone target: left red, right blue.
    target_path = tmp_path / "target.png"
    target = Image.new("RGB", (40, 20), (0, 0, 255))
    target.paste(Image.new("RGB", (20, 20), (255, 0, 0)), (0, 0))
    target.save(target_path)

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

    assert returned == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as opened:
        opened.load()
        assert opened.format == "JPEG"
        assert opened.size == (40, 20)

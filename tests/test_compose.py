"""Tests for ``mosaic.compose``.

The composition module is where the historical width/height swap bug
lived, so the tests here focus on the geometric invariants that would
silently produce a wrong-looking mosaic if regressed:

- ``compute_cell_size`` rounds down (so the splitter never asks for
  pixels past the image edge).
- ``split_image`` emits cells in row-major order (top-to-bottom,
  left-to-right) — this is the exact bug the original code had.
- ``split_image`` and ``paste_tiles_into_grid`` round-trip pixel-for-pixel
  on a divisible image. This is the strongest end-to-end check on both
  functions and on their shared ordering convention.
- ``paste_tiles_into_grid`` resizes mismatched tiles to fill the cell
  rather than leaving gaps.

Validation, no-op fast paths, and trivial cell-count checks are subsumed
by these and by the end-to-end tests in ``test_core.py``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from mosaic.compose import compute_cell_size, paste_tiles_into_grid, split_image


def test_compute_cell_size_rounds_down() -> None:
    # 101 wide / 10 cols = 10 (integer division), not 10.1. Rounding up
    # would cause split_image to read past the right edge of the image.
    assert compute_cell_size((101, 100), (10, 10)) == (10, 10)


def test_split_image_emits_cells_in_row_major_order() -> None:
    # Build a target where every cell has a distinct color, then verify
    # split_image returns them in row-major (top-to-bottom, left-to-right)
    # order. This is the exact regression that lived in the original code,
    # which iterated the outer loop over grid_width and the inner loop over
    # grid_height — the wrong way around.
    rows, cols = 2, 3
    cell_w, cell_h = 5, 4
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    expected_colors: list[tuple[int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            color = (r * 50, c * 50, 100)
            expected_colors.append(color)
            patch = Image.new("RGB", (cell_w, cell_h), color)
            image.paste(patch, (c * cell_w, r * cell_h))

    cells = split_image(image, (rows, cols))

    assert len(cells) == rows * cols
    for cell, expected in zip(cells, expected_colors, strict=True):
        arr = np.asarray(cell)
        assert (arr == np.array(expected, dtype=np.uint8)).all(), (
            f"Cell color mismatch: got {arr[0, 0].tolist()}, expected {expected}"
        )


def test_split_then_paste_round_trips_pixel_for_pixel() -> None:
    # Slice an image into cells and paste them straight back: the result
    # must be identical to the input. This catches any disagreement
    # between split_image's and paste_tiles_into_grid's ordering and is
    # the strongest single correctness check on the composition module.
    rows, cols = 3, 4
    cell_w, cell_h = 7, 5
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))

    # Paint each cell a distinct color so any swap is visible.
    for r in range(rows):
        for c in range(cols):
            patch = Image.new("RGB", (cell_w, cell_h), (r * 30 + 1, c * 40 + 1, 200))
            image.paste(patch, (c * cell_w, r * cell_h))

    cells = split_image(image, (rows, cols))
    rebuilt = paste_tiles_into_grid(cells, (rows, cols), (cell_w, cell_h))

    np.testing.assert_array_equal(np.asarray(image), np.asarray(rebuilt))


def test_paste_tiles_into_grid_resizes_mismatched_tiles() -> None:
    # Tiles whose size doesn't match cell_size must be resized to fill the
    # cell. The original implementation used the *maximum* tile dimensions
    # and silently produced gaps when tiles had different aspect ratios;
    # this test guards against that regression.
    rows, cols = 2, 2
    cell_w, cell_h = 10, 10
    tiles = [
        Image.new("RGB", (5, 5), (255, 0, 0)),  # smaller than cell
        Image.new("RGB", (20, 20), (0, 255, 0)),  # larger than cell
        Image.new("RGB", (10, 10), (0, 0, 255)),  # exact
        Image.new("RGB", (3, 7), (255, 255, 0)),  # different aspect
    ]

    canvas = paste_tiles_into_grid(tiles, (rows, cols), (cell_w, cell_h))

    # No gaps: canvas dims are exactly (cols*cw, rows*ch).
    assert canvas.size == (cols * cell_w, rows * cell_h)

    # The top-left cell should be predominantly red, despite tile being smaller.
    arr = np.asarray(canvas)
    top_left = arr[0:cell_h, 0:cell_w]
    assert top_left[..., 0].mean() > 200
    assert top_left[..., 1].mean() < 50
    assert top_left[..., 2].mean() < 50

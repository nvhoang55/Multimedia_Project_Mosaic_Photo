"""Tests for ``mosaic.compose``."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mosaic.compose import (
    compute_cell_size,
    fit_target_to_grid,
    paste_tiles_into_grid,
    split_image,
)

# ---------------------------------------------------------------------------
# compute_cell_size
# ---------------------------------------------------------------------------


def test_compute_cell_size_evenly_divisible() -> None:
    # 200 wide / 10 cols = 20 px wide per cell; 100 tall / 5 rows = 20 px tall.
    assert compute_cell_size((200, 100), (5, 10)) == (20, 20)


def test_compute_cell_size_rounds_down() -> None:
    # 101 / 10 = 10 (integer division), not 10.1.
    assert compute_cell_size((101, 100), (10, 10)) == (10, 10)


def test_compute_cell_size_non_square_grid() -> None:
    # Sanity check that (rows, cols) order is respected: rows=2, cols=4
    # with a 400x200 image -> 100 wide x 100 tall cells.
    assert compute_cell_size((400, 200), (2, 4)) == (100, 100)


def test_compute_cell_size_rejects_non_positive_grid() -> None:
    with pytest.raises(ValueError):
        compute_cell_size((100, 100), (0, 10))
    with pytest.raises(ValueError):
        compute_cell_size((100, 100), (10, -1))


# ---------------------------------------------------------------------------
# fit_target_to_grid
# ---------------------------------------------------------------------------


def test_fit_target_to_grid_no_op_when_already_divisible() -> None:
    # 200x100 with grid (5, 10) -> cell (20, 20) -> target size unchanged.
    image = Image.new("RGB", (200, 100), (10, 20, 30))
    fitted = fit_target_to_grid(image, (5, 10))
    assert fitted.size == (200, 100)
    # When no resize is needed, the function should return the same object
    # to avoid an unnecessary copy.
    assert fitted is image


def test_fit_target_to_grid_resizes_to_multiple() -> None:
    # 201x101 with grid (5, 10) -> cell (20, 20) -> resized to (200, 100).
    image = Image.new("RGB", (201, 101), (10, 20, 30))
    fitted = fit_target_to_grid(image, (5, 10))
    assert fitted.size == (200, 100)


def test_fit_target_to_grid_uses_explicit_cell_size() -> None:
    # Caller can override the derived cell size to upsample the target.
    image = Image.new("RGB", (50, 50), (0, 0, 0))
    fitted = fit_target_to_grid(image, (5, 5), cell_size=(20, 20))
    # 5 cols * 20 = 100, 5 rows * 20 = 100.
    assert fitted.size == (100, 100)


def test_fit_target_to_grid_rejects_image_smaller_than_grid() -> None:
    # 4x4 image with a 5x5 grid -> cell size would be (0, 0).
    image = Image.new("RGB", (4, 4), (0, 0, 0))
    with pytest.raises(ValueError, match="non-positive"):
        fit_target_to_grid(image, (5, 5))


# ---------------------------------------------------------------------------
# split_image
# ---------------------------------------------------------------------------


def test_split_image_produces_correct_count() -> None:
    image = Image.new("RGB", (40, 30), (0, 0, 0))
    cells = split_image(image, (3, 4))  # 3 rows, 4 cols
    assert len(cells) == 12


def test_split_image_cells_have_uniform_size() -> None:
    image = Image.new("RGB", (40, 30), (0, 0, 0))
    cells = split_image(image, (3, 4))
    # Cell size: 40/4 = 10 wide, 30/3 = 10 tall.
    for cell in cells:
        assert cell.size == (10, 10)


def test_split_image_row_major_order() -> None:
    # Build a target where every cell has a distinct color; verify that
    # split_image returns them in row-major (top-to-bottom, left-to-right)
    # order. This catches the historical width/height swap bug.
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
        # Every pixel of the cell should be the expected color.
        arr = np.asarray(cell)
        assert (arr == np.array(expected, dtype=np.uint8)).all(), (
            f"Cell color mismatch: got {arr[0, 0].tolist()}, expected {expected}"
        )


def test_split_image_non_square_grid_dimensions() -> None:
    # 4 rows, 2 cols on a 20x40 image -> cells are 10 wide x 10 tall.
    image = Image.new("RGB", (20, 40), (0, 0, 0))
    cells = split_image(image, (4, 2))
    assert len(cells) == 8
    for cell in cells:
        assert cell.size == (10, 10)


# ---------------------------------------------------------------------------
# paste_tiles_into_grid
# ---------------------------------------------------------------------------


def test_paste_tiles_into_grid_size_matches_grid_times_cell() -> None:
    rows, cols = 3, 5
    cell_w, cell_h = 8, 6
    tiles = [Image.new("RGB", (cell_w, cell_h), (0, 0, 0)) for _ in range(rows * cols)]
    canvas = paste_tiles_into_grid(tiles, (rows, cols), (cell_w, cell_h))
    assert canvas.size == (cols * cell_w, rows * cell_h)
    assert canvas.mode == "RGB"


def test_paste_tiles_into_grid_round_trips_through_split() -> None:
    # Split an image into cells, then paste them back: the result should
    # be pixel-identical to the input. This is the strongest correctness
    # check on the row-major ordering of both functions.
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
    # If a tile's size does not match cell_size, it should be resized
    # rather than producing gaps or being cropped.
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
    # Mean red channel is high; mean green/blue are low.
    assert top_left[..., 0].mean() > 200
    assert top_left[..., 1].mean() < 50
    assert top_left[..., 2].mean() < 50


def test_paste_tiles_into_grid_rejects_wrong_count() -> None:
    rows, cols = 2, 3  # expects 6 tiles
    tiles = [Image.new("RGB", (4, 4), (0, 0, 0)) for _ in range(5)]
    with pytest.raises(ValueError, match="Expected 6 tiles"):
        paste_tiles_into_grid(tiles, (rows, cols), (4, 4))

"""Tests for ``mosaic.matching``.

Matching is the heart of the mosaic algorithm. Only the behavioral
contracts that would silently produce wrong mosaics if regressed are
covered here:

- ``best_match_indices`` actually picks the closest tile per cell.
- ``best_match_indices`` happily reuses tiles when duplication is allowed.
- ``best_match_indices_unique`` never picks the same tile twice.
- ``best_match_indices_unique`` falls back to a second-choice tile when
  the preferred one is already taken.

The Lab color-space path is exercised end-to-end in ``test_core.py``;
the conversion math itself is a transparent helper, not core logic.
"""

from __future__ import annotations

import numpy as np

from mosaic.matching import best_match_indices, best_match_indices_unique


def test_best_match_indices_picks_closest_tile_per_cell() -> None:
    # Tiles are pure R, G, B; cells are slightly perturbed versions of each.
    # The matcher must return the perceptually-correct tile index for every
    # cell — this is the single most important invariant of the package.
    tile_colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.float64,
    )
    cell_colors = np.array(
        [
            [250, 5, 5],  # closest to red (index 0)
            [5, 250, 5],  # closest to green (index 1)
            [5, 5, 250],  # closest to blue (index 2)
            [240, 10, 10],  # also red
        ],
        dtype=np.float64,
    )

    indices = best_match_indices(cell_colors, tile_colors)

    np.testing.assert_array_equal(indices, [0, 1, 2, 0])


def test_best_match_indices_allows_duplicates() -> None:
    # Three cells, two tiles; the same tile must be reusable.
    tile_colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    cell_colors = np.array(
        [
            [10, 10, 10],
            [20, 20, 20],
            [200, 200, 200],
        ],
        dtype=np.float64,
    )

    indices = best_match_indices(cell_colors, tile_colors)

    # First two cells both pick the black tile (index 0); third picks white.
    np.testing.assert_array_equal(indices, [0, 0, 1])


def test_best_match_indices_unique_returns_distinct_tiles() -> None:
    # When duplication is forbidden, every cell must end up with a different
    # tile index. With three cells whose preferred tiles are all distinct,
    # the unique matcher should pick exactly those preferred tiles.
    tile_colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
        ],
        dtype=np.float64,
    )
    cell_colors = np.array(
        [
            [250, 5, 5],
            [5, 250, 5],
            [5, 5, 250],
        ],
        dtype=np.float64,
    )

    indices = best_match_indices_unique(cell_colors, tile_colors)

    assert len(set(indices.tolist())) == 3
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_best_match_indices_unique_falls_back_when_preferred_is_taken() -> None:
    # Two cells both prefer tile 0; the second cell must pick its
    # second-choice (tile 1) since duplication is forbidden. This is the
    # exact regression that broke in the original ``tiles.remove(int)``
    # implementation.
    tile_colors = np.array(
        [
            [0, 0, 0],
            [10, 10, 10],
            [255, 255, 255],
        ],
        dtype=np.float64,
    )
    cell_colors = np.array(
        [
            [1, 1, 1],  # closest to tile 0
            [2, 2, 2],  # also closest to tile 0 — but tile 0 is taken
        ],
        dtype=np.float64,
    )

    indices = best_match_indices_unique(cell_colors, tile_colors)

    np.testing.assert_array_equal(indices, [0, 1])

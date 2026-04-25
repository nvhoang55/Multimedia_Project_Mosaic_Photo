"""Tests for ``mosaic.matching``."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.matching import (
    best_match_indices,
    best_match_indices_unique,
    rgb_to_lab,
)

# ---------------------------------------------------------------------------
# best_match_indices (with duplication allowed)
# ---------------------------------------------------------------------------


def test_best_match_indices_picks_exact_color() -> None:
    # Tiles are pure R, G, B; cells are slightly perturbed versions of each.
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
    # Three cells, two tiles; the same tile should be reused.
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

    # First two cells both pick the black tile (index 0).
    np.testing.assert_array_equal(indices, [0, 0, 1])


def test_best_match_indices_returns_int_array() -> None:
    tile_colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    cell_colors = np.array([[10, 10, 10]], dtype=np.float64)

    indices = best_match_indices(cell_colors, tile_colors)

    assert indices.dtype == np.int64
    assert indices.shape == (1,)


def test_best_match_indices_lab_space_runs() -> None:
    # Smoke test: lab path must produce indices in range with the correct shape.
    rng = np.random.default_rng(0)
    tile_colors = rng.integers(0, 256, size=(8, 3)).astype(np.float64)
    cell_colors = rng.integers(0, 256, size=(20, 3)).astype(np.float64)

    indices = best_match_indices(cell_colors, tile_colors, color_space="lab")

    assert indices.shape == (20,)
    assert indices.min() >= 0
    assert indices.max() < tile_colors.shape[0]


def test_best_match_indices_lab_prefers_perceptual_match() -> None:
    # Construct a case where RGB and Lab disagree.
    # Cell is a mid-gray; tile A is the same gray, tile B is a vivid color
    # whose RGB-distance happens to be smaller than the gray's *channelwise*
    # distance but whose Lab-distance is much larger.
    cell_colors = np.array([[128, 128, 128]], dtype=np.float64)
    tile_colors = np.array(
        [
            [128, 128, 128],  # identical gray (perceptually closest)
            [200, 50, 50],  # a vivid red
        ],
        dtype=np.float64,
    )

    rgb_idx = best_match_indices(cell_colors, tile_colors, color_space="rgb")
    lab_idx = best_match_indices(cell_colors, tile_colors, color_space="lab")

    # In *both* spaces the identical color must win — this asserts the
    # tautology that exact matches are preferred regardless of space.
    np.testing.assert_array_equal(rgb_idx, [0])
    np.testing.assert_array_equal(lab_idx, [0])


def test_best_match_indices_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        best_match_indices(np.zeros((4,)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        best_match_indices(np.zeros((4, 3)), np.zeros((3, 4)))


def test_best_match_indices_rejects_empty_tiles() -> None:
    with pytest.raises(ValueError):
        best_match_indices(np.zeros((1, 3)), np.zeros((0, 3)))


def test_best_match_indices_rejects_unknown_color_space() -> None:
    with pytest.raises(ValueError):
        best_match_indices(
            np.zeros((1, 3)),
            np.ones((1, 3)),
            color_space="hsv",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# best_match_indices_unique (no duplicates)
# ---------------------------------------------------------------------------


def test_best_match_indices_unique_returns_unique_indices() -> None:
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

    # Each cell finds its perfect match and they are all distinct.
    assert len(set(indices.tolist())) == 3
    np.testing.assert_array_equal(indices, [0, 1, 2])


def test_best_match_indices_unique_forces_second_choice() -> None:
    # Two cells both prefer tile 0; second cell must fall back to tile 1.
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


def test_best_match_indices_unique_raises_when_too_few_tiles() -> None:
    tile_colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    cell_colors = np.zeros((5, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="Cannot match without duplication"):
        best_match_indices_unique(cell_colors, tile_colors)


def test_best_match_indices_unique_lab_runs() -> None:
    # Smoke test for the Lab path on the unique matcher.
    rng = np.random.default_rng(1)
    tile_colors = rng.integers(0, 256, size=(10, 3)).astype(np.float64)
    cell_colors = rng.integers(0, 256, size=(5, 3)).astype(np.float64)

    indices = best_match_indices_unique(cell_colors, tile_colors, color_space="lab")

    assert len(set(indices.tolist())) == 5
    assert indices.shape == (5,)


# ---------------------------------------------------------------------------
# rgb_to_lab
# ---------------------------------------------------------------------------


def test_rgb_to_lab_black() -> None:
    # sRGB black -> Lab (0, 0, 0).
    lab = rgb_to_lab(np.array([[0, 0, 0]], dtype=np.float64))
    np.testing.assert_allclose(lab[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_rgb_to_lab_white() -> None:
    # sRGB white -> Lab L=100, a=0, b=0.
    lab = rgb_to_lab(np.array([[255, 255, 255]], dtype=np.float64))
    np.testing.assert_allclose(lab[0], [100.0, 0.0, 0.0], atol=1e-3)


def test_rgb_to_lab_mid_gray_has_zero_chroma() -> None:
    # Any neutral gray must have a == 0 and b == 0 (no chroma).
    lab = rgb_to_lab(np.array([[128, 128, 128]], dtype=np.float64))
    assert lab.shape == (1, 3)
    np.testing.assert_allclose(lab[0, 1:], [0.0, 0.0], atol=1e-4)
    # L for mid-gray is around 53.4 (sRGB).
    assert 50.0 < lab[0, 0] < 60.0


def test_rgb_to_lab_preserves_shape() -> None:
    # (H, W, 3) -> (H, W, 3); function must broadcast cleanly.
    rgb = np.zeros((4, 5, 3), dtype=np.float64)
    lab = rgb_to_lab(rgb)
    assert lab.shape == (4, 5, 3)


def test_rgb_to_lab_red_has_positive_a() -> None:
    # Pure red has positive a* (red-green axis) and positive b* (yellow-blue).
    lab = rgb_to_lab(np.array([[255, 0, 0]], dtype=np.float64))
    assert lab[0, 1] > 0  # a*
    assert lab[0, 2] > 0  # b*


def test_rgb_to_lab_blue_has_negative_b() -> None:
    # Pure blue has negative b* (toward blue end of yellow-blue axis).
    lab = rgb_to_lab(np.array([[0, 0, 255]], dtype=np.float64))
    assert lab[0, 2] < 0

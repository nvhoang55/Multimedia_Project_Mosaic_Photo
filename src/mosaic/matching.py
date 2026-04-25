"""Nearest-tile matching.

Given an array of cell mean colors and an array of tile mean colors, return
for each cell the index of the closest tile. The implementation is fully
vectorized with numpy: for typical mosaic sizes (tens of thousands of cells,
hundreds to thousands of tiles) this is 50-500x faster than the original
pure-Python double loop.

Two color spaces are supported:

- ``"rgb"``: squared Euclidean distance in RGB. Fast, but does not match
  human perception especially well.
- ``"lab"``: Euclidean distance in CIE Lab (a reasonable approximation of
  perceptual ΔE76). Slightly slower but visibly better for photographic
  tiles. Conversion goes through sRGB -> linear RGB -> XYZ (D65) -> Lab
  using numpy only, so we don't pull in scikit-image as a dependency.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ColorSpace = Literal["rgb", "lab"]

# sRGB -> linear-RGB -> XYZ (D65) matrix, from IEC 61966-2-1 / Bruce Lindbloom.
_RGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# D65 reference white in XYZ, scaled so Y = 1.0.
_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Apply the inverse sRGB companding to values in ``[0, 1]``.

    Operates element-wise; preserves array shape.
    """
    a = 0.055
    threshold = 0.04045
    linear = np.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + a) / (1 + a)) ** 2.4,
    )
    return linear


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` XYZ array (D65) to CIE Lab."""
    # Normalize by the reference white.
    xyz_n = xyz / _D65_WHITE

    # Lab f() function with the standard CIE epsilon/kappa constants.
    epsilon = 216 / 24389
    kappa = 24389 / 27
    f = np.where(
        xyz_n > epsilon,
        np.cbrt(xyz_n),
        (kappa * xyz_n + 16) / 116,
    )

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB colors in ``[0, 255]`` to CIE Lab.

    Accepts any shape with a trailing dimension of size 3 (e.g. ``(N, 3)``
    or ``(H, W, 3)``); returns the same shape.
    """
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = _srgb_to_linear(rgb)
    # (..., 3) @ (3, 3).T -> (..., 3)
    xyz = linear @ _RGB_TO_XYZ_D65.T
    return _xyz_to_lab(xyz)


def _to_color_space(rgb: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    """Project an ``(N, 3)`` RGB array into the requested color space."""
    if color_space == "rgb":
        return np.asarray(rgb, dtype=np.float64)
    if color_space == "lab":
        return rgb_to_lab(rgb)
    raise ValueError(f"Unknown color_space: {color_space!r}")


def best_match_indices(
    cell_colors: np.ndarray,
    tile_colors: np.ndarray,
    color_space: ColorSpace = "rgb",
) -> np.ndarray:
    """Return, for each cell, the index of the closest tile.

    Parameters
    ----------
    cell_colors:
        ``(C, 3)`` array of mean RGB values for each target cell, in
        ``[0, 255]``.
    tile_colors:
        ``(T, 3)`` array of mean RGB values for each available tile, in
        ``[0, 255]``.
    color_space:
        ``"rgb"`` for squared-Euclidean matching in RGB, ``"lab"`` for
        Euclidean matching in CIE Lab (perceptually closer to ΔE76).

    Returns
    -------
    indices:
        ``(C,)`` int array; ``indices[i]`` is the row in ``tile_colors``
        whose color is closest to ``cell_colors[i]``.

    Notes
    -----
    Computes the full ``(C, T)`` distance matrix by broadcasting. Memory
    cost is ``8 * C * T`` bytes (float64); for 22500 cells x 1000 tiles
    that is ~180 MB, which is acceptable. If callers ever need to scale
    beyond that, this function can be chunked over cells without changing
    its API.
    """
    cell_colors = np.asarray(cell_colors, dtype=np.float64)
    tile_colors = np.asarray(tile_colors, dtype=np.float64)

    if cell_colors.ndim != 2 or cell_colors.shape[1] != 3:
        raise ValueError(f"cell_colors must have shape (C, 3); got {cell_colors.shape}")
    if tile_colors.ndim != 2 or tile_colors.shape[1] != 3:
        raise ValueError(f"tile_colors must have shape (T, 3); got {tile_colors.shape}")
    if tile_colors.shape[0] == 0:
        raise ValueError("tile_colors is empty; nothing to match against")

    cells = _to_color_space(cell_colors, color_space)
    tiles = _to_color_space(tile_colors, color_space)

    # Broadcast: (C, 1, 3) - (1, T, 3) -> (C, T, 3); square + sum -> (C, T).
    # argmin over the same monotonic transform of distance, so no sqrt needed.
    diff = cells[:, None, :] - tiles[None, :, :]
    dist_sq = np.einsum("ctk,ctk->ct", diff, diff)
    return np.argmin(dist_sq, axis=1).astype(np.int64)


def best_match_indices_unique(
    cell_colors: np.ndarray,
    tile_colors: np.ndarray,
    color_space: ColorSpace = "rgb",
) -> np.ndarray:
    """Match cells to tiles without reusing any tile (greedy).

    Requires ``len(tile_colors) >= len(cell_colors)``. Walks cells in their
    natural order and, for each one, picks the closest *unused* tile. This
    is greedy rather than globally optimal (the assignment problem would
    require Hungarian / linear_sum_assignment), but it is O(C * T) in time
    and matches the historical semantics of the ``duplicated_tile=False``
    flag in the original code.

    Returns ``(C,)`` int array of unique tile indices.
    """
    cell_colors = np.asarray(cell_colors, dtype=np.float64)
    tile_colors = np.asarray(tile_colors, dtype=np.float64)

    n_cells = cell_colors.shape[0]
    n_tiles = tile_colors.shape[0]
    if n_tiles < n_cells:
        raise ValueError(
            f"Cannot match without duplication: have {n_tiles} tiles "
            f"but need {n_cells} unique cells."
        )

    cells = _to_color_space(cell_colors, color_space)
    tiles = _to_color_space(tile_colors, color_space)

    # Precompute the full distance matrix once; mask out used columns
    # by overwriting them with +inf as we go.
    diff = cells[:, None, :] - tiles[None, :, :]
    dist_sq = np.einsum("ctk,ctk->ct", diff, diff)

    result = np.empty(n_cells, dtype=np.int64)
    for i in range(n_cells):
        idx = int(np.argmin(dist_sq[i]))
        result[i] = idx
        # Forbid this tile for all subsequent cells.
        dist_sq[i + 1 :, idx] = np.inf

    return result

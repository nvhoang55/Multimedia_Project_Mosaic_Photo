"""Splitting a target image into cells and pasting tiles back into a grid.

Convention used throughout this module (and the rest of the package):

- ``grid_size`` is ``(grid_rows, grid_cols)`` — the number of cells
  vertically and horizontally, in that order. This matches numpy's
  ``(rows, cols)`` shape convention.
- ``cell_size`` is ``(cell_width, cell_height)`` — pixel dimensions of
  a single cell, in PIL's ``(width, height)`` order.

Keeping these two orders explicit (and named consistently across the
codebase) avoids the silent width/height swap that existed in the
original implementation.
"""

from __future__ import annotations

from PIL import Image


def compute_cell_size(
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> tuple[int, int]:
    """Return ``(cell_width, cell_height)`` for a target image of ``image_size``.

    ``image_size`` is in PIL ``(width, height)`` order; ``grid_size`` is
    ``(rows, cols)``. Integer-divides the image dimensions by the grid
    dimensions; callers are expected to have resized the target so it
    divides evenly (see :func:`fit_target_to_grid`).
    """
    img_w, img_h = image_size
    rows, cols = grid_size
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid_size must be positive; got {grid_size!r}")
    return img_w // cols, img_h // rows


def fit_target_to_grid(
    image: Image.Image,
    grid_size: tuple[int, int],
    cell_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Resize ``image`` so its dimensions are exact multiples of ``grid_size``.

    If ``cell_size`` is provided, the result is exactly
    ``(cols * cell_w, rows * cell_h)``. Otherwise the cell size is derived
    from the input image and ``grid_size`` (rounding down). Returning a
    cleanly-divisible image lets the splitter avoid losing a strip of
    pixels along the right/bottom edges.
    """
    rows, cols = grid_size
    if cell_size is None:
        cell_size = compute_cell_size(image.size, grid_size)
    cell_w, cell_h = cell_size
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError(
            f"Derived cell_size {cell_size!r} is non-positive; "
            f"image is likely smaller than the grid {grid_size!r}."
        )
    target_size = (cols * cell_w, rows * cell_h)
    if image.size == target_size:
        return image
    return image.resize(target_size, Image.LANCZOS)


def split_image(
    image: Image.Image,
    grid_size: tuple[int, int],
) -> list[Image.Image]:
    """Slice ``image`` into ``rows * cols`` cells in row-major order.

    The image is expected to already be sized so that each axis divides
    evenly by the grid (call :func:`fit_target_to_grid` first if not).
    Returned cells are ordered ``[(r=0, c=0), (r=0, c=1), ..., (r=R-1, c=C-1)]``,
    which matches the order :func:`paste_tiles_into_grid` expects.
    """
    rows, cols = grid_size
    cell_w, cell_h = compute_cell_size(image.size, grid_size)

    cells: list[Image.Image] = []
    for row in range(rows):
        top = row * cell_h
        bottom = top + cell_h
        for col in range(cols):
            left = col * cell_w
            right = left + cell_w
            cells.append(image.crop((left, top, right, bottom)))
    return cells


def paste_tiles_into_grid(
    tiles: list[Image.Image],
    grid_size: tuple[int, int],
    cell_size: tuple[int, int],
) -> Image.Image:
    """Compose tiles back into a single image.

    Every tile is resized to ``cell_size`` before being pasted, so the
    output is exactly ``(cols * cell_w, rows * cell_h)`` regardless of
    individual tile dimensions. This is more predictable than the original
    implementation, which used the *maximum* tile dimensions and assumed
    every tile was that size — which silently produced gaps when tiles
    had different aspect ratios.

    Parameters
    ----------
    tiles:
        Length ``rows * cols`` list of PIL images in row-major order.
    grid_size:
        ``(rows, cols)``.
    cell_size:
        ``(cell_width, cell_height)`` for each cell of the output.
    """
    rows, cols = grid_size
    expected = rows * cols
    if len(tiles) != expected:
        raise ValueError(f"Expected {expected} tiles for grid {grid_size!r}; got {len(tiles)}.")

    cell_w, cell_h = cell_size
    out_w, out_h = cols * cell_w, rows * cell_h
    canvas = Image.new("RGB", (out_w, out_h))

    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        if tile.size != cell_size:
            tile = tile.resize(cell_size, Image.LANCZOS)
        canvas.paste(tile, (col * cell_w, row * cell_h))

    return canvas

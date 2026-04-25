"""Top-level orchestration for mosaic generation.

This module exposes the public API of the package:

- :class:`MosaicConfig` — dataclass holding all generation parameters.
- :func:`generate_mosaic_photo` — end-to-end entry point.
- :func:`build_mosaic` — same thing but takes an already-loaded PIL image
  for the target, useful in tests and library callers.

The pipeline is:

1. Open + resize the target image so its dimensions divide evenly by the
   grid (this is done *before* slicing rather than via a separate
   ``scale``-then-slice step, which keeps peak memory low).
2. Slice the target into row-major cells.
3. Load tile images, thumbnailing during decode where possible.
4. Compute mean RGB for every tile and every cell (vectorized).
5. Match cells to tiles in a single numpy broadcast (optionally in Lab).
6. Paste matched tiles back into a fresh canvas at the chosen cell size.
7. Convert to the requested color mode and write to disk.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.compose import (
    compute_cell_size,
    fit_target_to_grid,
    paste_tiles_into_grid,
    split_image,
)
from mosaic.matching import (
    ColorSpace,
    best_match_indices,
    best_match_indices_unique,
)
from mosaic.tiles import (
    average_rgb_batch,
    iter_image_paths,
    load_tiles,
)

logger = logging.getLogger(__name__)

# Color modes accepted by PIL's ``Image.convert``. We only expose the two
# that make sense for a mosaic: full color and grayscale.
ColorMode = str  # "RGB" or "L"; kept as str to stay compatible with PIL.


@dataclass(frozen=True)
class MosaicConfig:
    """Parameters for a single mosaic generation run.

    Attributes
    ----------
    target_image:
        Path to the source image that will be turned into a mosaic.
    tiles_path:
        Directory containing tile images. Only files with extensions in
        :data:`mosaic.tiles.SUPPORTED_EXTENSIONS` are considered.
    output_filename:
        Where the resulting mosaic will be written. The format is inferred
        from the extension (``.jpg``/``.jpeg`` -> JPEG, ``.png`` -> PNG).
    grid_size:
        ``(rows, cols)`` — how many tiles the mosaic has vertically and
        horizontally. The product is the total number of cells.
    scale:
        Multiplier applied to the *target* image dimensions before
        splitting. Larger ``scale`` -> larger output / more pixels per
        cell. Combined with ``grid_size``, this sets the final mosaic
        resolution.
    duplicated_tile:
        If ``False``, every tile in ``tiles_path`` is used at most once;
        requires ``rows * cols <= number_of_tiles``.
    color_mode:
        ``"RGB"`` for color output, ``"L"`` for grayscale.
    color_space:
        ``"rgb"`` for fast squared-Euclidean matching, ``"lab"`` for
        slower but more perceptually accurate Lab/ΔE76 matching.
    jpeg_quality:
        Encoder quality used when ``output_filename`` is JPEG.
    seed:
        Optional RNG seed. The tile order is shuffled before matching so
        that ties (multiple tiles equally close to a cell) are resolved
        non-deterministically; setting ``seed`` makes runs reproducible.
    """

    target_image: str | Path
    tiles_path: str | Path
    grid_size: tuple[int, int]
    output_filename: str | Path = "Result.jpg"
    scale: float = 3.0
    duplicated_tile: bool = True
    color_mode: ColorMode = "RGB"
    color_space: ColorSpace = "rgb"
    jpeg_quality: int = 90
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError(f"grid_size must be positive; got {self.grid_size!r}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive; got {self.scale}")
        if self.color_mode not in {"RGB", "L"}:
            raise ValueError(f"color_mode must be 'RGB' or 'L'; got {self.color_mode!r}")
        if self.color_space not in {"rgb", "lab"}:
            raise ValueError(f"color_space must be 'rgb' or 'lab'; got {self.color_space!r}")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError(f"jpeg_quality must be in [1, 95]; got {self.jpeg_quality}")


def generate_mosaic_photo(config: MosaicConfig) -> Path:
    """Build a mosaic image from ``config`` and write it to disk.

    Returns the path of the file that was written.
    """
    target_path = Path(config.target_image)
    if not target_path.is_file():
        raise FileNotFoundError(f"Target image not found: {target_path}")

    logger.info("Opening target image: %s", target_path)
    with Image.open(target_path) as src:
        src.load()
        target = src.convert("RGB")

    mosaic = build_mosaic(target, config)

    output_path = Path(config.output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, object] = {}
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs.update(format="JPEG", quality=config.jpeg_quality, optimize=True)
    elif suffix == ".png":
        save_kwargs.update(format="PNG", optimize=True)
    # Other extensions: let PIL infer.

    mosaic.save(output_path, **save_kwargs)
    logger.info("Wrote mosaic to %s", output_path)
    return output_path


def build_mosaic(target: Image.Image, config: MosaicConfig) -> Image.Image:
    """Build a mosaic ``Image`` from an already-opened ``target``.

    Splitting the disk-I/O entry point (:func:`generate_mosaic_photo`) from
    this in-memory function makes the pipeline straightforward to unit
    test without having to materialize files.
    """
    rows, cols = config.grid_size

    # Step 1: figure out the final mosaic size and resize the target to it.
    # We deliberately resize the target *directly* to the mosaic-grid size
    # rather than going through a (possibly huge) ``scale``-times intermediate.
    scaled_size = (
        max(1, round(target.size[0] * config.scale)),
        max(1, round(target.size[1] * config.scale)),
    )
    cell_size = compute_cell_size(scaled_size, config.grid_size)
    if cell_size[0] <= 0 or cell_size[1] <= 0:
        raise ValueError(
            f"Computed cell size {cell_size!r} is non-positive. "
            f"Try a smaller grid_size or a larger scale."
        )

    fitted = fit_target_to_grid(target, config.grid_size, cell_size=cell_size)
    logger.info(
        "Target sized to %s for a %dx%d grid (cell size %s)",
        fitted.size,
        rows,
        cols,
        cell_size,
    )

    # Step 2: slice into row-major cells.
    cells = split_image(fitted, config.grid_size)
    logger.info("Split target into %d cells", len(cells))

    # Step 3: discover and load tiles, thumbnailing them to cell size during
    # decode to keep peak RAM low.
    tile_paths = iter_image_paths(config.tiles_path)
    if not tile_paths:
        raise FileNotFoundError(f"No supported image files found in {config.tiles_path!r}")

    if not config.duplicated_tile and len(tile_paths) < rows * cols:
        raise ValueError(
            f"duplicated_tile=False requires at least {rows * cols} tiles "
            f"but only {len(tile_paths)} were found in {config.tiles_path!r}."
        )

    logger.info("Loading %d tiles from %s", len(tile_paths), config.tiles_path)
    tiles = load_tiles(tile_paths, target_size=cell_size)
    if not tiles:
        raise RuntimeError(f"None of the files in {config.tiles_path!r} could be decoded.")

    # Shuffle so that ties in matching are broken non-deterministically
    # (visually nicer than always picking the first match). Reproducible
    # via ``config.seed``.
    rng = random.Random(config.seed)
    rng.shuffle(tiles)

    # Step 4: compute mean colors for every tile and cell. The cells were
    # sliced from a ``convert('RGB')``-ed source, so they are already RGB.
    logger.info("Computing average colors")
    tile_colors = average_rgb_batch(tiles)
    cell_colors = average_rgb_batch(cells)

    # Step 5: pick the closest tile for every cell.
    logger.info(
        "Matching %d cells to %d tiles in %s space (duplicated_tile=%s)",
        cell_colors.shape[0],
        tile_colors.shape[0],
        config.color_space,
        config.duplicated_tile,
    )
    if config.duplicated_tile:
        match_indices = best_match_indices(cell_colors, tile_colors, color_space=config.color_space)
    else:
        match_indices = best_match_indices_unique(
            cell_colors, tile_colors, color_space=config.color_space
        )

    matched_tiles = [tiles[int(i)] for i in match_indices.tolist()]

    # Step 6: paste back into a fresh canvas, exactly cell-sized so we
    # never get gaps from tiles with mismatched aspect ratios.
    logger.info("Compositing mosaic")
    mosaic = paste_tiles_into_grid(matched_tiles, config.grid_size, cell_size=cell_size)

    # Step 7: optional grayscale conversion.
    if config.color_mode != "RGB":
        mosaic = mosaic.convert(config.color_mode)
        # Most downstream consumers (and JPEG) handle "L" fine; if a caller
        # asks for "L" but the file extension is JPEG, PIL writes a valid
        # grayscale JPEG without further work. Keep the array shape sane
        # for any tests that re-load the result.
        _ = np.asarray(mosaic)

    return mosaic

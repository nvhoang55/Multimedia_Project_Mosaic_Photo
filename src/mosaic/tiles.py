"""Tile loading and average-color computation.

Responsibilities:
    - Discover image files in a directory.
    - Load them safely (no leaked file handles, no silent failures).
    - Use JPEG draft decoding to keep peak memory low when tiles are large.
    - Compute per-tile mean RGB as a numpy array.

Note: Lab/ΔE conversion is handled by `mosaic.matching` to keep this module
focused on I/O and basic statistics.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Extensions PIL can read that we treat as candidate tiles.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
)


def iter_image_paths(directory: str | os.PathLike[str]) -> list[Path]:
    """Return sorted paths of supported image files directly inside ``directory``.

    Hidden files (starting with ``.``) and non-image extensions are skipped.
    Order is deterministic (sorted by name) so that runs are reproducible
    when callers seed the RNG.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Tiles path is not a directory: {dir_path}")

    paths: list[Path] = []
    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping non-image file: %s", entry)
            continue
        paths.append(entry)
    return paths


def load_tile(
    path: str | os.PathLike[str], target_size: tuple[int, int] | None = None
) -> Image.Image:
    """Load a single tile image as RGB, optionally thumbnailing during decode.

    For JPEG sources, ``Image.draft`` lets libjpeg decode at a smaller size
    very cheaply, which dramatically reduces peak memory and CPU when the
    final tile is much smaller than the source.

    The returned image has ``mode == 'RGB'`` and is fully loaded into memory
    so the caller does not need to keep the file handle open.
    """
    path = Path(path)
    with Image.open(path) as src:
        if target_size is not None and src.format == "JPEG":
            # draft() rounds down to a power-of-two scale; harmless if it can't.
            src.draft("RGB", target_size)

        # convert() returns a new image; close source after to free its handle.
        image = src.convert("RGB")

    if target_size is not None:
        # thumbnail() is in-place and preserves aspect ratio.
        image.thumbnail(target_size)

    image.load()
    return image


def load_tiles(
    paths: Iterable[str | os.PathLike[str]],
    target_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """Load every path; tiles that fail to decode are skipped with a warning."""
    tiles: list[Image.Image] = []
    for path in paths:
        try:
            tiles.append(load_tile(path, target_size=target_size))
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning("Skipping unreadable tile %s: %s", path, exc)
    return tiles


def average_rgb(image: Image.Image) -> np.ndarray:
    """Return the mean RGB color of ``image`` as a length-3 float64 array.

    The image is converted to RGB first if needed, so callers can pass any
    PIL image safely. Computing the mean directly with ``axis=(0, 1)`` is
    both clearer and slightly faster than reshaping into a 2D array.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float64)
    # arr.shape == (H, W, 3); collapse spatial axes.
    return arr.mean(axis=(0, 1))


def average_rgb_batch(images: Iterable[Image.Image]) -> np.ndarray:
    """Stack ``average_rgb`` results into an ``(N, 3)`` float64 array."""
    means = [average_rgb(img) for img in images]
    if not means:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(means, axis=0)

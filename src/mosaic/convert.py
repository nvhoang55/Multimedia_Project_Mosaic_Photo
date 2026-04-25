"""Re-encode images in a directory to JPEG.

The original ``Convert_Images_To_JPG.py`` simply renamed files to ``*.jpg``,
which left the underlying bytes in their original format (PNG, WEBP, etc.)
This module decodes each image and re-encodes it as real JPEG, then removes
the source file. It also avoids the rename-collision bug in the old code by
writing to a temporary name first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from mosaic.tiles import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Default JPEG quality. 90 is visually lossless for photographic content
# while keeping files reasonably small.
DEFAULT_QUALITY: int = 90


def _unique_destination(directory: Path, stem: str, suffix: str = ".jpg") -> Path:
    """Return a path inside ``directory`` whose name does not yet exist.

    Tries ``{stem}{suffix}`` first, then ``{stem}_1{suffix}``, ``{stem}_2{suffix}``,
    and so on. This protects against collisions when multiple source files
    would normalize to the same target name (e.g. ``photo.png`` and
    ``photo.PNG``).
    """
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def convert_directory_to_jpg(
    directory: str | os.PathLike[str],
    quality: int = DEFAULT_QUALITY,
    rename_sequentially: bool = False,
) -> list[Path]:
    """Re-encode every supported image in ``directory`` as a JPEG file.

    Parameters
    ----------
    directory:
        Path to a directory containing image files. Subdirectories are
        not traversed.
    quality:
        JPEG encoder quality (1-95 is the useful range for Pillow).
    rename_sequentially:
        If ``True``, output files are named ``0.jpg``, ``1.jpg``, ... in
        the order they are processed. If ``False`` (default), the source
        stem is preserved and only the extension changes. Sequential mode
        mirrors the behavior of the original ``Convert_Images_To_JPG.py``
        for callers that depend on it.

    Returns
    -------
    list[Path]
        Paths of the JPEG files that were successfully written.

    Notes
    -----
    - Images that are already valid JPEGs with a ``.jpg`` extension are
      left untouched.
    - Images with a transparent alpha channel are flattened onto a white
      background, since JPEG does not support transparency.
    - Files that cannot be decoded are skipped with a warning; they are
      not deleted, so the user can inspect them.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    if not 1 <= quality <= 95:
        raise ValueError(f"quality must be in [1, 95]; got {quality}")

    # Snapshot the listing first so files we create during iteration don't
    # get re-processed in the same run.
    sources = sorted(p for p in dir_path.iterdir() if p.is_file() and not p.name.startswith("."))

    written: list[Path] = []
    for index, source in enumerate(sources):
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping non-image file: %s", source)
            continue

        try:
            with Image.open(source) as src:
                src.load()
                image = _flatten_for_jpeg(src)
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning("Skipping unreadable image %s: %s", source, exc)
            continue

        stem = str(index) if rename_sequentially else source.stem
        already_jpeg = source.suffix.lower() in {".jpg", ".jpeg"} and not rename_sequentially

        if already_jpeg:
            # Source is already a JPEG with the right extension and we are
            # not renumbering; leave it alone.
            logger.debug("Already JPEG, leaving in place: %s", source)
            written.append(source)
            continue

        destination = _unique_destination(dir_path, stem, suffix=".jpg")

        # Write to a temp path first so a crash mid-encode cannot leave a
        # half-written file at the final destination.
        tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
        try:
            image.save(tmp_destination, format="JPEG", quality=quality, optimize=True)
            os.replace(tmp_destination, destination)
        except OSError as exc:
            logger.error("Failed to write %s: %s", destination, exc)
            if tmp_destination.exists():
                tmp_destination.unlink(missing_ok=True)
            continue

        # Only delete the source after the new file is safely on disk and
        # is not the same path we just wrote (case-insensitive filesystems
        # can make these identical).
        try:
            if source.resolve() != destination.resolve():
                source.unlink()
        except OSError as exc:
            logger.warning("Wrote %s but could not remove source %s: %s", destination, source, exc)

        written.append(destination)
        logger.info("Converted %s -> %s", source.name, destination.name)

    return written


def _flatten_for_jpeg(image: Image.Image) -> Image.Image:
    """Return a JPEG-safe RGB copy of ``image``.

    Images with an alpha channel are composited onto a white background;
    palette images and other modes are converted directly to RGB.
    """
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        # Composite onto white; matches what most viewers show for transparent PNGs.
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image

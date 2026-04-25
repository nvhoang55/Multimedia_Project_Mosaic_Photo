"""Command-line interface for the mosaic photo generator.

Two subcommands are exposed:

- ``mosaic build`` — generate a mosaic image from a target + tile folder.
- ``mosaic convert`` — re-encode a folder of images as JPEG (the rewrite
  of the old ``Convert_Images_To_JPG.py``).

Run ``mosaic --help`` after ``uv sync`` for the full list of options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mosaic import __version__
from mosaic.convert import DEFAULT_QUALITY, convert_directory_to_jpg
from mosaic.core import MosaicConfig, generate_mosaic_photo

logger = logging.getLogger("mosaic")


def _positive_int(value: str) -> int:
    """argparse type: parse a strictly-positive integer."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer; got {value!r}") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer; got {ivalue}")
    return ivalue


def _positive_float(value: str) -> float:
    """argparse type: parse a strictly-positive float."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number; got {value!r}") from exc
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number; got {fvalue}")
    return fvalue


def _quality(value: str) -> int:
    """argparse type: JPEG quality in [1, 95]."""
    ivalue = _positive_int(value)
    if not 1 <= ivalue <= 95:
        raise argparse.ArgumentTypeError(f"quality must be in [1, 95]; got {ivalue}")
    return ivalue


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mosaic",
        description="Photo mosaic generator: tile a target image with closest-match thumbnails.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show warnings and errors.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ---------- build ----------
    build = subparsers.add_parser(
        "build",
        help="Generate a mosaic image.",
        description="Generate a mosaic image from a target photo and a folder of tile images.",
    )
    build.add_argument(
        "target_image",
        type=Path,
        help="Path to the source image that will be turned into a mosaic.",
    )
    build.add_argument(
        "tiles_path",
        type=Path,
        help="Directory containing tile images (jpg/jpeg/png/webp/...).",
    )
    build.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        default=Path("Result.jpg"),
        help="Output path for the generated mosaic (default: %(default)s).",
    )
    build.add_argument(
        "--grid",
        nargs=2,
        type=_positive_int,
        metavar=("ROWS", "COLS"),
        default=[150, 150],
        help="Grid size as ROWS COLS (default: %(default)s).",
    )
    build.add_argument(
        "--scale",
        type=_positive_float,
        default=3.0,
        help="Scale factor applied to the target before splitting (default: %(default)s).",
    )
    build.add_argument(
        "--color-mode",
        choices=("RGB", "L"),
        default="RGB",
        help="'RGB' for color, 'L' for grayscale (default: %(default)s).",
    )
    build.add_argument(
        "--color-space",
        choices=("rgb", "lab"),
        default="rgb",
        help=(
            "Color space used for matching: 'rgb' is fast, "
            "'lab' is perceptually closer (default: %(default)s)."
        ),
    )
    build.add_argument(
        "--no-duplicates",
        dest="duplicated_tile",
        action="store_false",
        help=("Use each tile at most once. Requires at least ROWS*COLS tiles in the tiles folder."),
    )
    build.add_argument(
        "--jpeg-quality",
        type=_quality,
        default=90,
        help="JPEG encoder quality in [1, 95] (default: %(default)s).",
    )
    build.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible runs.",
    )

    # ---------- convert ----------
    convert = subparsers.add_parser(
        "convert",
        help="Re-encode a folder of images as JPEG.",
        description=(
            "Decode every supported image in DIRECTORY and re-encode it as JPEG. "
            "Unlike the historical script that only renamed extensions, this "
            "command writes real JPEG bytes."
        ),
    )
    convert.add_argument(
        "directory",
        type=Path,
        help="Directory containing image files to re-encode.",
    )
    convert.add_argument(
        "--quality",
        type=_quality,
        default=DEFAULT_QUALITY,
        help="JPEG encoder quality in [1, 95] (default: %(default)s).",
    )
    convert.add_argument(
        "--rename-sequentially",
        action="store_true",
        help=(
            "Rename outputs to 0.jpg, 1.jpg, ... in processing order, "
            "matching the legacy Convert_Images_To_JPG.py behavior."
        ),
    )

    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Set up root logging based on the -v / -q flags."""
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _cmd_build(args: argparse.Namespace) -> int:
    """Run the ``build`` subcommand. Returns a process exit code."""
    rows, cols = args.grid
    config = MosaicConfig(
        target_image=args.target_image,
        tiles_path=args.tiles_path,
        grid_size=(rows, cols),
        output_filename=args.output,
        scale=args.scale,
        duplicated_tile=args.duplicated_tile,
        color_mode=args.color_mode,
        color_space=args.color_space,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
    )
    output_path = generate_mosaic_photo(config)
    logger.info("Done. Wrote %s", output_path)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Run the ``convert`` subcommand. Returns a process exit code."""
    written = convert_directory_to_jpg(
        args.directory,
        quality=args.quality,
        rename_sequentially=args.rename_sequentially,
    )
    logger.info("Converted %d file(s) in %s", len(written), args.directory)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the ``mosaic`` console script.

    Returns an integer suitable for ``sys.exit``. Argument parsing errors
    bubble up as ``SystemExit(2)`` from argparse itself; everything else
    is caught and turned into a friendly stderr message + exit code 1.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    handlers = {
        "build": _cmd_build,
        "convert": _cmd_convert,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse already enforces this
        parser.error(f"Unknown command: {args.command!r}")

    try:
        return handler(args)
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

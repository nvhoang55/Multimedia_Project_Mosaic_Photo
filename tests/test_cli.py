"""Smoke tests for the ``mosaic`` command-line interface.

These tests exercise the argparse plumbing and the dispatch glue in
``mosaic.cli`` end-to-end. They deliberately avoid mocking: the build
subcommand really runs ``generate_mosaic_photo`` against a tiny on-disk
fixture, and the convert subcommand really re-encodes a PNG. Each test
keeps inputs small (a 4x4 grid, a handful of solid-color tiles) so the
suite stays fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mosaic.cli import main


def _write_solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Path:
    """Write a solid-color RGB image to ``path`` in PNG format."""
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_main_without_args_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    # argparse exits with status 2 when a required subcommand is missing.
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "COMMAND" in err or "command" in err


def test_main_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    # --version is handled by argparse and exits with status 0.
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    # Version line is "mosaic X.Y.Z".
    assert out.startswith("mosaic ")


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    # Top-level help should mention both subcommands.
    assert "build" in out
    assert "convert" in out


def test_main_unknown_subcommand_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["totally-not-a-command"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Argument validation (argparse types)
# ---------------------------------------------------------------------------


def test_main_build_rejects_non_positive_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    target = _write_solid(tmp_path / "t.png", (0, 0, 0))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "build",
                str(target),
                str(tiles_dir),
                "--grid",
                "0",
                "5",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "positive integer" in err


def test_main_build_rejects_non_positive_scale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    target = _write_solid(tmp_path / "t.png", (0, 0, 0))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "build",
                str(target),
                str(tiles_dir),
                "--scale",
                "0",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "positive number" in err


def test_main_build_rejects_bad_jpeg_quality(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    target = _write_solid(tmp_path / "t.png", (0, 0, 0))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "build",
                str(target),
                str(tiles_dir),
                "--jpeg-quality",
                "200",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "quality" in err


def test_main_build_rejects_invalid_color_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    target = _write_solid(tmp_path / "t.png", (0, 0, 0))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "build",
                str(target),
                str(tiles_dir),
                "--color-mode",
                "HSV",
            ]
        )
    assert excinfo.value.code == 2


def test_main_build_rejects_invalid_color_space(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    target = _write_solid(tmp_path / "t.png", (0, 0, 0))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "build",
                str(target),
                str(tiles_dir),
                "--color-space",
                "yuv",
            ]
        )
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# build subcommand
# ---------------------------------------------------------------------------


def test_main_build_produces_output_file(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))
    _write_solid(tiles_dir / "blue.png", (0, 0, 255))

    target = _write_solid(tmp_path / "target.png", (200, 0, 0), size=(20, 20))
    output = tmp_path / "out.jpg"

    rc = main(
        [
            "build",
            str(target),
            str(tiles_dir),
            "--output",
            str(output),
            "--grid",
            "2",
            "2",
            "--scale",
            "1",
            "--seed",
            "0",
        ]
    )

    assert rc == 0
    assert output.is_file()
    with Image.open(output) as img:
        img.load()
        assert img.format == "JPEG"
        # 2 rows * 2 cols cells, target was 20x20 -> output is 20x20.
        assert img.size == (20, 20)


def test_main_build_no_duplicates_flag(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    # Need 4 unique tiles for a 2x2 grid with --no-duplicates.
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        _write_solid(tiles_dir / f"t{i}.png", color)

    target = _write_solid(tmp_path / "target.png", (128, 128, 128), size=(20, 20))
    output = tmp_path / "no_dup.jpg"

    rc = main(
        [
            "build",
            str(target),
            str(tiles_dir),
            "--output",
            str(output),
            "--grid",
            "2",
            "2",
            "--scale",
            "1",
            "--no-duplicates",
            "--seed",
            "0",
        ]
    )

    assert rc == 0
    assert output.is_file()


def test_main_build_lab_color_space(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))
    _write_solid(tiles_dir / "blue.png", (0, 0, 255))

    target = _write_solid(tmp_path / "target.png", (250, 5, 5), size=(10, 10))
    output = tmp_path / "lab.jpg"

    rc = main(
        [
            "build",
            str(target),
            str(tiles_dir),
            "--output",
            str(output),
            "--grid",
            "2",
            "2",
            "--scale",
            "1",
            "--color-space",
            "lab",
            "--seed",
            "0",
        ]
    )

    assert rc == 0
    assert output.is_file()


def test_main_build_grayscale_output(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "dark.png", (50, 50, 50))
    _write_solid(tiles_dir / "light.png", (200, 200, 200))

    target = _write_solid(tmp_path / "target.png", (128, 128, 128), size=(10, 10))
    output = tmp_path / "gray.jpg"

    rc = main(
        [
            "build",
            str(target),
            str(tiles_dir),
            "--output",
            str(output),
            "--grid",
            "2",
            "2",
            "--scale",
            "1",
            "--color-mode",
            "L",
            "--seed",
            "0",
        ]
    )

    assert rc == 0
    with Image.open(output) as img:
        img.load()
        # JPEG saved from an "L" image stays as a single-channel JPEG.
        assert img.mode == "L"


def test_main_build_missing_target_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))

    rc = main(
        [
            "build",
            str(tmp_path / "does_not_exist.png"),
            str(tiles_dir),
            "--grid",
            "2",
            "2",
        ]
    )

    # FileNotFoundError is caught by main() and turned into exit code 1.
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "not found" in err or "no such" in err


def test_main_build_empty_tiles_dir_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    target = _write_solid(tmp_path / "target.png", (0, 0, 0), size=(10, 10))

    rc = main(
        [
            "build",
            str(target),
            str(tiles_dir),
            "--grid",
            "2",
            "2",
            "--scale",
            "1",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "no supported image" in err


def test_main_build_quiet_sets_warning_log_level(tmp_path: Path) -> None:
    """``--quiet`` should raise the root logger to WARNING.

    We don't try to assert on captured stderr text here, because pytest's
    log capturing intercepts records before they reach the stream handler
    that ``logging.basicConfig`` installed. Checking the effective level
    on the root logger is both more precise and more robust.
    """
    import logging

    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_solid(tiles_dir / "red.png", (255, 0, 0))

    target = _write_solid(tmp_path / "target.png", (200, 0, 0), size=(10, 10))
    output = tmp_path / "quiet.jpg"

    rc = main(
        [
            "--quiet",
            "build",
            str(target),
            str(tiles_dir),
            "--output",
            str(output),
            "--grid",
            "2",
            "2",
            "--scale",
            "1",
            "--seed",
            "0",
        ]
    )

    assert rc == 0
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING


# ---------------------------------------------------------------------------
# convert subcommand
# ---------------------------------------------------------------------------


def test_main_convert_re_encodes_png_to_jpeg(tmp_path: Path) -> None:
    src = _write_solid(tmp_path / "photo.png", (10, 20, 30))

    rc = main(["convert", str(tmp_path)])

    assert rc == 0
    # Source PNG is gone; a real JPEG with the same stem replaces it.
    assert not src.exists()
    out = tmp_path / "photo.jpg"
    assert out.is_file()
    with Image.open(out) as img:
        img.load()
        assert img.format == "JPEG"


def test_main_convert_sequential_renaming(tmp_path: Path) -> None:
    _write_solid(tmp_path / "a.png", (1, 1, 1))
    _write_solid(tmp_path / "b.png", (2, 2, 2))
    _write_solid(tmp_path / "c.png", (3, 3, 3))

    rc = main(["convert", str(tmp_path), "--rename-sequentially"])

    assert rc == 0
    for name in ["0.jpg", "1.jpg", "2.jpg"]:
        assert (tmp_path / name).is_file()


def test_main_convert_rejects_missing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["convert", str(tmp_path / "nope")])

    # NotADirectoryError is caught by main() -> exit code 1.
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "not a directory" in err


def test_main_convert_rejects_bad_quality(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["convert", str(tmp_path), "--quality", "999"])
    # argparse rejects via its type= validator before main() runs.
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "quality" in err

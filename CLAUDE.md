# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Photo mosaic generator: computes the average RGB of each tile image in a folder, splits the target image into a grid, and fills each cell with the closest-matching tile. Matching is fully vectorized with numpy and supports either squared-Euclidean RGB distance or perceptual ΔE76 in CIE Lab. Dependencies (`numpy`, `Pillow`) and the Python version are managed by [`uv`](https://docs.astral.sh/uv/) via `pyproject.toml`, `uv.lock`, and `.python-version`.

## Setup

Install `uv` (see https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```bash
uv sync
```

This fetches the pinned Python interpreter if needed, creates `.venv/`, installs the locked versions of `numpy` and `Pillow`, and installs this project in editable mode (which exposes the `mosaic` console script).

A project-local `uv.toml` pins the package index to public PyPI so resolution works regardless of any user-level config (e.g. a private CodeArtifact mirror with expired credentials).

## Commands

Generate a mosaic:

```bash
uv run mosaic build <target_image> <tiles_directory> \
    --output Result.jpg \
    --grid 150 150 \
    --scale 5
```

Re-encode a folder of images as real JPEGs (decode + re-encode, not just rename):

```bash
uv run mosaic convert <directory> [--quality 90] [--rename-sequentially]
```

Run `uv run mosaic --help` (and `... build --help` / `... convert --help`) for the full list of options.

Run the test suite:

```bash
uv run pytest
```

Lint and format:

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run ruff format --check src tests   # CI-style check, no writes
```

Dependency management:

- Add a runtime dependency: `uv add <package>`
- Add a dev-only dependency: `uv add --dev <package>`
- Remove a dependency: `uv remove <package>`
- Refresh the lock file: `uv lock`
- Re-sync the environment to match the lock file: `uv sync`

## Architecture

The package lives under `src/mosaic/`. Public API is exported from `mosaic/__init__.py`:

- `MosaicConfig` — frozen dataclass holding all generation parameters; validates in `__post_init__`.
- `generate_mosaic_photo(config)` — full disk-I/O entry point.

Internally, the pipeline is split across focused modules:

1. `mosaic.tiles` — discover image files (`iter_image_paths`), load them safely with optional thumbnail-during-decode for JPEGs (`load_tile`/`load_tiles`), and compute mean RGB (`average_rgb`/`average_rgb_batch`).
2. `mosaic.compose` — `compute_cell_size`, `fit_target_to_grid`, `split_image`, `paste_tiles_into_grid`. All use the convention `grid_size = (rows, cols)` and `cell_size = (width, height)` consistently to avoid the historical width/height swap.
3. `mosaic.matching` — `best_match_indices` (vectorized broadcast over the full `(C, T)` distance matrix) and `best_match_indices_unique` (greedy no-duplicates variant). Both accept `color_space="rgb"` or `"lab"`; Lab conversion is implemented in numpy via sRGB → linear RGB → XYZ (D65) → Lab using only `_RGB_TO_XYZ_D65` and `_D65_WHITE` constants in this module.
4. `mosaic.core` — `MosaicConfig` and the orchestration: `generate_mosaic_photo` (opens the file, calls `build_mosaic`, writes the output) and `build_mosaic` (in-memory, takes a PIL image; this is what tests target).
5. `mosaic.convert` — `convert_directory_to_jpg` for the `mosaic convert` subcommand. Decodes + re-encodes (writes to a `.tmp` first, then `os.replace` for atomicity) and flattens RGBA onto white because JPEG has no alpha.
6. `mosaic.cli` — argparse plumbing with `build` and `convert` subcommands; `main()` is the entry point declared in `[project.scripts]`.

The pipeline (`build_mosaic`) is:

1. Resize the target *directly* to the final mosaic-grid dimensions (`cols * cell_w, rows * cell_h`) — not through a separate `scale`-times intermediate, which keeps peak memory low.
2. `split_image` slices the fitted target into row-major cells.
3. `iter_image_paths` + `load_tiles` discover and decode tiles, thumbnailing to `cell_size` during decode.
4. The tile list is shuffled with a seeded `random.Random(config.seed)` so ties are broken non-deterministically by default but reproducibly when a seed is supplied.
5. `average_rgb_batch` reduces every tile and every cell to a `(N, 3)` mean color array.
6. `best_match_indices` (or `_unique` if `duplicated_tile=False`) broadcasts to a `(C, T)` distance matrix and `argmin`s along axis 1.
7. `paste_tiles_into_grid` resizes each matched tile to `cell_size` and pastes it into a fresh canvas. Resizing here means tiles with mismatched aspect ratios never produce gaps.
8. Optional `convert(color_mode)` for grayscale output; then write to disk with format inferred from the output extension.

## Conventions

- `grid_size` is always `(rows, cols)` (numpy-style). `cell_size` is always `(width, height)` (PIL-style). Both are documented in the docstrings of `mosaic.compose`.
- Public functions take and return numpy arrays where it makes sense (`average_rgb*`, matching). PIL `Image` objects are reserved for I/O and pasting.
- Errors are raised as standard exceptions (`FileNotFoundError`, `NotADirectoryError`, `ValueError`, `RuntimeError`); the CLI catches them in `main()` and turns them into exit code 1. Argument-parsing errors come from argparse with exit code 2.
- Logging goes through `logging.getLogger(__name__)` per module. The CLI calls `logging.basicConfig(force=True)` so it works under pytest too. Use `-v`/`-q` to change verbosity.

## Layout notes

- `src/mosaic/` — the package; this is what gets installed.
- `tests/` — pytest suite covering tiles, matching, compose, core (end-to-end), convert, and CLI. ~100 tests; runs in well under a second.
- `Assets/` — sample target (`Tree.jpg`) and tile folder (`Trees/`). Gitignored; not shipped.
- `pyproject.toml` — project metadata, runtime deps, `[dependency-groups].dev`, `[project.scripts]`, ruff config, pytest config.
- `uv.lock` — locked dependency versions; commit it.
- `uv.toml` — project-local uv config that pins the index to public PyPI.
- `.python-version` — pinned interpreter (currently 3.12).
- `.venv/` — local virtual environment created by `uv sync` (gitignored).
- `.github/workflows/ci.yml` — runs ruff + pytest on Python 3.10, 3.11, and 3.12 on every push and PR against `master`/`main`.

## Adding features

- New CLI flags: extend the relevant subparser in `mosaic.cli._build_parser`, plumb the value through into `MosaicConfig`, and add a smoke test in `tests/test_cli.py` plus a unit test on the underlying behavior.
- New color spaces: add a branch in `mosaic.matching._to_color_space` and a `Literal` member to `ColorSpace`. Add tests in `tests/test_matching.py`.
- New file formats: `mosaic.tiles.SUPPORTED_EXTENSIONS` is the single source of truth for what the tile loader and the convert subcommand will consider an image.
- Always run `uv run ruff check src tests`, `uv run ruff format src tests`, and `uv run pytest` before committing.
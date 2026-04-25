# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Photo mosaic generator: computes the average RGB of each tile image in a folder, splits the target image into a grid, and fills each cell with the closest-matching tile. Dependencies (`numpy`, `Pillow`) and the Python version are managed by [`uv`](https://docs.astral.sh/uv/) via `pyproject.toml`, `uv.lock`, and `.python-version`.

## Setup

Install `uv` (see https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```bash
uv sync
```

This fetches the pinned Python interpreter if needed, creates `.venv/`, and installs the locked versions of `numpy` and `Pillow`.

## Commands

Run the mosaic generator (must be executed from `scripts/` because paths are relative to that CWD):

```bash
cd scripts
uv run python Mosaic_Photo_Generator.py
```

Normalize tile filenames/extensions to `.jpg` before generating:

```bash
cd scripts
uv run python Convert_Images_To_JPG.py
```

Both scripts are parameter-less entry points — edit the `if __name__ == '__main__':` block at the bottom of each file to change the target image, tile folder, or grid/scale settings. There is no test suite, lint config, or build step.

Dependency management:

- Add a dependency: `uv add <package>`
- Remove a dependency: `uv remove <package>`
- Refresh the lock file: `uv lock`
- Re-sync the environment to match the lock file: `uv sync`

## Architecture

`scripts/Mosaic_Photo_Generator.py` is the whole pipeline. `generate_mosaic_photo` is the top-level entry; internally the stages are:

1. `split_image` — slice the (upscaled) target into a `grid_size` lattice of cells.
2. `get_tiles_from` + `resize_all_tiles` — load every tile in `tiles_path`, convert to RGB, thumbnail each to the cell size.
3. `get_average_rgb` — reduce each tile and each target cell to a mean `(R,G,B)` via numpy reshape + mean.
4. `get_best_match_index` — for each cell, linear-scan all tile averages and pick the minimum squared Euclidean distance in RGB space. This is the hot loop and is O(cells × tiles).
5. `create_image_grid` — paste matched tiles into a new blank `Image.new('RGB', ...)` sized by the max tile dim × grid dim.

`scale` in `generate_mosaic_photo` multiplies the target's resolution *before* splitting; combined with `grid_size`, it sets the final mosaic resolution. `duplicated_tile=False` requires `grid_h * grid_w <= len(tiles)`.

### Known issues to be aware of when editing

- `create_mosaic_photo` uses `count % batch_size is 0` (identity check on ints) — should be `==`.
- In the `duplicated_tile=False` branch, `tiles.remove(match_index)` passes an int to `list.remove` which expects the element; the removal is also not reflected in `all_tile_average_rgb`, so indices desync. This path is effectively broken.
- `get_tiles_from` will raise on any non-image file in `tiles_path`; the README notes the directory must contain only `.jpg`/`.jpeg`.

## Layout notes

- `scripts/` — the two Python entry points and the generated `Result.jpg` (gitignored).
- `Assets/` — sample target (`Tree.jpg`) and tile folder (`Trees/`). Gitignored; not shipped.
- `pyproject.toml` / `uv.lock` / `.python-version` — `uv` project metadata, locked dependencies, and pinned Python version.
- `.venv/` — local virtual environment created by `uv sync` (gitignored).
- The `__main__` examples in both scripts reference `../data/Face.jpg` and `../data/Face/`, which don't exist in this repo. Use `../Assets/Tree.jpg` and `../Assets/Trees/` (or your own paths) when running locally.
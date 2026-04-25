# Mosaic Photo Generator

This program generates a mosaic photo from a folder of tile images.

It computes the average RGB color of each tile, splits the target image into a grid, and fills each cell with the closest-matching tile (optionally in CIE Lab color space for perceptually-better matches). Matching is fully vectorized with numpy, so even a 150×150 grid against a few hundred tiles finishes in a couple of seconds.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management. The required Python version is pinned in `.python-version` and the runtime dependencies (`numpy`, `Pillow`) are declared in `pyproject.toml` and locked in `uv.lock`.

1. Install `uv` (see the [official docs](https://docs.astral.sh/uv/getting-started/installation/)).

   On macOS / Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   On Windows (PowerShell):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. From the repository root, create the virtual environment and install dependencies:

   ```bash
   uv sync
   ```

   `uv` will fetch the pinned Python interpreter (if needed), create `.venv/`, and install the locked versions of `numpy` and `Pillow`. The project is also installed in editable mode, which exposes the `mosaic` console script.

## Usage

The package exposes a single CLI entry point, `mosaic`, with two subcommands.

### Generate a mosaic

```bash
uv run mosaic build <target_image> <tiles_directory> [options]
```

Example:

```bash
uv run mosaic build Assets/Tree.jpg Assets/Trees/ \
    --output Result.jpg \
    --grid 150 150 \
    --scale 5 \
    --color-mode RGB \
    --color-space rgb
```

Options for `build`:

| Flag | Default | Description |
| --- | --- | --- |
| `--output PATH` | `Result.jpg` | Output path. Format inferred from extension (`.jpg` or `.png`). |
| `--grid ROWS COLS` | `150 150` | Number of cells vertically (rows) and horizontally (cols). |
| `--scale FLOAT` | `3.0` | Multiplier applied to the target's resolution before splitting. Larger = more pixels per cell. |
| `--color-mode {RGB,L}` | `RGB` | `RGB` for color, `L` for grayscale. |
| `--color-space {rgb,lab}` | `rgb` | `rgb` is fast; `lab` is perceptually closer (CIE Lab / ΔE76). |
| `--no-duplicates` | (off) | Use each tile at most once. Requires at least `ROWS*COLS` tiles. |
| `--jpeg-quality INT` | `90` | JPEG encoder quality (1-95) when output is JPEG. |
| `--seed INT` | (random) | RNG seed for reproducible runs. |
| `-v` / `-q` | | Verbose (DEBUG) or quiet (WARNING+) logging. |

Run `uv run mosaic build --help` for the full list.

### Re-encode a folder of images as JPEG

The tile folder must contain only images PIL can decode. The `convert` subcommand decodes each file and re-encodes it as a real JPEG (unlike the legacy script, which only renamed extensions):

```bash
uv run mosaic convert <directory> [--quality 90] [--rename-sequentially]
```

- `--quality`: JPEG encoder quality (1-95, default 90).
- `--rename-sequentially`: Rename outputs to `0.jpg`, `1.jpg`, ... in processing order, matching the legacy behavior.

Files that aren't recognized as images are left in place; transparent PNGs are flattened onto a white background; existing JPEGs are left untouched unless `--rename-sequentially` is set.

## Library API

If you'd rather call the generator from Python, the public API is two symbols:

```python
from mosaic import MosaicConfig, generate_mosaic_photo

config = MosaicConfig(
    target_image="Assets/Tree.jpg",
    tiles_path="Assets/Trees/",
    grid_size=(150, 150),    # (rows, cols)
    output_filename="Result.jpg",
    scale=5.0,
    duplicated_tile=True,
    color_mode="RGB",        # "RGB" | "L"
    color_space="rgb",       # "rgb" | "lab"
    seed=0,                  # optional, makes runs reproducible
)
generate_mosaic_photo(config)
```

`MosaicConfig` is a frozen dataclass and validates all parameters in `__post_init__`.

## Development

Set up dev dependencies (`pytest`, `ruff`):

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Run the linter / formatter:

```bash
uv run ruff check src tests
uv run ruff format src tests
```

CI runs both on every push and pull request against `master`/`main` (see `.github/workflows/ci.yml`).

### Managing dependencies

- Add a runtime dependency: `uv add <package>`
- Add a dev-only dependency: `uv add --dev <package>`
- Remove a dependency: `uv remove <package>`
- Refresh the lock file: `uv lock`
- Re-sync the environment to match the lock file: `uv sync`

## Project layout

```
.
├── pyproject.toml          # project metadata, deps, ruff/pytest config
├── uv.lock                 # locked dependency versions
├── uv.toml                 # project-local uv config (pins index to PyPI)
├── .python-version         # pinned interpreter version
├── src/
│   └── mosaic/
│       ├── __init__.py     # exports MosaicConfig, generate_mosaic_photo
│       ├── cli.py          # `mosaic` console-script entry point
│       ├── core.py         # MosaicConfig + top-level pipeline
│       ├── tiles.py        # tile loading and average-RGB
│       ├── matching.py     # vectorized matching (RGB / Lab)
│       ├── compose.py      # split target / paste tiles back into a grid
│       └── convert.py      # JPEG re-encoding (`mosaic convert`)
├── tests/                  # pytest suite (~100 tests)
└── Assets/                 # sample target/tiles (gitignored, not shipped)
```

## Result

Click on the image and zoom in to check out the individual tile images.

![demo mosaic](demo.jpg)
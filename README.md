# Mosaic Photo Generator

This program generates a mosaic photo based on a bunch of tile images (from a specific folder).

Basically, it calculates the average RGB color of each tile and compares it to the average RGB of each section of the original image. It then puts all the matched tiles in a list and builds a new photo based on those matches.

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

   `uv` will fetch the pinned Python interpreter (if needed), create `.venv/`, and install the locked versions of `numpy` and `Pillow`.

## Usage

1. Make sure the folder for tile images only contains image files and that their extensions are `.jpg` or `.jpeg`. You can convert any other extensions to `.jpg` by running `Convert_Images_To_JPG.py`.

   ```python
   if __name__ == '__main__':
       convert_tile_extension_to_jpg(directory_path='../Assets/Trees/')
   ```

   Run it with:

   ```bash
   cd scripts
   uv run python Convert_Images_To_JPG.py
   ```

2. Adjust the output mosaic photo by configuring these parameters in the `generate_mosaic_photo()` function in main.

   ```python
   if __name__ == '__main__':
       generate_mosaic_photo(target_image='../Assets/Tree.jpg',
                             tiles_path='../Assets/Trees/',
                             output_filename='Result.jpg',
                             scale=5,
                             grid_size=(150, 150),
                             duplicated_tile=True,
                             color_mode='RGB')
   ```

   Breakdown:
   - `target_image`: path to the image you want to turn into a mosaic.
   - `tiles_path`: path to the folder of tile images.
   - `output_filename`: path for the generated mosaic photo.
   - `scale`: zoom level of the image after it's converted.
   - `grid_size`: determines how many times you want to split the original image.
   - `duplicated_tile`: whether the program can use a tile image multiple times or not (if set to false, make sure you have enough unique tiles in the folder to fill the `grid_size`).
   - `color_mode`: `'RGB'` for color, `'L'` for grayscale.

3. Run `Mosaic_Photo_Generator.py` (paths in the script are relative to `scripts/`, so run it from there):

   ```bash
   cd scripts
   uv run python Mosaic_Photo_Generator.py
   ```

## Managing dependencies

- Add a dependency: `uv add <package>`
- Remove a dependency: `uv remove <package>`
- Refresh the lock file: `uv lock`
- Re-sync the environment to match the lock file: `uv sync`

## Result
Click on the image and zoom in to check out the individual tile images.

![alt text](https://github.com/Charlotte-Miller/Multimedia_Project_Mosaic_Photo/blob/master/scripts/Result.jpg?raw=true)
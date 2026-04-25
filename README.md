# Mosaic Photo Generator

This program generates a mosaic photo based on a bunch of tile images (from a specific folder).

Basically, it calculates the average RGB color of each tile and compares it to the average RGB of each section of the original image. It then puts all the matched tiles in a list and builds a new photo based on those matches.

## Installation
Python version: 3.8.7

PIP version: 21.0

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install these packages.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy
python3 -m pip install --upgrade Pillow
```

## Usage
1. Make sure the folder for tile images only contains image files and that their extensions are .jpg or .jpeg. You can convert any other extensions to .jpg by running `Convert_Images_To_JPG.py`.

```python
if __name__ == '__main__':
    convert_tile_extension_to_jpg(directory_path='../data/Face/')
```

2. Adjust the output mosaic photo by configuring these parameters in the `generate_mosaic_photo()` function in main.

```python
if __name__ == '__main__':
    generate_mosaic_photo(target_image='../data/Face.jpg',
                          tiles_path='../data/Face/',
                          output_filename='Result.jpg',
                          scale=5,
                          grid_size=(150, 150),
                          duplicated_tile=True,
                          color_mode='RGB')
```
Breakdown:
- target_image: path to the image you want to turn into a mosaic.
- tiles_path: path to the folder of tile images.
- output_filename: path for the generated mosaic photo.
- scale: zoom level of the image after it's converted.
- grid_size: determines how many times you want to split the original image.
- duplicated_tile: whether the program can use a tile image multiple times or not (if set to false, make sure you have enough unique tiles in the folder to fill the grid_size).
- color_mode: 'RGB' for color, 'L' for grayscale.

3. Run `Mosaic_Photo_Generator.py`

## Result
Click on the image and zoom in to check out the individual tile images.

![alt text](https://github.com/Charlotte-Miller/Multimedia_Project_Mosaic_Photo/blob/master/scripts/Result.jpg?raw=true)

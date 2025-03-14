# ImageTileProcessor

This is a in-house package to deal with lazy tiled image tile loading to save RAM and various methods to post process tiled outpus.

## Installation

```sh
poetry install
```
or
```sh
pip install ImageTileProcessor
```
## Usage
There are several command line tools that can be used to process tiled images (and their processed results).
```sh
merge-peaks

merge-polygons

tile-2d-image
```

And if you just want to fetch the raw pixel data from a tile, you can use the following function:
```python
from imagetileprocessor import slice_and_crop_image
slice_and_crop_image(
    image_p, x_min, x_max, y_min, y_max, zs, channel, resolution_level
)
```
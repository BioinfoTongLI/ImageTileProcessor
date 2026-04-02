#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

import logging
import os
import tifffile
from aicsimageio import AICSImage
import zarr

logging.basicConfig(level=logging.INFO)


def get_tile_from_tifffile(
    image: str,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    channel: list[int] = [0],
    zplane: list[int] = [0],
    timepoint: list[int] = [0],
    resolution_level: int = 0,
):
    """
    Extract a tile from a TIFF file using zarr backend for lazy loading.

    Args:
        image: Path to the TIFF file.
        xmin: Left boundary of the tile (inclusive).
        xmax: Right boundary of the tile (exclusive).
        ymin: Top boundary of the tile (inclusive).
        ymax: Bottom boundary of the tile (exclusive).
        channel: Channel indices to extract.
        zplane: Z-plane indices to extract.
        timepoint: Timepoint indices to extract.
        resolution_level: Pyramid resolution level (0 = full resolution).

    Returns:
        numpy.ndarray: The extracted tile.
    """
    if not os.path.exists(image):
        raise FileNotFoundError(f"Image file not found: {image}")
    if xmin < 0 or ymin < 0:
        raise ValueError(f"Tile coordinates must be non-negative, got xmin={xmin}, ymin={ymin}")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Min coordinates must be less than max: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

    store = tifffile.imread(image, aszarr=True)
    zgroup = zarr.open(store, mode="r")
    if isinstance(zgroup, zarr.core.Array):
        dimension_order = zgroup.attrs["_ARRAY_DIMENSIONS"]
        if len(zgroup.shape) == 2:
            dimension_order = "YX"
        elif dimension_order == ["Q", "Q", "Q", "Y", "X"]:
            dimension_order = "QQQYX"
        elif dimension_order == ["C", "Y", "X"]:
            dimension_order = "CYX"
        else:
            logging.error(f"Unknown dimension order {zgroup.shape}")
        arr = zgroup
    else:
        arr = zgroup[resolution_level]
        dimension_order = [d[0] for d in arr.attrs["_ARRAY_DIMENSIONS"]]
        dimension_order = "".join(dimension_order)

    # Extract the tile based on the dimension order
    if dimension_order == "YX":
        tile = arr[ymin:ymax, xmin:xmax]
    elif dimension_order in ("YXC", "YXS"):
        tile = arr[ymin:ymax, xmin:xmax, channel]
    elif dimension_order in ("CYX", "SYX"):
        tile = arr[channel, ymin:ymax, xmin:xmax]
    elif dimension_order == "ZYX":
        tile = arr[zplane, ymin:ymax, xmin:xmax]
    elif dimension_order == "ZYXC":
        tile = arr[zplane, ymin:ymax, xmin:xmax, channel]
    elif dimension_order == "YXCZ":
        tile = arr[ymin:ymax, xmin:xmax, channel, zplane]
    elif dimension_order == "XYCZT":
        tile = arr[ymin:ymax, xmin:xmax, channel, zplane, timepoint]
    elif dimension_order == "QQQYX":
        tile = arr[0, channel, 0, ymin:ymax, xmin:xmax]
    else:
        raise ValueError(f"Unsupported dimension order: {dimension_order}")

    logging.debug(f"tile shape {tile.shape}")
    return tile


def slice_and_crop_image(
    image_p: str,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    zs: list[int],
    channel: list[int],
    resolution_level: int,
):
    """
    Extract a tile from any supported image format.

    For TIFF files this uses a zarr-backed lazy load so only the requested
    tile is read from disk.  For other formats the whole Z/C plane is loaded
    first and then cropped (higher memory usage).

    Args:
        image_p: Path to the image file.
        x_min: Left boundary of the tile.
        x_max: Right boundary of the tile.
        y_min: Top boundary of the tile.
        y_max: Bottom boundary of the tile.
        zs: Z-plane indices to extract.
        channel: Channel indices to extract.
        resolution_level: Pyramid resolution level (TIFF only).

    Returns:
        numpy.ndarray: The extracted tile.
    """
    if image_p.endswith(".tif") or image_p.endswith(".tiff"):
        crop = get_tile_from_tifffile(
            image_p,
            x_min,
            x_max,
            y_min,
            y_max,
            zplane=zs,
            channel=channel,
            resolution_level=resolution_level,
        )
    else:
        # Loads the whole plane then crops — higher memory footprint for non-TIFF files
        img = AICSImage(image_p)
        lazy_one_plane = img.get_image_dask_data(
            "ZCYX", T=0, C=channel, Z=zs
        )
        crop = lazy_one_plane[:, :, y_min:y_max, x_min:x_max].compute()
    return crop

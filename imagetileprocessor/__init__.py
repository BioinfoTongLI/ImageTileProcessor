#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

import logging
import tifffile
from aicsimageio import AICSImage
import zarr

logging.basicConfig(level=logging.INFO)

# Function to extract a tile from a TIFF file
def get_tile_from_tifffile(
        image, xmin, xmax, ymin, ymax,
        channel:list[int]=[0],
        zplane:list[int]=[0],
        timepoint:list[int]=[0],
        resolution_level=0):
    store = tifffile.imread(image, aszarr=True)
    zgroup = zarr.open(store, mode="r")
    if isinstance(zgroup, zarr.core.Array):
        dimension_order = zgroup.attrs["_ARRAY_DIMENSIONS"]
        if len(zgroup.shape) == 2:
            dimension_order = "YX"
        elif dimension_order == ["Q", "Q", "Q", "Y", "X"]:
            dimension_order = "QQQYX"
        else:
            logging.error(f"Unknown dimension order {image.shape}")
        image = zgroup
    else:
        image = zgroup[resolution_level]
        dimension_order = [d[0] for d in image.attrs["_ARRAY_DIMENSIONS"]]
        dimension_order = "".join(dimension_order)

    # Extract the tile based on the dimension order
    if dimension_order=="YX":
        tile = image[ymin:ymax, xmin:xmax]
    elif dimension_order=="YXC" or dimension_order=="YXS":
        tile = image[ymin:ymax, xmin:xmax, channel]
    elif dimension_order=="CYX" or dimension_order=="SYX":
        tile = image[channel, ymin:ymax, xmin:xmax]
    elif dimension_order=="ZYX":
        tile = image[zplane, ymin:ymax, xmin:xmax]
    elif dimension_order=="ZYXC":
        tile = image[zplane, ymin:ymax, xmin:xmax, channel]
    elif dimension_order=="YXCZ":
        tile = image[ymin:ymax, xmin:xmax, channel, zplane]
    elif dimension_order=="XYCZT":
        tile = image[ymin:ymax, xmin:xmax, channel, zplane, timepoint]
    elif dimension_order=="QQQYX":
        tile = image[0, channel, 0,  ymin:ymax, xmin:xmax]
    else:
        raise Exception(f"Unknown dimension order {dimension_order}")
    
    logging.debug(f"tile shape {tile.shape}")
    return tile

# Function to slice and crop an image based on the provided parameters
def slice_and_crop_image(image_p, x_min, x_max, y_min, y_max, zs, channel, resolution_level):
    if image_p.endswith(".tif") or image_p.endswith(".tiff"):
        crop = get_tile_from_tifffile(
            image_p, x_min, x_max, y_min, y_max, zplane=zs, channel=channel, resolution_level=resolution_level, 
        )
    else:
        # This will load the whole slice first and then crop it. So, large memory footprint
        img = AICSImage(image_p)
        lazy_one_plane = img.get_image_dask_data(
            "ZCYX",
            T=0, # only one time point is allowed for now
            C=channel,
            Z=zs)
        crop = lazy_one_plane[:, :, y_min:y_max, x_min:x_max].compute()
    return crop
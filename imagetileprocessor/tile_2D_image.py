#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

"""
Slice an image in XY and save the tile coordinates to a CSV file.
"""

import csv

import fire
from aicsimageio import AICSImage


def calculate_slices(
    image_size: tuple[int, int],
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """
    Calculate tile coordinates for the given image size.

    Args:
        image_size: (width, height) of the image in pixels.
        chunk_size: Side length of each square tile in pixels.
        overlap: Number of pixels each tile overlaps its neighbour.

    Returns:
        List of (x_min, y_min, x_max, y_max) tuples, one per tile.

    Raises:
        ValueError: If overlap >= chunk_size (would produce zero or negative step).
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    width, height = image_size
    step = chunk_size - overlap
    slices = []
    for i in range(0, width, step):
        for j in range(0, height, step):
            box = (i, j, min(i + chunk_size, width), min(j + chunk_size, height))
            slices.append(box)
    return slices


def write_slices_to_csv(slices: list[tuple[int, int, int, int]], output_name: str):
    """
    Write tile coordinates to a CSV file.

    Args:
        slices: List of (x_min, y_min, x_max, y_max) tuples.
        output_name: Path for the output CSV file.
    """
    with open(output_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tile", "X_MIN", "Y_MIN", "X_MAX", "Y_MAX"])
        for i, (x1, y1, x2, y2) in enumerate(slices):
            writer.writerow([i + 1, x1, y1, x2, y2])


def main(
    image: str,
    output_name: str,
    overlap: int = 30,
    chunk_size: int = 4096,
    C: int = 0,
    S: int = 0,
    T: int = 0,
):
    """
    Slice an image into tiles and save coordinates to a CSV file.

    Args:
        image: Path to the input image.
        output_name: Path for the output CSV file.
        overlap: Overlap between adjacent tiles in pixels. Defaults to 30.
        chunk_size: Tile side length in pixels. Defaults to 4096.
        C: Channel index. Defaults to 0.
        S: Scene index. Defaults to 0.
        T: Timepoint index. Defaults to 0.
    """
    img = AICSImage(image)
    lazy_one_plane = img.get_image_dask_data("XY", S=S, T=T, C=C)
    slices = calculate_slices(lazy_one_plane.shape, chunk_size, overlap)
    write_slices_to_csv(slices, output_name)


def run():
    options = {
        "run": main,
        "version": "0.2.1",
    }
    fire.Fire(options)


if __name__ == "__main__":
    run()

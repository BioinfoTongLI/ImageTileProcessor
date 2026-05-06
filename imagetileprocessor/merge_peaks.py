#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

"""
A simple script to merge peaks from adjacent & partially overlapping tiles.

This script reads multiple CSV files containing peak coordinates,
merges overlapping peaks, and writes the merged peaks to an output file.

Functions:
    main(*csvs, output_name: str, peak_radius: float = 1.5)
"""

import pandas as pd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union


def main(*csvs: str, output_name: str, peak_radius: float = 1.5):
    """
    Merge peaks from multiple CSV files and write the result to an output file.

    Reads CSV files where the first column is the Y coordinate and the second
    column is the X coordinate.  Points within ``peak_radius`` of each other
    are merged and replaced by their centroid.

    Args:
        *csvs: Paths to the input CSV files.
        output_name: Path for the output WKT file.
        peak_radius: Radius used to buffer points before merging. Default 1.5.
    """
    if not csvs:
        raise ValueError("At least one input CSV file must be provided")

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    points = []
    for coord in df.values:
        points.append(Point(coord[1], coord[0]))

    # Buffer each point and merge overlapping buffers
    buffers = [point.buffer(peak_radius) for point in points]
    merged = unary_union(buffers)

    # Replace each merged region with its centroid.
    # unary_union may return a bare Polygon when all buffers overlap into one.
    regions = merged.geoms if hasattr(merged, "geoms") else [merged]
    peaks = MultiPoint([g.centroid for g in regions])

    with open(output_name, "w") as f:
        f.write(peaks.wkt)


def run():
    import fire

    options = {
        "run": main,
        "version": "0.2.1",
    }
    fire.Fire(options)


if __name__ == "__main__":
    run()

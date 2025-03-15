#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

"""
A simple script to merge peaks from adjacent&partially overlaping tiles.

This script reads multiple CSV files containing peak coordinates,
merges overlapping peaks, and writes the merged peaks to an output file.

Functions:
    main(*csvs, output_name: str, peak_radius: float = 1.5)
    version()
"""

import dask.dataframe as dd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union

# Main function to merge peaks from multiple CSV files
def main(*csvs, output_name: str, peak_radius: float = 1.5):
    """
    Merge peaks from multiple CSV files and write the result to an output file.

    Parameters:
    csvs (str): Paths to the input CSV files.
    output_name (str): Path to the output file.
    peak_radius (float): Radius to use for merging peaks. Default is 1.5.

    Returns:
    None
    """
    df = dd.read_csv(csvs).compute()
    points= []
    for coord in df.values:
        points.append(Point(coord[1], coord[0]))
    # Create a buffer around each point
    buffers = [point.buffer(peak_radius) for point in points]

    # Merge overlapping buffers
    merged = unary_union(buffers)

    # Get the centroid of each merged geometry
    peaks = MultiPoint([g.centroid for g in merged.geoms])

    # Dump the merged multipolygon in WKT format
    with open(output_name, "w") as file:
        file.write(peaks.wkt)

def run():
    import fire
    options = {
        "run" : main,
        "version" : "0.0.3", 
    }
    fire.Fire(options)

if __name__ == "__main__":
    run() 
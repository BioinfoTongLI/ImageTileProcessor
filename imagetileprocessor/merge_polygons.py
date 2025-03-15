#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

import argparse
import logging
import os
import shapely
import shapely.geometry
import shapely.strtree
import shapely.ops
import shapely.wkt
import tqdm
import multiprocessing

# configure logging
logging.basicConfig(level=logging.INFO)

# Function to read polygons from a WKT file
def read_polygons(wkt_file):
    """
    Read polygons from a WKT file.

    :param wkt_file: Path to the WKT file.
    :type wkt_file: str
    :return: List of polygons.
    :rtype: list
    """
    polygons = []
    with open(wkt_file, "rt") as wkt:
        wkt_str = wkt.read().strip()
        if wkt_str:
            for polygon in shapely.wkt.loads(wkt_str).geoms:
                polygons.append(polygon.buffer(0))
    return polygons

# Function to merge overlapping polygons
def merge_overlapping_polygons(polygons):
    """
    Merge overlapping polygons.

    :param polygons: List of polygons.
    :type polygons: list
    :return: List of merged polygons.
    :rtype: list
    """
    tree = shapely.strtree.STRtree(polygons)
    processed = set()
    polygon_groups = []

    logging.info("finding overlapping polygons")
    for pdi, poly in tqdm.tqdm(enumerate(polygons), total=len(polygons)):
        if pdi in processed:
            continue
        shapely.prepare(poly)
        shrinked = poly.buffer(-1)
        intersect_candidates = tree.query(geometry=poly, predicate="intersects").tolist()
        intersected = {j for j in intersect_candidates if polygons[j].intersects(shrinked) and j not in processed}
        processed.update(intersected)
        polygon_groups.append(intersected)

    final_polygons = []
    non_overlapping_polygons = 0
    overlapping_polygons = 0
    logging.info(f"merging overlapping polygons")
    for group in tqdm.tqdm(polygon_groups):
        if len(group) == 1:
            non_overlapping_polygons += len(group)
            final_polygons.extend([polygons[i] for i in group])
        else:
            overlapping_polygons += len(group)
            final_polygons.append(shapely.ops.unary_union([polygons[i] for i in group]))

    logging.info(f"non-overlapping polygons: {non_overlapping_polygons}")
    logging.info(f"overlapping polygons: {overlapping_polygons}")
    logging.info(f"final polygons: {len(final_polygons)}")

    return final_polygons

# Function to convert a polygon geometry to a GeoJSON feature
def read_polygon_as_geojson_feature(geometry):
    """
    Convert a polygon geometry to a GeoJSON feature.

    :param geometry: Polygon geometry.
    :type geometry: shapely.geometry.Polygon
    :return: GeoJSON feature as a string.
    :rtype: str
    """
    return f'{{ "type": "Feature", "geometry": {geometry} }}'

def load_polygons_from_wkts(wkts, cpus):
    """
    Load polygons from WKT files using multiprocessing.

    :param wkts: List of WKT file paths.
    :type wkts: list
    :param cpus: Number of CPUs to use.
    :type cpus: int
    :return: List of polygons.
    :rtype: list
    """
    logging.info(f"loading polygons from segmented tiles")
    with multiprocessing.Pool(cpus) as pool:
        polygons = list(tqdm.tqdm(pool.imap(read_polygons, wkts), total=len(wkts)))
    polygons = [p for ps in polygons for p in ps]
    logging.info(f"loaded {len(polygons)} polygons from {len(wkts)} tiles")
    return polygons

def drop_empty_polygons(polygons):
    """
    Drop empty polygons from the list.

    :param polygons: List of polygons.
    :type polygons: list
    :return: List of non-empty polygons.
    :rtype: list
    """
    logging.info(f"dropping empty polygons")
    return [p for p in polygons if not p.is_empty]

def convert_to_geojson(output_prefix, stitched_polygons, cpus):
    """
    Convert stitched polygons to GeoJSON format and save to a file.

    :param stitched_polygons: List of stitched polygons.
    :type stitched_polygons: list
    :param cpus: Number of CPUs to use.
    :type cpus: int
    """
    geojson_output_filename = f"{output_prefix}.geojson"
    logging.info(f"reading polygons as GeoJSON (may take a while)")
    geojson_list = shapely.to_geojson(stitched_polygons).tolist()
    logging.info(f"converting segmentations to GeoJSON Feature")
    with multiprocessing.Pool(cpus) as pool:
        geojson_features = list(
            tqdm.tqdm(
                pool.imap(read_polygon_as_geojson_feature, geojson_list),
                total=len(geojson_list),
            )
        )
    logging.info(f"writing GeoJSON FeatureCollection as '{geojson_output_filename}'")
    with open(geojson_output_filename, "wt") as f:
        geojson_output = f"""{{
            "type": "FeatureCollection",
            "features": [{",".join(geojson_features)}]
        }}"""
        f.write(geojson_output)
    del geojson_features
    del geojson_output

def convert_to_wkt(output_prefix, stitched_polygons, cpus):
    """
    Convert stitched polygons to WKT format and save to a file.

    :param stitched_polygons: List of stitched polygons.
    :type stitched_polygons: list
    :param cpus: Number of CPUs to use.
    :type cpus: int
    """
    wkt_output_filename = f"{output_prefix}.wkt"
    logging.info(f"converting segmentations to well-known-text polygons")
    with multiprocessing.Pool(cpus) as pool:
        wkt_strings = list(
            tqdm.tqdm(
                pool.imap(shapely.wkt.dumps, stitched_polygons),
                total=len(stitched_polygons),
            )
        )
    logging.info(f"writing segmentation as well-known-text as '{wkt_output_filename}'")
    with open(wkt_output_filename, "w") as f:
        for wkt_line in tqdm.tqdm(wkt_strings):
            f.write(wkt_line + "\n")

# Main function to process image and WKT files
def main(
        output_prefix: str,
        wkts: list,
        # resolution_level: int = 0
    ):
    """
    Main function to process image and WKT files.

    :param image_path: Path to the image file.
    :type image_path: str
    :param wkts: List of WKT file paths.
    :type wkts: list
    :param resolution_level: Resolution level.
    :type resolution_level: int
    """
    cpus = len(os.sched_getaffinity(0))
    logging.info(f"available cpus = {cpus}")

    polygons = load_polygons_from_wkts(wkts, cpus)
    stitched_polygons = merge_overlapping_polygons(polygons)
    stitched_polygons = drop_empty_polygons(stitched_polygons)
    del polygons

    convert_to_geojson(output_prefix, stitched_polygons, cpus)
    convert_to_wkt(output_prefix, stitched_polygons, cpus)

def run():
    # define argument parser for command line arguments
    parser = argparse.ArgumentParser(description="merge tiled segmentations")
    # argument declarations
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--wkts", nargs="+")
    # parser.add_argument("--image_path", default=None, type=str)
    # parser.add_argument("--resolution_level", default=0, type=int)
    parser.add_argument("--version", action="version", version="0.0.2")
    
    # parse command line arguments
    try:
        args = parser.parse_args()
    except Exception as ex:
        parser.print_help()
        SystemExit(ex)

    if "version" in args:
        print(args.version)
    else:
        # invoke main function with parsed arguments
        main(**vars(args))

# Entry point for the script
if __name__ == "__main__":
    run() 

#
# cd /nfs/cellgeni/prete/segmentation/segmentation_benchmark/modules/sanger/merge_outlines/resources/usr/bin
# singularity shell -B /lustre,/nfs /nfs/cellgeni/prete/segmentation/segmentation_benchmark/containers/segmentation-benchmark-utils.sif
#

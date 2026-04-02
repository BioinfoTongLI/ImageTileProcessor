#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

import argparse
import json
import logging
import multiprocessing
import os

import shapely
import shapely.geometry
import shapely.ops
import shapely.strtree
import shapely.wkt
import tqdm

logging.basicConfig(level=logging.INFO)


def read_polygons(wkt_file: str) -> list:
    """
    Read polygons from a WKT file.

    The file is expected to contain a single WKT geometry collection
    (e.g. MULTIPOLYGON or GEOMETRYCOLLECTION).

    Args:
        wkt_file: Path to the WKT file.

    Returns:
        List of shapely geometries, each buffered by 0 to fix any topology.
    """
    polygons = []
    with open(wkt_file, "rt") as f:
        wkt_str = f.read().strip()
        if wkt_str:
            for polygon in shapely.wkt.loads(wkt_str).geoms:
                polygons.append(polygon.buffer(0))
    return polygons


def merge_overlapping_polygons(polygons: list) -> list:
    """
    Merge overlapping polygons into single geometries.

    Uses an STR-tree for fast intersection queries.  Polygons that overlap
    by more than a 1-pixel inset are grouped and merged with unary_union.

    Args:
        polygons: List of shapely polygon geometries.

    Returns:
        List of merged geometries.
    """
    if not polygons:
        return []

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
        intersected = {
            j for j in intersect_candidates
            if polygons[j].intersects(shrinked) and j not in processed
        }
        processed.update(intersected)
        polygon_groups.append(intersected)

    final_polygons = []
    non_overlapping = 0
    overlapping = 0
    logging.info("merging overlapping polygons")
    for group in tqdm.tqdm(polygon_groups):
        if len(group) == 1:
            non_overlapping += len(group)
            final_polygons.extend([polygons[i] for i in group])
        else:
            overlapping += len(group)
            final_polygons.append(shapely.ops.unary_union([polygons[i] for i in group]))

    logging.info(f"non-overlapping polygons: {non_overlapping}")
    logging.info(f"overlapping polygons: {overlapping}")
    logging.info(f"final polygons: {len(final_polygons)}")

    return final_polygons


def _polygon_to_geojson_feature(geometry_str: str) -> str:
    """
    Wrap a GeoJSON geometry string in a GeoJSON Feature object.

    Args:
        geometry_str: A valid GeoJSON geometry string (from shapely.to_geojson).

    Returns:
        A valid GeoJSON Feature string.
    """
    return json.dumps({"type": "Feature", "geometry": json.loads(geometry_str)})


def parallel_load_polygons_from_wkts(wkts: list, cpus: int) -> list:
    """
    Load polygons from WKT files using multiprocessing.

    Args:
        wkts: List of WKT file paths.
        cpus: Number of worker processes.

    Returns:
        Flat list of all polygons across all files.
    """
    logging.info("loading polygons from segmented tiles")
    with multiprocessing.Pool(cpus) as pool:
        polygons = list(tqdm.tqdm(pool.imap(read_polygons, wkts), total=len(wkts)))
    polygons = [p for ps in polygons for p in ps]
    logging.info(f"loaded {len(polygons)} polygons from {len(wkts)} tiles")
    return polygons


def load_polygons_from_wkts(wkts: list) -> list:
    """
    Load polygons from WKT files sequentially.

    Args:
        wkts: List of WKT file paths.

    Returns:
        Flat list of all polygons across all files.
    """
    logging.info("loading polygons from segmented tiles")
    polygons = []
    for wkt_file in tqdm.tqdm(wkts, desc="Loading WKT files"):
        polygons.extend(read_polygons(wkt_file))
    logging.info(f"loaded {len(polygons)} polygons from {len(wkts)} tiles")
    return polygons


def drop_empty_polygons(polygons: list) -> list:
    """
    Remove empty geometries from a list.

    Args:
        polygons: List of shapely geometries.

    Returns:
        List with empty geometries removed.
    """
    logging.info("dropping empty polygons")
    return [p for p in polygons if not p.is_empty]


def convert_to_geojson(output_prefix: str, stitched_polygons: list, cpus: int):
    """
    Convert stitched polygons to a GeoJSON FeatureCollection file.

    Args:
        output_prefix: Output file path without extension.
        stitched_polygons: List of shapely geometries to export.
        cpus: Number of worker processes for parallel serialisation.
    """
    geojson_output_filename = f"{output_prefix}.geojson"
    logging.info("reading polygons as GeoJSON (may take a while)")
    geojson_list = shapely.to_geojson(stitched_polygons).tolist()

    logging.info("converting segmentations to GeoJSON Features")
    with multiprocessing.Pool(cpus) as pool:
        geojson_features = list(
            tqdm.tqdm(
                pool.imap(_polygon_to_geojson_feature, geojson_list),
                total=len(geojson_list),
            )
        )

    logging.info(f"writing GeoJSON FeatureCollection as '{geojson_output_filename}'")
    feature_collection = (
        '{"type":"FeatureCollection","features":['
        + ",".join(geojson_features)
        + "]}"
    )
    with open(geojson_output_filename, "wt") as f:
        f.write(feature_collection)


def convert_to_wkt(output_prefix: str, stitched_polygons: list, cpus: int):
    """
    Convert stitched polygons to a WKT file, one polygon per line.

    Args:
        output_prefix: Output file path without extension.
        stitched_polygons: List of shapely geometries to export.
        cpus: Number of worker processes for parallel serialisation.
    """
    wkt_output_filename = f"{output_prefix}.wkt"
    logging.info("converting segmentations to well-known-text polygons")
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


def main(output_prefix: str, wkts: list):
    """
    Merge tiled polygon segmentations and write GeoJSON and WKT outputs.

    Args:
        output_prefix: Prefix for output files (without extension).
            Produces ``<output_prefix>.geojson`` and ``<output_prefix>.wkt``.
        wkts: List of WKT file paths, one per image tile.
    """
    cpus = len(os.sched_getaffinity(0))
    logging.info(f"available cpus = {cpus}")

    polygons = load_polygons_from_wkts(wkts)
    stitched_polygons = merge_overlapping_polygons(polygons)
    stitched_polygons = drop_empty_polygons(stitched_polygons)

    convert_to_geojson(output_prefix, stitched_polygons, cpus)
    convert_to_wkt(output_prefix, stitched_polygons, cpus)


def run():
    parser = argparse.ArgumentParser(description="Merge tiled polygon segmentations")
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--wkts", nargs="+", required=True)
    parser.add_argument("--version", action="version", version="0.1.16")

    try:
        args = parser.parse_args()
    except SystemExit:
        raise
    except Exception as ex:
        parser.print_help()
        raise SystemExit(ex)

    main(**vars(args))


if __name__ == "__main__":
    run()

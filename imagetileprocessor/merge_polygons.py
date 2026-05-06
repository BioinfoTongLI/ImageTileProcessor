#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute
"""
merge_polygons.py — stitch tiled polygon segmentations into a single dataset.

Workflow
--------
1. Load per-tile WKT files (in parallel or sequentially).
2. Validate and repair any geometrically invalid polygons.
3. Merge polygons that overlap by more than a 1-pixel margin.
4. Drop any empty geometries produced during merging.
5. Write the result as both GeoJSON (FeatureCollection) and WKT.

Each GeoJSON Feature is assigned a deterministic UUID5 in its top-level ``id``
field, conforming to RFC 7946.
"""

import json
import logging
import multiprocessing
import os
import uuid

import click

import shapely
import shapely.geometry
import shapely.ops
import shapely.strtree
import shapely.validation
import shapely.wkt
import tqdm

logging.basicConfig(level=logging.INFO)

DETERMINISTIC_FEATURE_NAMESPACE = uuid.UUID(
    "df0c588a-4c4b-5f72-80be-f39cc4d4ba6e"
)


def _normalised_wkb_hex(geometry: shapely.geometry.base.BaseGeometry) -> str:
    """Return canonical WKB hex for stable sorting and ID generation."""
    return shapely.to_wkb(shapely.normalize(geometry), hex=True)


def _geometry_sort_key(
    geometry: shapely.geometry.base.BaseGeometry,
) -> tuple[tuple[float, ...], float, float, str, str]:
    """Build a deterministic ordering key for shapely geometries."""
    bounds = tuple(geometry.bounds) if not geometry.is_empty else (float("inf"),) * 4
    return (
        bounds,
        geometry.area,
        geometry.length,
        geometry.geom_type,
        _normalised_wkb_hex(geometry),
    )


def sort_polygons_deterministically(
    polygons: list[shapely.geometry.base.BaseGeometry],
) -> list[shapely.geometry.base.BaseGeometry]:
    """Return geometries in a stable, content-derived order."""
    return sorted(polygons, key=_geometry_sort_key)


def _overlaps_with_margin(
    first: shapely.geometry.base.BaseGeometry,
    second: shapely.geometry.base.BaseGeometry,
) -> bool:
    """Return true when either polygon intersects the other's 1-pixel inset."""
    first_inset = first.buffer(-1)
    second_inset = second.buffer(-1)
    return (
        (not first_inset.is_empty and second.intersects(first_inset))
        or (not second_inset.is_empty and first.intersects(second_inset))
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_polygons(wkt_file: str) -> list[shapely.geometry.base.BaseGeometry]:
    """Read polygons from a WKT file.

    The file must contain a single WKT geometry collection (e.g.
    ``MULTIPOLYGON`` or ``GEOMETRYCOLLECTION``).  Each component geometry is
    buffered by 0 to normalise ring orientation and close any minor gaps left
    by the serialiser.

    Args:
        wkt_file: Path to the WKT file.

    Returns:
        List of shapely geometries, one per component of the collection.
        Returns an empty list if the file is empty.
    """
    polygons = []
    with open(wkt_file, "rt") as f:
        wkt_str = f.read().strip()
    if wkt_str:
        for polygon in shapely.wkt.loads(wkt_str).geoms:
            polygons.append(polygon.buffer(0))
    return polygons


def parallel_load_polygons_from_wkts(
    wkts: list[str], cpus: int
) -> list[shapely.geometry.base.BaseGeometry]:
    """Load polygons from multiple WKT files using a multiprocessing pool.

    Args:
        wkts: WKT file paths, one per image tile.
        cpus: Number of worker processes.

    Returns:
        Flat list of all polygons from all tiles.
    """
    logging.info("loading polygons from segmented tiles (parallel)")
    with multiprocessing.Pool(cpus) as pool:
        per_tile = list(tqdm.tqdm(pool.imap(read_polygons, wkts), total=len(wkts)))
    polygons = [p for tile in per_tile for p in tile]
    logging.info(f"loaded {len(polygons)} polygons from {len(wkts)} tiles")
    return polygons


def load_polygons_from_wkts(
    wkts: list[str],
) -> list[shapely.geometry.base.BaseGeometry]:
    """Load polygons from multiple WKT files sequentially.

    Prefer this over the parallel variant when the number of tiles is small
    or when multiprocessing overhead would dominate.

    Args:
        wkts: WKT file paths, one per image tile.

    Returns:
        Flat list of all polygons from all tiles.
    """
    logging.info("loading polygons from segmented tiles (sequential)")
    polygons = []
    for wkt_file in tqdm.tqdm(wkts, desc="Loading WKT files"):
        polygons.extend(read_polygons(wkt_file))
    logging.info(f"loaded {len(polygons)} polygons from {len(wkts)} tiles")
    return polygons


# ---------------------------------------------------------------------------
# Geometry validation & cleaning
# ---------------------------------------------------------------------------


def check_polygon_validity(
    polygons: list[shapely.geometry.base.BaseGeometry],
) -> list[shapely.geometry.base.BaseGeometry]:
    """Validate polygons and repair any that are geometrically invalid.

    A polygon is invalid when it violates the OGC Simple Features rules —
    common causes include self-intersections, repeated points, and unclosed
    rings.  Invalid geometries cause silent failures in spatial operations
    such as intersection tests and union.

    Each invalid polygon is repaired in-place using
    :func:`shapely.validation.make_valid`, which may split a self-intersecting
    polygon into a ``GEOMETRYCOLLECTION`` of smaller, valid parts.  Those
    parts are flattened back into the output list so the polygon count may
    increase slightly.

    Args:
        polygons: List of shapely geometries to validate.

    Returns:
        New list where every geometry satisfies ``polygon.is_valid``.
        Valid geometries are passed through unchanged.
    """
    n_invalid = sum(1 for p in polygons if not p.is_valid)
    if n_invalid == 0:
        logging.info("all polygons are valid")
        return polygons

    logging.warning(
        f"{n_invalid}/{len(polygons)} invalid polygons found — repairing with make_valid"
    )

    repaired: list[shapely.geometry.base.BaseGeometry] = []
    for poly in tqdm.tqdm(polygons, desc="Validating polygons"):
        if poly.is_valid:
            repaired.append(poly)
            continue

        fixed = shapely.validation.make_valid(poly)

        # make_valid may return a GeometryCollection containing the repaired
        # parts; flatten those into individual polygons.
        if hasattr(fixed, "geoms"):
            repaired.extend(part for part in fixed.geoms if not part.is_empty)
        elif not fixed.is_empty:
            repaired.append(fixed)

    n_still_invalid = sum(1 for p in repaired if not p.is_valid)
    if n_still_invalid:
        logging.error(
            f"{n_still_invalid} polygon(s) could not be repaired and will be kept as-is"
        )

    logging.info(
        f"validity check complete: {len(polygons)} → {len(repaired)} geometries"
    )
    return repaired


def drop_empty_polygons(
    polygons: list[shapely.geometry.base.BaseGeometry],
) -> list[shapely.geometry.base.BaseGeometry]:
    """Remove empty geometries from a list.

    Empty geometries (``GEOMETRYCOLLECTION EMPTY``, etc.) can arise after
    merging or repairing polygons.  They carry no spatial information and
    would produce degenerate GeoJSON features.

    Args:
        polygons: List of shapely geometries.

    Returns:
        New list with all empty geometries removed.
    """
    before = len(polygons)
    result = [p for p in polygons if not p.is_empty]
    dropped = before - len(result)
    if dropped:
        logging.info(f"dropped {dropped} empty geometry/geometries")
    return result


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_overlapping_polygons(
    polygons: list[shapely.geometry.base.BaseGeometry],
) -> list[shapely.geometry.base.BaseGeometry]:
    """Merge polygons that overlap by more than a 1-pixel margin.

    Two polygons are considered overlapping when one intersects the other
    *after* the other has been inset by 1 pixel (``buffer(-1)``).  This
    tolerance prevents merging polygons that share only a common edge or a
    single point — artefacts that naturally arise at tile boundaries.

    The algorithm uses a Shapely STR-tree for fast bounding-box pre-filtering,
    then performs exact intersection tests only on the candidates returned by
    the tree.  Overlapping polygons are grouped transitively and merged into a
    single geometry with :func:`shapely.ops.unary_union`.

    Args:
        polygons: List of shapely polygon geometries.

    Returns:
        List of merged geometries.  Polygons with no overlapping neighbours
        are returned unchanged.
    """
    if not polygons:
        return []

    polygons = sort_polygons_deterministically(polygons)
    tree = shapely.strtree.STRtree(polygons)
    processed: set[int] = set()
    polygon_groups: list[set[int]] = []

    logging.info("finding overlapping polygons")
    for idx, poly in tqdm.tqdm(enumerate(polygons), total=len(polygons)):
        if idx in processed:
            continue
        group: set[int] = set()
        pending = [idx]

        while pending:
            current = pending.pop(0)
            if current in group:
                continue

            group.add(current)
            current_poly = polygons[current]
            shapely.prepare(current_poly)

            candidates = sorted(
                tree.query(geometry=current_poly, predicate="intersects").tolist()
            )
            for candidate in candidates:
                if (
                    candidate not in processed
                    and candidate not in group
                    and _overlaps_with_margin(current_poly, polygons[candidate])
                ):
                    pending.append(candidate)

        processed.update(group)
        polygon_groups.append(group)

    merged: list[shapely.geometry.base.BaseGeometry] = []
    n_non_overlapping = 0
    n_overlapping = 0

    logging.info("merging overlapping polygon groups")
    for group in tqdm.tqdm(polygon_groups):
        ordered_group = sorted(group)
        if len(group) == 1:
            n_non_overlapping += 1
            merged.extend(polygons[i] for i in ordered_group)
        else:
            n_overlapping += len(group)
            merged.append(
                shapely.ops.unary_union([polygons[i] for i in ordered_group])
            )

    logging.info(f"non-overlapping polygons: {n_non_overlapping}")
    logging.info(f"polygons merged away:     {n_overlapping}")
    logging.info(f"final polygon count:      {len(merged)}")
    return sort_polygons_deterministically(merged)


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------


def _polygon_to_geojson_feature(feature_input: tuple[int, str] | str) -> str:
    """Serialise one GeoJSON geometry string as a GeoJSON Feature.

    A UUID5 is generated from the geometry string and deterministic output
    position, then stored in the Feature's top-level ``id`` member following
    RFC 7946 §3.2.  This gives every exported polygon a stable, unique
    identifier that downstream tools (e.g. QuPath, QGIS) can use for selection
    and cross-referencing.

    Args:
        feature_input: Either a valid GeoJSON geometry string as returned by
            :func:`shapely.to_geojson`, or ``(index, geometry_str)``.

    Returns:
        A JSON-serialised GeoJSON Feature string with ``type``, ``id``,
        ``geometry``, and ``properties`` (``null``) members, as required
        by RFC 7946 §3.2.
    """
    if isinstance(feature_input, tuple):
        index, geometry_str = feature_input
    else:
        index = 0
        geometry_str = feature_input

    feature_id = uuid.uuid5(
        DETERMINISTIC_FEATURE_NAMESPACE,
        f"{index}:{geometry_str}",
    )
    feature = {
        "type": "Feature",
        "id": str(feature_id),
        "geometry": json.loads(geometry_str),
        "properties": None,
    }
    return json.dumps(feature)


def convert_to_geojson(
    output_prefix: str,
    stitched_polygons: list[shapely.geometry.base.BaseGeometry],
    cpus: int,
) -> None:
    """Write stitched polygons to a GeoJSON FeatureCollection file.

    Each feature is assigned a deterministic UUID5 ``id`` and empty
    ``properties``. Serialisation is parallelised across ``cpus`` workers.

    Args:
        output_prefix: Output path without extension.
            The file is written to ``<output_prefix>.geojson``.
        stitched_polygons: Merged shapely geometries to export.
        cpus: Number of worker processes for parallel serialisation.
    """
    out_path = f"{output_prefix}.geojson"
    logging.info("serialising polygons to GeoJSON geometry strings")
    ordered_polygons = [
        shapely.normalize(p) for p in sort_polygons_deterministically(stitched_polygons)
    ]
    geojson_list = sorted(shapely.to_geojson(ordered_polygons).tolist())

    logging.info("wrapping geometries in GeoJSON Features")
    with multiprocessing.Pool(cpus) as pool:
        features = list(
            tqdm.tqdm(
                pool.imap(_polygon_to_geojson_feature, enumerate(geojson_list)),
                total=len(geojson_list),
            )
        )

    feature_collection = (
        '{"type":"FeatureCollection","features":[' + ",".join(features) + "]}"
    )
    logging.info(f"writing GeoJSON FeatureCollection to '{out_path}'")
    with open(out_path, "wt") as f:
        f.write(feature_collection)


def convert_to_wkt(
    output_prefix: str,
    stitched_polygons: list[shapely.geometry.base.BaseGeometry],
    cpus: int,
) -> None:
    """Write stitched polygons to a WKT file, one polygon per line.

    Args:
        output_prefix: Output path without extension.
            The file is written to ``<output_prefix>.wkt``.
        stitched_polygons: Merged shapely geometries to export.
        cpus: Number of worker processes for parallel serialisation.
    """
    out_path = f"{output_prefix}.wkt"
    ordered_polygons = [
        shapely.normalize(p) for p in sort_polygons_deterministically(stitched_polygons)
    ]
    logging.info("serialising polygons to WKT")
    with multiprocessing.Pool(cpus) as pool:
        wkt_strings = sorted(
            tqdm.tqdm(
                pool.imap(shapely.wkt.dumps, ordered_polygons),
                total=len(ordered_polygons),
            )
        )
    logging.info(f"writing WKT to '{out_path}'")
    with open(out_path, "w") as f:
        for line in wkt_strings:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def main(output_prefix: str, wkts: list[str]) -> None:
    """Run the full merge pipeline and write GeoJSON and WKT outputs.

    Steps:

    1. Load per-tile WKT files sequentially.
    2. Validate and repair invalid geometries.
    3. Merge overlapping polygons (1-pixel inset tolerance).
    4. Drop any empty geometries.
    5. Write ``<output_prefix>.geojson`` and ``<output_prefix>.wkt``.

    Args:
        output_prefix: Prefix for output files (without extension).
        wkts: WKT file paths, one per image tile.
    """
    cpus = len(os.sched_getaffinity(0))
    logging.info(f"available CPUs: {cpus}")

    polygons = load_polygons_from_wkts(sorted(wkts))
    polygons = check_polygon_validity(polygons)
    merged = merge_overlapping_polygons(polygons)
    merged = sort_polygons_deterministically(drop_empty_polygons(merged))

    convert_to_geojson(output_prefix, merged, cpus)
    convert_to_wkt(output_prefix, merged, cpus)


@click.command()
@click.version_option("0.2.1")
@click.option(
    "--output_prefix",
    required=True,
    help="Output file path without extension.",
)
@click.argument("wkts", nargs=-1, required=True)
def run(output_prefix: str, wkts: tuple[str, ...]) -> None:
    """Merge tiled polygon segmentations into a single GeoJSON/WKT dataset.

    WKTS are one or more WKT files produced by per-tile segmentation.
    """
    main(output_prefix=output_prefix, wkts=list(wkts))


if __name__ == "__main__":
    run()

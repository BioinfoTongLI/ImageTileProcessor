#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

import json
import os
import tempfile

import pytest
import shapely
from shapely.geometry import Polygon

from imagetileprocessor.tile_2D_image import calculate_slices, write_slices_to_csv
from imagetileprocessor.merge_polygons import (
    _polygon_to_geojson_feature,
    drop_empty_polygons,
    merge_overlapping_polygons,
    read_polygons,
)


# ---------------------------------------------------------------------------
# calculate_slices
# ---------------------------------------------------------------------------

class TestCalculateSlices:
    def test_exact_fit_no_overlap(self):
        slices = calculate_slices((100, 100), 50, 0)
        assert slices == [(0, 0, 50, 50), (0, 50, 50, 100), (50, 0, 100, 50), (50, 50, 100, 100)]

    def test_last_tile_clamped_to_image_boundary(self):
        slices = calculate_slices((70, 80), 50, 0)
        x_maxes = [s[2] for s in slices]
        y_maxes = [s[3] for s in slices]
        assert max(x_maxes) == 70
        assert max(y_maxes) == 80

    def test_with_overlap_produces_correct_step(self):
        # step = 50 - 10 = 40; image 100 wide → tiles start at 0, 40, 80
        slices = calculate_slices((100, 50), 50, 10)
        x_mins = [s[0] for s in slices]
        assert sorted(set(x_mins)) == [0, 40, 80]

    def test_single_tile_when_chunk_larger_than_image(self):
        slices = calculate_slices((100, 100), 200, 0)
        assert slices == [(0, 0, 100, 100)]

    def test_overlap_equals_chunk_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            calculate_slices((100, 100), 50, 50)

    def test_overlap_greater_than_chunk_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            calculate_slices((100, 100), 50, 60)

    def test_returns_list_of_4_tuples(self):
        slices = calculate_slices((200, 200), 100, 10)
        assert all(len(s) == 4 for s in slices)

    def test_tiles_cover_full_width(self):
        slices = calculate_slices((256, 256), 64, 8)
        # Every x pixel in [0, 256) must be covered by at least one tile
        for x in range(256):
            assert any(s[0] <= x < s[2] for s in slices), f"pixel x={x} not covered"

    def test_tiles_cover_full_height(self):
        slices = calculate_slices((256, 256), 64, 8)
        for y in range(256):
            assert any(s[1] <= y < s[3] for s in slices), f"pixel y={y} not covered"


# ---------------------------------------------------------------------------
# write_slices_to_csv
# ---------------------------------------------------------------------------

class TestWriteSlicesToCsv:
    def test_csv_header_and_rows(self):
        slices = [(0, 0, 50, 50), (50, 0, 100, 50)]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as f:
            fname = f.name
        try:
            write_slices_to_csv(slices, fname)
            with open(fname) as f:
                lines = f.readlines()
            assert lines[0].strip() == "Tile,X_MIN,Y_MIN,X_MAX,Y_MAX"
            assert lines[1].strip() == "1,0,0,50,50"
            assert lines[2].strip() == "2,50,0,100,50"
        finally:
            os.unlink(fname)

    def test_tile_numbering_starts_at_one(self):
        slices = [(0, 0, 10, 10)]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as f:
            fname = f.name
        try:
            write_slices_to_csv(slices, fname)
            with open(fname) as f:
                rows = f.readlines()
            assert rows[1].startswith("1,")
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# merge_overlapping_polygons
# ---------------------------------------------------------------------------

class TestMergeOverlappingPolygons:
    def test_non_overlapping_polygons_unchanged(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
        result = merge_overlapping_polygons([p1, p2])
        assert len(result) == 2

    def test_overlapping_polygons_merged(self):
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p2 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])
        result = merge_overlapping_polygons([p1, p2])
        assert len(result) == 1
        # Merged area should be larger than either individual polygon
        assert result[0].area > p1.area

    def test_empty_input_returns_empty(self):
        assert merge_overlapping_polygons([]) == []

    def test_single_polygon_returned_unchanged(self):
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = merge_overlapping_polygons([p])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# drop_empty_polygons
# ---------------------------------------------------------------------------

class TestDropEmptyPolygons:
    def test_empty_polygon_removed(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon()
        result = drop_empty_polygons([p1, p2])
        assert len(result) == 1
        assert result[0].equals(p1)

    def test_all_valid_polygons_kept(self):
        polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(3)]
        assert drop_empty_polygons(polys) == polys

    def test_all_empty_returns_empty(self):
        assert drop_empty_polygons([Polygon(), Polygon()]) == []


# ---------------------------------------------------------------------------
# _polygon_to_geojson_feature
# ---------------------------------------------------------------------------

class TestPolygonToGeoJsonFeature:
    def test_output_is_valid_json(self):
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        geojson_str = shapely.to_geojson(p)
        feature = _polygon_to_geojson_feature(geojson_str)
        parsed = json.loads(feature)  # must not raise
        assert parsed["type"] == "Feature"
        assert "geometry" in parsed
        assert parsed["geometry"]["type"] == "Polygon"

    def test_geometry_coordinates_preserved(self):
        p = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        geojson_str = shapely.to_geojson(p)
        feature = _polygon_to_geojson_feature(geojson_str)
        parsed = json.loads(feature)
        assert parsed["geometry"]["coordinates"] is not None


# ---------------------------------------------------------------------------
# read_polygons
# ---------------------------------------------------------------------------

class TestReadPolygons:
    def test_reads_multipolygon_wkt(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        from shapely.geometry import MultiPolygon
        mp = MultiPolygon([p1, p2])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wkt", delete=False) as f:
            f.write(mp.wkt)
            fname = f.name
        try:
            result = read_polygons(fname)
            assert len(result) == 2
        finally:
            os.unlink(fname)

    def test_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".wkt", delete=False) as f:
            f.write("")
            fname = f.name
        try:
            result = read_polygons(fname)
            assert result == []
        finally:
            os.unlink(fname)

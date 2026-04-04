#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Wellcome Sanger Institute

import csv
import json
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest
import shapely
from click.testing import CliRunner
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon

from imagetileprocessor.merge_peaks import main as peaks_main
from imagetileprocessor.merge_polygons import (
    _polygon_to_geojson_feature,
    check_polygon_validity,
    convert_to_geojson,
    convert_to_wkt,
    drop_empty_polygons,
    load_polygons_from_wkts,
    merge_overlapping_polygons,
    read_polygons,
    run as polygons_cli,
)
from imagetileprocessor.tile_2D_image import calculate_slices, write_slices_to_csv
from imagetileprocessor.tile_2D_image import run as tile_cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wkt(polygons: list, suffix: str = ".wkt") -> str:
    """Write a MultiPolygon WKT to a temp file and return its path."""
    mp = MultiPolygon(polygons)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(mp.wkt)
    f.close()
    return f.name


def _write_csv(rows: list[list], suffix: str = ".csv") -> str:
    """Write rows to a CSV temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, newline=""
    )
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
    f.close()
    return f.name


def _tmp_path(suffix: str = "") -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    os.unlink(f.name)
    return f.name


# ---------------------------------------------------------------------------
# calculate_slices
# ---------------------------------------------------------------------------


class TestCalculateSlices:
    def test_exact_fit_no_overlap(self):
        slices = calculate_slices((100, 100), 50, 0)
        assert slices == [
            (0, 0, 50, 50),
            (0, 50, 50, 100),
            (50, 0, 100, 50),
            (50, 50, 100, 100),
        ]

    def test_last_tile_clamped_to_image_boundary(self):
        slices = calculate_slices((70, 80), 50, 0)
        assert max(s[2] for s in slices) == 70
        assert max(s[3] for s in slices) == 80

    def test_with_overlap_produces_correct_step(self):
        # step = 50 - 10 = 40; width 100 → tiles start at 0, 40, 80
        slices = calculate_slices((100, 50), 50, 10)
        x_mins = sorted(set(s[0] for s in slices))
        assert x_mins == [0, 40, 80]

    def test_single_tile_when_chunk_larger_than_image(self):
        assert calculate_slices((100, 100), 200, 0) == [(0, 0, 100, 100)]

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
        fname = _tmp_path(".csv")
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
        fname = _tmp_path(".csv")
        try:
            write_slices_to_csv([(0, 0, 10, 10)], fname)
            with open(fname) as f:
                rows = f.readlines()
            assert rows[1].startswith("1,")
        finally:
            os.unlink(fname)

    def test_empty_slices_writes_only_header(self):
        fname = _tmp_path(".csv")
        try:
            write_slices_to_csv([], fname)
            with open(fname) as f:
                lines = f.readlines()
            assert len(lines) == 1
            assert "Tile" in lines[0]
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# tile_2D_image CLI (mocked AICSImage)
# ---------------------------------------------------------------------------


class TestTile2DImageCli:
    def test_cli_produces_csv(self):
        mock_img = MagicMock()
        mock_img.get_image_dask_data.return_value = MagicMock(shape=(200, 300))

        fname = _tmp_path(".csv")
        try:
            with patch("imagetileprocessor.tile_2D_image.AICSImage", return_value=mock_img):
                runner = CliRunner()
                # fire-based CLI; call main directly instead
                from imagetileprocessor.tile_2D_image import main as tile_main
                tile_main.__wrapped__ = None  # ensure no wrapping issues
                with patch("imagetileprocessor.tile_2D_image.AICSImage", return_value=mock_img):
                    from imagetileprocessor.tile_2D_image import main as tile_main2
                    tile_main2("fake.tif", fname, overlap=10, chunk_size=100)

            with open(fname) as f:
                lines = f.readlines()
            assert lines[0].strip() == "Tile,X_MIN,Y_MIN,X_MAX,Y_MAX"
            assert len(lines) > 1
        finally:
            if os.path.exists(fname):
                os.unlink(fname)


# ---------------------------------------------------------------------------
# read_polygons
# ---------------------------------------------------------------------------


class TestReadPolygons:
    def test_reads_multipolygon_wkt(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        fname = _write_wkt([p1, p2])
        try:
            result = read_polygons(fname)
            assert len(result) == 2
        finally:
            os.unlink(fname)

    def test_empty_file_returns_empty_list(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".wkt", delete=False)
        f.write("")
        f.close()
        try:
            assert read_polygons(f.name) == []
        finally:
            os.unlink(f.name)

    def test_returned_polygons_are_valid(self):
        p = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        fname = _write_wkt([p])
        try:
            result = read_polygons(fname)
            assert all(r.is_valid for r in result)
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# load_polygons_from_wkts
# ---------------------------------------------------------------------------


class TestLoadPolygonsFromWkts:
    def test_loads_from_multiple_files(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        p3 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
        f1 = _write_wkt([p1])
        f2 = _write_wkt([p2, p3])
        try:
            result = load_polygons_from_wkts([f1, f2])
            assert len(result) == 3
        finally:
            os.unlink(f1)
            os.unlink(f2)

    def test_empty_wkt_file_contributes_zero_polygons(self):
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        f1 = _write_wkt([p])
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".wkt", delete=False)
        f2.write("")
        f2.close()
        try:
            result = load_polygons_from_wkts([f1, f2.name])
            assert len(result) == 1
        finally:
            os.unlink(f1)
            os.unlink(f2.name)


# ---------------------------------------------------------------------------
# check_polygon_validity
# ---------------------------------------------------------------------------


class TestCheckPolygonValidity:
    def test_valid_polygons_pass_through_unchanged(self):
        polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]) for _ in range(3)]
        result = check_polygon_validity(polys)
        assert len(result) == 3
        assert all(p.is_valid for p in result)

    def test_invalid_bowtie_polygon_is_repaired(self):
        # Self-intersecting "bowtie" — classic invalid polygon
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        assert not bowtie.is_valid
        result = check_polygon_validity([bowtie])
        assert len(result) >= 1
        assert all(p.is_valid for p in result)

    def test_empty_input_returns_empty(self):
        assert check_polygon_validity([]) == []

    def test_mixed_valid_invalid_all_repaired(self):
        valid = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        result = check_polygon_validity([valid, bowtie])
        assert all(p.is_valid for p in result)


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
        assert result[0].area > p1.area

    def test_empty_input_returns_empty(self):
        assert merge_overlapping_polygons([]) == []

    def test_single_polygon_returned_unchanged(self):
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = merge_overlapping_polygons([p])
        assert len(result) == 1

    def test_touching_edge_only_not_merged(self):
        # Two squares sharing only an edge — inset by 1 px means they should NOT merge
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        result = merge_overlapping_polygons([p1, p2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# drop_empty_polygons
# ---------------------------------------------------------------------------


class TestDropEmptyPolygons:
    def test_empty_polygon_removed(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = drop_empty_polygons([p1, Polygon()])
        assert len(result) == 1
        assert result[0].equals(p1)

    def test_all_valid_polygons_kept(self):
        polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(3)]
        assert drop_empty_polygons(polys) == polys

    def test_all_empty_returns_empty(self):
        assert drop_empty_polygons([Polygon(), Polygon()]) == []


# ---------------------------------------------------------------------------
# _polygon_to_geojson_feature
# ---------------------------------------------------------------------------


class TestPolygonToGeoJsonFeature:
    def _feature(self, poly: Polygon) -> dict:
        return json.loads(_polygon_to_geojson_feature(shapely.to_geojson(poly)))

    def test_output_is_valid_json(self):
        parsed = self._feature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        assert parsed["type"] == "Feature"
        assert "geometry" in parsed
        assert parsed["geometry"]["type"] == "Polygon"

    def test_geometry_coordinates_preserved(self):
        parsed = self._feature(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
        assert parsed["geometry"]["coordinates"] is not None

    def test_id_is_present(self):
        parsed = self._feature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        assert "id" in parsed

    def test_id_is_valid_uuid4(self):
        parsed = self._feature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        # Must not raise
        val = uuid.UUID(parsed["id"], version=4)
        assert val.version == 4

    def test_each_call_produces_unique_id(self):
        geojson_str = shapely.to_geojson(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        ids = {json.loads(_polygon_to_geojson_feature(geojson_str))["id"] for _ in range(10)}
        assert len(ids) == 10

    def test_properties_is_null(self):
        parsed = self._feature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        assert "properties" in parsed
        assert parsed["properties"] is None


# ---------------------------------------------------------------------------
# convert_to_geojson
# ---------------------------------------------------------------------------


class TestConvertToGeoJson:
    def test_produces_valid_feature_collection(self):
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ]
        prefix = _tmp_path()
        try:
            convert_to_geojson(prefix, polys, cpus=1)
            with open(f"{prefix}.geojson") as f:
                fc = json.load(f)
            assert fc["type"] == "FeatureCollection"
            assert len(fc["features"]) == 2
        finally:
            path = f"{prefix}.geojson"
            if os.path.exists(path):
                os.unlink(path)

    def test_each_feature_has_uuid_id(self):
        polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        prefix = _tmp_path()
        try:
            convert_to_geojson(prefix, polys, cpus=1)
            with open(f"{prefix}.geojson") as f:
                fc = json.load(f)
            feat = fc["features"][0]
            assert "id" in feat
            uuid.UUID(feat["id"], version=4)  # must not raise
        finally:
            path = f"{prefix}.geojson"
            if os.path.exists(path):
                os.unlink(path)

    def test_all_feature_ids_are_unique(self):
        polys = [
            Polygon([(i * 10, 0), (i * 10 + 5, 0), (i * 10 + 5, 5), (i * 10, 5)])
            for i in range(5)
        ]
        prefix = _tmp_path()
        try:
            convert_to_geojson(prefix, polys, cpus=1)
            with open(f"{prefix}.geojson") as f:
                fc = json.load(f)
            ids = [feat["id"] for feat in fc["features"]]
            assert len(ids) == len(set(ids))
        finally:
            path = f"{prefix}.geojson"
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# convert_to_wkt
# ---------------------------------------------------------------------------


class TestConvertToWkt:
    def test_produces_one_line_per_polygon(self):
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ]
        prefix = _tmp_path()
        try:
            convert_to_wkt(prefix, polys, cpus=1)
            with open(f"{prefix}.wkt") as f:
                lines = [ln for ln in f.readlines() if ln.strip()]
            assert len(lines) == 2
        finally:
            path = f"{prefix}.wkt"
            if os.path.exists(path):
                os.unlink(path)

    def test_wkt_lines_are_parseable(self):
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        prefix = _tmp_path()
        try:
            convert_to_wkt(prefix, [poly], cpus=1)
            with open(f"{prefix}.wkt") as f:
                line = f.readline().strip()
            parsed = shapely.wkt.loads(line)
            assert parsed.is_valid
        finally:
            path = f"{prefix}.wkt"
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# merge_polygons CLI (click)
# ---------------------------------------------------------------------------


class TestMergePolygonsCli:
    def test_cli_runs_and_produces_outputs(self):
        p1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        p2 = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])
        wkt_file = _write_wkt([p1, p2])
        prefix = _tmp_path()
        runner = CliRunner()
        try:
            result = runner.invoke(polygons_cli, ["--output_prefix", prefix, wkt_file])
            assert result.exit_code == 0, result.output
            assert os.path.exists(f"{prefix}.geojson")
            assert os.path.exists(f"{prefix}.wkt")
        finally:
            os.unlink(wkt_file)
            for ext in (".geojson", ".wkt"):
                if os.path.exists(f"{prefix}{ext}"):
                    os.unlink(f"{prefix}{ext}")

    def test_cli_missing_required_args_fails(self):
        runner = CliRunner()
        result = runner.invoke(polygons_cli, [])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# merge_peaks.main
# ---------------------------------------------------------------------------


class TestMergePeaksMain:
    def _make_csv(self, coords: list[tuple[float, float]]) -> str:
        """Write (y, x) rows to a CSV with header, return path."""
        rows = [["y", "x"]] + [list(c) for c in coords]
        return _write_csv(rows)

    def test_single_csv_writes_wkt(self):
        csv_path = self._make_csv([(0.0, 0.0), (10.0, 10.0)])
        out = _tmp_path(".wkt")
        try:
            peaks_main(csv_path, output_name=out)
            with open(out) as f:
                content = f.read().strip()
            assert content.startswith("MULTIPOINT")
        finally:
            os.unlink(csv_path)
            if os.path.exists(out):
                os.unlink(out)

    def test_nearby_peaks_are_merged(self):
        # Two points close together should collapse to one centroid
        csv_path = self._make_csv([(0.0, 0.0), (0.5, 0.5)])
        out = _tmp_path(".wkt")
        try:
            peaks_main(csv_path, output_name=out, peak_radius=2.0)
            with open(out) as f:
                result = shapely.wkt.loads(f.read().strip())
            # Two nearby points → one merged region → one centroid
            assert len(result.geoms) == 1
        finally:
            os.unlink(csv_path)
            if os.path.exists(out):
                os.unlink(out)

    def test_distant_peaks_not_merged(self):
        # Points far apart should remain separate
        csv_path = self._make_csv([(0.0, 0.0), (100.0, 100.0)])
        out = _tmp_path(".wkt")
        try:
            peaks_main(csv_path, output_name=out, peak_radius=1.5)
            with open(out) as f:
                result = shapely.wkt.loads(f.read().strip())
            assert len(result.geoms) == 2
        finally:
            os.unlink(csv_path)
            if os.path.exists(out):
                os.unlink(out)

    def test_multiple_csvs_concatenated(self):
        f1 = self._make_csv([(0.0, 0.0)])
        f2 = self._make_csv([(100.0, 100.0)])
        out = _tmp_path(".wkt")
        try:
            peaks_main(f1, f2, output_name=out, peak_radius=1.5)
            with open(out) as f:
                result = shapely.wkt.loads(f.read().strip())
            assert len(result.geoms) == 2
        finally:
            os.unlink(f1)
            os.unlink(f2)
            if os.path.exists(out):
                os.unlink(out)

    def test_no_csvs_raises_value_error(self):
        with pytest.raises(ValueError, match="At least one"):
            peaks_main(output_name="irrelevant.wkt")

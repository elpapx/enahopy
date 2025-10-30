"""
Tests for GeographicMerger Module
=================================

Tests for the lightweight geographic merger class.
"""

import logging
import unittest

import pandas as pd
import pytest

from enahopy.merger.config import GeoMergeConfiguration
from enahopy.merger.geographic.merger import GeographicMerger, ENAHOGeographicMerger


class TestGeographicMerger(unittest.TestCase):
    """Tests for GeographicMerger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.merger = GeographicMerger()

        # Sample data
        self.df_data = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "150103"],
                "conglome": ["001", "002", "003"],
                "ingreso": [2000, 1500, 1800],
            }
        )

        self.df_geo = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "150103"],
                "departamento": ["Lima", "Lima", "Lima"],
                "provincia": ["Lima", "Lima", "Lima"],
                "distrito": ["Lima", "Ancón", "Ate"],
            }
        )

    def test_init_default(self):
        """Should initialize with default config."""
        merger = GeographicMerger()
        self.assertIsNotNone(merger.config)
        self.assertIsNotNone(merger.logger)

    def test_init_custom_config(self):
        """Should initialize with custom config."""
        config = GeoMergeConfiguration(columna_union="codigo")
        merger = GeographicMerger(config=config)
        self.assertEqual(merger.config.columna_union, "codigo")

    def test_init_custom_logger(self):
        """Should initialize with custom logger."""
        logger = logging.getLogger("test_logger")
        merger = GeographicMerger(logger=logger)
        self.assertEqual(merger.logger.name, "test_logger")

    def test_merge_basic(self):
        """Should perform basic geographic merge."""
        result, report = self.merger.merge(self.df_data, self.df_geo)

        # Check result structure
        self.assertEqual(len(result), 3)
        self.assertIn("ubigeo", result.columns)
        self.assertIn("departamento", result.columns)
        self.assertIn("ingreso", result.columns)

        # Check report
        self.assertEqual(report["input_rows"], 3)
        self.assertEqual(report["geography_rows"], 3)
        self.assertEqual(report["output_rows"], 3)
        self.assertEqual(report["match_rate"], 100.0)

    def test_merge_custom_column(self):
        """Should merge using custom column name."""
        # Rename columns
        df_data = self.df_data.rename(columns={"ubigeo": "codigo"})
        df_geo = self.df_geo.rename(columns={"ubigeo": "codigo"})

        result, report = self.merger.merge(df_data, df_geo, columna_union="codigo")

        self.assertEqual(len(result), 3)
        self.assertIn("codigo", result.columns)
        self.assertIn("departamento", result.columns)

    def test_merge_partial_matches(self):
        """Should handle partial geographic matches."""
        # Geography data missing one UBIGEO
        df_partial_geo = pd.DataFrame(
            {"ubigeo": ["150101", "150102"], "departamento": ["Lima", "Lima"]}  # Missing 150103
        )

        result, report = self.merger.merge(self.df_data, df_partial_geo)

        # All input records should be present
        self.assertEqual(len(result), 3)

        # Record with missing geography should have NaN
        missing_geo = result[result["ubigeo"] == "150103"]
        self.assertTrue(pd.isna(missing_geo["departamento"].iloc[0]))

    def test_merge_no_matches(self):
        """Should handle case with no geographic matches."""
        df_no_match_geo = pd.DataFrame(
            {
                "ubigeo": ["080101", "080102"],  # Different UBIGEOs
                "departamento": ["Cusco", "Cusco"],
            }
        )

        result, report = self.merger.merge(self.df_data, df_no_match_geo)

        # All input records present
        self.assertEqual(len(result), 3)

        # All geography columns should be NaN
        self.assertTrue(result["departamento"].isna().all())

    def test_merge_duplicate_ubigeos_in_geography(self):
        """Should handle duplicate UBIGEOs in geography data."""
        df_dup_geo = pd.DataFrame(
            {
                "ubigeo": ["150101", "150101", "150102"],  # Duplicate 150101
                "departamento": ["Lima", "Lima", "Lima"],
                "detalle": ["A", "B", "C"],
            }
        )

        result, report = self.merger.merge(self.df_data, df_dup_geo)

        # Left join with duplicates creates extra rows
        self.assertGreaterEqual(len(result), 3)

    def test_merge_empty_geography(self):
        """Should handle empty geography DataFrame."""
        df_empty_geo = pd.DataFrame(columns=["ubigeo", "departamento"])

        result, report = self.merger.merge(self.df_data, df_empty_geo)

        # All input records preserved
        self.assertEqual(len(result), 3)
        self.assertEqual(report["geography_rows"], 0)

        # Geography columns should be NaN
        self.assertTrue(result["departamento"].isna().all())

    def test_merge_preserves_original_columns(self):
        """Should preserve all original columns from principal DataFrame."""
        result, report = self.merger.merge(self.df_data, self.df_geo)

        # Check original columns present
        for col in self.df_data.columns:
            self.assertIn(col, result.columns)

        # Check original values preserved
        self.assertEqual(result["ingreso"].tolist(), self.df_data["ingreso"].tolist())

    def test_merge_with_validate_parameter(self):
        """Should accept validate parameter (for API compatibility)."""
        # Parameter accepted but not currently used
        result, report = self.merger.merge(self.df_data, self.df_geo, validate=True)

        self.assertEqual(len(result), 3)

        result2, report2 = self.merger.merge(self.df_data, self.df_geo, validate=False)

        self.assertEqual(len(result2), 3)

    def test_merge_with_columnas_geograficas_parameter(self):
        """Should accept columnas_geograficas parameter (for API compatibility)."""
        # Parameter accepted but not currently used
        result, report = self.merger.merge(
            self.df_data, self.df_geo, columnas_geograficas=["departamento", "provincia"]
        )

        self.assertEqual(len(result), 3)


class TestGeographicMergerAlias(unittest.TestCase):
    """Tests for ENAHOGeographicMerger alias."""

    def test_alias_works(self):
        """Should work with ENAHOGeographicMerger alias."""
        merger = ENAHOGeographicMerger()
        self.assertIsInstance(merger, GeographicMerger)

    def test_alias_functionality(self):
        """Should have same functionality as GeographicMerger."""
        df_data = pd.DataFrame({"ubigeo": ["150101", "150102"], "value": [100, 200]})
        df_geo = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["Lima", "Lima"]})

        merger = ENAHOGeographicMerger()
        result, report = merger.merge(df_data, df_geo)

        self.assertEqual(len(result), 2)
        self.assertIn("departamento", result.columns)


class TestGeographicMergerEdgeCases(unittest.TestCase):
    """Edge case tests for GeographicMerger."""

    def setUp(self):
        """Set up test fixtures."""
        self.merger = GeographicMerger()

    def test_merge_with_nan_ubigeos(self):
        """Should handle NaN UBIGEOs in principal data."""
        df_data = pd.DataFrame({"ubigeo": ["150101", None, "150103"], "value": [100, 200, 300]})
        df_geo = pd.DataFrame({"ubigeo": ["150101", "150103"], "departamento": ["Lima", "Lima"]})

        result, report = self.merger.merge(df_data, df_geo)

        # Should preserve all rows
        self.assertEqual(len(result), 3)

    def test_merge_single_row(self):
        """Should handle single row DataFrames."""
        df_data = pd.DataFrame({"ubigeo": ["150101"], "value": [100]})
        df_geo = pd.DataFrame({"ubigeo": ["150101"], "departamento": ["Lima"]})

        result, report = self.merger.merge(df_data, df_geo)

        self.assertEqual(len(result), 1)
        self.assertEqual(report["input_rows"], 1)
        self.assertEqual(report["output_rows"], 1)

    def test_merge_large_geography(self):
        """Should handle large geography DataFrame."""
        # Create data with few UBIGEOs
        df_data = pd.DataFrame({"ubigeo": ["150101", "150102"], "value": [100, 200]})

        # Create large geography with many UBIGEOs
        ubigeos = [f"{i:06d}" for i in range(1, 1001)]
        df_geo = pd.DataFrame({"ubigeo": ubigeos, "departamento": ["Lima"] * 1000})

        result, report = self.merger.merge(df_data, df_geo)

        # Should only return matched rows (left join)
        self.assertEqual(len(result), 2)
        self.assertEqual(report["geography_rows"], 1000)

    def test_merge_with_numeric_ubigeos(self):
        """Should handle numeric UBIGEO codes."""
        df_data = pd.DataFrame({"ubigeo": [150101, 150102, 150103], "value": [100, 200, 300]})
        df_geo = pd.DataFrame(
            {"ubigeo": [150101, 150102, 150103], "departamento": ["Lima", "Lima", "Lima"]}
        )

        result, report = self.merger.merge(df_data, df_geo)

        self.assertEqual(len(result), 3)
        self.assertEqual(report["match_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()

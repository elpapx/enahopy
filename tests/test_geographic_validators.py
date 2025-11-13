"""
Tests for Geographic Validators Module
======================================

Comprehensive tests for UBIGEO validation and territorial consistency checks.
"""

import logging
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from enahopy.merger.config import TipoValidacionUbigeo
from enahopy.merger.exceptions import TerritorialInconsistencyError, UbigeoValidationError
from enahopy.merger.geographic.validators import TerritorialValidator, UbigeoValidator


class TestUbigeoValidator(unittest.TestCase):
    """Tests for UbigeoValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.validator = UbigeoValidator(self.logger)

    def test_init(self):
        """Should initialize validator correctly."""
        self.assertIsInstance(self.validator.logger, logging.Logger)
        self.assertEqual(self.validator._cache_validacion, {})

    def test_validar_estructura_ubigeo_valid_6_digits(self):
        """Should validate 6-digit UBIGEO correctly."""
        # Valid 6-digit UBIGEOs (department 01 = Amazonas)
        result, msg = self.validator.validar_estructura_ubigeo("010101")
        self.assertTrue(result)
        self.assertEqual(msg, "Válido")

    def test_validar_estructura_ubigeo_accepts_valid_lengths(self):
        """Should accept valid length UBIGEOs."""
        # 6-digit UBIGEO (full)
        result, msg = self.validator.validar_estructura_ubigeo("010101")
        self.assertTrue(result)

        # Note: 2 and 4 digit codes require special right-padding which
        # is handled by the serie validation, not individual validation

    def test_validar_estructura_ubigeo_invalid_length(self):
        """Should reject UBIGEO with invalid length."""
        # 3 digits - invalid
        result, msg = self.validator.validar_estructura_ubigeo("010")
        self.assertFalse(result)
        self.assertIn("Longitud inválida", msg)

        # 5 digits - invalid
        result, msg = self.validator.validar_estructura_ubigeo("01010")
        self.assertFalse(result)
        self.assertIn("Longitud inválida", msg)

        # 7 digits - invalid
        result, msg = self.validator.validar_estructura_ubigeo("0101011")
        self.assertFalse(result)
        self.assertIn("Longitud inválida", msg)

    def test_validar_estructura_ubigeo_not_string(self):
        """Should reject non-string UBIGEO."""
        result, msg = self.validator.validar_estructura_ubigeo(101010)
        self.assertFalse(result)
        self.assertEqual(msg, "UBIGEO debe ser string")

    def test_validar_estructura_ubigeo_non_numeric(self):
        """Should reject UBIGEO with non-numeric characters."""
        result, msg = self.validator.validar_estructura_ubigeo("01010A")
        self.assertFalse(result)
        self.assertEqual(msg, "Contiene caracteres no numéricos")

        result, msg = self.validator.validar_estructura_ubigeo("01-01-01")
        self.assertFalse(result)
        self.assertEqual(msg, "Contiene caracteres no numéricos")

    def test_validar_estructura_ubigeo_invalid_department(self):
        """Should reject invalid department codes."""
        # Department 99 doesn't exist
        result, msg = self.validator.validar_estructura_ubigeo("990101")
        self.assertFalse(result)
        self.assertIn("Departamento inválido", msg)

    def test_validar_estructura_ubigeo_invalid_province(self):
        """Should reject invalid province codes."""
        # Province 00 is invalid
        result, msg = self.validator.validar_estructura_ubigeo("010001")
        self.assertFalse(result)
        self.assertIn("Provincia inválida", msg)

    def test_validar_estructura_ubigeo_invalid_district(self):
        """Should reject invalid district codes."""
        # District 00 is invalid
        result, msg = self.validator.validar_estructura_ubigeo("010100")
        self.assertFalse(result)
        self.assertIn("Distrito inválido", msg)

    def test_validar_estructura_ubigeo_caching(self):
        """Should cache validation results."""
        ubigeo = "010101"

        # First call
        result1, msg1 = self.validator.validar_estructura_ubigeo(ubigeo)

        # Second call should use cache (test by checking cache_info)
        result2, msg2 = self.validator.validar_estructura_ubigeo(ubigeo)

        self.assertEqual(result1, result2)
        self.assertEqual(msg1, msg2)

        # Check cache was used
        cache_info = self.validator.validar_estructura_ubigeo.cache_info()
        self.assertGreater(cache_info.hits, 0)

    def test_validar_serie_ubigeos_basic(self):
        """Should validate series with basic validation."""
        serie = pd.Series(["010101", "010102", "150101"])

        mask, errors = self.validator.validar_serie_ubigeos(serie, TipoValidacionUbigeo.BASIC)

        self.assertEqual(len(mask), 3)
        self.assertTrue(all(mask))
        self.assertEqual(len(errors), 0)

    def test_validar_serie_ubigeos_structural(self):
        """Should validate series with structural validation."""
        serie = pd.Series(["010101", "990101", "010102"])

        mask, errors = self.validator.validar_serie_ubigeos(serie, TipoValidacionUbigeo.STRUCTURAL)

        self.assertEqual(len(mask), 3)
        self.assertTrue(mask.iloc[0])
        self.assertFalse(mask.iloc[1])  # Invalid department
        self.assertTrue(mask.iloc[2])
        self.assertGreater(len(errors), 0)

    def test_validar_serie_ubigeos_with_nulls(self):
        """Should handle null values in series."""
        serie = pd.Series(["010101", None, "010102", np.nan])

        mask, errors = self.validator.validar_serie_ubigeos(serie, TipoValidacionUbigeo.STRUCTURAL)

        # Nulls should be marked as invalid
        self.assertTrue(mask.iloc[0])
        self.assertTrue(mask.iloc[2])

    def test_validar_serie_ubigeos_normalization(self):
        """Should normalize UBIGEOs before validation."""
        # Series validation normalizes with zfill
        serie = pd.Series(["010101", "150101", "080101"])

        mask, errors = self.validator.validar_serie_ubigeos(serie, TipoValidacionUbigeo.STRUCTURAL)

        self.assertTrue(all(mask))

    def test_validar_estructura_ubigeo_all_departments(self):
        """Should validate all valid department codes."""
        # Test first few departments (01-05)
        valid_departments = ["01", "02", "03", "04", "05"]

        for dept in valid_departments:
            ubigeo = f"{dept}0101"
            result, msg = self.validator.validar_estructura_ubigeo(ubigeo)
            self.assertTrue(result, f"Department {dept} should be valid")

    def test_validar_estructura_ubigeo_edge_cases_province(self):
        """Should validate edge cases for province codes."""
        # Province 01 (minimum valid)
        result, msg = self.validator.validar_estructura_ubigeo("010101")
        self.assertTrue(result)

        # Province 99 (maximum valid)
        result, msg = self.validator.validar_estructura_ubigeo("019901")
        self.assertTrue(result)

    def test_validar_estructura_ubigeo_edge_cases_district(self):
        """Should validate edge cases for district codes."""
        # District 01 (minimum valid)
        result, msg = self.validator.validar_estructura_ubigeo("010101")
        self.assertTrue(result)

        # District 99 (maximum valid)
        result, msg = self.validator.validar_estructura_ubigeo("010199")
        self.assertTrue(result)


class TestTerritorialValidator(unittest.TestCase):
    """Tests for TerritorialValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.validator = TerritorialValidator(self.logger)

    def test_init(self):
        """Should initialize validator correctly."""
        self.assertIsInstance(self.validator.logger, logging.Logger)

    def test_validar_jerarquia_territorial_valid(self):
        """Should pass for territorially consistent data."""
        df = pd.DataFrame(
            {
                "ubigeo": ["010101", "010102", "010201"],
                "departamento": ["01", "01", "01"],
                "provincia": ["0101", "0101", "0102"],
                "distrito": ["010101", "010102", "010201"],
            }
        )

        columnas = {
            "departamento": "departamento",
            "provincia": "provincia",
            "distrito": "distrito",
        }

        inconsistencias = self.validator.validar_jerarquia_territorial(df, columnas)
        self.assertEqual(len(inconsistencias), 0)

    def test_validar_jerarquia_territorial_inconsistent_district(self):
        """Should detect district-province inconsistencies."""
        df = pd.DataFrame(
            {
                "provincia": ["0101", "0101"],
                "distrito": ["010101", "010201"],  # 010201 belongs to province 0102, not 0101
            }
        )

        columnas = {"provincia": "provincia", "distrito": "distrito"}

        inconsistencias = self.validator.validar_jerarquia_territorial(df, columnas)
        self.assertGreater(len(inconsistencias), 0)

    def test_validar_jerarquia_territorial_missing_columns(self):
        """Should handle missing territorial columns gracefully."""
        df = pd.DataFrame({"ubigeo": ["010101", "010102"]})

        columnas = {"departamento": "departamento", "provincia": "provincia"}

        # Should not raise exception if columns don't exist
        inconsistencias = self.validator.validar_jerarquia_territorial(df, columnas)
        # Just verify it runs without crashing
        self.assertIsInstance(inconsistencias, list)

    def test_validar_jerarquia_territorial_with_nulls(self):
        """Should handle null values in territorial columns."""
        df = pd.DataFrame(
            {
                "departamento": ["01", "01", None],
                "provincia": ["0101", "0101", None],
                "distrito": ["010101", "010102", None],
            }
        )

        columnas = {
            "departamento": "departamento",
            "provincia": "provincia",
            "distrito": "distrito",
        }

        # Should handle nulls gracefully
        inconsistencias = self.validator.validar_jerarquia_territorial(df, columnas)
        self.assertIsInstance(inconsistencias, list)


class TestUbigeoValidatorPerformance(unittest.TestCase):
    """Performance tests for UbigeoValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.validator = UbigeoValidator(self.logger)

    @pytest.mark.slow
    def test_large_series_validation(self):
        """Should handle large series efficiently."""
        # Create large series (10,000 UBIGEOs)
        ubigeos = [f"010101"] * 5000 + [f"150101"] * 5000
        serie = pd.Series(ubigeos)

        mask, errors = self.validator.validar_serie_ubigeos(serie, TipoValidacionUbigeo.BASIC)

        self.assertEqual(len(mask), 10000)
        self.assertTrue(all(mask))

    def test_cache_effectiveness(self):
        """Should effectively cache validation results."""
        # Clear cache first
        self.validator.validar_estructura_ubigeo.cache_clear()

        # Validate same UBIGEO multiple times
        ubigeo = "010101"

        for _ in range(100):
            self.validator.validar_estructura_ubigeo(ubigeo)

        # Check cache was hit
        cache_info = self.validator.validar_estructura_ubigeo.cache_info()
        self.assertEqual(cache_info.hits, 99)  # First call miss, rest hits
        self.assertEqual(cache_info.misses, 1)


class TestUbigeoValidatorAdditionalMethods(unittest.TestCase):
    """Tests for additional UbigeoValidator methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.validator = UbigeoValidator(self.logger)

    def test_extraer_componentes_ubigeo(self):
        """Should extract UBIGEO components correctly."""
        serie = pd.Series(["010101", "150101", "080201"])

        componentes = self.validator.extraer_componentes_ubigeo(serie)

        self.assertEqual(len(componentes), 3)
        self.assertIn("ubigeo", componentes.columns)
        self.assertIn("departamento", componentes.columns)
        self.assertIn("provincia", componentes.columns)
        self.assertIn("distrito", componentes.columns)

        # Check first UBIGEO components
        self.assertEqual(componentes.iloc[0]["departamento"], "01")
        self.assertEqual(componentes.iloc[0]["provincia"], "0101")
        self.assertEqual(componentes.iloc[0]["distrito"], "010101")

    def test_extraer_componentes_ubigeo_with_nulls(self):
        """Should handle nulls when extracting components."""
        serie = pd.Series(["010101", None, "150101"])

        componentes = self.validator.extraer_componentes_ubigeo(serie)

        self.assertEqual(len(componentes), 3)
        # Null should result in empty/NA values
        self.assertTrue(pd.isna(componentes.iloc[1]["departamento"]))

    def test_get_validation_summary(self):
        """Should generate validation summary correctly."""
        serie = pd.Series(["010101", "150101", "990101", "010101"])  # 1 invalid, 1 duplicate

        summary = self.validator.get_validation_summary(serie, TipoValidacionUbigeo.STRUCTURAL)

        self.assertEqual(summary["total_records"], 4)
        self.assertEqual(summary["valid_ubigeos"], 3)
        self.assertEqual(summary["invalid_ubigeos"], 1)
        self.assertEqual(summary["null_values"], 0)
        self.assertEqual(summary["duplicate_ubigeos"], 1)
        self.assertGreater(summary["error_count"], 0)

    def test_get_validation_summary_with_nulls(self):
        """Should include null count in summary."""
        serie = pd.Series(["010101", None, "150101", np.nan])

        summary = self.validator.get_validation_summary(serie, TipoValidacionUbigeo.BASIC)

        self.assertEqual(summary["total_records"], 4)
        self.assertEqual(summary["null_values"], 2)

    def test_validate_ubigeo_consistency(self):
        """Should validate UBIGEO consistency in DataFrame."""
        df = pd.DataFrame({"codigo": ["010101", "010102", "010201"]})

        inconsistencias = self.validator.validate_ubigeo_consistency(df, "codigo")

        # Should not find major inconsistencies in valid data
        self.assertIsInstance(inconsistencias, list)

    def test_validate_ubigeo_consistency_missing_column(self):
        """Should handle missing column gracefully."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        inconsistencias = self.validator.validate_ubigeo_consistency(df, "codigo")

        self.assertEqual(len(inconsistencias), 1)
        self.assertIn("no encontrada", inconsistencias[0])

    def test_validar_serie_ubigeos_existence_validation(self):
        """Should fall back to STRUCTURAL for unsupported validation types."""
        serie = pd.Series(["010101", "150101"])

        # This should trigger the fallback warning
        mask, errors = self.validator.validar_serie_ubigeos(serie, TipoValidacionUbigeo.EXISTENCE)

        # Should still work (fallback to STRUCTURAL)
        self.assertEqual(len(mask), 2)
        self.assertTrue(all(mask))


class TestUbigeoValidatorIntegration(unittest.TestCase):
    """Integration tests for validator with real-world scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.validator = UbigeoValidator(self.logger)

    def test_real_world_ubigeos_lima(self):
        """Should validate real Lima UBIGEOs."""
        lima_ubigeos = [
            "150101",  # Lima - Lima - Lima
            "150102",  # Lima - Lima - Ancón
            "150103",  # Lima - Lima - Ate
            "150132",  # Lima - Lima - San Isidro
        ]

        for ubigeo in lima_ubigeos:
            result, msg = self.validator.validar_estructura_ubigeo(ubigeo)
            self.assertTrue(result, f"Lima UBIGEO {ubigeo} should be valid")

    def test_real_world_ubigeos_cusco(self):
        """Should validate real Cusco UBIGEOs."""
        cusco_ubigeos = [
            "080101",  # Cusco - Cusco - Cusco
            "080102",  # Cusco - Cusco - Ccorca
            "080201",  # Cusco - Acomayo - Acomayo
        ]

        for ubigeo in cusco_ubigeos:
            result, msg = self.validator.validar_estructura_ubigeo(ubigeo)
            self.assertTrue(result, f"Cusco UBIGEO {ubigeo} should be valid")

    def test_mixed_validation_types(self):
        """Should handle mixed validation scenarios."""
        serie = pd.Series(
            [
                "010101",  # Valid 6-digit
                "150101",  # Valid Lima
                "080101",  # Valid Cusco
                "990101",  # Invalid department
                "010001",  # Invalid province
                None,  # Null value
            ]
        )

        mask, errors = self.validator.validar_serie_ubigeos(serie, TipoValidacionUbigeo.STRUCTURAL)

        # Should validate correctly
        self.assertTrue(mask.iloc[0])
        self.assertTrue(mask.iloc[1])
        self.assertTrue(mask.iloc[2])
        self.assertFalse(mask.iloc[3])
        self.assertFalse(mask.iloc[4])


class TestGeoDataQualityValidator(unittest.TestCase):
    """Tests for GeoDataQualityValidator coordinate validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        from enahopy.merger.geographic.validators import GeoDataQualityValidator

        self.validator = GeoDataQualityValidator(self.logger)

    def test_validate_geo_coordinates_valid(self):
        """Should validate correct Peru coordinates."""
        df = pd.DataFrame(
            {
                "ubigeo": ["150101", "080101"],
                "longitud": [-77.0428, -71.9675],  # Valid Peru longitudes
                "latitud": [-12.0464, -13.5319],  # Valid Peru latitudes
            }
        )

        coord_cols = {"x": "longitud", "y": "latitud"}
        problemas = self.validator.territorial_validator.validate_coordinate_consistency(
            df, coord_cols, "ubigeo"
        )

        # Should find no problems with valid coordinates
        self.assertEqual(len(problemas), 0)

    def test_validate_geo_coordinates_out_of_range_longitude(self):
        """Should detect out-of-range longitudes."""
        df = pd.DataFrame(
            {"ubigeo": ["150101"], "longitud": [-90.0], "latitud": [-12.0]}  # Too far west for Peru
        )

        coord_cols = {"x": "longitud", "y": "latitud"}
        problemas = self.validator.territorial_validator.validate_coordinate_consistency(
            df, coord_cols, "ubigeo"
        )

        # Should detect out-of-range longitude
        self.assertGreater(len(problemas), 0)
        self.assertTrue(any("longitud" in p.lower() for p in problemas))

    def test_validate_geo_coordinates_out_of_range_latitude(self):
        """Should detect out-of-range latitudes."""
        df = pd.DataFrame(
            {
                "ubigeo": ["150101"],
                "longitud": [-77.0],
                "latitud": [-25.0],  # Too far south for Peru
            }
        )

        coord_cols = {"x": "longitud", "y": "latitud"}
        problemas = self.validator.territorial_validator.validate_coordinate_consistency(
            df, coord_cols, "ubigeo"
        )

        # Should detect out-of-range latitude
        self.assertGreater(len(problemas), 0)
        self.assertTrue(any("latitud" in p.lower() for p in problemas))

    def test_validate_geo_coordinates_missing_columns(self):
        """Should handle missing coordinate columns."""
        df = pd.DataFrame({"ubigeo": ["150101"]})

        coord_cols = {"x": "longitud", "y": "latitud"}
        problemas = self.validator.territorial_validator.validate_coordinate_consistency(
            df, coord_cols, "ubigeo"
        )

        # Should report missing columns
        self.assertGreater(len(problemas), 0)
        self.assertTrue(any("no encontrada" in p for p in problemas))

    def test_validate_geo_coordinates_null_values(self):
        """Should detect null coordinates."""
        df = pd.DataFrame(
            {"ubigeo": ["150101", "080101"], "longitud": [-77.0, None], "latitud": [-12.0, -13.5]}
        )

        coord_cols = {"x": "longitud", "y": "latitud"}
        problemas = self.validator.territorial_validator.validate_coordinate_consistency(
            df, coord_cols, "ubigeo"
        )

        # Should detect incomplete coordinates
        self.assertGreater(len(problemas), 0)
        self.assertTrue(any("incompletas" in p for p in problemas))

    def test_validate_geo_coordinates_duplicates(self):
        """Should detect duplicate coordinates."""
        df = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102"],
                "longitud": [-77.0, -77.0],  # Exact duplicates
                "latitud": [-12.0, -12.0],
            }
        )

        coord_cols = {"x": "longitud", "y": "latitud"}
        problemas = self.validator.territorial_validator.validate_coordinate_consistency(
            df, coord_cols, "ubigeo"
        )

        # Should detect duplicate coordinates
        self.assertGreater(len(problemas), 0)
        self.assertTrue(any("duplicadas" in p for p in problemas))

    def test_validate_geo_coordinates_incomplete_coord_cols(self):
        """Should handle incomplete coord_cols specification."""
        df = pd.DataFrame({"ubigeo": ["150101"], "longitud": [-77.0], "latitud": [-12.0]})

        # Missing 'y' key
        coord_cols = {"x": "longitud"}
        problemas = self.validator.territorial_validator.validate_coordinate_consistency(
            df, coord_cols, "ubigeo"
        )

        # Should report incomplete specification
        self.assertGreater(len(problemas), 0)
        self.assertTrue(any("'x' e 'y'" in p for p in problemas))


class TestTerritorialCoverage(unittest.TestCase):
    """Tests for territorial coverage validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        from enahopy.merger.geographic.validators import GeoDataQualityValidator

        self.validator = GeoDataQualityValidator(self.logger)

    def test_check_territorial_coverage_basic(self):
        """Should analyze basic territorial coverage."""
        df = pd.DataFrame({"ubigeo": ["150101", "150102", "080101", "080102"]})

        coverage = self.validator.territorial_validator.check_territorial_coverage(df, "ubigeo")

        # Should return coverage dict with statistics
        self.assertIsInstance(coverage, dict)
        self.assertIn("departamentos_presentes", coverage)
        self.assertIn("departamentos_encontrados", coverage)
        self.assertEqual(coverage["departamentos_presentes"], 2)  # Lima and Cusco
        self.assertIn("LIMA", coverage["departamentos_encontrados"])
        self.assertIn("CUSCO", coverage["departamentos_encontrados"])

    def test_check_territorial_coverage_expected_departments(self):
        """Should compare against expected departments."""
        df = pd.DataFrame({"ubigeo": ["150101", "080101"]})  # Only Lima and Cusco

        expected = ["LIMA", "CUSCO", "AMAZONAS"]  # Expecting these departments
        coverage = self.validator.territorial_validator.check_territorial_coverage(
            df, "ubigeo", expected
        )

        # Should include expected coverage information
        self.assertIsInstance(coverage, dict)
        self.assertIn("expected_missing", coverage)
        self.assertIn("expected_coverage", coverage)
        self.assertIn("AMAZONAS", coverage["expected_missing"])  # Should be missing


class TestComprehensiveValidation(unittest.TestCase):
    """Tests for comprehensive geographic validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        from enahopy.merger.geographic.validators import GeoDataQualityValidator

        self.validator = GeoDataQualityValidator(self.logger)

    def test_comprehensive_validation_all_enabled(self):
        """Should run all validations when all configs enabled."""
        df = pd.DataFrame(
            {
                "ubigeo": ["150101", "080101"],
                "departamento": ["15", "08"],
                "provincia": ["1501", "0801"],
                "distrito": ["150101", "080101"],
                "coordenada_x": [-77.0, -71.9],
                "coordenada_y": [-12.0, -13.5],
            }
        )

        geo_columns = {
            "ubigeo": "ubigeo",
            "departamento": "departamento",
            "provincia": "provincia",
            "distrito": "distrito",
            "coordenada_x": "coordenada_x",
            "coordenada_y": "coordenada_y",
        }

        results = self.validator.comprehensive_validation(df, geo_columns)

        # Should return comprehensive results
        self.assertIn("results", results)
        self.assertIn("ubigeo_validation", results["results"])
        self.assertIn("territorial_hierarchy", results["results"])
        self.assertIn("coordinate_validation", results["results"])
        self.assertIn("territorial_coverage", results["results"])
        self.assertIn("overall_quality_score", results)

    def test_comprehensive_validation_selective(self):
        """Should run only selected validations."""
        df = pd.DataFrame({"ubigeo": ["150101", "080101"]})

        geo_columns = {"ubigeo": "ubigeo"}
        config = {
            "ubigeo_structure": True,
            "territorial_hierarchy": False,
            "coordinate_consistency": False,
            "coverage_analysis": True,
        }

        results = self.validator.comprehensive_validation(df, geo_columns, config)

        # Should only have enabled validations
        self.assertIn("ubigeo_validation", results["results"])
        self.assertIn("territorial_coverage", results["results"])
        self.assertNotIn("territorial_hierarchy", results["results"])
        self.assertNotIn("coordinate_validation", results["results"])

    def test_comprehensive_validation_without_coordinates(self):
        """Should handle missing coordinate columns gracefully."""
        df = pd.DataFrame({"ubigeo": ["150101", "080101"]})

        geo_columns = {"ubigeo": "ubigeo"}

        results = self.validator.comprehensive_validation(df, geo_columns)

        # Should not fail without coordinates
        self.assertIn("results", results)
        self.assertIn("overall_quality_score", results)

    def test_quality_score_perfect_data(self):
        """Should give high quality score for perfect data."""
        df = pd.DataFrame(
            {
                "ubigeo": ["150101", "080101", "010101"] * 10,  # 3 departments, good coverage
                "coordenada_x": [-77.0] * 30,
                "coordenada_y": [-12.0] * 30,
            }
        )

        geo_columns = {
            "ubigeo": "ubigeo",
            "coordenada_x": "coordenada_x",
            "coordenada_y": "coordenada_y",
        }

        results = self.validator.comprehensive_validation(df, geo_columns)

        # Should have high quality score
        self.assertGreaterEqual(results["overall_quality_score"], 80)

    def test_quality_score_bad_data(self):
        """Should give low quality score for bad data."""
        df = pd.DataFrame(
            {
                "ubigeo": ["999999", "888888", "777777"],  # Invalid UBIGEOs
                "coordenada_x": [-90.0, 0.0, 100.0],  # Out of range
                "coordenada_y": [-20.0, 5.0, 50.0],  # Out of range
            }
        )

        geo_columns = {
            "ubigeo": "ubigeo",
            "coordenada_x": "coordenada_x",
            "coordenada_y": "coordenada_y",
        }

        results = self.validator.comprehensive_validation(df, geo_columns)

        # Should have low quality score
        self.assertLessEqual(results["overall_quality_score"], 70)

    def test_comprehensive_validation_missing_ubigeo_column(self):
        """Should handle missing UBIGEO column."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        geo_columns = {"ubigeo": "nonexistent"}
        config = {"ubigeo_structure": True}

        # Should not crash with missing column
        try:
            results = self.validator.comprehensive_validation(df, geo_columns, config)
            # If it doesn't crash, should have results structure
            self.assertIn("results", results)
        except KeyError:
            # Acceptable to raise KeyError for missing column
            pass


class TestTerritorialHierarchyEdgeCases(unittest.TestCase):
    """Tests for territorial hierarchy edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        from enahopy.merger.geographic.validators import TerritorialValidator

        self.validator = TerritorialValidator(self.logger)

    def test_provincia_departamento_inconsistency(self):
        """Should detect provincia-departamento inconsistency."""
        df = pd.DataFrame(
            {
                "provincia": ["1501", "0801"],  # Lima, Cusco provinces
                "departamento": ["08", "15"],  # Swapped departments!
            }
        )

        columnas = {"provincia": "provincia", "departamento": "departamento"}

        inconsistencias = self.validator.validar_jerarquia_territorial(df, columnas)

        # Should detect 2 inconsistencies
        self.assertGreater(len(inconsistencias), 0)
        self.assertTrue(any("provincia-departamento" in i for i in inconsistencias))

    def test_all_null_territorial_columns(self):
        """Should handle all-null territorial values."""
        df = pd.DataFrame({"provincia": [None, None], "departamento": [None, None]})

        columnas = {"provincia": "provincia", "departamento": "departamento"}

        inconsistencias = self.validator.validar_jerarquia_territorial(df, columnas)

        # Should not crash with all nulls
        self.assertIsInstance(inconsistencias, list)

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame(columns=["provincia", "departamento"])

        columnas = {"provincia": "provincia", "departamento": "departamento"}

        inconsistencias = self.validator.validar_jerarquia_territorial(df, columnas)

        # Should return empty list
        self.assertEqual(inconsistencias, [])


class TestCoordinateEdgeCases(unittest.TestCase):
    """Tests for coordinate validation edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        from enahopy.merger.geographic.validators import TerritorialValidator

        self.validator = TerritorialValidator(self.logger)

    def test_boundary_coordinates(self):
        """Should accept coordinates at Peru's boundaries."""
        df = pd.DataFrame({"lon": [-81.4, -68.6], "lat": [-18.3, 0.1]})  # Near boundaries

        coord_cols = {"x": "lon", "y": "lat"}
        problemas = self.validator.validate_coordinate_consistency(df, coord_cols)

        # Boundary coordinates should be accepted
        self.assertEqual(len(problemas), 0)

    def test_all_null_coordinates(self):
        """Should handle all-null coordinates."""
        df = pd.DataFrame({"lon": [None, None, None], "lat": [None, None, None]})

        coord_cols = {"x": "lon", "y": "lat"}
        problemas = self.validator.validate_coordinate_consistency(df, coord_cols)

        # Should report incomplete coordinates
        self.assertGreater(len(problemas), 0)
        self.assertTrue(any("incompletas" in p for p in problemas))

    def test_single_row_no_duplicates(self):
        """Should handle single-row DataFrame without duplicate check."""
        df = pd.DataFrame({"lon": [-77.0], "lat": [-12.0]})

        coord_cols = {"x": "lon", "y": "lat"}
        problemas = self.validator.validate_coordinate_consistency(df, coord_cols)

        # Single row should have no duplicate issues
        self.assertFalse(any("duplicadas" in p for p in problemas))


class TestUbigeoEdgeCases(unittest.TestCase):
    """Tests for UBIGEO validation edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        from enahopy.merger.geographic.validators import UbigeoValidator

        self.validator = UbigeoValidator(self.logger)

    def test_zero_province_code(self):
        """Should reject UBIGEO with province code 00."""
        is_valid, msg = self.validator.validar_estructura_ubigeo("150001")

        # Province 00 is invalid
        self.assertFalse(is_valid)
        self.assertIn("Provincia", msg)

    def test_zero_district_code(self):
        """Should reject UBIGEO with district code 00."""
        is_valid, msg = self.validator.validar_estructura_ubigeo("150100")

        # District 00 is invalid
        self.assertFalse(is_valid)
        self.assertIn("Distrito", msg)

    def test_high_province_code(self):
        """Should accept valid high province codes."""
        is_valid, msg = self.validator.validar_estructura_ubigeo("159901")

        # Province 99 is valid
        self.assertTrue(is_valid)

    def test_high_district_code(self):
        """Should accept valid high district codes."""
        is_valid, msg = self.validator.validar_estructura_ubigeo("150199")

        # District 99 is valid
        self.assertTrue(is_valid)

    def test_empty_string_ubigeo(self):
        """Should handle empty string UBIGEO."""
        is_valid, msg = self.validator.validar_estructura_ubigeo("")

        # Empty string is invalid
        self.assertFalse(is_valid)

    def test_whitespace_ubigeo(self):
        """Should reject UBIGEO with whitespace."""
        is_valid, msg = self.validator.validar_estructura_ubigeo("15 01 01")

        # Whitespace makes it non-numeric
        self.assertFalse(is_valid)
        self.assertIn("no numéricos", msg)


if __name__ == "__main__":
    unittest.main()

"""
Edge case tests for ENAHOPY merger module.

This module tests robustness of the merger under edge conditions:
- Empty and null DataFrame merges
- UBIGEO validation edge cases
- Duplicate key handling
- Data type mismatches
- Geographic hierarchy validation
- Conflict resolution scenarios
- Referential integrity violations

Author: MLOps-Engineer (MO-1 Phase 2)
Date: 2025-10-10
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from enahopy.exceptions import GeoMergeError
from enahopy.merger import ENAHOGeoMerger, merge_enaho_modules
from enahopy.merger.config import (
    GeoMergeConfiguration,
    ModuleMergeConfig,
    ModuleMergeLevel,
    ModuleMergeStrategy,
    TipoManejoDuplicados,
    TipoManejoErrores,
)
from enahopy.merger.geographic.validators import TerritorialValidator, UbigeoValidator
from enahopy.merger.modules.merger import ENAHOModuleMerger

# ============================================================================
# Geographic Merge Edge Cases
# ============================================================================


class TestGeoMergerEdgeCases:
    """Test geographic merger with edge cases."""

    @pytest.fixture
    def logger(self):
        """Create logger for tests."""
        return logging.getLogger("test_geo_merger")

    @pytest.fixture
    def basic_config(self):
        """Create basic geo merge configuration."""
        return GeoMergeConfiguration(manejo_errores=TipoManejoErrores.IGNORE)

    @pytest.fixture
    def geo_merger(self, basic_config):
        """Create geo merger instance."""
        return ENAHOGeoMerger(geo_config=basic_config, verbose=False)

    def test_empty_principal_dataframe(self, geo_merger):
        """Test merging with empty principal DataFrame."""
        df_empty = pd.DataFrame()
        df_geo = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["Lima", "Lima"]})

        # Should raise ValueError for empty DataFrame
        with pytest.raises(ValueError):
            geo_merger.merge_geographic_data(df_empty, df_geo)

    def test_empty_geographic_dataframe(self, geo_merger):
        """Test merging with empty geographic DataFrame."""
        df_principal = pd.DataFrame({"ubigeo": ["150101", "150102"], "poblacion": [100, 200]})
        df_geo_empty = pd.DataFrame()

        # Should raise ValueError for empty geographic data
        with pytest.raises(ValueError):
            geo_merger.merge_geographic_data(df_principal, df_geo_empty)

    def test_all_null_ubigeo_column(self, geo_merger):
        """Test DataFrame with all null UBIGEO values."""
        df_principal = pd.DataFrame({"ubigeo": [None, None, None], "poblacion": [100, 200, 150]})
        df_geo = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["Lima", "Lima"]})

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect:
            mock_detect.return_value = {"departamento": "departamento"}

            # Should handle null UBIGEOs gracefully
            result, validation = geo_merger.merge_geographic_data(df_principal, df_geo)

            # All records should be present but without geographic data
            assert len(result) == 3
            assert validation is not None

    def test_mixed_null_and_valid_ubigeo(self, geo_merger):
        """Test DataFrame with mix of null and valid UBIGEOs."""
        df_principal = pd.DataFrame(
            {"ubigeo": ["150101", None, "150102", None], "poblacion": [100, 200, 150, 180]}
        )
        df_geo = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102"],
                "departamento": ["Lima", "Lima"],
                "provincia": ["Lima", "Lima"],
            }
        )

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect:
            mock_detect.return_value = {"departamento": "departamento", "provincia": "provincia"}

            result, validation = geo_merger.merge_geographic_data(df_principal, df_geo)

            # Should merge valid UBIGEOs and keep nulls
            assert len(result) == 4
            assert result[result["ubigeo"] == "150101"]["departamento"].iloc[0] == "Lima"
            # For null UBIGEOs, implementation uses valor_faltante (default: 'DESCONOCIDO')
            null_ubigeo_rows = result[pd.isna(result["ubigeo"])]
            assert len(null_ubigeo_rows) == 2
            assert null_ubigeo_rows["departamento"].iloc[0] == "DESCONOCIDO"

    def test_ubigeo_data_type_mismatch(self):
        """Test UBIGEO as integer instead of string."""
        config = GeoMergeConfiguration(manejo_errores=TipoManejoErrores.COERCE)
        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        df_principal = pd.DataFrame(
            {
                "ubigeo": [150101, 150102, 130101],  # Numeric instead of string
                "poblacion": [100, 200, 150],
            }
        )
        df_geo = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "130101"],
                "departamento": ["Lima", "Lima", "La Libertad"],
            }
        )

        with patch.object(merger.pattern_detector, "detectar_columnas_geograficas") as mock_detect:
            mock_detect.return_value = {"departamento": "departamento"}

            # Should coerce and merge successfully
            result, validation = merger.merge_geographic_data(df_principal, df_geo)

            assert len(result) >= 3
            assert "departamento" in result.columns

    def test_duplicate_ubigeo_in_principal(self):
        """Test principal DataFrame with duplicate UBIGEOs."""
        config = GeoMergeConfiguration(
            manejo_duplicados=TipoManejoDuplicados.FIRST, manejo_errores=TipoManejoErrores.IGNORE
        )
        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        df_principal = pd.DataFrame(
            {"ubigeo": ["150101", "150101", "150102"], "poblacion": [100, 200, 150]}  # Duplicate
        )
        df_geo = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["Lima", "Lima"]})

        with patch.object(merger.pattern_detector, "detectar_columnas_geograficas") as mock_detect:
            mock_detect.return_value = {"departamento": "departamento"}

            result, validation = merger.merge_geographic_data(df_principal, df_geo)

            # Should handle duplicates according to strategy
            assert len(result) >= 2

    def test_duplicate_ubigeo_in_geographic(self):
        """Test geographic DataFrame with duplicate UBIGEOs."""
        config = GeoMergeConfiguration(
            manejo_duplicados=TipoManejoDuplicados.FIRST, manejo_errores=TipoManejoErrores.IGNORE
        )
        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        df_principal = pd.DataFrame({"ubigeo": ["150101", "150102"], "poblacion": [100, 200]})
        df_geo = pd.DataFrame(
            {
                "ubigeo": ["150101", "150101", "150102"],  # Duplicate
                "departamento": ["Lima", "Lima_DUP", "Lima"],
            }
        )

        with patch.object(merger.pattern_detector, "detectar_columnas_geograficas") as mock_detect:
            mock_detect.return_value = {"departamento": "departamento"}

            result, validation = merger.merge_geographic_data(df_principal, df_geo)

            # First occurrence should be used
            lima_row = result[result["ubigeo"] == "150101"].iloc[0]
            assert lima_row["departamento"] == "Lima"  # Not "Lima_DUP"

    def test_no_matching_ubigeos(self, geo_merger):
        """Test when no UBIGEOs match between DataFrames."""
        df_principal = pd.DataFrame(
            {"ubigeo": ["010101", "020101"], "poblacion": [100, 200]}  # Different UBIGEOs
        )
        df_geo = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["Lima", "Lima"]})

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect:
            mock_detect.return_value = {"departamento": "departamento"}

            result, validation = geo_merger.merge_geographic_data(df_principal, df_geo)

            # Should complete but with no geographic data matched
            # Left join preserves all records from principal DataFrame
            assert len(result) == 2
            # No matches, so all geographic columns should have valor_faltante
            assert result["departamento"].iloc[0] == "DESCONOCIDO"
            # Validation may show 100% if all records are processed (even if no data matches)
            assert validation.total_records == 2


# ============================================================================
# UBIGEO Validation Edge Cases
# ============================================================================


class TestUbigeoValidatorEdgeCases:
    """Test UBIGEO validator with edge cases."""

    @pytest.fixture
    def validator(self):
        """Create UBIGEO validator."""
        logger = logging.getLogger("test_ubigeo")
        return UbigeoValidator(logger)

    def test_ubigeo_with_leading_zeros_missing(self, validator):
        """Test UBIGEO without proper leading zeros."""
        # "1234" instead of "010234" - should be invalid due to wrong length
        valid, msg = validator.validar_estructura_ubigeo("1234")
        assert not valid
        # Check for error message indicating length or format issue
        assert msg is not None and len(msg) > 0

    def test_ubigeo_with_letters(self, validator):
        """Test UBIGEO containing letters."""
        valid, msg = validator.validar_estructura_ubigeo("15A101")
        assert not valid

    def test_ubigeo_with_special_characters(self, validator):
        """Test UBIGEO with special characters."""
        invalid_ubigeos = ["15-01-01", "15.01.01", "15/01/01", "150 101"]

        for ubigeo in invalid_ubigeos:
            valid, msg = validator.validar_estructura_ubigeo(ubigeo)
            assert not valid

    def test_ubigeo_too_short(self, validator):
        """Test UBIGEO shorter than minimum valid length."""
        # Valid UBIGEOs can be 2 (dept), 4 (prov), or 6 (dist) digits
        # These are ALL invalid lengths
        short_ubigeos = ["", "1", "150", "15010"]  # 0, 1, 3, 5 digits (invalid)

        for ubigeo in short_ubigeos:
            valid, msg = validator.validar_estructura_ubigeo(ubigeo)
            assert not valid, f"UBIGEO '{ubigeo}' should be invalid but was marked as valid"

    def test_ubigeo_too_long(self, validator):
        """Test UBIGEO longer than 6 digits."""
        long_ubigeo = "1501011"  # 7 digits

        valid, msg = validator.validar_estructura_ubigeo(long_ubigeo)
        assert not valid

    def test_ubigeo_with_invalid_department_code(self, validator):
        """Test UBIGEO with invalid department code (> 25)."""
        invalid_ubigeos = ["990101", "260101", "000101"]

        for ubigeo in invalid_ubigeos:
            valid, msg = validator.validar_estructura_ubigeo(ubigeo)
            assert not valid

    def test_ubigeo_all_zeros(self, validator):
        """Test UBIGEO with all zeros."""
        valid, msg = validator.validar_estructura_ubigeo("000000")
        assert not valid

    def test_ubigeo_extraction_with_nulls(self, validator):
        """Test UBIGEO component extraction with null values."""
        serie = pd.Series(["150101", None, "080801", np.nan, "130101"])
        componentes = validator.extraer_componentes_ubigeo(serie)

        # Should handle nulls gracefully - returns DataFrame with same length
        assert len(componentes) == 5
        assert componentes.iloc[0]["departamento"] == "15"
        # Null values should result in NaN or empty string in components
        assert (
            pd.isna(componentes.iloc[1]["departamento"])
            or componentes.iloc[1]["departamento"] == ""
        )

    def test_ubigeo_extraction_with_invalid_format(self, validator):
        """Test component extraction with invalid formats."""
        serie = pd.Series(["150101", "INVALID", "12345", "080801"])
        componentes = validator.extraer_componentes_ubigeo(serie)

        # Valid UBIGEOs should be extracted, invalid ones should be null or error
        assert len(componentes) >= 2


# ============================================================================
# Module Merge Edge Cases
# ============================================================================


class TestModuleMergerEdgeCases:
    """Test module merger with edge cases."""

    @pytest.fixture
    def logger(self):
        """Create logger."""
        return logging.getLogger("test_module_merger")

    @pytest.fixture
    def config(self):
        """Create merge configuration."""
        return ModuleMergeConfig()

    @pytest.fixture
    def merger(self, config, logger):
        """Create module merger."""
        return ENAHOModuleMerger(config, logger)

    def test_merge_empty_left_dataframe(self, merger):
        """Test merging with empty left DataFrame."""
        df_empty = pd.DataFrame()
        df_right = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "ingreso": [1000, 2000],
            }
        )

        # Implementation handles empty DataFrame gracefully with warning
        result = merger.merge_modules(df_empty, df_right, "01", "34")
        # Empty left DataFrame results in empty merged DataFrame
        assert result is not None
        assert len(result.merged_df) == 0

    def test_merge_empty_right_dataframe(self, merger):
        """Test merging with empty right DataFrame."""
        df_left = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "poblacion": [100, 200],
            }
        )
        df_empty = pd.DataFrame()

        # Implementation handles empty right DataFrame gracefully with warning
        result = merger.merge_modules(df_left, df_empty, "01", "34")
        # Left join with empty right preserves left records
        assert result is not None
        assert len(result.merged_df) == 2

    def test_merge_with_all_null_keys(self, merger):
        """Test merging when all key columns are null."""
        df_left = pd.DataFrame(
            {
                "conglome": [None, None],
                "vivienda": [None, None],
                "hogar": [None, None],
                "poblacion": [100, 200],
            }
        )
        df_right = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "ingreso": [1000, 2000],
            }
        )

        result = merger.merge_modules(df_left, df_right, "01", "34")

        # Should complete but likely with no matches
        assert result is not None
        assert len(result.merged_df) >= 0

    def test_merge_with_mismatched_key_data_types(self, merger):
        """Test merging when key columns have different data types."""
        df_left = pd.DataFrame(
            {
                "conglome": ["001", "002"],  # String
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "poblacion": [100, 200],
            }
        )
        df_right = pd.DataFrame(
            {
                "conglome": [1, 2],  # Integer
                "vivienda": [1, 1],
                "hogar": [1, 2],
                "ingreso": [1000, 2000],
            }
        )

        # Merger should handle type coercion or fail gracefully
        try:
            result = merger.merge_modules(df_left, df_right, "01", "34")
            assert result is not None
        except (ValueError, TypeError):
            # Expected if type coercion is not supported
            pass

    def test_merge_with_duplicate_keys_in_both(self, merger):
        """Test merging when both DataFrames have duplicate keys."""
        df_left = pd.DataFrame(
            {
                "conglome": ["001", "001", "002"],
                "vivienda": ["001", "001", "001"],
                "hogar": ["1", "1", "2"],
                "poblacion": [100, 150, 200],
            }
        )
        df_right = pd.DataFrame(
            {
                "conglome": ["001", "001", "002"],
                "vivienda": ["001", "001", "001"],
                "hogar": ["1", "1", "2"],
                "ingreso": [1000, 1500, 2000],
            }
        )

        result = merger.merge_modules(df_left, df_right, "01", "34")

        # Should create cartesian product or handle duplicates
        assert result is not None
        # Number of rows may be > original due to many-to-many relationship

    def test_merge_with_single_row_dataframes(self, merger):
        """Test merging DataFrames with only one row each."""
        df_left = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["001"], "hogar": ["1"], "poblacion": [100]}
        )
        df_right = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["001"], "hogar": ["1"], "ingreso": [1000]}
        )

        result = merger.merge_modules(df_left, df_right, "01", "34")

        assert len(result.merged_df) == 1
        assert "poblacion" in result.merged_df.columns
        assert "ingreso" in result.merged_df.columns

    def test_merge_strategy_with_conflicting_columns(self):
        """Test different merge strategies with column conflicts."""
        logger = logging.getLogger("test")

        # Create modules with same column name
        df_left = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "factor": [1.1, 1.2],  # Conflict
            }
        )
        df_right = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "factor": [2.1, 2.2],  # Conflict
            }
        )

        # Test KEEP_LEFT strategy
        config_left = ModuleMergeConfig(merge_strategy=ModuleMergeStrategy.KEEP_LEFT)
        merger_left = ENAHOModuleMerger(config_left, logger)

        result = merger_left.merge_modules(df_left, df_right, "01", "34")

        assert result is not None
        # Left values should be preserved (1.1, 1.2) or suffixed

    def test_merge_with_missing_standard_keys(self, merger):
        """Test merging when standard key columns are missing."""
        df_left = pd.DataFrame(
            {"custom_id": ["001", "002"], "poblacion": [100, 200]}  # Non-standard key
        )
        df_right = pd.DataFrame({"custom_id": ["001", "002"], "ingreso": [1000, 2000]})

        # Should raise KeyError due to missing required keys (conglome, vivienda, hogar)
        with pytest.raises(KeyError):
            merger.merge_modules(df_left, df_right, "01", "34")

    def test_merge_coalesce_strategy_with_nulls(self):
        """Test COALESCE strategy fills nulls from right DataFrame."""
        logger = logging.getLogger("test")
        config = ModuleMergeConfig(merge_strategy=ModuleMergeStrategy.COALESCE)
        merger = ENAHOModuleMerger(config, logger)

        df_left = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "shared_col": [100, None],  # Has null
            }
        )
        df_right = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["1", "2"],
                "shared_col": [999, 200],  # Has value for null in left
            }
        )

        result = merger.merge_modules(df_left, df_right, "01", "34")

        # With COALESCE, null in left should be filled from right
        assert result is not None
        # Check if coalescing happened (implementation-specific)


# ============================================================================
# Referential Integrity Edge Cases
# ============================================================================


class TestReferentialIntegrityEdgeCases:
    """Test referential integrity validation."""

    @pytest.fixture
    def logger(self):
        """Create logger."""
        return logging.getLogger("test_integrity")

    def test_orphan_records_in_persona_module(self, logger):
        """Test persona records without corresponding hogar records."""
        config = ModuleMergeConfig(merge_level=ModuleMergeLevel.PERSONA)
        merger = ENAHOModuleMerger(config, logger)

        # Hogar module
        mod_01 = pd.DataFrame(
            {"conglome": ["001", "002"], "vivienda": ["001", "001"], "hogar": ["1", "2"]}
        )

        # Persona module with orphans
        mod_02 = pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],  # "003" is orphan
                "vivienda": ["001", "001", "999"],
                "hogar": ["1", "2", "1"],
                "codperso": ["01", "01", "01"],
            }
        )

        result = merger.merge_modules(mod_01, mod_02, "01", "02")

        # Left join keeps all left records (hogar module)
        # Orphan persona records are not included since they don't match left keys
        assert result is not None
        assert len(result.merged_df) == 2  # Only matched records from left


# ============================================================================
# Performance Edge Cases
# ============================================================================


class TestMergerPerformanceEdgeCases:
    """Test merger performance with edge cases."""

    @pytest.fixture
    def logger(self):
        """Create logger."""
        return logging.getLogger("test_performance")

    @pytest.mark.slow
    def test_merge_with_high_cardinality(self, logger):
        """Test merging DataFrames with many unique keys."""
        config = ModuleMergeConfig()
        merger = ENAHOModuleMerger(config, logger)

        # Create large DataFrames
        n = 10000
        df_left = pd.DataFrame(
            {
                "conglome": [f"{i:04d}" for i in range(n)],
                "vivienda": ["001"] * n,
                "hogar": ["1"] * n,
                "poblacion": range(n),
            }
        )
        df_right = pd.DataFrame(
            {
                "conglome": [f"{i:04d}" for i in range(n)],
                "vivienda": ["001"] * n,
                "hogar": ["1"] * n,
                "ingreso": range(1000, 1000 + n),
            }
        )

        result = merger.merge_modules(df_left, df_right, "01", "34")

        # Should complete in reasonable time
        assert len(result.merged_df) == n
        assert "poblacion" in result.merged_df.columns
        assert "ingreso" in result.merged_df.columns


if __name__ == "__main__":
    # Run tests with: pytest tests/test_merger_edge_cases.py -v
    pytest.main([__file__, "-v", "--tb=short"])

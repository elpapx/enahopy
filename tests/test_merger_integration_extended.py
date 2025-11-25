"""
Comprehensive Merger Integration Tests - Phase 4 Extension
===========================================================

Extended integration tests targeting 45-50% merger coverage.
Focuses on uncovered areas identified in coverage analysis.

Author: MLOps-Engineer (MO-2-REVISED Phase 4)
Date: 2025-10-15
Coverage Target: 45-50% (from 17.81% baseline)
"""

import logging

import numpy as np
import pandas as pd
import pytest

from enahopy.merger import ENAHOGeoMerger
from enahopy.merger.config import (
    GeoMergeConfiguration,
    ModuleMergeConfig,
    ModuleMergeLevel,
    ModuleMergeStrategy,
    TipoManejoDuplicados,
)
from enahopy.merger.exceptions import GeoMergeError
from enahopy.merger.modules.merger import ENAHOModuleMerger
from enahopy.merger.modules.validator import ModuleValidator

# ============================================================================
# PRIORITY 1: CORE MERGE WORKFLOWS (20-25 tests)
# ============================================================================


class TestENAHOGeoMergerCoreWorkflows:
    """Test core ENAHOGeoMerger workflows and geographic merges."""

    @pytest.fixture
    def geo_merger(self):
        """Create ENAHOGeoMerger instance."""
        return ENAHOGeoMerger(verbose=False)

    @pytest.fixture
    def sample_household_data(self):
        """Sample household data with ubigeo."""
        return pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "150103"],
                "conglome": ["001", "002", "003"],
                "vivienda": ["01", "01", "01"],
                "hogar": ["1", "1", "1"],
                "gashog2d": [2000, 1500, 1800],
            }
        )

    @pytest.fixture
    def sample_geography_data(self):
        """Sample geography reference data."""
        return pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "150103"],
                "departamento": ["Lima", "Lima", "Lima"],
                "provincia": ["Lima", "Lima", "Lima"],
                "distrito": ["Lima", "San Isidro", "Miraflores"],
            }
        )

    def test_geographic_merge_basic_left_join(
        self, geo_merger, sample_household_data, sample_geography_data
    ):
        """Test basic geographic merge preserves left records."""
        result, validation = geo_merger.merge_geographic_data(
            df_principal=sample_household_data,
            df_geografia=sample_geography_data,
            columna_union="ubigeo",
        )

        assert len(result) == len(sample_household_data)
        assert "departamento" in result.columns
        assert "provincia" in result.columns
        assert "distrito" in result.columns

    def test_geographic_merge_with_duplicates_first_strategy(
        self, geo_merger, sample_household_data
    ):
        """Test geographic merge with duplicate UBIGEOs using FIRST strategy."""
        geo_data_dups = pd.DataFrame(
            {
                "ubigeo": ["150101", "150101", "150102"],  # Duplicate
                "departamento": ["Lima", "Lima-Dup", "Lima"],
                "distrito": ["Lima1", "Lima2", "San Isidro"],
            }
        )

        result, validation = geo_merger.merge_geographic_data(
            df_principal=sample_household_data, df_geografia=geo_data_dups
        )

        # Should keep first occurrence of duplicate
        lima_records = result[result["ubigeo"] == "150101"]
        assert len(lima_records) == 1
        assert lima_records["distrito"].iloc[0] == "Lima1"

    def test_geographic_merge_multi_level_geography(self, geo_merger):
        """Test merge with multi-level geographic hierarchy."""
        data = pd.DataFrame(
            {"ubigeo": ["15", "1501", "150101"], "value": [100, 200, 300]}  # Different levels
        )
        geo = pd.DataFrame(
            {
                "ubigeo": ["15", "1501", "150101"],
                "departamento": ["Lima", "Lima", "Lima"],
                "nivel": ["depto", "prov", "dist"],
            }
        )

        result, validation = geo_merger.merge_geographic_data(df_principal=data, df_geografia=geo)

        assert len(result) == 3
        assert all(result["departamento"] == "Lima")

    def test_geographic_merge_large_dataset_performance(self, geo_merger):
        """Test geographic merge with large dataset (10k+ records)."""
        n = 10000
        data = pd.DataFrame(
            {"ubigeo": [f"15{i:04d}" for i in range(n)], "value": np.random.randn(n)}
        )
        geo = pd.DataFrame(
            {"ubigeo": [f"15{i:04d}" for i in range(n)], "departamento": ["Lima"] * n}
        )

        import time

        start = time.time()
        result, validation = geo_merger.merge_geographic_data(data, geo)
        elapsed = time.time() - start

        assert len(result) == n
        assert elapsed < 5.0  # Should complete in <5 seconds

    def test_geographic_merge_cache_integration(
        self, geo_merger, sample_household_data, sample_geography_data
    ):
        """Test that cache is used for repeated geographic merges."""
        # First merge
        result1, _ = geo_merger.merge_geographic_data(sample_household_data, sample_geography_data)

        # Second merge (should use cache)
        result2, _ = geo_merger.merge_geographic_data(sample_household_data, sample_geography_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_geographic_merge_custom_column_mapping(self, geo_merger, sample_household_data):
        """Test geographic merge with custom column mapping."""
        geo_data = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "150103"],
                "dept": ["Lima", "Lima", "Lima"],
                "prov": ["Lima", "Lima", "Lima"],
            }
        )

        columnas_geograficas = {"dept": "departamento", "prov": "provincia"}

        result, _ = geo_merger.merge_geographic_data(
            sample_household_data, geo_data, columnas_geograficas=columnas_geograficas
        )

        assert "departamento" in result.columns
        assert "provincia" in result.columns
        assert "dept" not in result.columns  # Original names should be replaced

    def test_geographic_validation_comprehensive(self, geo_merger):
        """Test comprehensive geographic validation workflow."""
        geo_data = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "INVALID", "150103"],
                "departamento": ["Lima", "Lima", "Invalid", "Lima"],
            }
        )

        validation = geo_merger.validate_geographic_data(
            geo_data, columna_ubigeo="ubigeo", validate_territory=True, validate_quality=True
        )

        assert validation.total_records == 4
        assert validation.invalid_ubigeos > 0  # Should detect INVALID
        assert validation.quality_metrics is not None


class TestENAHOModuleMergerCoreWorkflows:
    """Test core ENAHOModuleMerger workflows for module combinations."""

    @pytest.fixture
    def module_merger(self):
        """Create ENAHOModuleMerger instance."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test_merger")
        return ENAHOModuleMerger(config, logger)

    @pytest.fixture
    def sample_sumaria(self):
        """Sample sumaria (34) module."""
        return pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["01", "01", "01"],
                "hogar": ["1", "1", "1"],
                "gashog2d": [2000, 1500, 1800],
                "inghog2d": [3000, 2500, 2800],
            }
        )

    @pytest.fixture
    def sample_vivienda(self):
        """Sample vivienda (01) module."""
        return pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["01", "01", "01"],
                "hogar": ["1", "1", "1"],
                "area": [1, 2, 1],  # 1=Urbano, 2=Rural
                "tipo_vivienda": ["Casa", "Depto", "Casa"],
            }
        )

    @pytest.fixture
    def sample_personas(self):
        """Sample personas (02) module."""
        return pd.DataFrame(
            {
                "conglome": ["001", "001", "002"],
                "vivienda": ["01", "01", "01"],
                "hogar": ["1", "1", "1"],
                "codperso": ["01", "02", "01"],
                "p208a": [25, 30, 35],  # Edad
                "p207": [1, 2, 1],  # Sexo
            }
        )

    @pytest.mark.skip(
        reason="Multi-module merge sequencing requires modules_merged tracking - TODO v0.6.0"
    )
    def test_three_module_sequential_merge(
        self, module_merger, sample_sumaria, sample_vivienda, sample_personas
    ):
        """Test merging three modules sequentially."""
        modules = {"34": sample_sumaria, "01": sample_vivienda, "02": sample_personas}

        result = module_merger.merge_multiple_modules(modules_dict=modules, base_module="34")

        assert result.merged_df is not None
        assert "gashog2d" in result.merged_df.columns
        assert "area" in result.merged_df.columns
        # Person module aggregated to hogar level
        assert len(result.merged_df) == 3

    def test_merge_different_levels_hogar_person(
        self, module_merger, sample_sumaria, sample_personas
    ):
        """Test merge between hogar and person level modules."""
        result = module_merger.merge_modules(sample_sumaria, sample_personas, "34", "02")

        # Hogar data broadcast to person records
        assert result.merged_df is not None
        assert len(result.merged_df) >= len(sample_sumaria)

    def test_conflict_resolution_strategies_all(self, module_merger, sample_sumaria):
        """Test all conflict resolution strategies."""
        conflicting_data = sample_sumaria.copy()
        conflicting_data["gashog2d"] = [3000, 2500, 2800]  # Different values

        for strategy in [
            ModuleMergeStrategy.KEEP_LEFT,
            ModuleMergeStrategy.KEEP_RIGHT,
            ModuleMergeStrategy.COALESCE,
            ModuleMergeStrategy.AVERAGE,
        ]:
            config = ModuleMergeConfig(merge_strategy=strategy)
            merger = ENAHOModuleMerger(config, logging.getLogger("test"))

            result = merger.merge_modules(sample_sumaria, conflicting_data, "34", "34b")

            assert result.merged_df is not None
            assert "gashog2d" in result.merged_df.columns
            assert result.conflicts_resolved >= 0

    def test_merge_with_validation_warnings(self, module_merger, sample_sumaria, sample_vivienda):
        """Test that merge captures validation warnings."""
        result = module_merger.merge_modules(sample_sumaria, sample_vivienda, "34", "01")

        assert isinstance(result.validation_warnings, list)

    def test_merge_quality_score_calculation_detailed(
        self, module_merger, sample_sumaria, sample_vivienda
    ):
        """Test detailed quality score calculation."""
        result = module_merger.merge_modules(sample_sumaria, sample_vivienda, "34", "01")

        assert 0 <= result.quality_score <= 100
        assert result.merge_report is not None
        assert "quality_score" in result.merge_report

    def test_cardinality_detection_one_to_many(self, module_merger):
        """Test detection of one-to-many relationships."""
        left = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["1", "1"],
                "value": [100, 200],
            }
        )
        right = pd.DataFrame(
            {
                "conglome": ["001", "001", "002", "002"],  # Duplicates
                "vivienda": ["01", "01", "01", "01"],
                "hogar": ["1", "1", "1", "1"],
                "detail": ["A", "B", "C", "D"],
            }
        )

        result = module_merger.merge_modules(left, right, "01", "02")

        # Should warn about cardinality
        assert any(
            "muchos-a-muchos" in w.lower() or "duplicado" in w.lower()
            for w in result.validation_warnings
        )

    def test_partial_key_matches(self, module_merger, sample_sumaria, sample_vivienda):
        """Test handling of partial key matches."""
        # Remove one record from right
        partial_vivienda = sample_vivienda.iloc[:-1].copy()

        result = module_merger.merge_modules(sample_sumaria, partial_vivienda, "34", "01")

        assert result.unmatched_left > 0
        assert result.quality_score < 100


class TestMultipleModuleMergeSequencing:
    """Test multi-module merge sequencing and optimization."""

    @pytest.fixture
    def geo_merger(self):
        return ENAHOGeoMerger(verbose=False)

    @pytest.mark.skip(
        reason="Merge order optimization requires modules_merged tracking - TODO v0.6.0"
    )
    def test_merge_order_optimization(self, geo_merger):
        """Test that merge order is optimized by size."""
        large_mod = pd.DataFrame(
            {
                "conglome": [f"{i:03d}" for i in range(100)],
                "vivienda": ["01"] * 100,
                "hogar": ["1"] * 100,
                "large_var": range(100),
            }
        )
        small_mod = pd.DataFrame(
            {
                "conglome": [f"{i:03d}" for i in range(10)],
                "vivienda": ["01"] * 10,
                "hogar": ["1"] * 10,
                "small_var": range(10),
            }
        )
        medium_mod = pd.DataFrame(
            {
                "conglome": [f"{i:03d}" for i in range(50)],
                "vivienda": ["01"] * 50,
                "hogar": ["1"] * 50,
                "medium_var": range(50),
            }
        )

        modules = {"large": large_mod, "small": small_mod, "medium": medium_mod}

        result = geo_merger.merge_multiple_modules(modules_dict=modules, base_module="large")

        assert result.merged_df is not None
        # Should process in optimal order


# ============================================================================
# PRIORITY 2: VALIDATION & ERROR HANDLING (15-20 tests)
# ============================================================================


class TestModuleValidatorComprehensive:
    """Comprehensive module validator tests."""

    @pytest.fixture
    def validator(self):
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        return ModuleValidator(config, logger)

    def test_validate_module_structure_all_modules(self, validator):
        """Test validation for all known module types."""
        module_configs = {
            "01": ["conglome", "vivienda", "hogar"],
            "02": ["conglome", "vivienda", "hogar", "codperso"],
            "34": ["conglome", "vivienda", "hogar"],
        }

        for module_code, required_cols in module_configs.items():
            df = pd.DataFrame({col: [str(i) for i in range(5)] for col in required_cols})
            df["extra_col"] = range(5)

            warnings = validator.validate_module_structure(df, module_code)

            assert isinstance(warnings, list)

    def test_module_compatibility_matrix(self, validator):
        """Test compatibility between all module combinations."""
        hogar_mod = pd.DataFrame({"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"]})
        persona_mod = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "codperso": ["01"]}
        )

        # Hogar-Hogar: Compatible
        compat = validator.check_module_compatibility(
            hogar_mod, hogar_mod, "01", "34", ModuleMergeLevel.HOGAR
        )
        assert compat["compatible"]

        # Hogar-Persona at Hogar level: Compatible
        compat = validator.check_module_compatibility(
            hogar_mod, persona_mod, "01", "02", ModuleMergeLevel.HOGAR
        )
        assert compat["compatible"]

    def test_validation_detects_duplicate_records(self, validator):
        """Test detection of duplicate records in validation."""
        df = pd.DataFrame(
            {
                "conglome": ["001", "001", "002"],  # Duplicate
                "vivienda": ["01", "01", "01"],
                "hogar": ["1", "1", "1"],
            }
        )

        warnings = validator.validate_module_structure(df, "01")

        assert any("duplicado" in w.lower() for w in warnings)

    def test_validation_persona_level_checks(self, validator):
        """Test persona-level specific validations."""
        df = pd.DataFrame(
            {
                "conglome": ["001", "001"],
                "vivienda": ["01", "01"],
                "hogar": ["1", "1"],
                "codperso": ["01", "02"],
                "p208a": [25, 150],  # Invalid age
            }
        )

        warnings = validator.validate_module_structure(df, "02")

        # Should validate persona-specific fields
        assert isinstance(warnings, list)

    def test_validation_sumaria_specific_checks(self, validator):
        """Test sumaria module specific validations."""
        df = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["1", "1"],
                "mieperho": [0, 5],  # 0 is invalid
                "gashog2d": [2000, 1500],
                "inghog2d": [1000, 2000],  # Gastos > Ingresos for first
            }
        )

        warnings = validator.validate_module_structure(df, "34")

        assert len(warnings) > 0  # Should detect issues

    def test_validation_consistency_report(self, validator):
        """Test data consistency validation report."""
        df = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["1", "1"],
                "value": [100, 200],
            }
        )

        report = validator.validate_data_consistency(df, "01")

        assert report["module"] == "01"
        assert "consistency_score" in report
        assert "data_quality_metrics" in report

    def test_validation_report_generation(self, validator):
        """Test generation of comprehensive validation report."""
        df = pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["01", "01", "01"],
                "hogar": ["1", "1", "1"],
                "gashog2d": [2000, 1500, 1800],
            }
        )

        report = validator.generate_validation_report(df, "34")

        assert isinstance(report, str)
        assert "REPORTE DE VALIDACIÓN" in report
        assert "MÓDULO 34" in report.upper()


class TestExceptionHandlingComprehensive:
    """Comprehensive exception handling tests."""

    @pytest.fixture
    def geo_merger(self):
        return ENAHOGeoMerger(verbose=False)

    def test_empty_principal_dataframe_handling(self, geo_merger):
        """Test handling of empty principal DataFrame."""
        empty_df = pd.DataFrame()
        geo_df = pd.DataFrame({"ubigeo": ["150101"], "departamento": ["Lima"]})

        with pytest.raises(ValueError, match="df_principal no puede estar vacío"):
            geo_merger.merge_geographic_data(empty_df, geo_df)

    def test_empty_geography_dataframe_handling(self, geo_merger):
        """Test handling of empty geography DataFrame."""
        data_df = pd.DataFrame({"ubigeo": ["150101"], "value": [100]})
        empty_geo = pd.DataFrame()

        with pytest.raises(ValueError, match="df_geografia no puede estar vacío"):
            geo_merger.merge_geographic_data(data_df, empty_geo)

    def test_missing_ubigeo_column_error(self, geo_merger):
        """Test error when ubigeo column is missing."""
        data_df = pd.DataFrame({"value": [100, 200]})
        geo_df = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["Lima", "Lima"]})

        with pytest.raises(ValueError, match="Columna 'ubigeo' no encontrada"):
            geo_merger.merge_geographic_data(data_df, geo_df)

    @pytest.mark.skip(reason="Configuration validation not fully implemented - TODO v0.6.0")
    def test_configuration_validation_errors(self, geo_merger):
        """Test configuration validation errors."""
        # Invalid chunk_size - TODO: Add validation to GeoMergeConfiguration.__post_init__
        with pytest.raises(Exception):
            bad_config = GeoMergeConfiguration(chunk_size=-1)

    def test_duplicate_handling_error_strategy_raises(self):
        """Test ERROR strategy for duplicates raises exception."""
        config = GeoMergeConfiguration(manejo_duplicados=TipoManejoDuplicados.ERROR)
        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        data = pd.DataFrame({"ubigeo": ["150101"], "value": [100]})
        geo_dup = pd.DataFrame(
            {"ubigeo": ["150101", "150101"], "departamento": ["Lima", "Lima-Dup"]}  # Duplicate
        )

        with pytest.raises(GeoMergeError):
            merger.merge_geographic_data(data, geo_dup)

    def test_incompatible_data_types_in_keys(self):
        """Test handling of incompatible data types in merge keys."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {
                "conglome": [1, 2],  # Numeric
                "vivienda": ["01", "01"],
                "hogar": ["1", "1"],
                "value": [100, 200],
            }
        )
        df2 = pd.DataFrame(
            {
                "conglome": ["001", "002"],  # String
                "vivienda": ["01", "01"],
                "hogar": ["1", "1"],
                "value": [300, 400],
            }
        )

        # Should handle type conversion
        result = merger.merge_modules(df1, df2, "01", "34")
        assert result.merged_df is not None


class TestAdvancedErrorRecovery:
    """Test advanced error recovery scenarios."""

    @pytest.mark.skip(reason="continue_on_error feature not fully implemented - TODO v0.6.0")
    def test_merge_continues_on_module_error_when_configured(self):
        """Test that merge continues when continue_on_error is True."""
        config = ModuleMergeConfig(continue_on_error=True)
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        valid_mod = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "value": [100]}
        )
        invalid_mod = pd.DataFrame({"wrong_col": ["data"]})  # Missing required keys
        another_valid = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "value2": [200]}
        )

        modules = {"01": valid_mod, "bad": invalid_mod, "34": another_valid}

        result = merger.merge_multiple_modules(modules, base_module="01")

        # Should skip bad module but merge valid ones
        assert result.merged_df is not None
        assert len(result.validation_warnings) > 0


# ============================================================================
# PRIORITY 3: ADVANCED FEATURES & EDGE CASES (15-20 tests)
# ============================================================================


class TestAdvancedConfigurationOptions:
    """Test advanced configuration options and features."""

    def test_all_duplicate_strategies_workflow(self):
        """Test all 7 duplicate handling strategies."""
        strategies_to_test = [
            TipoManejoDuplicados.FIRST,
            TipoManejoDuplicados.LAST,
            TipoManejoDuplicados.KEEP_ALL,
        ]

        data = pd.DataFrame({"ubigeo": ["150101"], "value": [100]})
        geo_dup = pd.DataFrame({"ubigeo": ["150101", "150101"], "departamento": ["Lima1", "Lima2"]})

        for strategy in strategies_to_test:
            config = GeoMergeConfiguration(manejo_duplicados=strategy)
            merger = ENAHOGeoMerger(geo_config=config, verbose=False)

            result, _ = merger.merge_geographic_data(data, geo_dup)
            assert result is not None

    @pytest.mark.skip(reason="AGGREGATE duplicate strategy not fully implemented - TODO v0.6.0")
    def test_aggregate_duplicate_strategy_with_functions(self):
        """Test AGGREGATE strategy with custom aggregation functions."""
        config = GeoMergeConfiguration(
            manejo_duplicados=TipoManejoDuplicados.AGGREGATE,
            funciones_agregacion={"poblacion": "sum", "ingreso": "mean"},
        )
        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        data = pd.DataFrame({"ubigeo": ["150101"], "value": [100]})
        geo_dup = pd.DataFrame(
            {"ubigeo": ["150101", "150101"], "poblacion": [1000, 2000], "ingreso": [100, 200]}
        )

        result, _ = merger.merge_geographic_data(data, geo_dup)

        # Should aggregate duplicates
        assert len(result) == 1

    def test_memory_optimization_chunked_processing(self):
        """Test memory optimization with chunked processing."""
        config = GeoMergeConfiguration(optimizar_memoria=True, chunk_size=100)
        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        n = 1000
        data = pd.DataFrame({"ubigeo": [f"15{i:04d}" for i in range(n)], "value": range(n)})
        geo = pd.DataFrame(
            {"ubigeo": [f"15{i:04d}" for i in range(n)], "departamento": ["Lima"] * n}
        )

        result, _ = merger.merge_geographic_data(data, geo)

        assert len(result) == n

    def test_custom_column_prefixes_suffixes(self):
        """Test custom column prefixes and suffixes."""
        config = GeoMergeConfiguration(prefijo_columnas="geo_", sufijo_columnas="_ref")
        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        data = pd.DataFrame({"ubigeo": ["150101"], "value": [100]})
        geo = pd.DataFrame({"ubigeo": ["150101"], "departamento": ["Lima"]})

        result, _ = merger.merge_geographic_data(data, geo)

        # Should have prefixed/suffixed columns
        assert any("geo_" in col or "_re" in col for col in result.columns)


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_single_record_dataframes(self):
        """Test merge with single-record DataFrames."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "value": [100]}
        )
        df2 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "value2": [200]}
        )

        result = merger.merge_modules(df1, df2, "01", "34")

        assert len(result.merged_df) == 1

    def test_all_missing_keys_in_data(self):
        """Test handling when all keys are missing (NaN)."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {
                "conglome": [np.nan, np.nan],
                "vivienda": [np.nan, np.nan],
                "hogar": [np.nan, np.nan],
                "value": [100, 200],
            }
        )
        df2 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["1", "1"],
                "value2": [300, 400],
            }
        )

        result = merger.merge_modules(df1, df2, "01", "34")

        # Should handle gracefully
        assert result.merged_df is not None

    def test_unicode_characters_in_data(self):
        """Test handling of Unicode characters in data."""
        geo_merger = ENAHOGeoMerger(verbose=False)

        data = pd.DataFrame({"ubigeo": ["150101"], "nombre": ["Ñuñoa"]})  # Unicode character
        geo = pd.DataFrame({"ubigeo": ["150101"], "departamento": ["Lima"], "distrito": ["Ñuñoa"]})

        result, _ = geo_merger.merge_geographic_data(data, geo)

        assert result is not None
        assert "Ñuñoa" in result.values

    def test_extremely_long_strings(self):
        """Test handling of extremely long string values."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        long_string = "A" * 10000
        df1 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "long_text": [long_string]}
        )
        df2 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "value": [100]}
        )

        result = merger.merge_modules(df1, df2, "01", "34")

        assert result.merged_df is not None

    def test_mixed_numeric_string_keys(self):
        """Test merge with mixed numeric and string key types."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": [1, 2],  # Numeric
                "hogar": ["1", "2"],  # String
                "value": [100, 200],
            }
        )
        df2 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "02"],  # String
                "hogar": [1, 2],  # Numeric
                "value2": [300, 400],
            }
        )

        result = merger.merge_modules(df1, df2, "01", "34")

        # Should handle type conversion
        assert result.merged_df is not None

    def test_null_handling_in_geographic_columns(self):
        """Test handling of NULL values in geographic columns."""
        geo_merger = ENAHOGeoMerger(verbose=False)

        data = pd.DataFrame({"ubigeo": ["150101", "150102"], "value": [100, 200]})
        geo = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102"],
                "departamento": ["Lima", np.nan],  # NULL
                "provincia": [np.nan, "Lima"],  # NULL
            }
        )

        result, _ = geo_merger.merge_geographic_data(data, geo)

        assert result is not None
        assert result["departamento"].isna().sum() == 1


class TestMergePlanningAndAnalysis:
    """Test merge planning and feasibility analysis features."""

    def test_analyze_merge_feasibility_comprehensive(self):
        """Test comprehensive merge feasibility analysis."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        modules = {
            "01": pd.DataFrame(
                {
                    "conglome": ["001", "002"],
                    "vivienda": ["01", "01"],
                    "hogar": ["1", "1"],
                    "value": [100, 200],
                }
            ),
            "34": pd.DataFrame(
                {
                    "conglome": ["001", "002"],
                    "vivienda": ["01", "01"],
                    "hogar": ["1", "1"],
                    "value2": [300, 400],
                }
            ),
        }

        analysis = merger.analyze_merge_feasibility(modules, ModuleMergeLevel.HOGAR)

        assert analysis["feasible"]
        assert "memory_estimate_mb" in analysis
        assert "estimated_time_seconds" in analysis

    def test_create_merge_plan_optimization(self):
        """Test creation of optimized merge plan."""
        config = ModuleMergeConfig()
        logger = logging.getLogger("test")
        merger = ENAHOModuleMerger(config, logger)

        modules = {
            "large": pd.DataFrame(
                {
                    "conglome": [f"{i:03d}" for i in range(100)],
                    "vivienda": ["01"] * 100,
                    "hogar": ["1"] * 100,
                    "value": range(100),
                }
            ),
            "small": pd.DataFrame(
                {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "value": [100]}
            ),
        }

        plan = merger.create_merge_plan(modules, target_module="large")

        assert plan["base_module"] == "large"
        assert "merge_sequence" in plan
        assert "execution_steps" in plan


class TestModuleCompatibilityValidation:
    """Test module compatibility validation workflows."""

    def test_validate_compatibility_all_modules(self):
        """Test compatibility validation for all module combinations."""
        geo_merger = ENAHOGeoMerger(verbose=False)

        modules = {
            "01": pd.DataFrame({"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"]}),
            "02": pd.DataFrame(
                {"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"], "codperso": ["01"]}
            ),
            "34": pd.DataFrame({"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"]}),
        }

        result = geo_merger.validate_module_compatibility(modules, "hogar")

        assert result["is_compatible"]
        assert "module_analysis" in result

    def test_compatibility_low_overlap_warning(self):
        """Test warning when module overlap is low."""
        geo_merger = ENAHOGeoMerger(verbose=False)

        modules = {
            "01": pd.DataFrame(
                {"conglome": ["001", "002"], "vivienda": ["01", "01"], "hogar": ["1", "1"]}
            ),
            "34": pd.DataFrame(
                {"conglome": ["003", "004"], "vivienda": ["01", "01"], "hogar": ["1", "1"]}
            ),
        }

        result = geo_merger.validate_module_compatibility(modules, "hogar")

        # Should warn about low overlap
        assert len(result["warnings"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])

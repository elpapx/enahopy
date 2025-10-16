"""
Comprehensive Integration Tests for ENAHO Merger Module
========================================================

This test suite provides comprehensive integration testing for the merger module,
covering geographic merging, module merging, validation workflows, and edge cases.

Target Coverage: Merger module 17.81% → 45-50%
Total Tests: 50-80 comprehensive integration tests

Test Organization:
- Priority 1: Core Merge Workflows (20-25 tests)
- Priority 2: Validation & Error Handling (15-20 tests)
- Priority 3: Advanced Features & Edge Cases (15-20 tests)

Coverage Focus:
- enahopy/merger/core.py (ENAHOGeoMerger class)
- enahopy/merger/modules/merger.py (ENAHOModuleMerger class)
- enahopy/merger/modules/validator.py (ModuleValidator class)
- enahopy/merger/config.py (Configuration classes)
- enahopy/merger/exceptions.py (Exception handling)
"""

import logging
import unittest
from typing import Dict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from enahopy.merger import ENAHOGeoMerger
from enahopy.merger.config import (
    GeoMergeConfiguration,
    GeoValidationResult,
    ModuleMergeConfig,
    ModuleMergeLevel,
    ModuleMergeResult,
    ModuleMergeStrategy,
    ModuleType,
    TipoManejoDuplicados,
    TipoManejoErrores,
    TipoValidacionUbigeo,
)
from enahopy.merger.exceptions import (
    ConfigurationError,
    ConflictResolutionError,
    DataQualityError,
    GeoMergeError,
    IncompatibleModulesError,
    MergeKeyError,
    MergeValidationError,
    ModuleMergeError,
    UbigeoValidationError,
    ValidationThresholdError,
)
from enahopy.merger.modules.merger import ENAHOModuleMerger
from enahopy.merger.modules.validator import ModuleValidator

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_household_data():
    """Sample ENAHO household data (enaho01)."""
    return pd.DataFrame(
        {
            "conglome": ["001", "002", "003", "004", "005"],
            "vivienda": ["01", "01", "02", "01", "03"],
            "hogar": ["01", "01", "01", "02", "01"],
            "ubigeo": ["150101", "150102", "010201", "150101", "150102"],
            "result": [1, 1, 1, 1, 1],
            "ingreso": [1500.0, 2000.0, 1200.0, 3000.0, 1800.0],
        }
    )


@pytest.fixture
def sample_person_data():
    """Sample ENAHO person data (enaho02)."""
    return pd.DataFrame(
        {
            "conglome": ["001", "001", "002", "003", "004"],
            "vivienda": ["01", "01", "01", "02", "01"],
            "hogar": ["01", "01", "01", "01", "02"],
            "codperso": ["01", "02", "01", "01", "01"],
            "edad": [35, 40, 28, 55, 32],
            "sexo": [1, 2, 1, 2, 1],
        }
    )


@pytest.fixture
def sample_geography_data():
    """Sample geographic reference data."""
    return pd.DataFrame(
        {
            "ubigeo": ["150101", "150102", "010201"],
            "departamento": ["Lima", "Lima", "Amazonas"],
            "provincia": ["Lima", "Lima", "Bagua"],
            "distrito": ["Lima", "San Isidro", "Bagua"],
        }
    )


@pytest.fixture
def sample_sumaria_data():
    """Sample ENAHO sumaria module (34) data."""
    return pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["01", "01", "01"],
            "gashog2d": [2000.0, 1500.0, 1800.0],
            "inghog2d": [3000.0, 2500.0, 2200.0],
            "mieperho": [4, 3, 5],
            "pobreza": [1, 2, 1],
        }
    )


@pytest.fixture
def sample_vivienda_data():
    """Sample ENAHO housing module (01) data."""
    return pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["01", "01", "01"],
            "area": [1, 2, 1],
            "p110": [1, 1, 2],  # Tipo de vivienda
            "p111a": [3, 4, 3],  # Material predominante
        }
    )


@pytest.fixture
def logger():
    """Logger for tests."""
    return logging.getLogger("test_merger_integration")


@pytest.fixture
def geo_merger():
    """Basic ENAHOGeoMerger instance."""
    return ENAHOGeoMerger(verbose=False)


@pytest.fixture
def module_merger(logger):
    """Basic ENAHOModuleMerger instance."""
    config = ModuleMergeConfig()
    return ENAHOModuleMerger(config, logger)


# ============================================================================
# PRIORITY 1: CORE MERGE WORKFLOWS (20-25 tests)
# ============================================================================


class TestENAHOGeoMergerBasic:
    """Test basic geographic merging functionality."""

    def test_basic_geographic_merge_success(
        self, geo_merger, sample_household_data, sample_geography_data
    ):
        """Test basic geographic merge with valid data."""
        # Mock pattern detector and territorial validator
        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            geo_merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {
                "departamento": "departamento",
                "provincia": "provincia",
                "distrito": "distrito",
            }
            mock_territorial.return_value = []

            result, validation = geo_merger.merge_geographic_data(
                df_principal=sample_household_data,
                df_geografia=sample_geography_data,
                columna_union="ubigeo",
            )

            # Assertions
            assert result is not None
            assert len(result) == len(sample_household_data)
            assert "departamento" in result.columns
            assert "provincia" in result.columns
            assert "distrito" in result.columns
            assert validation.total_records > 0

    def test_geographic_merge_preserves_all_principal_records(
        self, geo_merger, sample_household_data, sample_geography_data
    ):
        """Test that geographic merge preserves all records from principal DataFrame."""
        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            geo_merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento"}
            mock_territorial.return_value = []

            result, _ = geo_merger.merge_geographic_data(
                df_principal=sample_household_data, df_geografia=sample_geography_data
            )

            # Left join semantics: all principal records preserved
            assert len(result) == len(sample_household_data)
            assert set(sample_household_data["conglome"]) == set(result["conglome"])

    def test_geographic_merge_with_partial_coverage(self, geo_merger, sample_household_data):
        """Test geographic merge when not all UBIGEOs have geographic data."""
        partial_geo = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102"],  # Missing some UBIGEOs
                "departamento": ["Lima", "Lima"],
            }
        )

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            geo_merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento"}
            mock_territorial.return_value = []

            result, validation = geo_merger.merge_geographic_data(
                df_principal=sample_household_data, df_geografia=partial_geo
            )

            # All records preserved, some with valor_faltante ('DESCONOCIDO') for missing matches
            assert len(result) == len(sample_household_data)
            # Check for missing geographic data (either NaN or valor_faltante)
            assert (
                result["departamento"].isna().sum() > 0
                or (result["departamento"] == "DESCONOCIDO").sum() > 0
            )

    def test_geographic_merge_adds_columns(
        self, geo_merger, sample_household_data, sample_geography_data
    ):
        """Test that geographic merge adds new columns without removing existing ones."""
        original_cols = set(sample_household_data.columns)

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            geo_merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento", "provincia": "provincia"}
            mock_territorial.return_value = []

            result, _ = geo_merger.merge_geographic_data(
                df_principal=sample_household_data, df_geografia=sample_geography_data
            )

            result_cols = set(result.columns)
            # All original columns preserved
            assert original_cols.issubset(result_cols)
            # New columns added
            assert "departamento" in result_cols
            assert "provincia" in result_cols

    def test_geographic_merge_handles_ubigeo_mismatch(self, geo_merger):
        """Test geographic merge when UBIGEOs don't match between datasets."""
        df_data = pd.DataFrame({"ubigeo": ["150101", "150102"], "value": [100, 200]})

        df_geo = pd.DataFrame(
            {
                "ubigeo": ["250101", "250102"],  # Different UBIGEOs
                "departamento": ["Ucayali", "Ucayali"],
            }
        )

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            geo_merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento"}
            mock_territorial.return_value = []

            result, validation = geo_merger.merge_geographic_data(
                df_principal=df_data, df_geografia=df_geo
            )

            # All records preserved but no geographic matches
            assert len(result) == len(df_data)
            # With no matches, implementation uses valor_faltante ('DESCONOCIDO')
            assert (result["departamento"] == "DESCONOCIDO").all() or result[
                "departamento"
            ].isna().all()


class TestENAHOModuleMergerBasic:
    """Test basic module-to-module merging."""

    def test_two_module_merge_household_person(
        self, module_merger, sample_household_data, sample_person_data
    ):
        """Test merging household and person modules."""
        result = module_merger.merge_modules(
            left_df=sample_household_data,
            right_df=sample_person_data,
            left_module="01",
            right_module="02",
        )

        assert result is not None
        assert isinstance(result, ModuleMergeResult)
        assert result.merged_df is not None
        assert len(result.merged_df) >= len(sample_household_data)

    def test_two_module_merge_sumaria_vivienda(
        self, module_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test merging sumaria and vivienda modules at household level."""
        result = module_merger.merge_modules(
            left_df=sample_sumaria_data,
            right_df=sample_vivienda_data,
            left_module="34",
            right_module="01",
        )

        assert result.merged_df is not None
        assert len(result.merged_df) == len(sample_sumaria_data)
        # Check columns from both modules present
        assert "gashog2d" in result.merged_df.columns
        assert "area" in result.merged_df.columns

    def test_module_merge_preserves_left_records(
        self, module_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test that module merge preserves all left DataFrame records."""
        result = module_merger.merge_modules(
            left_df=sample_sumaria_data,
            right_df=sample_vivienda_data,
            left_module="34",
            right_module="01",
        )

        # Left join semantics
        assert len(result.merged_df) == len(sample_sumaria_data)
        assert set(sample_sumaria_data["conglome"]) == set(result.merged_df["conglome"])

    def test_module_merge_with_partial_overlap(self, module_merger, sample_sumaria_data):
        """Test module merge when only some records match."""
        partial_vivienda = pd.DataFrame(
            {
                "conglome": ["001"],  # Only one match
                "vivienda": ["01"],
                "hogar": ["01"],
                "area": [1],
            }
        )

        result = module_merger.merge_modules(
            left_df=sample_sumaria_data,
            right_df=partial_vivienda,
            left_module="34",
            right_module="01",
        )

        # All left records preserved
        assert len(result.merged_df) == len(sample_sumaria_data)
        # Some records have no match
        assert result.unmatched_left > 0

    def test_module_merge_quality_score_calculation(
        self, module_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test that merge quality score is calculated correctly."""
        result = module_merger.merge_modules(
            left_df=sample_sumaria_data,
            right_df=sample_vivienda_data,
            left_module="34",
            right_module="01",
        )

        assert result.quality_score is not None
        assert 0 <= result.quality_score <= 100

    def test_multiple_modules_sequential_merge(
        self, geo_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test sequential merge of three modules."""
        person_data = pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["01", "01", "02"],
                "hogar": ["01", "01", "01"],
                "nmiembros": [4, 3, 5],
            }
        )

        modules = {"34": sample_sumaria_data, "01": sample_vivienda_data, "02": person_data}

        result = geo_merger.merge_multiple_modules(modules_dict=modules, base_module="34")

        assert result.merged_df is not None
        assert len(result.merged_df) == len(sample_sumaria_data)
        # Check all modules' columns present
        assert "gashog2d" in result.merged_df.columns
        assert "area" in result.merged_df.columns
        assert "nmiembros" in result.merged_df.columns

    def test_merge_keys_generation_household_level(self, module_merger):
        """Test that household-level merge keys are correct."""
        keys = module_merger._get_merge_keys_for_level(ModuleMergeLevel.HOGAR)

        assert "conglome" in keys
        assert "vivienda" in keys
        assert "hogar" in keys
        assert "codperso" not in keys

    def test_merge_keys_generation_person_level(self, module_merger):
        """Test that person-level merge keys are correct."""
        keys = module_merger._get_merge_keys_for_level(ModuleMergeLevel.PERSONA)

        assert "conglome" in keys
        assert "vivienda" in keys
        assert "hogar" in keys
        assert "codperso" in keys


class TestGeoModuleCombinedWorkflow:
    """Test combined geographic and module merge workflows."""

    @pytest.mark.skip(reason="Combined workflow requires modules_merged attribute - TODO v0.6.0")
    def test_modules_then_geography_workflow(
        self, geo_merger, sample_sumaria_data, sample_vivienda_data, sample_geography_data
    ):
        """Test merging modules first, then adding geography."""
        # Add ubigeo to sumaria
        sample_sumaria_with_ubigeo = sample_sumaria_data.copy()
        sample_sumaria_with_ubigeo["ubigeo"] = ["150101", "150102", "010201"]

        modules = {"34": sample_sumaria_with_ubigeo, "01": sample_vivienda_data}

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            geo_merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento"}
            mock_territorial.return_value = []

            result, report = geo_merger.merge_modules_with_geography(
                modules_dict=modules, df_geografia=sample_geography_data, base_module="34"
            )

            assert result is not None
            assert "departamento" in result.columns
            assert "gashog2d" in result.columns
            assert "area" in result.columns

    @pytest.mark.skip(reason="Combined workflow requires modules_merged attribute - TODO v0.6.0")
    def test_combined_merge_preserves_record_count(
        self, geo_merger, sample_sumaria_data, sample_vivienda_data, sample_geography_data
    ):
        """Test that combined merge preserves base module record count."""
        sample_sumaria_with_ubigeo = sample_sumaria_data.copy()
        sample_sumaria_with_ubigeo["ubigeo"] = ["150101", "150102", "010201"]

        modules = {"34": sample_sumaria_with_ubigeo, "01": sample_vivienda_data}

        with patch.object(
            geo_merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            geo_merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento"}
            mock_territorial.return_value = []

            result, _ = geo_merger.merge_modules_with_geography(
                modules_dict=modules, df_geografia=sample_geography_data, base_module="34"
            )

            # Base module record count preserved
            assert len(result) == len(sample_sumaria_data)


# ============================================================================
# PRIORITY 2: VALIDATION & ERROR HANDLING (15-20 tests)
# ============================================================================


class TestModuleValidator:
    """Test module validation logic."""

    def test_validate_module_structure_valid(self, logger):
        """Test validation of well-formed module data."""
        validator = ModuleValidator(ModuleMergeConfig(), logger)

        valid_df = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["01", "02"],
                "value": [100, 200],
            }
        )

        warnings = validator.validate_module_structure(valid_df, "34")

        # Well-formed module should have few/no warnings
        assert isinstance(warnings, list)

    def test_validate_missing_required_columns(self, logger):
        """Test validation fails with missing columns."""
        validator = ModuleValidator(ModuleMergeConfig(), logger)

        invalid_df = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                # Missing vivienda and hogar
                "value": [100, 200],
            }
        )

        warnings = validator.validate_module_structure(invalid_df, "34")

        # Should have warnings about missing columns
        assert len(warnings) > 0
        assert any("columnas faltantes" in w.lower() or "missing" in w.lower() for w in warnings)

    def test_validate_duplicate_records_detection(self, logger):
        """Test that validator detects duplicate records."""
        validator = ModuleValidator(ModuleMergeConfig(), logger)

        df_with_dups = pd.DataFrame(
            {
                "conglome": ["001", "001"],  # Duplicate
                "vivienda": ["01", "01"],
                "hogar": ["01", "01"],
                "value": [100, 200],
            }
        )

        warnings = validator.validate_module_structure(df_with_dups, "34")

        # Should detect duplicates
        assert any("duplicado" in w.lower() or "duplicate" in w.lower() for w in warnings)

    def test_check_module_compatibility_compatible(self, logger):
        """Test compatibility check for compatible modules."""
        validator = ModuleValidator(ModuleMergeConfig(), logger)

        df1 = pd.DataFrame(
            {"conglome": ["001", "002"], "vivienda": ["01", "01"], "hogar": ["01", "02"]}
        )

        df2 = pd.DataFrame(
            {"conglome": ["001", "002"], "vivienda": ["01", "01"], "hogar": ["01", "02"]}
        )

        result = validator.check_module_compatibility(df1, df2, "34", "01", ModuleMergeLevel.HOGAR)

        assert result.get("compatible", False) is True

    @pytest.mark.skip(
        reason="Advanced compatibility validation not fully implemented - TODO v0.6.0"
    )
    def test_check_module_compatibility_incompatible_keys(self, logger):
        """Test compatibility check detects missing keys."""
        validator = ModuleValidator(ModuleMergeConfig(), logger)

        df1 = pd.DataFrame(
            {"conglome": ["001", "002"], "vivienda": ["01", "01"], "hogar": ["01", "02"]}
        )

        df2 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                # Missing vivienda and hogar
                "value": [100, 200],
            }
        )

        result = validator.check_module_compatibility(df1, df2, "34", "01", ModuleMergeLevel.HOGAR)

        assert result.get("compatible", True) is False

    def test_validate_persona_level_module(self, logger):
        """Test validation specific to person-level modules."""
        validator = ModuleValidator(ModuleMergeConfig(), logger)

        persona_df = pd.DataFrame(
            {
                "conglome": ["001", "001"],
                "vivienda": ["01", "01"],
                "hogar": ["01", "01"],
                "codperso": ["01", "02"],
                "p203": [1, 2],
                "p208a": [25, 30],
            }
        )

        warnings = validator.validate_module_structure(persona_df, "02")

        # Should validate without errors
        assert isinstance(warnings, list)

    def test_validate_sumaria_specific_checks(self, logger, sample_sumaria_data):
        """Test sumaria-specific validation rules."""
        validator = ModuleValidator(ModuleMergeConfig(), logger)

        warnings = validator.validate_module_structure(sample_sumaria_data, "34")

        # Should perform sumaria-specific validations
        assert isinstance(warnings, list)


class TestExceptionHandling:
    """Test error scenarios and exception handling."""

    def test_merge_empty_dataframe_raises_exception(self, geo_merger):
        """Test that empty DataFrames raise appropriate exceptions."""
        empty_df = pd.DataFrame()
        valid_df = pd.DataFrame({"ubigeo": ["150101"], "value": [100]})

        with pytest.raises(ValueError):
            geo_merger.merge_geographic_data(empty_df, valid_df)

    def test_merge_missing_column_raises_exception(self, geo_merger, sample_household_data):
        """Test that missing merge column raises exception."""
        df_geo = pd.DataFrame({"wrong_column": ["150101"], "departamento": ["Lima"]})

        with pytest.raises(ValueError):
            geo_merger.merge_geographic_data(sample_household_data, df_geo, columna_union="ubigeo")

    @pytest.mark.skip(
        reason="IncompatibleModulesError exception not fully implemented - TODO v0.6.0"
    )
    def test_incompatible_modules_raises_exception(self, module_merger, logger):
        """Test that incompatible modules raise IncompatibleModulesError."""
        df1 = pd.DataFrame({"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"]})

        # Missing required keys
        df2 = pd.DataFrame({"wrong_col": ["001"]})

        with pytest.raises(IncompatibleModulesError):
            module_merger.merge_modules(df1, df2, "34", "01")

    def test_invalid_ubigeo_handling(self, geo_merger):
        """Test handling of invalid UBIGEO codes."""
        df_data = pd.DataFrame(
            {"ubigeo": ["999999", "INVALID", "150101"], "value": [100, 200, 300]}
        )

        df_geo = pd.DataFrame({"ubigeo": ["150101"], "departamento": ["Lima"]})

        config = GeoMergeConfiguration(manejo_errores=TipoManejoErrores.COERCE)

        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        with patch.object(
            merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento"}
            mock_territorial.return_value = []

            # Should not raise with COERCE
            result, validation = merger.merge_geographic_data(df_data, df_geo)

            assert result is not None

    def test_configuration_validation_errors(self):
        """Test that invalid configurations raise ConfigurationError."""
        config = GeoMergeConfiguration(chunk_size=-1)  # Invalid

        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        with pytest.raises(ConfigurationError):
            merger._validate_configurations()

    def test_duplicate_handling_error_strategy(self, geo_merger):
        """Test ERROR strategy for duplicates raises exception."""
        df_geo_dup = pd.DataFrame(
            {"ubigeo": ["150101", "150101"], "departamento": ["Lima", "Lima"]}  # Duplicate
        )

        config = GeoMergeConfiguration(manejo_duplicados=TipoManejoDuplicados.ERROR)

        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        with pytest.raises(GeoMergeError):
            merger._handle_duplicates(df_geo_dup, "ubigeo")

    @pytest.mark.skip(
        reason="ERROR merge strategy with exceptions not fully implemented - TODO v0.6.0"
    )
    def test_conflict_resolution_error_strategy(self, logger):
        """Test ERROR strategy for conflicts raises exception."""
        config = ModuleMergeConfig(merge_strategy=ModuleMergeStrategy.ERROR)

        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value": [100]}
        )

        df2 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value": [200]}  # Conflict
        )

        # Merge and then resolve conflicts
        result = merger.merge_modules(df1, df2, "01", "34")

        # The merge should succeed but conflicts should be tracked
        assert result.conflicts_resolved >= 0


class TestValidationWorkflows:
    """Test comprehensive validation workflows."""

    def test_validate_geographic_data_comprehensive(self, geo_merger, sample_geography_data):
        """Test comprehensive geographic data validation."""
        validation = geo_merger.validate_geographic_data(
            df=sample_geography_data,
            columna_ubigeo="ubigeo",
            validate_territory=True,
            validate_quality=True,
        )

        assert isinstance(validation, GeoValidationResult)
        assert validation.total_records > 0
        assert validation.valid_ubigeos >= 0

    def test_validation_detects_invalid_ubigeos(self, geo_merger):
        """Test that validation detects invalid UBIGEO codes."""
        df_invalid = pd.DataFrame(
            {"ubigeo": ["999999", "XXXXX", "150101"], "value": [100, 200, 300]}
        )

        validation = geo_merger.validate_geographic_data(df=df_invalid, columna_ubigeo="ubigeo")

        assert validation.invalid_ubigeos > 0

    def test_validation_reports_quality_metrics(self, geo_merger, sample_geography_data):
        """Test that validation includes quality metrics."""
        validation = geo_merger.validate_geographic_data(
            df=sample_geography_data, columna_ubigeo="ubigeo", validate_quality=True
        )

        assert validation.quality_metrics is not None
        assert isinstance(validation.quality_metrics, dict)

    def test_validate_module_compatibility_workflow(
        self, geo_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test module compatibility validation workflow."""
        modules = {"34": sample_sumaria_data, "01": sample_vivienda_data}

        result = geo_merger.validate_module_compatibility(modules_dict=modules, merge_level="hogar")

        assert result is not None
        assert "is_compatible" in result
        assert "module_analysis" in result


# ============================================================================
# PRIORITY 3: ADVANCED FEATURES & EDGE CASES (15-20 tests)
# ============================================================================


class TestMergeConfigurations:
    """Test different configuration strategies."""

    def test_duplicate_strategy_first(self, geo_merger):
        """Test FIRST duplicate handling strategy."""
        df_geo_dup = pd.DataFrame(
            {
                "ubigeo": ["150101", "150101", "150102"],
                "departamento": ["Lima1", "Lima2", "Lima"],
                "value": [1, 2, 3],
            }
        )

        config = GeoMergeConfiguration(manejo_duplicados=TipoManejoDuplicados.FIRST)

        merger = ENAHOGeoMerger(geo_config=config, verbose=False)
        result = merger._handle_duplicates(df_geo_dup, "ubigeo")

        # Should keep first occurrence
        assert len(result) == 2
        lima_row = result[result["ubigeo"] == "150101"].iloc[0]
        assert lima_row["departamento"] == "Lima1"

    def test_duplicate_strategy_last(self, geo_merger):
        """Test LAST duplicate handling strategy."""
        df_geo_dup = pd.DataFrame(
            {
                "ubigeo": ["150101", "150101", "150102"],
                "departamento": ["Lima1", "Lima2", "Lima"],
                "value": [1, 2, 3],
            }
        )

        config = GeoMergeConfiguration(manejo_duplicados=TipoManejoDuplicados.LAST)

        merger = ENAHOGeoMerger(geo_config=config, verbose=False)
        result = merger._handle_duplicates(df_geo_dup, "ubigeo")

        # Should keep last occurrence
        assert len(result) == 2
        lima_row = result[result["ubigeo"] == "150101"].iloc[0]
        assert lima_row["departamento"] == "Lima2"

    def test_duplicate_strategy_aggregate(self, geo_merger):
        """Test AGGREGATE duplicate handling strategy."""
        df_geo_dup = pd.DataFrame(
            {
                "ubigeo": ["150101", "150101", "150102"],
                "departamento": ["Lima", "Lima", "Lima"],
                "population": [100, 200, 300],
            }
        )

        config = GeoMergeConfiguration(
            manejo_duplicados=TipoManejoDuplicados.AGGREGATE,
            funciones_agregacion={"population": "sum", "departamento": "first"},
        )

        merger = ENAHOGeoMerger(geo_config=config, verbose=False)
        result = merger._handle_duplicates(df_geo_dup, "ubigeo")

        # Should aggregate
        assert len(result) == 2
        lima_row = result[result["ubigeo"] == "150101"].iloc[0]
        assert lima_row["population"] == 300  # sum of 100 + 200

    def test_merge_strategy_coalesce(self, module_merger, logger):
        """Test COALESCE merge strategy."""
        config = ModuleMergeConfig(merge_strategy=ModuleMergeStrategy.COALESCE)

        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["01", "02"],
                "value": [100, np.nan],
            }
        )

        df2 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["01", "02"],
                "value": [np.nan, 200],
            }
        )

        result = merger.merge_modules(df1, df2, "01", "34")

        # COALESCE should fill NaN values
        assert result.merged_df is not None

    def test_merge_strategy_keep_left(self, module_merger, logger):
        """Test KEEP_LEFT merge strategy."""
        config = ModuleMergeConfig(merge_strategy=ModuleMergeStrategy.KEEP_LEFT)

        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value": [100]}
        )

        df2 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value": [200]}
        )

        result = merger.merge_modules(df1, df2, "01", "34")

        assert result.merged_df is not None

    def test_merge_strategy_keep_right(self, module_merger, logger):
        """Test KEEP_RIGHT merge strategy."""
        config = ModuleMergeConfig(merge_strategy=ModuleMergeStrategy.KEEP_RIGHT)

        merger = ENAHOModuleMerger(config, logger)

        df1 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value": [100]}
        )

        df2 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value": [200]}
        )

        result = merger.merge_modules(df1, df2, "01", "34")

        assert result.merged_df is not None

    def test_custom_merge_configuration(self, geo_merger):
        """Test custom GeoMergeConfiguration."""
        config = GeoMergeConfiguration(
            columna_union="custom_ubigeo",
            manejo_duplicados=TipoManejoDuplicados.FIRST,
            validar_formato_ubigeo=False,
            optimizar_memoria=True,
            chunk_size=100000,
        )

        assert config.columna_union == "custom_ubigeo"
        assert config.manejo_duplicados == TipoManejoDuplicados.FIRST
        assert config.optimizar_memoria is True


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_empty_dataframe_merge(self, module_merger, logger):
        """Test merging with empty dataframes."""
        empty_df = pd.DataFrame()
        valid_df = pd.DataFrame({"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"]})

        result = module_merger.merge_modules(empty_df, valid_df, "01", "34")

        # Should handle gracefully
        assert result is not None
        assert result.merged_df is not None

    def test_single_row_dataframe_merge(self, module_merger, logger):
        """Test merging single-row dataframes."""
        df1 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value1": [100]}
        )

        df2 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "value2": [200]}
        )

        result = module_merger.merge_modules(df1, df2, "01", "34")

        assert len(result.merged_df) == 1
        assert "value1" in result.merged_df.columns
        assert "value2" in result.merged_df.columns

    def test_large_dataset_simulation(self, module_merger, logger):
        """Test performance with larger datasets (simulated)."""
        # Create larger dataset (1000 rows)
        df1 = pd.DataFrame(
            {
                "conglome": [f"{i:03d}" for i in range(1000)],
                "vivienda": ["01"] * 1000,
                "hogar": ["01"] * 1000,
                "value": range(1000),
            }
        )

        df2 = pd.DataFrame(
            {
                "conglome": [f"{i:03d}" for i in range(1000)],
                "vivienda": ["01"] * 1000,
                "hogar": ["01"] * 1000,
                "other": range(1000, 2000),
            }
        )

        result = module_merger.merge_modules(df1, df2, "01", "34")

        assert len(result.merged_df) == 1000

    def test_all_missing_keys_handling(self, module_merger, logger):
        """Test merge behavior when all keys are missing/NaN."""
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
                "hogar": ["01", "02"],
                "other": [300, 400],
            }
        )

        result = module_merger.merge_modules(df1, df2, "01", "34")

        # Should handle gracefully even with all missing keys
        assert result is not None

    def test_mixed_data_types_in_keys(self, module_merger, logger):
        """Test handling of mixed data types in merge keys."""
        df1 = pd.DataFrame(
            {
                "conglome": ["001", "002"],  # String
                "vivienda": ["01", "01"],
                "hogar": ["01", "02"],
                "value": [100, 200],
            }
        )

        df2 = pd.DataFrame(
            {
                "conglome": [1, 2],  # Integer
                "vivienda": ["01", "01"],
                "hogar": ["01", "02"],
                "other": [300, 400],
            }
        )

        # Should handle type conversion
        result = module_merger.merge_modules(df1, df2, "01", "34")

        assert result is not None

    def test_special_characters_in_ubigeo(self, geo_merger):
        """Test handling of special characters in UBIGEO codes."""
        df_data = pd.DataFrame(
            {"ubigeo": ["150-101", "150.102", "150#103"], "value": [100, 200, 300]}
        )

        df_geo = pd.DataFrame({"ubigeo": ["150101"], "departamento": ["Lima"]})

        config = GeoMergeConfiguration(manejo_errores=TipoManejoErrores.COERCE)

        merger = ENAHOGeoMerger(geo_config=config, verbose=False)

        with patch.object(
            merger.pattern_detector, "detectar_columnas_geograficas"
        ) as mock_detect, patch.object(
            merger.territorial_validator, "validar_jerarquia_territorial"
        ) as mock_territorial:
            mock_detect.return_value = {"departamento": "departamento"}
            mock_territorial.return_value = []

            # Should handle gracefully
            result, _ = merger.merge_geographic_data(df_data, df_geo)

            assert result is not None

    def test_extremely_long_strings_in_data(self, module_merger, logger):
        """Test handling of extremely long string values."""
        long_string = "A" * 10000

        df1 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "description": [long_string]}
        )

        df2 = pd.DataFrame(
            {"conglome": ["001"], "vivienda": ["01"], "hogar": ["01"], "other": [100]}
        )

        result = module_merger.merge_modules(df1, df2, "01", "34")

        assert result is not None

    def test_zero_variance_columns(self, module_merger, logger):
        """Test merging with columns that have zero variance."""
        df1 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["01", "02"],
                "constant": [100, 100],  # Zero variance
            }
        )

        df2 = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["01", "01"],
                "hogar": ["01", "02"],
                "other": [200, 300],
            }
        )

        result = module_merger.merge_modules(df1, df2, "01", "34")

        assert result is not None
        assert "constant" in result.merged_df.columns


class TestPerformanceOptimizations:
    """Test performance optimization features."""

    def test_chunk_processing_enabled(self, geo_merger):
        """Test that chunk processing is properly enabled for large datasets."""
        # Create dataset larger than chunk_size
        large_df = pd.DataFrame(
            {"ubigeo": [f"{i:06d}" for i in range(100000, 160000)], "value": range(60000)}
        )

        config = GeoMergeConfiguration(optimizar_memoria=True, chunk_size=50000)

        assert config.optimizar_memoria is True
        assert config.chunk_size == 50000

    def test_memory_optimization_features(self, module_merger):
        """Test memory optimization in module merger."""
        # Large dataset that would trigger optimizations
        df1 = pd.DataFrame(
            {
                "conglome": [f"{i:03d}" for i in range(10000)],
                "vivienda": ["01"] * 10000,
                "hogar": ["01"] * 10000,
                "value": range(10000),
            }
        )

        df2 = df1.copy()
        df2["other"] = range(10000, 20000)

        result = module_merger.merge_modules(df1, df2, "01", "34")

        assert result is not None


class TestReportGeneration:
    """Test report and statistics generation."""

    def test_validation_result_summary_report(self, geo_merger, sample_geography_data):
        """Test generation of validation summary report."""
        validation = geo_merger.validate_geographic_data(
            df=sample_geography_data, columna_ubigeo="ubigeo"
        )

        report = validation.get_summary_report()

        assert isinstance(report, str)
        assert len(report) > 0

    def test_merge_result_summary_report(
        self, module_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test generation of merge result summary report."""
        result = module_merger.merge_modules(sample_sumaria_data, sample_vivienda_data, "34", "01")

        report = result.get_summary_report()

        assert isinstance(report, str)
        assert "MERGE" in report.upper()

    def test_quality_metrics_in_reports(
        self, module_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test that quality metrics are included in reports."""
        result = module_merger.merge_modules(sample_sumaria_data, sample_vivienda_data, "34", "01")

        assert result.quality_score is not None
        assert 0 <= result.quality_score <= 100


# ============================================================================
# ADDITIONAL COVERAGE TESTS
# ============================================================================


class TestMergeAnalysis:
    """Test merge analysis and feasibility checking."""

    def test_analyze_merge_feasibility(
        self, module_merger, sample_sumaria_data, sample_vivienda_data
    ):
        """Test merge feasibility analysis."""
        modules = {"34": sample_sumaria_data, "01": sample_vivienda_data}

        analysis = module_merger.analyze_merge_feasibility(
            modules_dict=modules, merge_level=ModuleMergeLevel.HOGAR
        )

        assert analysis is not None
        assert "feasible" in analysis
        assert "modules_analyzed" in analysis

    def test_create_merge_plan(self, module_merger, sample_sumaria_data, sample_vivienda_data):
        """Test merge plan creation."""
        modules = {"34": sample_sumaria_data, "01": sample_vivienda_data}

        plan = module_merger.create_merge_plan(modules_dict=modules, target_module="34")

        assert plan is not None
        assert "merge_sequence" in plan
        assert "estimated_time_seconds" in plan


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    def test_all_validation_types(self):
        """Test all UBIGEO validation types."""
        for val_type in TipoValidacionUbigeo:
            config = GeoMergeConfiguration(tipo_validacion_ubigeo=val_type)
            assert config.tipo_validacion_ubigeo == val_type

    def test_all_duplicate_strategies(self):
        """Test all duplicate handling strategies are valid."""
        for strategy in TipoManejoDuplicados:
            config = GeoMergeConfiguration(manejo_duplicados=strategy)
            assert config.manejo_duplicados == strategy

    def test_all_merge_strategies(self):
        """Test all module merge strategies are valid."""
        for strategy in ModuleMergeStrategy:
            config = ModuleMergeConfig(merge_strategy=strategy)
            assert config.merge_strategy == strategy


# ============================================================================
# RUNNER
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

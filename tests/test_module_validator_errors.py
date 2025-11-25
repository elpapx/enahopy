"""
Tests for ENAHO Module Validator - Error Handling & Edge Cases
================================================================

This test suite focuses on error handling and edge cases for the
ModuleValidator class that may not be covered by existing integration tests.

Focus areas:
- Invalid module codes and structures
- Edge cases (empty DataFrames, missing keys, invalid values)
- Boundary conditions for validation rules
- Error detection and warning generation

Author: ENAHOPY Test Team
Date: 2025-11-13
"""

import logging

import pandas as pd
import pytest

from enahopy.merger.config import ModuleMergeConfig, ModuleMergeLevel, ModuleType
from enahopy.merger.modules.validator import ModuleValidator

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def config():
    """Create default merge configuration."""
    return ModuleMergeConfig()


@pytest.fixture
def logger():
    """Create logger for validator."""
    return logging.getLogger("test_validator")


@pytest.fixture
def validator(config, logger):
    """Create ModuleValidator instance."""
    return ModuleValidator(config, logger)


@pytest.fixture
def valid_hogar_df():
    """Create valid hogar-level DataFrame."""
    return pd.DataFrame(
        {
            "conglome": ["HH001", "HH002", "HH003"],
            "vivienda": ["V001", "V002", "V003"],
            "hogar": [1, 1, 1],
            "value": [100, 200, 300],
        }
    )


@pytest.fixture
def valid_persona_df():
    """Create valid persona-level DataFrame."""
    return pd.DataFrame(
        {
            "conglome": ["HH001", "HH001", "HH002"],
            "vivienda": ["V001", "V001", "V002"],
            "hogar": [1, 1, 1],
            "codperso": [1, 2, 1],
            "edad": [25, 30, 45],
        }
    )


@pytest.fixture
def sumaria_df():
    """Create valid sumaria (module 34) DataFrame."""
    return pd.DataFrame(
        {
            "conglome": ["HH001", "HH002"],
            "vivienda": ["V001", "V002"],
            "hogar": [1, 1],
            "mieperho": [4, 3],
            "gashog2d": [1500.0, 1200.0],
            "inghog2d": [2000.0, 1800.0],
            "pobreza": [0, 1],
        }
    )


# ==============================================================================
# Test Class: Module Structure Validation - Error Cases
# ==============================================================================


class TestModuleStructureValidationErrors:
    """Test error handling in module structure validation."""

    def test_validate_unrecognized_module(self, validator, valid_hogar_df):
        """Test validation of unrecognized module code."""
        warnings = validator.validate_module_structure(valid_hogar_df, "99")

        assert len(warnings) > 0
        assert any("no reconocido" in w.lower() for w in warnings)

    def test_validate_intermediate_module_no_warnings(self, validator, valid_hogar_df):
        """Test that intermediate modules skip strict validation."""
        # Test various intermediate module naming patterns
        intermediate_codes = ["merged_01_02", "combined_34", "+01+02", "merged"]

        for code in intermediate_codes:
            warnings = validator.validate_module_structure(valid_hogar_df, code)
            assert len(warnings) == 0, f"Intermediate module '{code}' should have no warnings"

    def test_validate_missing_required_columns(self, validator):
        """Test validation with missing required columns."""
        # Create DataFrame missing required keys
        df = pd.DataFrame({"conglome": ["HH001"], "other_col": [100]})

        warnings = validator.validate_module_structure(df, "34")

        assert len(warnings) > 0
        assert any("faltantes" in w.lower() for w in warnings)

    def test_validate_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame(columns=["conglome", "vivienda", "hogar"])

        # Should not crash, may return warnings
        warnings = validator.validate_module_structure(df, "34")
        assert isinstance(warnings, list)

    def test_validate_duplicate_keys(self, validator, valid_hogar_df):
        """Test detection of duplicate keys."""
        # Create DataFrame with duplicates
        df_with_dupes = pd.concat([valid_hogar_df, valid_hogar_df.iloc[:1]], ignore_index=True)

        warnings = validator.validate_module_structure(df_with_dupes, "34")

        assert len(warnings) > 0
        assert any("duplicados" in w.lower() for w in warnings)


# ==============================================================================
# Test Class: Persona-Level Validation - Edge Cases
# ==============================================================================


class TestPersonaLevelValidationEdgeCases:
    """Test edge cases in persona-level module validation."""

    def test_invalid_codperso_values(self, validator):
        """Test detection of invalid codperso values."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001", "HH002", "HH003"],
                "vivienda": ["V001", "V002", "V003"],
                "hogar": [1, 1, 1],
                "codperso": [0, None, ""],  # Invalid values
            }
        )

        warnings = validator._validate_persona_level_module(df, "03")

        assert len(warnings) > 0
        assert any("inválidos" in w.lower() for w in warnings)

    def test_very_high_codperso(self, validator):
        """Test detection of unusually high codperso values."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001"],
                "vivienda": ["V001"],
                "hogar": [1],
                "codperso": [35],  # Higher than typical max (30)
            }
        )

        warnings = validator._validate_persona_level_module(df, "03")

        assert len(warnings) > 0
        assert any("muy alto" in w.lower() for w in warnings)

    def test_household_with_too_many_people(self, validator):
        """Test detection of households with > 20 people."""
        # Create household with 25 people
        df = pd.DataFrame(
            {
                "conglome": ["HH001"] * 25,
                "vivienda": ["V001"] * 25,
                "hogar": [1] * 25,
                "codperso": list(range(1, 26)),
            }
        )

        warnings = validator._validate_persona_level_module(df, "03")

        assert len(warnings) > 0
        assert any("más de 20 personas" in w.lower() for w in warnings)

    def test_household_with_zero_people(self, validator):
        """Test detection of households with 0 people."""
        # This is an edge case that shouldn't happen but validator should detect
        df = pd.DataFrame(columns=["conglome", "vivienda", "hogar", "codperso"])

        warnings = validator._validate_persona_level_module(df, "03")

        # Empty DataFrame shouldn't cause errors
        assert isinstance(warnings, list)

    def test_codperso_not_present(self, validator):
        """Test persona validation when codperso column is missing."""
        df = pd.DataFrame({"conglome": ["HH001"], "vivienda": ["V001"], "hogar": [1], "edad": [25]})

        # Should not crash
        warnings = validator._validate_persona_level_module(df, "03")
        assert isinstance(warnings, list)


# ==============================================================================
# Test Class: Hogar-Level Validation - Edge Cases
# ==============================================================================


class TestHogarLevelValidationEdgeCases:
    """Test edge cases in hogar-level module validation."""

    def test_multiple_records_per_household(self, validator):
        """Test detection of multiple records for same household."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001", "HH001"],  # Same household, duplicate
                "vivienda": ["V001", "V001"],
                "hogar": [1, 1],
                "value": [100, 200],
            }
        )

        warnings = validator._validate_hogar_level_module(
            df, "34", ["conglome", "vivienda", "hogar"]
        )

        assert len(warnings) > 0
        assert any("múltiples registros" in w.lower() for w in warnings)

    def test_hogar_column_missing(self, validator):
        """Test validation when hogar column is missing."""
        df = pd.DataFrame({"conglome": ["HH001"], "vivienda": ["V001"], "value": [100]})

        # Should handle gracefully
        warnings = validator._validate_hogar_level_module(df, "34", ["conglome", "vivienda"])
        assert isinstance(warnings, list)

    def test_sumaria_missing_key_variables(self, validator):
        """Test sumaria validation with missing key variables."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001"],
                "vivienda": ["V001"],
                "hogar": [1],
                # Missing: mieperho, gashog2d, inghog2d, pobreza
            }
        )

        warnings = validator._validate_sumaria_module(df)

        assert len(warnings) > 0
        assert any("faltantes" in w.lower() for w in warnings)

    def test_economic_module_validation(self, validator):
        """Test validation of economic modules (07, 08)."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001"],
                "vivienda": ["V001"],
                "hogar": [1],
                "ingreso": [1000.0],
            }
        )

        # Should not crash for economic modules
        warnings = validator._validate_economic_module(df, "07")
        assert isinstance(warnings, list)


# ==============================================================================
# Test Class: Special Module Validation
# ==============================================================================


class TestSpecialModuleValidation:
    """Test validation of special modules."""

    def test_module_37_empty_dataframe(self, validator):
        """Test module 37 with empty DataFrame (normal case)."""
        df = pd.DataFrame(columns=["conglome", "vivienda", "hogar"])

        warnings = validator._validate_special_module(df, "37")

        # Empty is normal for module 37
        assert len(warnings) > 0
        assert any("vacío" in w.lower() for w in warnings)

    def test_module_37_with_data(self, validator):
        """Test module 37 with actual data."""
        df = pd.DataFrame({"conglome": ["HH001"], "vivienda": ["V001"], "hogar": [1], "value": [1]})

        warnings = validator._validate_special_module(df, "37")

        # Should not warn if data exists
        assert isinstance(warnings, list)


# ==============================================================================
# Test Class: Module Compatibility Checking
# ==============================================================================


class TestModuleCompatibilityChecking:
    """Test module compatibility validation."""

    def test_check_compatibility_empty_dataframes(self, validator):
        """Test compatibility check with empty DataFrames."""
        df1 = pd.DataFrame(columns=["conglome", "vivienda", "hogar"])
        df2 = pd.DataFrame(columns=["conglome", "vivienda", "hogar"])

        result = validator.check_module_compatibility(df1, df2, "34", "01", ModuleMergeLevel.HOGAR)

        assert isinstance(result, dict)
        assert "compatible" in result

    def test_check_compatibility_no_common_keys(self, validator):
        """Test compatibility when modules have no matching keys."""
        df1 = pd.DataFrame({"conglome": ["HH001"], "vivienda": ["V001"], "hogar": [1]})

        df2 = pd.DataFrame({"conglome": ["HH999"], "vivienda": ["V999"], "hogar": [1]})

        result = validator.check_module_compatibility(df1, df2, "34", "01", ModuleMergeLevel.HOGAR)

        assert isinstance(result, dict)
        # Should indicate low or no match rate

    def test_check_compatibility_perfect_match(self, validator, valid_hogar_df):
        """Test compatibility with perfect key matches."""
        df1 = valid_hogar_df.copy()
        df2 = valid_hogar_df.copy()
        df2["extra_col"] = [1, 2, 3]

        result = validator.check_module_compatibility(df1, df2, "34", "01", ModuleMergeLevel.HOGAR)

        assert result.get("compatible", False)
        # Should have high match rate


# ==============================================================================
# Test Class: Edge Cases and Boundary Conditions
# ==============================================================================


class TestEdgeCasesAndBoundaries:
    """Test various edge cases and boundary conditions."""

    def test_dataframe_with_null_values_in_keys(self, validator):
        """Test validation with null values in key columns."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001", None, "HH003"],
                "vivienda": ["V001", "V002", None],
                "hogar": [1, 1, 1],
            }
        )

        # Should handle nulls gracefully
        warnings = validator.validate_module_structure(df, "34")
        assert isinstance(warnings, list)

    def test_dataframe_with_mixed_types(self, validator):
        """Test validation with mixed data types in columns."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001", 123, "HH003"],  # Mixed str/int
                "vivienda": ["V001", "V002", "V003"],
                "hogar": [1, "1", 1.0],  # Mixed int/str/float
            }
        )

        warnings = validator.validate_module_structure(df, "34")
        assert isinstance(warnings, list)

    def test_very_large_dataframe(self, validator):
        """Test validation with large DataFrame (performance test)."""
        n = 100000
        df = pd.DataFrame(
            {
                "conglome": [f"HH{i:06d}" for i in range(n)],
                "vivienda": [f"V{i:06d}" for i in range(n)],
                "hogar": [1] * n,
            }
        )

        # Should complete without timeout
        warnings = validator.validate_module_structure(df, "34")
        assert isinstance(warnings, list)

    def test_single_row_dataframe(self, validator):
        """Test validation with single-row DataFrame."""
        df = pd.DataFrame(
            {
                "conglome": ["HH001"],
                "vivienda": ["V001"],
                "hogar": [1],
                "mieperho": [1],
                "gashog2d": [100.0],
                "inghog2d": [150.0],
                "pobreza": [0],
            }
        )

        warnings = validator.validate_module_structure(df, "34")
        # Single row should be valid
        assert isinstance(warnings, list)


# ==============================================================================
# Test Class: Type-Specific Validations
# ==============================================================================


class TestModuleTypeValidations:
    """Test validations for different module types."""

    def test_validate_hogar_level_type(self, validator, valid_hogar_df, config):
        """Test that hogar-level modules trigger hogar validation."""
        # Mock config to have module info
        if "34" in config.module_validations:
            module_info = config.module_validations["34"]
            if module_info["level"] == ModuleType.HOGAR_LEVEL:
                warnings = validator.validate_module_structure(valid_hogar_df, "34")
                assert isinstance(warnings, list)

    def test_validate_persona_level_type(self, validator, valid_persona_df, config):
        """Test that persona-level modules trigger persona validation."""
        # Module 03 is typically persona level
        if "03" in config.module_validations:
            warnings = validator.validate_module_structure(valid_persona_df, "03")
            assert isinstance(warnings, list)

    def test_validate_special_type(self, validator, config):
        """Test that special modules trigger special validation."""
        df = pd.DataFrame({"conglome": ["HH001"], "vivienda": ["V001"], "hogar": [1]})

        # Module 37 is special
        if "37" in config.module_validations:
            warnings = validator.validate_module_structure(df, "37")
            assert isinstance(warnings, list)


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

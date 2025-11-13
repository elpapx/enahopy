"""
Tests for ENAHO Loader I/O Utilities
=====================================

This test suite validates the convenience functions and utilities
in enahopy.loader.utils.io_utils module.

Focus areas:
- Error handling for invalid inputs
- Edge cases (empty data, missing files, etc.)
- Validation functions
- Utility helper methods

Author: ENAHOPY Test Team
Date: 2025-11-13
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from enahopy.loader.core.exceptions import ENAHOError, ENAHOValidationError
from enahopy.loader.utils.io_utils import (
    ENAHOUtils,
    download_enaho_data,
    find_enaho_files,
    get_available_data,
    get_file_info,
    read_enaho_file,
    validate_download_request,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_io_utils_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataframe():
    """Create sample ENAHO-like DataFrame."""
    return pd.DataFrame(
        {
            "conglome": ["HH001", "HH002", "HH003"],
            "vivienda": ["V001", "V002", "V003"],
            "hogar": [1, 1, 1],
            "codperso": ["P01", "P02", "P03"],
            "value": [100, 200, 300],
        }
    )


@pytest.fixture
def sample_test_file(temp_test_dir, sample_dataframe):
    """Create a sample test file."""
    test_file = temp_test_dir / "test_data.csv"
    sample_dataframe.to_csv(test_file, index=False)
    return test_file


# ==============================================================================
# Test Class: ENAHOUtils - Estimate Download Size
# ==============================================================================


class TestENAHOUtilsEstimateSize:
    """Test download size estimation functionality."""

    def test_estimate_download_size_single_module_single_year(self):
        """Test estimation for single module and single year."""
        result = ENAHOUtils.estimate_download_size(["01"], ["2023"])

        assert "total_mb" in result
        assert "total_gb" in result
        assert "by_module" in result
        assert "compressed_size" in result
        assert result["total_mb"] > 0
        assert result["by_module"]["01"] > 0

    def test_estimate_download_size_multiple_modules(self):
        """Test estimation for multiple modules."""
        result = ENAHOUtils.estimate_download_size(["01", "02", "34"], ["2023"])

        assert result["total_mb"] > 0
        assert len(result["by_module"]) == 3
        assert "01" in result["by_module"]
        assert "02" in result["by_module"]
        assert "34" in result["by_module"]

    def test_estimate_download_size_multiple_years(self):
        """Test estimation scales with multiple years."""
        result_one_year = ENAHOUtils.estimate_download_size(["01"], ["2023"])
        result_three_years = ENAHOUtils.estimate_download_size(["01"], ["2023", "2022", "2021"])

        # Total should be approximately 3x for 3 years
        assert result_three_years["total_mb"] > result_one_year["total_mb"] * 2.5

    def test_estimate_download_size_unknown_module(self):
        """Test estimation with unknown module uses default size."""
        result = ENAHOUtils.estimate_download_size(["99"], ["2023"])

        assert result["total_mb"] > 0  # Should use default value
        assert "99" in result["by_module"]

    def test_estimate_download_size_empty_lists(self):
        """Test estimation with empty inputs."""
        result = ENAHOUtils.estimate_download_size([], [])

        assert result["total_mb"] == 0
        assert result["total_gb"] == 0
        assert len(result["by_module"]) == 0

    def test_estimate_download_size_compressed_calculation(self):
        """Test compressed size is reasonable fraction of total."""
        result = ENAHOUtils.estimate_download_size(["01", "34"], ["2023"])

        # Compressed should be ~30% of total
        assert result["compressed_size"] < result["total_mb"]
        assert result["compressed_size"] > 0


# ==============================================================================
# Test Class: ENAHOUtils - Module Description
# ==============================================================================


class TestENAHOUtilsModuleDescription:
    """Test module description retrieval."""

    def test_get_module_description_valid_module(self):
        """Test getting description for valid module."""
        desc = ENAHOUtils.get_module_description("34")

        assert desc is not None
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_module_description_with_leading_zero(self):
        """Test module with leading zero normalization."""
        desc1 = ENAHOUtils.get_module_description("01")
        desc2 = ENAHOUtils.get_module_description("1")

        # Both should return same description (after normalization)
        assert desc1 == desc2

    def test_get_module_description_unknown_module(self):
        """Test unknown module returns default message."""
        desc = ENAHOUtils.get_module_description("99")

        assert "desconocido" in desc.lower() or desc == "Módulo desconocido"


# ==============================================================================
# Test Class: ENAHOUtils - Parallel Settings Recommendations
# ==============================================================================


class TestENAHOUtilsParallelRecommendations:
    """Test parallel download settings recommendations."""

    def test_recommend_parallel_few_files(self):
        """Test recommendation for few files (≤2)."""
        result = ENAHOUtils.recommend_parallel_settings(2)

        assert result["parallel"] is False
        assert result["max_workers"] == 1
        assert "reason" in result

    def test_recommend_parallel_moderate_files(self):
        """Test recommendation for moderate files (3-8)."""
        result = ENAHOUtils.recommend_parallel_settings(5)

        assert result["parallel"] is True
        assert result["max_workers"] == 2
        assert "moderada" in result["reason"].lower()

    def test_recommend_parallel_high_files(self):
        """Test recommendation for high files (9-20)."""
        result = ENAHOUtils.recommend_parallel_settings(15)

        assert result["parallel"] is True
        assert result["max_workers"] == 4
        assert "alta" in result["reason"].lower()

    def test_recommend_parallel_very_high_files(self):
        """Test recommendation for very high files (>20)."""
        result = ENAHOUtils.recommend_parallel_settings(25)

        assert result["parallel"] is True
        assert result["max_workers"] == 6
        assert "muy alta" in result["reason"].lower()

    def test_recommend_parallel_boundary_values(self):
        """Test boundary values for recommendations."""
        # Test exact boundaries
        result_2 = ENAHOUtils.recommend_parallel_settings(2)
        result_3 = ENAHOUtils.recommend_parallel_settings(3)

        assert result_2["parallel"] is False
        assert result_3["parallel"] is True


# ==============================================================================
# Test Class: ENAHOUtils - Merge DataFrames
# ==============================================================================


class TestENAHOUtilsMergeDataFrames:
    """Test DataFrame merging utilities."""

    def test_merge_dataframes_default_keys(self, sample_dataframe):
        """Test merge with default ENAHO keys."""
        df1 = sample_dataframe.copy()
        df2 = sample_dataframe.copy()
        df2["new_col"] = [1, 2, 3]

        result = ENAHOUtils.merge_enaho_dataframes({"df1": df1, "df2": df2})

        assert not result.empty
        assert "new_col" in result.columns
        assert len(result) > 0

    def test_merge_dataframes_custom_keys(self, sample_dataframe):
        """Test merge with custom keys."""
        df1 = sample_dataframe.copy()
        df2 = sample_dataframe.copy()

        result = ENAHOUtils.merge_enaho_dataframes(
            {"df1": df1, "df2": df2}, on=["conglome", "vivienda"], how="inner"
        )

        assert not result.empty
        assert len(result) > 0

    def test_merge_dataframes_empty_dict(self):
        """Test merge with empty dictionary."""
        result = ENAHOUtils.merge_enaho_dataframes({})

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_merge_dataframes_missing_keys(self, sample_dataframe):
        """Test merge when some keys are missing."""
        df1 = sample_dataframe.copy()
        df2 = sample_dataframe.copy()
        df2 = df2.drop(columns=["codperso"])  # Remove one key column

        # Should still work with warning
        with pytest.warns(UserWarning):
            result = ENAHOUtils.merge_enaho_dataframes(
                {"df1": df1, "df2": df2}, on=["conglome", "vivienda", "hogar", "codperso"]
            )

        # Should still produce results with available keys
        assert isinstance(result, pd.DataFrame)

    def test_merge_dataframes_left_join(self, sample_dataframe):
        """Test left join behavior."""
        df1 = sample_dataframe.copy()
        df2 = sample_dataframe.copy().iloc[:2]  # Only 2 rows

        result = ENAHOUtils.merge_enaho_dataframes({"df1": df1, "df2": df2}, how="left")

        # Should preserve all rows from df1
        assert len(result) >= len(df1)

    def test_merge_dataframes_with_codperso(self, sample_dataframe):
        """Test merge includes codperso when available in all DataFrames."""
        df1 = sample_dataframe.copy()
        df2 = sample_dataframe.copy()

        result = ENAHOUtils.merge_enaho_dataframes({"df1": df1, "df2": df2})

        # codperso should be used in the merge
        assert "codperso" in result.columns


# ==============================================================================
# Test Class: ENAHOUtils - Validate Keys
# ==============================================================================


class TestENAHOUtilsValidateKeys:
    """Test ENAHO key validation functionality."""

    def test_validate_keys_hogar_level_valid(self, sample_dataframe):
        """Test validation at hogar level with valid keys."""
        result = ENAHOUtils.validate_enaho_keys(sample_dataframe, level="hogar")

        assert result["is_valid"] == True
        assert result["level"] == "hogar"
        assert result["duplicates"] == 0
        assert result["total_records"] == len(sample_dataframe)

    def test_validate_keys_persona_level_valid(self, sample_dataframe):
        """Test validation at persona level with valid keys."""
        result = ENAHOUtils.validate_enaho_keys(sample_dataframe, level="persona")

        assert result["is_valid"] == True
        assert result["level"] == "persona"
        assert result["duplicates"] == 0

    def test_validate_keys_vivienda_level_valid(self, sample_dataframe):
        """Test validation at vivienda level with valid keys."""
        result = ENAHOUtils.validate_enaho_keys(sample_dataframe, level="vivienda")

        assert result["is_valid"] == True
        assert result["level"] == "vivienda"

    def test_validate_keys_missing_columns(self, sample_dataframe):
        """Test validation with missing required columns."""
        df = sample_dataframe.drop(columns=["hogar"])

        result = ENAHOUtils.validate_enaho_keys(df, level="hogar")

        assert result["is_valid"] is False
        assert "missing_keys" in result
        assert "hogar" in result["missing_keys"]
        assert "error" in result

    def test_validate_keys_duplicates(self, sample_dataframe):
        """Test validation detects duplicate keys."""
        # Create DataFrame with duplicates
        df = pd.concat([sample_dataframe, sample_dataframe.iloc[:1]], ignore_index=True)

        result = ENAHOUtils.validate_enaho_keys(df, level="hogar")

        assert result["is_valid"] == False
        assert result["duplicates"] > 0

    def test_validate_keys_invalid_level(self, sample_dataframe):
        """Test validation with invalid level raises error."""
        with pytest.raises(ValueError, match="Nivel no soportado"):
            ENAHOUtils.validate_enaho_keys(sample_dataframe, level="invalid_level")

    def test_validate_keys_completeness_calculation(self, sample_dataframe):
        """Test completeness is calculated correctly."""
        result = ENAHOUtils.validate_enaho_keys(sample_dataframe, level="hogar")

        assert "completeness" in result
        assert result["completeness"] == 100.0  # All keys are complete

    def test_validate_keys_with_nulls(self, sample_dataframe):
        """Test validation with null values in keys."""
        df = sample_dataframe.copy()
        df.loc[0, "hogar"] = None

        result = ENAHOUtils.validate_enaho_keys(df, level="hogar")

        assert result["completeness"] < 100.0  # Should detect incomplete keys


# ==============================================================================
# Test Class: Convenience Functions - Error Handling
# ==============================================================================


class TestConvenienceFunctionsErrorHandling:
    """Test error handling in convenience functions."""

    def test_read_enaho_file_nonexistent_file(self):
        """Test reading non-existent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, ENAHOError)):
            read_enaho_file("/nonexistent/path/file.dta")

    def test_get_file_info_nonexistent_file(self):
        """Test getting info for non-existent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, ENAHOError)):
            get_file_info("/nonexistent/path/file.dta")

    def test_find_enaho_files_nonexistent_directory(self):
        """Test finding files in non-existent directory raises error."""
        with pytest.raises((FileNotFoundError, ENAHOError)):
            find_enaho_files("/nonexistent/directory/")

    def test_get_available_data_transversal(self):
        """Test getting available data for transversal dataset."""
        result = get_available_data(is_panel=False)

        assert "years" in result
        assert "modules" in result
        assert "dataset_type" in result
        assert result["dataset_type"] == "transversal"

    def test_get_available_data_panel(self):
        """Test getting available data for panel dataset."""
        result = get_available_data(is_panel=True)

        assert "years" in result
        assert "modules" in result
        assert result["dataset_type"] == "panel"


# ==============================================================================
# Test Class: Download Validation
# ==============================================================================


class TestDownloadValidation:
    """Test download request validation."""

    def test_validate_download_request_valid(self):
        """Test validation of valid download request."""
        result = validate_download_request(["01", "34"], ["2022", "2023"])

        assert isinstance(result, dict)
        # Should have validation information

    def test_validate_download_request_empty_modules(self):
        """Test validation with empty modules list."""
        # Should handle gracefully or raise appropriate error
        result = validate_download_request([], ["2023"])
        assert isinstance(result, dict)

    def test_validate_download_request_empty_years(self):
        """Test validation with empty years list."""
        result = validate_download_request(["01"], [])
        assert isinstance(result, dict)


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Test edge cases across utility functions."""

    def test_estimate_download_size_zero_modules(self):
        """Test estimation with zero modules."""
        result = ENAHOUtils.estimate_download_size([], ["2023"])

        assert result["total_mb"] == 0
        assert len(result["by_module"]) == 0

    def test_merge_dataframes_single_dataframe(self):
        """Test merge with only one DataFrame."""
        df = pd.DataFrame({"conglome": ["HH001"], "vivienda": ["V001"], "hogar": [1]})

        result = ENAHOUtils.merge_enaho_dataframes({"df1": df})

        assert not result.empty
        assert len(result) == 1

    def test_validate_keys_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame(columns=["conglome", "vivienda", "hogar"])

        result = ENAHOUtils.validate_enaho_keys(df, level="hogar")

        assert result["total_records"] == 0
        assert result["duplicates"] == 0

    def test_validate_keys_single_row_dataframe(self):
        """Test validation with single row DataFrame."""
        df = pd.DataFrame({"conglome": ["HH001"], "vivienda": ["V001"], "hogar": [1]})

        result = ENAHOUtils.validate_enaho_keys(df, level="hogar")

        assert result["is_valid"] == True
        assert result["total_records"] == 1


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

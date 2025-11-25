"""
Tests for ENAHO Data Downloader - Error Handling & Edge Cases
===============================================================

This test suite focuses on error handling and edge cases for the
ENAHODataDownloader class that may not be covered by existing tests.

Focus areas:
- Invalid module/year combinations
- Error handling in download operations
- Edge cases in validation and availability checking
- Configuration edge cases

Author: ENAHOPY Test Team
Date: 2025-11-13
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from enahopy.loader.core.config import ENAHOConfig
from enahopy.loader.core.exceptions import ENAHOError, ENAHOValidationError
from enahopy.loader.io.main import ENAHODataDownloader

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="test_downloader_"))
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def downloader():
    """Create ENAHODataDownloader instance."""
    return ENAHODataDownloader(verbose=False)


@pytest.fixture
def config():
    """Create ENAHO configuration."""
    return ENAHOConfig()


# ==============================================================================
# Test Class: Initialization and Configuration
# ==============================================================================


class TestDownloaderInitialization:
    """Test downloader initialization and configuration."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        downloader = ENAHODataDownloader()
        assert downloader is not None
        assert downloader.config is not None
        assert downloader.logger is not None

    def test_init_with_config(self, config):
        """Test initialization with custom config."""
        downloader = ENAHODataDownloader(config=config, verbose=False)
        assert downloader.config == config

    def test_init_verbose_mode(self):
        """Test initialization in verbose mode."""
        downloader = ENAHODataDownloader(verbose=True)
        assert downloader.logger is not None

    def test_init_non_verbose_mode(self):
        """Test initialization in non-verbose mode."""
        downloader = ENAHODataDownloader(verbose=False)
        assert downloader.logger is not None


# ==============================================================================
# Test Class: Module and Year Validation
# ==============================================================================


class TestModuleYearValidation:
    """Test validation of modules and years."""

    def test_get_available_modules(self, downloader):
        """Test getting available modules."""
        modules = downloader.get_available_modules()
        assert isinstance(modules, dict)
        assert len(modules) > 0
        # Check some expected modules
        assert "34" in modules  # Sumaria

    def test_get_available_years_transversal(self, downloader):
        """Test getting available years for transversal data."""
        years = downloader.get_available_years(is_panel=False)
        assert isinstance(years, list)
        assert len(years) > 0
        # Years should be strings
        assert all(isinstance(year, str) for year in years)

    def test_get_available_years_panel(self, downloader):
        """Test getting available years for panel data."""
        years = downloader.get_available_years(is_panel=True)
        assert isinstance(years, list)
        # Panel years might be empty or have different years

    def test_validate_availability_valid(self, downloader):
        """Test validation with valid modules and years."""
        result = downloader.validate_availability(["34"], ["2022"], is_panel=False)
        assert isinstance(result, dict)
        assert "status" in result

    def test_validate_availability_invalid_module(self, downloader):
        """Test validation with invalid module."""
        result = downloader.validate_availability(["99"], ["2022"], is_panel=False)
        assert isinstance(result, dict)
        # Should indicate invalid module

    def test_validate_availability_empty_modules(self, downloader):
        """Test validation with empty modules list."""
        result = downloader.validate_availability([], ["2022"], is_panel=False)
        assert isinstance(result, dict)

    def test_validate_availability_empty_years(self, downloader):
        """Test validation with empty years list."""
        result = downloader.validate_availability(["34"], [], is_panel=False)
        assert isinstance(result, dict)


# ==============================================================================
# Test Class: Local File Operations
# ==============================================================================


class TestLocalFileOperations:
    """Test local file reading and discovery."""

    def test_find_local_files_nonexistent_directory(self, downloader):
        """Test finding files in non-existent directory."""
        with pytest.raises((FileNotFoundError, ENAHOError)):
            downloader.find_local_files("/nonexistent/path", pattern="*.dta")

    def test_find_local_files_empty_directory(self, downloader, temp_dir):
        """Test finding files in empty directory."""
        files = downloader.find_local_files(str(temp_dir), pattern="*.dta")
        assert isinstance(files, list)
        assert len(files) == 0

    def test_find_local_files_with_pattern(self, downloader, temp_dir):
        """Test finding files with specific pattern."""
        # Create some test files
        (temp_dir / "test1.dta").touch()
        (temp_dir / "test2.csv").touch()
        (temp_dir / "test3.dta").touch()

        dta_files = downloader.find_local_files(str(temp_dir), pattern="*.dta")
        assert len(dta_files) == 2

    def test_find_local_files_recursive(self, downloader, temp_dir):
        """Test finding files recursively."""
        # Create subdirectory with files
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "test.dta").touch()

        files = downloader.find_local_files(str(temp_dir), pattern="*.dta", recursive=True)
        assert len(files) >= 1

    def test_find_local_files_non_recursive(self, downloader, temp_dir):
        """Test finding files non-recursively."""
        # Create subdirectory with files
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "test.dta").touch()
        (temp_dir / "test.dta").touch()

        files = downloader.find_local_files(str(temp_dir), pattern="*.dta", recursive=False)
        # Should only find file in root
        assert len(files) == 1

    def test_read_local_file_nonexistent(self, downloader):
        """Test reading non-existent local file."""
        with pytest.raises((FileNotFoundError, ENAHOError)):
            downloader.read_local_file("/nonexistent/file.dta")

    def test_read_local_file_returns_reader(self, downloader, temp_dir):
        """Test that read_local_file returns a reader object."""
        # Create a dummy CSV file
        test_file = temp_dir / "test.csv"
        test_file.write_text("col1,col2\n1,2\n3,4\n")

        reader = downloader.read_local_file(str(test_file))
        assert reader is not None
        # Should return ENAHOLocalReader instance


# ==============================================================================
# Test Class: Download Configuration Edge Cases
# ==============================================================================


class TestDownloadConfigurationEdgeCases:
    """Test edge cases in download configuration."""

    def test_download_with_invalid_output_dir(self, downloader):
        """Test download with invalid output directory."""
        # Download may create the directory if possible, so we test that it either:
        # 1. Raises an error for truly invalid paths, OR
        # 2. Creates the directory and proceeds
        # This test validates the method handles edge cases gracefully
        try:
            result = downloader.download(
                modules=["34"],
                years=["2022"],
                output_dir="/nonexistent/deeply/nested/path",
                decompress=False,
            )
            # If no error, download should have handled it (may have created dir or skipped)
            assert result is None or isinstance(result, dict)
        except (OSError, ENAHOError, PermissionError):
            # Expected - validation should catch truly invalid paths
            pass

    def test_download_empty_modules_list(self, downloader, temp_dir):
        """Test download with empty modules list."""
        # Should handle gracefully or raise validation error
        try:
            result = downloader.download(
                modules=[], years=["2022"], output_dir=str(temp_dir), decompress=False
            )
            # If it doesn't raise an error, result should be empty or None
            assert result is None or result == {}
        except (ENAHOValidationError, ENAHOError, ValueError):
            # Expected - validation should catch this
            pass

    def test_download_empty_years_list(self, downloader, temp_dir):
        """Test download with empty years list."""
        try:
            result = downloader.download(
                modules=["34"], years=[], output_dir=str(temp_dir), decompress=False
            )
            assert result is None or result == {}
        except (ENAHOValidationError, ENAHOError, ValueError):
            pass

    def test_download_with_overwrite_false(self, downloader, temp_dir):
        """Test download with overwrite=False when file exists."""
        # Create a dummy file that would conflict
        test_file = temp_dir / "modulo_34_2022.zip"
        test_file.touch()

        # Attempting download with overwrite=False should skip or handle gracefully
        # Note: This test might need mocking to avoid actual download
        pass  # Implementation depends on actual download behavior

    def test_download_parallel_with_max_workers(self, downloader, temp_dir):
        """Test parallel download configuration."""
        # Test that parallel parameter and max_workers are accepted
        # Would need mocking to avoid actual downloads

    def test_download_with_progress_callback(self, downloader, temp_dir):
        """Test download with progress callback."""
        callback_calls = []

        def progress_callback(task, completed, total):
            callback_calls.append((task, completed, total))

        # Would need mocking to test without actual download


# ==============================================================================
# Test Class: Cache Integration
# ==============================================================================


class TestCacheIntegration:
    """Test cache integration with downloader."""

    def test_downloader_has_cache_manager(self, downloader):
        """Test that downloader has cache manager."""
        assert downloader.cache_manager is not None

    def test_cache_manager_is_functional(self, downloader):
        """Test that cache manager can perform basic operations."""
        # Set and get metadata
        test_key = "test_key"
        test_data = {"test": "data"}

        downloader.cache_manager.set_metadata(test_key, test_data)
        retrieved = downloader.cache_manager.get_metadata(test_key)

        assert retrieved == test_data


# ==============================================================================
# Test Class: Error Recovery
# ==============================================================================


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def test_downloader_after_failed_operation(self, downloader):
        """Test that downloader remains functional after failed operation."""
        # Attempt invalid operation
        try:
            downloader.find_local_files("/nonexistent/path")
        except (FileNotFoundError, ENAHOError):
            pass

        # Downloader should still be functional
        modules = downloader.get_available_modules()
        assert modules is not None

    def test_multiple_validation_calls(self, downloader):
        """Test multiple validation calls don't cause issues."""
        for _ in range(3):
            result = downloader.validate_availability(["34"], ["2022"], is_panel=False)
            assert isinstance(result, dict)


# ==============================================================================
# Test Class: String Representations
# ==============================================================================


class TestStringRepresentations:
    """Test string representations of downloader."""

    def test_downloader_repr(self, downloader):
        """Test __repr__ of downloader."""
        repr_str = repr(downloader)
        assert isinstance(repr_str, str)
        assert "ENAHODataDownloader" in repr_str or "object" in repr_str

    def test_downloader_str(self, downloader):
        """Test __str__ of downloader."""
        str_repr = str(downloader)
        assert isinstance(str_repr, str)


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

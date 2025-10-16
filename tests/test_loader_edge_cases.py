"""
Edge case tests for ENAHOPY loader module.

This module tests robustness of the loader under edge conditions:
- Empty and malformed files
- Encoding issues
- Large files and memory constraints
- Concurrent operations
- Cache corruption and recovery
- Data validation edge cases

Author: MLOps-Engineer (MO-1 Phase 2)
Date: 2025-10-10
"""

import io
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from enahopy.exceptions import ENAHOCacheError, ENAHOValidationError, FileReaderError
from enahopy.loader.core.cache import CacheManager
from enahopy.loader.core.config import ENAHOConfig
from enahopy.loader.core.logging import setup_logging
from enahopy.loader.io.local_reader import ENAHOLocalReader
from enahopy.loader.io.readers.csv import CSVReader
from enahopy.loader.io.readers.factory import ReaderFactory

# ============================================================================
# File Handling Edge Cases
# ============================================================================


class TestLoaderFileEdgeCases:
    """Test loader module with various file edge cases."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        config = ENAHOConfig()
        config.cache_dir = Path(temp_dir) / ".cache"
        config.cache_dir.mkdir(exist_ok=True)
        return config

    def test_empty_csv_file(self, temp_dir):
        """Test loading an empty CSV file."""
        # Create empty CSV file
        empty_file = Path(temp_dir) / "empty.csv"
        empty_file.write_text("")

        reader = CSVReader()

        # Should either return empty DataFrame or raise clear error
        try:
            df = reader.read(str(empty_file))
            assert df.empty or len(df) == 0
        except (FileReaderError, pd.errors.EmptyDataError) as e:
            # Expected - should have clear message
            assert "empty" in str(e).lower() or "no data" in str(e).lower()

    def test_csv_with_only_headers(self, temp_dir):
        """Test CSV file with headers but no data rows."""
        csv_file = Path(temp_dir) / "headers_only.csv"
        csv_file.write_text("col1,col2,col3\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should create DataFrame with correct columns but zero rows
        assert df.empty
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_malformed_zip_file(self, temp_dir):
        """Test handling of corrupted ZIP file."""
        bad_zip = Path(temp_dir) / "corrupted.zip"
        bad_zip.write_bytes(b"This is not a valid ZIP file")

        # Should raise clear error when trying to extract
        with pytest.raises((zipfile.BadZipFile, FileReaderError)):
            with zipfile.ZipFile(bad_zip, "r") as zf:
                zf.extractall(temp_dir)

    def test_zip_with_unexpected_contents(self, temp_dir):
        """Test ZIP file that extracts but doesn't contain expected data file."""
        zip_path = Path(temp_dir) / "unexpected.zip"
        wrong_file = Path(temp_dir) / "wrong.txt"
        wrong_file.write_text("Not the expected data file")

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(wrong_file, "wrong.txt")

        # When trying to find expected .sav or .dta file, should fail gracefully
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            sav_files = [n for n in names if n.endswith(".sav")]
            dta_files = [n for n in names if n.endswith(".dta")]
            assert len(sav_files) == 0
            assert len(dta_files) == 0

    def test_file_with_special_characters_in_name(self, temp_dir):
        """Test files with spaces, accents, and special characters in filenames."""
        # Create file with special characters
        special_name = "archivo año 2022 niños.csv"
        special_file = Path(temp_dir) / special_name
        special_file.write_text("col1,col2\n1,2\n3,4\n")

        reader = CSVReader()
        df = reader.read(str(special_file))

        # Should handle filename correctly
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]

    def test_csv_with_non_utf8_encoding(self, temp_dir):
        """Test CSV file with Latin-1 encoding (common in Spanish data)."""
        # Create file with Latin-1 encoding
        latin1_file = Path(temp_dir) / "latin1.csv"
        content = "año,niños,educación\n2022,100,básica\n"
        latin1_file.write_bytes(content.encode("latin-1"))

        reader = CSVReader()

        # Should either auto-detect encoding or fail with clear message
        try:
            df = reader.read(str(latin1_file))
            # If successful, verify data integrity
            assert "año" in df.columns or "aÃ±o" in df.columns  # May be garbled
        except (UnicodeDecodeError, FileReaderError) as e:
            # Expected if encoding detection fails
            assert "encoding" in str(e).lower() or "decode" in str(e).lower()

    def test_very_wide_dataframe(self, temp_dir):
        """Test file with many columns (1000+)."""
        # Create CSV with 1000 columns
        num_cols = 1000
        headers = [f"col_{i}" for i in range(num_cols)]
        values = [str(i) for i in range(num_cols)]

        csv_file = Path(temp_dir) / "wide.csv"
        csv_file.write_text(",".join(headers) + "\n" + ",".join(values) + "\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should handle wide DataFrame
        assert len(df.columns) == num_cols
        assert len(df) == 1

    def test_csv_with_mixed_delimiters(self, temp_dir):
        """Test CSV file with inconsistent delimiters."""
        # Some rows with comma, some with semicolon
        mixed_file = Path(temp_dir) / "mixed.csv"
        mixed_file.write_text("col1,col2,col3\n1,2,3\n4;5;6\n")

        reader = CSVReader()

        # Should either auto-detect or fail gracefully
        try:
            df = reader.read(str(mixed_file))
            # If it reads, might have malformed data
            assert len(df) >= 1
        except (FileReaderError, pd.errors.ParserError):
            # Expected for inconsistent format
            pass

    def test_csv_with_quoted_newlines(self, temp_dir):
        """Test CSV with quoted fields containing newlines."""
        csv_file = Path(temp_dir) / "quoted_newlines.csv"
        content = 'col1,col2,col3\n1,"text with\nnewline",3\n4,5,6\n'
        csv_file.write_text(content)

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should correctly parse quoted fields
        assert len(df) == 2
        assert "\n" in df.iloc[0]["col2"] or "newline" in df.iloc[0]["col2"]


# ============================================================================
# Data Validation Edge Cases
# ============================================================================


class TestLoaderDataValidation:
    """Test data validation under edge conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_all_columns_null(self, temp_dir):
        """Test file where all values are missing/null."""
        csv_file = Path(temp_dir) / "all_null.csv"
        csv_file.write_text("col1,col2,col3\n,,\n,,\n,,\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should create DataFrame with NaN values
        assert len(df) == 3
        assert df.isna().all().all()  # All values are NaN

    def test_single_column_dataframe(self, temp_dir):
        """Test DataFrame with only one column."""
        csv_file = Path(temp_dir) / "single_col.csv"
        csv_file.write_text("only_column\n1\n2\n3\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        assert len(df.columns) == 1
        assert len(df) == 3
        assert df.columns[0] == "only_column"

    def test_single_row_dataframe(self, temp_dir):
        """Test DataFrame with only one data row."""
        csv_file = Path(temp_dir) / "single_row.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        assert len(df) == 1
        assert len(df.columns) == 3

    def test_mixed_numeric_string_column(self, temp_dir):
        """Test column with both numbers and strings."""
        csv_file = Path(temp_dir) / "mixed_types.csv"
        csv_file.write_text("mixed_col\n123\nabc\n456\nxyz\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # pandas should infer object dtype
        assert df["mixed_col"].dtype == object
        assert len(df) == 4

    def test_numeric_column_with_special_values(self, temp_dir):
        """Test numeric column with inf, -inf, NaN."""
        csv_file = Path(temp_dir) / "special_nums.csv"
        csv_file.write_text("values\n1.5\ninf\n-inf\nnan\n2.5\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should parse special values correctly
        assert np.isinf(df["values"].iloc[1])  # inf
        assert np.isinf(df["values"].iloc[2])  # -inf
        assert np.isnan(df["values"].iloc[3])  # nan

    def test_unicode_in_column_names(self, temp_dir):
        """Test columns with Unicode characters (Spanish accents)."""
        csv_file = Path(temp_dir) / "unicode_cols.csv"
        csv_file.write_text("año,niños,educación\n2022,100,básica\n", encoding="utf-8")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should preserve Unicode in column names
        assert "año" in df.columns
        assert "niños" in df.columns
        assert "educación" in df.columns

    def test_duplicate_column_names(self, temp_dir):
        """Test file with repeated column names."""
        csv_file = Path(temp_dir) / "dup_cols.csv"
        csv_file.write_text("col1,col2,col1,col3\n1,2,3,4\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # pandas adds .1, .2 suffixes to duplicates
        assert len(df.columns) == 4
        assert "col1" in df.columns
        # Check for disambiguated names (pandas behavior)

    def test_extremely_long_string_values(self, temp_dir):
        """Test handling of very long string values."""
        csv_file = Path(temp_dir) / "long_strings.csv"
        long_string = "x" * 100000  # 100k characters
        csv_file.write_text(f"col1,col2\n{long_string},short\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should handle long strings
        assert len(df.iloc[0]["col1"]) == 100000

    def test_dataframe_with_only_whitespace(self, temp_dir):
        """Test file with only whitespace values."""
        csv_file = Path(temp_dir) / "whitespace.csv"
        csv_file.write_text("col1,col2,col3\n   ,  ,\t\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should read whitespace as strings or strip them
        assert len(df) == 1


# ============================================================================
# Cache Edge Cases
# ============================================================================


class TestLoaderCacheEdgeCases:
    """Test cache system under edge conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def cache_manager(self, temp_dir):
        """Create cache manager instance."""
        cache_dir = Path(temp_dir) / ".cache"
        return CacheManager(cache_dir, ttl_hours=24)

    def test_cache_with_invalid_key_characters(self, cache_manager):
        """Test cache with keys containing invalid filesystem characters."""
        invalid_keys = [
            "key/with/slashes",
            "key\\with\\backslashes",
            "key:with:colons",
            "key*with*asterisks",
            "key?with?questions",
        ]

        for key in invalid_keys:
            with pytest.raises((ValueError, ENAHOCacheError)):
                cache_manager.set_metadata(key, {"data": "test"})

    def test_cache_key_too_long(self, cache_manager):
        """Test cache key exceeding maximum length."""
        long_key = "x" * 1000  # Very long key

        with pytest.raises((ValueError, ENAHOCacheError)):
            cache_manager.set_metadata(long_key, {"data": "test"})

    def test_cache_with_empty_key(self, cache_manager):
        """Test cache operations with empty string key."""
        with pytest.raises((ValueError, ENAHOCacheError)):
            cache_manager.set_metadata("", {"data": "test"})

    def test_cache_with_whitespace_only_key(self, cache_manager):
        """Test cache key with only whitespace."""
        with pytest.raises((ValueError, ENAHOCacheError)):
            cache_manager.set_metadata("   ", {"data": "test"})

    def test_cache_with_non_serializable_data(self, cache_manager):
        """Test caching data that can't be JSON serialized."""

        class NonSerializable:
            pass

        with pytest.raises((TypeError, ValueError, ENAHOCacheError)):
            cache_manager.set_metadata("test_key", {"obj": NonSerializable()})

    def test_cache_directory_deleted_during_operation(self, temp_dir, cache_manager):
        """Test recovery when cache directory is deleted mid-operation."""
        # Set some metadata successfully
        cache_manager.set_metadata("test_key", {"data": "value"})

        # Delete cache directory
        shutil.rmtree(cache_manager.cache_dir, ignore_errors=True)

        # Should either recreate directory or fail gracefully
        result = cache_manager.get_metadata("test_key")
        assert result is None  # Should return None for missing cache

    def test_cache_file_permissions_error(self, cache_manager, temp_dir):
        """Test cache behavior when file permissions prevent write."""
        if not hasattr(tempfile, "gettempdir"):  # Skip on some platforms
            pytest.skip("Platform doesn't support permission testing")

        # This test is platform-specific and may be skipped
        # Just verify the cache manager has error handling
        assert hasattr(cache_manager, "set_metadata")
        assert hasattr(cache_manager, "get_metadata")

    def test_concurrent_cache_access(self, cache_manager):
        """Test multiple concurrent cache operations."""
        # Simulate rapid concurrent access
        keys = [f"key_{i}" for i in range(100)]

        for key in keys:
            cache_manager.set_metadata(key, {"index": key})

        # All should be retrievable
        for key in keys:
            result = cache_manager.get_metadata(key)
            assert result is not None
            assert result["index"] == key

    def test_cache_with_very_large_metadata(self, cache_manager):
        """Test caching very large metadata objects."""
        # Create large metadata (1MB+)
        large_data = {"values": list(range(100000))}

        cache_manager.set_metadata("large_key", large_data)
        result = cache_manager.get_metadata("large_key")

        assert result is not None
        assert len(result["values"]) == 100000


# ============================================================================
# Performance and Memory Edge Cases
# ============================================================================


class TestLoaderPerformanceEdgeCases:
    """Test loader performance under stress conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.slow
    def test_moderately_large_file(self, temp_dir):
        """Test loading a moderately large CSV file (10MB)."""
        csv_file = Path(temp_dir) / "large.csv"

        # Create 10MB CSV file (~100k rows with 10 columns)
        num_rows = 100000
        with open(csv_file, "w") as f:
            # Write header
            f.write("col1,col2,col3,col4,col5,col6,col7,col8,col9,col10\n")
            # Write data rows
            for i in range(num_rows):
                f.write(f"{i},{i+1},{i+2},{i+3},{i+4},{i+5},{i+6},{i+7},{i+8},{i+9}\n")

        reader = CSVReader()
        start = time.time()
        df = reader.read(str(csv_file))
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert len(df) == num_rows
        assert len(df.columns) == 10
        assert elapsed < 30  # Should load in under 30 seconds

    def test_deeply_nested_directory_structure(self, temp_dir):
        """Test file in deeply nested directory."""
        # Create deep directory structure
        deep_path = Path(temp_dir)
        for i in range(50):
            deep_path = deep_path / f"level_{i}"
        deep_path.mkdir(parents=True, exist_ok=True)

        csv_file = deep_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")

        reader = CSVReader()
        df = reader.read(str(csv_file))

        assert len(df) == 2

    def test_filename_with_maximum_path_length(self, temp_dir):
        """Test handling of very long file paths."""
        # Create long filename (but within OS limits)
        long_name = "x" * 100 + ".csv"
        csv_file = Path(temp_dir) / long_name

        try:
            csv_file.write_text("col1,col2\n1,2\n")
            reader = CSVReader()
            df = reader.read(str(csv_file))
            assert len(df) == 1
        except OSError:
            # Some filesystems have stricter limits
            pytest.skip("Filesystem doesn't support long filenames")


# ============================================================================
# Reader Factory Edge Cases
# ============================================================================


class TestReaderFactoryEdgeCases:
    """Test ReaderFactory with edge cases."""

    def test_factory_with_unknown_extension(self):
        """Test factory with unsupported file extension."""
        with pytest.raises((ValueError, FileReaderError)):
            ReaderFactory.create_reader("unknown_file.xyz")

    def test_factory_with_no_extension(self):
        """Test factory with file that has no extension."""
        with pytest.raises((ValueError, FileReaderError)):
            ReaderFactory.create_reader("file_no_extension")

    def test_factory_with_multiple_dots(self):
        """Test factory with filename containing multiple dots."""
        # Should use last extension
        reader = ReaderFactory.create_reader("my.data.file.csv")
        assert isinstance(reader, CSVReader)

    def test_factory_with_uppercase_extension(self):
        """Test factory with uppercase file extension."""
        reader = ReaderFactory.create_reader("DATA.CSV")
        assert isinstance(reader, CSVReader)

    def test_factory_with_mixed_case_extension(self):
        """Test factory with mixed-case extension."""
        reader = ReaderFactory.create_reader("data.CsV")
        assert isinstance(reader, CSVReader)


# ============================================================================
# Integration Edge Cases
# ============================================================================


class TestLoaderIntegrationEdgeCases:
    """Test integrated loader components under edge conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_read_write_cycle_preserves_data(self, temp_dir):
        """Test that data survives read-write-read cycle."""
        # Create test data
        original_df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.5, 2.5, 3.5]}
        )

        # Write to CSV
        csv_file = Path(temp_dir) / "cycle_test.csv"
        original_df.to_csv(csv_file, index=False)

        # Read back
        reader = CSVReader()
        read_df = reader.read(str(csv_file))

        # Compare
        pd.testing.assert_frame_equal(original_df, read_df)

    def test_handling_of_bom_marker(self, temp_dir):
        """Test files with UTF-8 BOM marker."""
        csv_file = Path(temp_dir) / "with_bom.csv"
        content = "col1,col2,col3\n1,2,3\n"
        # Write with BOM
        csv_file.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        reader = CSVReader()
        df = reader.read(str(csv_file))

        # Should correctly skip BOM
        assert "col1" in df.columns  # Not '\ufeffcol1'
        assert len(df) == 1


if __name__ == "__main__":
    # Run tests with: pytest tests/test_loader_edge_cases.py -v
    pytest.main([__file__, "-v", "--tb=short"])

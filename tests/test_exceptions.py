"""
Unit tests for ENAHOPY Exception System  
======================================

Tests for the unified exception hierarchy and utility functions.

Author: ENAHOPY Team
Date: 2024-12-09
"""

import unittest
from datetime import datetime

from enahopy.exceptions import (
    DataQualityError,
    ENAHOCacheError,
    ENAHOConfigError,
    ENAHODownloadError,
    ENAHOError,
    ENAHOFileError,
    ENAHOIntegrityError,
    ENAHOMergeError,
    ENAHONullAnalysisError,
    ENAHOTimeoutError,
    ENAHOValidationError,
    FileReaderError,
    GeoMergeError,
    ImputationError,
    IncompatibleModulesError,
    MergeKeyError,
    ModuleMergeError,
    PatternDetectionError,
    TerritorialInconsistencyError,
    UbigeoValidationError,
    UnsupportedFormatError,
    VisualizationError,
    create_error_report,
    format_exception_for_logging,
    get_error_recommendations,
)


class TestENAHOExceptions(unittest.TestCase):
    """Test cases for ENAHOPY exception hierarchy."""

    def test_base_exception_creation(self):
        """Test ENAHOError base exception creation."""
        exc = ENAHOError(
            "Test message",
            error_code="TEST_ERROR",
            operation="test_operation",
            context_key="context_value",
        )

        self.assertEqual(str(exc), "Test message [TEST_ERROR] during 'test_operation'")
        self.assertEqual(exc.message, "Test message")
        self.assertEqual(exc.error_code, "TEST_ERROR")
        self.assertEqual(exc.operation, "test_operation")
        self.assertEqual(exc.context["context_key"], "context_value")
        self.assertIsInstance(exc.timestamp, str)

        # Test timestamp format
        datetime.fromisoformat(exc.timestamp)  # Should not raise

    def test_base_exception_minimal(self):
        """Test ENAHOError with minimal parameters."""
        exc = ENAHOError("Simple message")

        self.assertEqual(str(exc), "Simple message")
        self.assertEqual(exc.message, "Simple message")
        self.assertIsNone(exc.error_code)
        self.assertIsNone(exc.operation)
        self.assertEqual(exc.context, {})

    def test_exception_to_dict(self):
        """Test exception serialization to dictionary."""
        exc = ENAHOError(
            "Test message",
            error_code="TEST_ERROR",
            operation="test_operation",
            test_context="value",
        )

        exc_dict = exc.to_dict()

        expected_keys = {
            "exception_type",
            "message",
            "error_code",
            "operation",
            "context",
            "timestamp",
            "is_enahopy_exception",
        }
        self.assertEqual(set(exc_dict.keys()), expected_keys)

        self.assertEqual(exc_dict["exception_type"], "ENAHOError")
        self.assertEqual(exc_dict["message"], "Test message")
        self.assertEqual(exc_dict["error_code"], "TEST_ERROR")
        self.assertEqual(exc_dict["operation"], "test_operation")
        self.assertEqual(exc_dict["context"]["test_context"], "value")
        self.assertTrue(exc_dict["is_enahopy_exception"])

    def test_download_error_specific_attributes(self):
        """Test ENAHODownloadError specific attributes."""
        exc = ENAHODownloadError(
            "Download failed",
            url="https://example.com/data.zip",
            status_code=404,
            error_code="DOWNLOAD_NOT_FOUND",
        )

        self.assertIsInstance(exc, ENAHOError)
        self.assertEqual(exc.url, "https://example.com/data.zip")
        self.assertEqual(exc.status_code, 404)

    def test_validation_error_specific_attributes(self):
        """Test ENAHOValidationError specific attributes."""
        failures = ["Invalid year", "Missing module"]
        exc = ENAHOValidationError(
            "Validation failed", validation_failures=failures, error_code="VALIDATION_FAILED"
        )

        self.assertIsInstance(exc, ENAHOError)
        self.assertEqual(exc.validation_failures, failures)

    def test_integrity_error_specific_attributes(self):
        """Test ENAHOIntegrityError specific attributes."""
        exc = ENAHOIntegrityError(
            "Checksum mismatch",
            expected_checksum="abc123",
            actual_checksum="def456",
            error_code="INTEGRITY_CHECK_FAILED",
        )

        self.assertIsInstance(exc, ENAHOError)
        self.assertEqual(exc.expected_checksum, "abc123")
        self.assertEqual(exc.actual_checksum, "def456")

    def test_timeout_error_specific_attributes(self):
        """Test ENAHOTimeoutError specific attributes."""
        exc = ENAHOTimeoutError("Operation timed out", timeout_seconds=30.0, error_code="TIMEOUT")

        self.assertIsInstance(exc, ENAHOError)
        self.assertEqual(exc.timeout_seconds, 30.0)

    def test_file_error_hierarchy(self):
        """Test file error hierarchy."""
        # Test base file error
        file_exc = ENAHOFileError("File error", file_path="/path/to/file.dta")
        self.assertIsInstance(file_exc, ENAHOError)
        self.assertEqual(file_exc.file_path, "/path/to/file.dta")

        # Test file reader error
        reader_exc = FileReaderError("Cannot read file", file_path="/path/to/file.dta")
        self.assertIsInstance(reader_exc, ENAHOFileError)
        self.assertIsInstance(reader_exc, ENAHOError)

        # Test unsupported format error
        format_exc = UnsupportedFormatError(
            "Unsupported format",
            format_attempted="xyz",
            supported_formats=["dta", "sav", "csv"],
            file_path="/path/to/file.xyz",
        )
        self.assertIsInstance(format_exc, FileReaderError)
        self.assertEqual(format_exc.format_attempted, "xyz")
        self.assertEqual(format_exc.supported_formats, ["dta", "sav", "csv"])

    def test_cache_error_specific_attributes(self):
        """Test ENAHOCacheError specific attributes."""
        exc = ENAHOCacheError(
            "Cache operation failed",
            cache_operation="write",
            cache_key="test_key",
            error_code="CACHE_WRITE_FAILED",
        )

        self.assertIsInstance(exc, ENAHOError)
        self.assertEqual(exc.cache_operation, "write")
        self.assertEqual(exc.cache_key, "test_key")

    def test_merge_error_hierarchy(self):
        """Test merge error hierarchy."""
        # Base merge error
        merge_exc = ENAHOMergeError("Merge failed")
        self.assertIsInstance(merge_exc, ENAHOError)

        # Geographic merge error
        geo_exc = GeoMergeError("Geographic merge failed", merge_column="ubigeo", affected_rows=100)
        self.assertIsInstance(geo_exc, ENAHOMergeError)
        self.assertEqual(geo_exc.merge_column, "ubigeo")
        self.assertEqual(geo_exc.affected_rows, 100)

        # UBIGEO validation error
        ubigeo_exc = UbigeoValidationError(
            "Invalid UBIGEO codes", invalid_ubigeos=["123", "abc"], validation_type="format"
        )
        self.assertIsInstance(ubigeo_exc, ENAHOValidationError)
        self.assertEqual(ubigeo_exc.invalid_ubigeos, ["123", "abc"])
        self.assertEqual(ubigeo_exc.validation_type, "format")

    def test_module_merge_error_hierarchy(self):
        """Test module merge error hierarchy."""
        # Module merge error
        module_exc = ModuleMergeError(
            "Module merge failed", modules_involved=["01", "02"], merge_level="household"
        )
        self.assertIsInstance(module_exc, ENAHOMergeError)
        self.assertEqual(module_exc.modules_involved, ["01", "02"])
        self.assertEqual(module_exc.merge_level, "household")

        # Incompatible modules error
        incompat_exc = IncompatibleModulesError(
            "Modules are incompatible",
            module1="01",
            module2="34",
            compatibility_issues=["Different years", "Missing keys"],
        )
        self.assertIsInstance(incompat_exc, ModuleMergeError)
        self.assertEqual(incompat_exc.module1, "01")
        self.assertEqual(incompat_exc.module2, "34")
        self.assertEqual(incompat_exc.compatibility_issues, ["Different years", "Missing keys"])

        # Merge key error
        key_exc = MergeKeyError(
            "Merge key problems", missing_keys=["conglome"], invalid_keys=["bad_key"]
        )
        self.assertIsInstance(key_exc, ModuleMergeError)
        self.assertEqual(key_exc.missing_keys, ["conglome"])
        self.assertEqual(key_exc.invalid_keys, ["bad_key"])

    def test_null_analysis_error_hierarchy(self):
        """Test null analysis error hierarchy."""
        # Base null analysis error
        null_exc = ENAHONullAnalysisError("Null analysis failed")
        self.assertIsInstance(null_exc, ENAHOError)

        # Pattern detection error
        pattern_exc = PatternDetectionError(
            "Pattern detection failed", pattern_type="monotone", affected_columns=["col1", "col2"]
        )
        self.assertIsInstance(pattern_exc, ENAHONullAnalysisError)
        self.assertEqual(pattern_exc.pattern_type, "monotone")
        self.assertEqual(pattern_exc.affected_columns, ["col1", "col2"])

        # Visualization error
        viz_exc = VisualizationError("Visualization failed", visualization_type="heatmap")
        self.assertIsInstance(viz_exc, ENAHONullAnalysisError)
        self.assertEqual(viz_exc.visualization_type, "heatmap")

        # Imputation error
        imp_exc = ImputationError(
            "Imputation failed", imputation_strategy="mean", affected_columns=["income"]
        )
        self.assertIsInstance(imp_exc, ENAHONullAnalysisError)
        self.assertEqual(imp_exc.imputation_strategy, "mean")
        self.assertEqual(imp_exc.affected_columns, ["income"])

    def test_config_error_specific_attributes(self):
        """Test ENAHOConfigError specific attributes."""
        exc = ENAHOConfigError(
            "Configuration error",
            config_section="database",
            invalid_parameters={"host": "invalid", "port": -1},
        )

        self.assertIsInstance(exc, ENAHOError)
        self.assertEqual(exc.config_section, "database")
        self.assertEqual(exc.invalid_parameters["host"], "invalid")
        self.assertEqual(exc.invalid_parameters["port"], -1)

    def test_data_quality_error_specific_attributes(self):
        """Test DataQualityError specific attributes."""
        metrics = {"completeness": 0.5, "consistency": 0.3}
        exc = DataQualityError(
            "Data quality below threshold", quality_metrics=metrics, quality_threshold=0.8
        )

        self.assertIsInstance(exc, ENAHOError)
        self.assertEqual(exc.quality_metrics, metrics)
        self.assertEqual(exc.quality_threshold, 0.8)


class TestExceptionUtilities(unittest.TestCase):
    """Test cases for exception utility functions."""

    def test_format_enahopy_exception_for_logging(self):
        """Test formatting ENAHOPY exception for logging."""
        exc = ENAHODownloadError(
            "Download failed",
            error_code="DOWNLOAD_FAILED",
            operation="download_module",
            url="https://example.com/data.zip",
            status_code=404,
        )

        formatted = format_exception_for_logging(exc)

        self.assertEqual(formatted["exception_type"], "ENAHODownloadError")
        self.assertEqual(formatted["message"], "Download failed")
        self.assertEqual(formatted["error_code"], "DOWNLOAD_FAILED")
        self.assertEqual(formatted["operation"], "download_module")
        self.assertIn("url", formatted["context"])
        self.assertIn("status_code", formatted["context"])
        self.assertTrue(formatted["is_enahopy_exception"])

    def test_format_standard_exception_for_logging(self):
        """Test formatting standard Python exception for logging."""
        exc = ValueError("Invalid value provided")

        formatted = format_exception_for_logging(exc)

        self.assertEqual(formatted["exception_type"], "ValueError")
        self.assertEqual(formatted["message"], "Invalid value provided")
        self.assertNotIn("error_code", formatted)
        self.assertEqual(formatted["is_enahopy_exception"], False)
        self.assertIn("timestamp", formatted)

    def test_create_error_report_enahopy_exception(self):
        """Test creating error report for ENAHOPY exception."""
        exc = ModuleMergeError(
            "Cannot merge modules",
            error_code="MERGE_FAILED",
            operation="merge_modules",
            modules_involved=["01", "02"],
            merge_level="household",
        )

        context = {"year": 2023, "data_size": 10000}
        report = create_error_report(exc, context)

        self.assertIn("ENAHOPY ERROR REPORT", report)
        self.assertIn("Exception Type: ModuleMergeError", report)
        self.assertIn("Message: Cannot merge modules", report)
        self.assertIn("Error Code: MERGE_FAILED", report)
        self.assertIn("Operation: merge_modules", report)
        self.assertIn("Exception Context:", report)
        self.assertIn("modules_involved", report)
        self.assertIn("Operation Context:", report)
        self.assertIn("year: 2023", report)
        self.assertIn("Recommendations:", report)

    def test_create_error_report_standard_exception(self):
        """Test creating error report for standard exception."""
        exc = FileNotFoundError("File not found: data.csv")
        context = {"file_path": "/path/to/data.csv"}

        report = create_error_report(exc, context)

        self.assertIn("Exception Type: FileNotFoundError", report)
        self.assertIn("Message: File not found: data.csv", report)
        self.assertNotIn("Error Code:", report)
        self.assertIn("Operation Context:", report)
        self.assertIn("file_path: /path/to/data.csv", report)

    def test_get_error_recommendations_download_error(self):
        """Test recommendations for download errors."""
        exc = ENAHODownloadError("Network timeout")
        recommendations = get_error_recommendations(exc)

        self.assertIn("Check your internet connection", recommendations)
        self.assertIn("Verify INEI servers are accessible", recommendations)
        self.assertIn("Try again with a smaller request", recommendations)

    def test_get_error_recommendations_ubigeo_error(self):
        """Test recommendations for UBIGEO validation errors."""
        exc = UbigeoValidationError("Invalid UBIGEO format")
        recommendations = get_error_recommendations(exc)

        self.assertTrue(any("Verify UBIGEO codes are 6 digits" in rec for rec in recommendations))
        self.assertIn("Check department codes are in range 01-25", recommendations)
        self.assertTrue(any("province/district codes are valid" in rec for rec in recommendations))

    def test_get_error_recommendations_incompatible_modules(self):
        """Test recommendations for incompatible modules."""
        exc = IncompatibleModulesError("Modules cannot be merged")
        recommendations = get_error_recommendations(exc)

        self.assertIn("Verify modules are from the same survey year", recommendations)
        self.assertIn("Check if modules have compatible merge keys", recommendations)
        self.assertTrue(
            any("Consider using household-level merge" in rec for rec in recommendations)
        )

    def test_get_error_recommendations_file_reader_error(self):
        """Test recommendations for file reader errors."""
        exc = FileReaderError("Cannot read file")
        recommendations = get_error_recommendations(exc)

        self.assertIn("Verify file exists and is not corrupted", recommendations)
        self.assertTrue(any("Check file format is supported" in rec for rec in recommendations))
        self.assertTrue(any("file is not currently open" in rec for rec in recommendations))

    def test_get_error_recommendations_cache_error(self):
        """Test recommendations for cache errors."""
        exc = ENAHOCacheError("Cache write failed")
        recommendations = get_error_recommendations(exc)

        self.assertIn("Clear cache directory and retry", recommendations)
        self.assertIn("Check disk space availability", recommendations)
        self.assertIn("Verify cache directory permissions", recommendations)

    def test_get_error_recommendations_pattern_detection_error(self):
        """Test recommendations for pattern detection errors."""
        exc = PatternDetectionError("Pattern detection failed")
        recommendations = get_error_recommendations(exc)

        self.assertIn("Check data has sufficient non-null values", recommendations)
        self.assertIn("Verify column data types are appropriate", recommendations)
        self.assertIn("Try with simpler pattern detection settings", recommendations)

    def test_get_error_recommendations_generic(self):
        """Test recommendations for generic errors."""
        exc = RuntimeError("Generic runtime error")
        recommendations = get_error_recommendations(exc)

        self.assertIn("Check input parameters for correctness", recommendations)
        self.assertIn("Verify data integrity and format", recommendations)
        self.assertTrue(any("Review system resources" in rec for rec in recommendations))
        self.assertTrue(any("Consult documentation" in rec for rec in recommendations))


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for ENAHOPY Logging System
====================================

Tests for the centralized logging system and formatters.

Author: ENAHOPY Team
Date: 2024-12-09
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from enahopy.exceptions import ENAHODownloadError
from enahopy.logging import (
    ENAHOFormatter,
    ENAHOLogManager,
    get_logger,
    log_exception,
    log_performance,
    setup_logging,
)


class TestENAHOFormatter(unittest.TestCase):
    """Test cases for ENAHOFormatter."""

    def setUp(self):
        """Set up test environment."""
        self.record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        self.record.module = "test_module"
        self.record.funcName = "test_function"

    def test_human_readable_format(self):
        """Test human-readable formatter."""
        formatter = ENAHOFormatter(structured=False)
        formatted = formatter.format(self.record)

        self.assertIn("Test message", formatted)
        self.assertIn("INFO", formatted)
        self.assertIn("test_logger", formatted)
        # Should contain timestamp
        self.assertRegex(formatted, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_structured_format(self):
        """Test structured JSON formatter."""
        formatter = ENAHOFormatter(structured=True)
        formatted = formatter.format(self.record)

        # Should be valid JSON
        log_data = json.loads(formatted)

        self.assertEqual(log_data["level"], "INFO")
        self.assertEqual(log_data["logger"], "test_logger")
        self.assertEqual(log_data["message"], "Test message")
        self.assertEqual(log_data["module"], "test_module")
        self.assertEqual(log_data["function"], "test_function")
        self.assertEqual(log_data["line"], 42)
        self.assertIn("timestamp", log_data)

    def test_structured_format_with_exception(self):
        """Test structured formatter with exception info."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            self.record.exc_info = sys.exc_info()

        formatter = ENAHOFormatter(structured=True)
        formatted = formatter.format(self.record)

        log_data = json.loads(formatted)

        self.assertIn("exception", log_data)
        self.assertEqual(log_data["exception"]["type"], "ValueError")
        self.assertEqual(log_data["exception"]["message"], "Test exception")
        self.assertIn("traceback", log_data["exception"])

    def test_structured_format_with_extra_fields(self):
        """Test structured formatter with extra fields."""
        # Add extra fields to record
        self.record.custom_field = "custom_value"
        self.record.numeric_field = 123

        formatter = ENAHOFormatter(structured=True)
        formatted = formatter.format(self.record)

        log_data = json.loads(formatted)

        self.assertIn("extra", log_data)
        self.assertEqual(log_data["extra"]["custom_field"], "custom_value")
        self.assertEqual(log_data["extra"]["numeric_field"], 123)


class TestENAHOLogManager(unittest.TestCase):
    """Test cases for ENAHOLogManager singleton."""

    def test_singleton_behavior(self):
        """Test that ENAHOLogManager is a singleton."""
        manager1 = ENAHOLogManager()
        manager2 = ENAHOLogManager()

        self.assertIs(manager1, manager2)

    def test_setup_basic_logging(self):
        """Test basic logging setup."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            manager = ENAHOLogManager()
            logger = manager.setup_logging(verbose=True, structured=False)

            # Should configure logger
            mock_logger.setLevel.assert_called_with(logging.DEBUG)
            self.assertTrue(mock_logger.addHandler.called)

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            manager = ENAHOLogManager()
            logger = manager.setup_logging(verbose=True, structured=True, log_file=str(log_file))

            # Test that we can log to the file
            logger.info("Test message")

            # Verify file was created
            self.assertTrue(log_file.exists())

            # Close all handlers to release file handles on Windows
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def test_get_logger(self):
        """Test getting logger instances."""
        manager = ENAHOLogManager()

        # Test default logger
        logger1 = manager.get_logger()
        self.assertEqual(logger1.name, "enahopy")

        # Test named logger
        logger2 = manager.get_logger("test_module")
        self.assertEqual(logger2.name, "enahopy.test_module")

        # Same name should return same logger
        logger3 = manager.get_logger("test_module")
        self.assertIs(logger2, logger3)

    def test_log_exception_enahopy_exception(self):
        """Test logging ENAHOPY exception."""
        manager = ENAHOLogManager()
        mock_logger = Mock()

        exc = ENAHODownloadError(
            "Download failed", error_code="DOWNLOAD_FAILED", url="https://example.com/data.zip"
        )

        context = {"year": 2023, "module": "01"}

        manager.log_exception(mock_logger, exc, context, "ERROR")

        # Should call logger.error with extra data
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args

        self.assertIn("Exception occurred", call_args[0][0])
        self.assertIn("exception_details", call_args[1]["extra"])
        self.assertIn("operation_context", call_args[1]["extra"])
        self.assertEqual(call_args[1]["extra"]["operation_context"], context)
        self.assertTrue(call_args[1]["exc_info"])

    def test_log_exception_standard_exception(self):
        """Test logging standard Python exception."""
        manager = ENAHOLogManager()
        mock_logger = Mock()

        exc = ValueError("Invalid value")
        context = {"input_value": "bad_value"}

        manager.log_exception(mock_logger, exc, context, "WARNING")

        # Should call logger.warning
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args

        self.assertIn("Exception occurred", call_args[0][0])
        self.assertIn("exception_details", call_args[1]["extra"])
        self.assertEqual(call_args[1]["extra"]["operation_context"], context)


class TestLoggingConvenienceFunctions(unittest.TestCase):
    """Test cases for logging convenience functions."""

    @patch("enahopy.logging._log_manager")
    def test_setup_logging_convenience(self, mock_manager):
        """Test setup_logging convenience function."""
        mock_logger = Mock()
        mock_manager.setup_logging.return_value = mock_logger

        result = setup_logging(
            verbose=True,
            structured=True,
            log_file="/path/to/log.txt",
            log_level="DEBUG",
            enable_performance=True,
        )

        mock_manager.setup_logging.assert_called_once_with(
            verbose=True,
            structured=True,
            log_file="/path/to/log.txt",
            log_level="DEBUG",
            enable_performance=True,
        )
        self.assertEqual(result, mock_logger)

    @patch("enahopy.logging._log_manager")
    def test_get_logger_convenience(self, mock_manager):
        """Test get_logger convenience function."""
        mock_logger = Mock()
        mock_manager.get_logger.return_value = mock_logger

        result = get_logger("test_module")

        mock_manager.get_logger.assert_called_once_with("test_module")
        self.assertEqual(result, mock_logger)

    @patch("enahopy.logging._log_manager")
    @patch("enahopy.logging.get_logger")
    def test_log_exception_convenience(self, mock_get_logger, mock_manager):
        """Test log_exception convenience function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        exc = RuntimeError("Test error")
        context = {"test": "context"}

        log_exception(exc, context, "ERROR", "test_logger")

        mock_get_logger.assert_called_once_with("test_logger")
        mock_manager.log_exception.assert_called_once_with(mock_logger, exc, context, "ERROR")


class TestPerformanceLogging(unittest.TestCase):
    """Test cases for performance logging decorator."""

    def test_log_performance_decorator_success(self):
        """Test performance logging decorator on successful function."""
        with patch("enahopy.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @log_performance(threshold=0.0)  # Always log
            def test_function(x, y):
                return x + y

            result = test_function(1, 2)

            self.assertEqual(result, 3)
            # Should log performance info
            mock_logger.info.assert_called()
            log_call = mock_logger.info.call_args

            self.assertIn("test_function executed", log_call[0][0])
            self.assertIn("execution_time", log_call[1]["extra"])
            self.assertIn("success", log_call[1]["extra"])
            self.assertTrue(log_call[1]["extra"]["success"])

    def test_log_performance_decorator_exception(self):
        """Test performance logging decorator on function that raises exception."""
        with patch("enahopy.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @log_performance(threshold=0.0)  # Always log
            def failing_function():
                raise ValueError("Test error")

            with self.assertRaises(ValueError):
                failing_function()

            # Should log performance info even on failure
            mock_logger.info.assert_called()
            log_call = mock_logger.info.call_args

            self.assertIn("failing_function executed", log_call[0][0])
            self.assertIn("execution_time", log_call[1]["extra"])
            self.assertIn("success", log_call[1]["extra"])
            self.assertFalse(log_call[1]["extra"]["success"])
            self.assertIn("error", log_call[1]["extra"])

    def test_log_performance_with_threshold(self):
        """Test performance logging with time threshold."""
        with patch("enahopy.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # High threshold - should not log
            @log_performance(threshold=10.0)
            def fast_function():
                return "done"

            fast_function()

            # Should not log because execution time < threshold
            mock_logger.info.assert_not_called()

    def test_log_performance_with_custom_logger(self):
        """Test performance logging with custom logger name."""
        with patch("enahopy.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            @log_performance(logger_name="custom.logger", threshold=0.0)  # Always log
            def test_function():
                return "done"

            test_function()

            mock_get_logger.assert_called_with("custom.logger")

    def test_log_performance_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""

        @log_performance
        def documented_function(arg1, arg2="default"):
            """This is a test function."""
            return arg1

        self.assertEqual(documented_function.__name__, "documented_function")
        self.assertEqual(documented_function.__doc__, "This is a test function.")

        # Test function still works normally
        result = documented_function("test")
        self.assertEqual(result, "test")


if __name__ == "__main__":
    unittest.main()

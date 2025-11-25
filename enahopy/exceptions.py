"""
ENAHOPY - Unified Exception Hierarchy
====================================

Centralized exception system for all ENAHOPY operations.
All exceptions inherit from ENAHOError and include context, error codes,
and timestamps for debugging and logging.

Author: ENAHOPY Team
Date: 2024-12-09
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class ENAHOError(Exception):
    """
    Base exception for all ENAHOPY operations.

    All ENAHOPY exceptions should inherit from this class to ensure
    consistent error handling and logging across the library.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        operation: Optional[str] = None,
        **context,
    ):
        """
        Initialize the exception with context information.

        Args:
            message: Human-readable error description
            error_code: Machine-readable error code (e.g., "DOWNLOAD_FAILED")
            operation: Operation that failed (e.g., "download_module")
            **context: Additional context information for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.operation = operation
        self.context = context
        self.timestamp = datetime.now().isoformat()

    def __str__(self) -> str:
        """Return formatted error message with context."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"[{self.error_code}]")

        if self.operation:
            parts.append(f"during '{self.operation}'")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "operation": self.operation,
            "context": self.context,
            "timestamp": self.timestamp,
            "is_enahopy_exception": True,
        }


# =====================================================
# LOADER SYSTEM EXCEPTIONS
# =====================================================


class ENAHODownloadError(ENAHOError):
    """Error during data download operations."""

    def __init__(
        self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.url = url
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        result = super().to_dict()
        # Add download-specific context
        if self.url is not None:
            result["context"]["url"] = self.url
        if self.status_code is not None:
            result["context"]["status_code"] = self.status_code
        return result


class ENAHOValidationError(ENAHOError):
    """Error during parameter or data validation."""

    def __init__(self, message: str, validation_failures: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_failures = validation_failures or []


class ENAHOIntegrityError(ENAHOError):
    """Error related to data integrity checks."""

    def __init__(
        self,
        message: str,
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum


class ENAHOTimeoutError(ENAHOError):
    """Error due to operation timeout."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds


class ENAHOFileError(ENAHOError):
    """Base class for file-related errors."""

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.file_path = file_path


class FileReaderError(ENAHOFileError):
    """Error during file reading operations."""


class UnsupportedFormatError(FileReaderError):
    """Error for unsupported file formats."""

    def __init__(
        self,
        message: str,
        format_attempted: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.format_attempted = format_attempted
        self.supported_formats = supported_formats or []


class ENAHOCacheError(ENAHOError):
    """Error during cache operations."""

    def __init__(
        self,
        message: str,
        cache_operation: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.cache_operation = cache_operation
        self.cache_key = cache_key


# =====================================================
# MERGER SYSTEM EXCEPTIONS
# =====================================================


class ENAHOMergeError(ENAHOError):
    """Base class for all merge-related errors."""


class GeoMergeError(ENAHOMergeError):
    """Error during geographic data merging."""

    def __init__(
        self,
        message: str,
        merge_column: Optional[str] = None,
        affected_rows: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.merge_column = merge_column
        self.affected_rows = affected_rows


class UbigeoValidationError(ENAHOValidationError):
    """Error during UBIGEO code validation."""

    def __init__(
        self,
        message: str,
        invalid_ubigeos: Optional[List[str]] = None,
        validation_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.invalid_ubigeos = invalid_ubigeos or []
        self.validation_type = validation_type


class TerritorialInconsistencyError(GeoMergeError):
    """Error due to territorial data inconsistencies."""

    def __init__(
        self, message: str, inconsistencies: Optional[List[Dict[str, Any]]] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.inconsistencies = inconsistencies or []


class ModuleMergeError(ENAHOMergeError):
    """Error during module merging operations."""

    def __init__(
        self,
        message: str,
        modules_involved: Optional[List[str]] = None,
        merge_level: Optional[str] = None,
        **kwargs,
    ):
        # Add module-specific data to context
        if modules_involved is not None:
            kwargs["modules_involved"] = modules_involved
        if merge_level is not None:
            kwargs["merge_level"] = merge_level

        super().__init__(message, **kwargs)
        self.modules_involved = modules_involved or []
        self.merge_level = merge_level


class IncompatibleModulesError(ModuleMergeError):
    """Error when modules are incompatible for merging."""

    def __init__(
        self,
        message: str,
        module1: Optional[str] = None,
        module2: Optional[str] = None,
        compatibility_issues: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.module1 = module1
        self.module2 = module2
        self.compatibility_issues = compatibility_issues or []


class MergeKeyError(ModuleMergeError, KeyError):
    """Error related to merge keys.

    Inherits from both ModuleMergeError (for ENAHO context) and KeyError
    (for compatibility with standard Python exception handling).
    """

    def __init__(
        self,
        message: str,
        missing_keys: Optional[List[str]] = None,
        invalid_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Initialize ModuleMergeError with our custom logic
        ModuleMergeError.__init__(self, message, **kwargs)
        # Initialize KeyError with the message (for compatibility)
        KeyError.__init__(self, message)
        self.missing_keys = missing_keys or []
        self.invalid_keys = invalid_keys or []


class DuplicateHandlingError(GeoMergeError):
    """Error during handling of duplicate geographic entries."""

    def __init__(
        self,
        message: str,
        duplicates_info: Optional[Dict[str, Any]] = None,
        duplicate_count: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.duplicates_info = duplicates_info or {}
        self.duplicate_count = duplicate_count


# =====================================================
# NULL ANALYSIS SYSTEM EXCEPTIONS
# =====================================================


class ENAHONullAnalysisError(ENAHOError):
    """Base class for null analysis errors."""


class PatternDetectionError(ENAHONullAnalysisError):
    """Error during pattern detection in null analysis."""

    def __init__(
        self,
        message: str,
        pattern_type: Optional[str] = None,
        affected_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.pattern_type = pattern_type
        self.affected_columns = affected_columns or []


class VisualizationError(ENAHONullAnalysisError):
    """Error during visualization generation."""

    def __init__(self, message: str, visualization_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.visualization_type = visualization_type


class ImputationError(ENAHONullAnalysisError):
    """Error during data imputation operations."""

    def __init__(
        self,
        message: str,
        imputation_strategy: Optional[str] = None,
        affected_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.imputation_strategy = imputation_strategy
        self.affected_columns = affected_columns or []


# =====================================================
# CONFIGURATION AND SYSTEM EXCEPTIONS
# =====================================================


class ENAHOConfigError(ENAHOError):
    """Error in system configuration."""

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        invalid_parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.invalid_parameters = invalid_parameters or {}


class DataQualityError(ENAHOError):
    """Error related to data quality issues."""

    def __init__(
        self,
        message: str,
        quality_metrics: Optional[Dict[str, float]] = None,
        quality_threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.quality_metrics = quality_metrics or {}
        self.quality_threshold = quality_threshold


# =====================================================
# UTILITY FUNCTIONS
# =====================================================


def format_exception_for_logging(exception: Exception) -> Dict[str, Any]:
    """
    Format any exception for structured logging.

    Args:
        exception: Exception to format

    Returns:
        Dictionary with structured exception information
    """
    if isinstance(exception, ENAHOError):
        return exception.to_dict()

    # Handle non-ENAHOPY exceptions
    return {
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "module": getattr(exception, "__module__", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "is_enahopy_exception": False,
    }


def create_error_report(
    exception: Exception, operation_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a comprehensive error report for debugging.

    Args:
        exception: Exception that occurred
        operation_context: Additional context about the operation

    Returns:
        Formatted error report as string
    """
    lines = [
        "=" * 60,
        "ENAHOPY ERROR REPORT",
        "=" * 60,
        f"Exception Type: {type(exception).__name__}",
        f"Message: {str(exception)}",
        f"Timestamp: {datetime.now().isoformat()}",
    ]

    # Add ENAHOPY-specific information
    if isinstance(exception, ENAHOError):
        if exception.error_code:
            lines.append(f"Error Code: {exception.error_code}")
        if exception.operation:
            lines.append(f"Operation: {exception.operation}")
        if exception.context:
            lines.append("\nException Context:")
            for key, value in exception.context.items():
                lines.append(f"  {key}: {value}")

    # Add operation context
    if operation_context:
        lines.append("\nOperation Context:")
        for key, value in operation_context.items():
            lines.append(f"  {key}: {value}")

    # Add recommendations
    recommendations = get_error_recommendations(exception)
    if recommendations:
        lines.append("\nRecommendations:")
        for rec in recommendations:
            lines.append(f"  • {rec}")

    lines.append("=" * 60)
    return "\n".join(lines)


def get_error_recommendations(exception: Exception) -> List[str]:
    """
    Generate actionable recommendations based on exception type.

    Args:
        exception: Exception to analyze

    Returns:
        List of recommendation strings
    """
    if isinstance(exception, ENAHODownloadError):
        return [
            "Check your internet connection",
            "Verify INEI servers are accessible",
            "Try again with a smaller request",
            "Check if the requested year/module combination exists",
        ]

    elif isinstance(exception, UbigeoValidationError):
        return [
            "Verify UBIGEO codes are 6 digits (DDPPDD format)",
            "Check department codes are in range 01-25",
            "Ensure province/district codes are valid for the department",
            "Review data source for UBIGEO formatting issues",
        ]

    elif isinstance(exception, IncompatibleModulesError):
        return [
            "Verify modules are from the same survey year",
            "Check if modules have compatible merge keys",
            "Consider using household-level merge instead of individual-level",
            "Review module documentation for merge requirements",
        ]

    elif isinstance(exception, FileReaderError):
        return [
            "Verify file exists and is not corrupted",
            "Check file format is supported (DTA, SAV, CSV, Parquet)",
            "Ensure file is not currently open in another application",
            "Try reading with different encoding settings",
        ]

    elif isinstance(exception, ENAHOCacheError):
        return [
            "Clear cache directory and retry",
            "Check disk space availability",
            "Verify cache directory permissions",
            "Consider disabling cache if issues persist",
        ]

    elif isinstance(exception, PatternDetectionError):
        return [
            "Check data has sufficient non-null values",
            "Verify column data types are appropriate",
            "Try with simpler pattern detection settings",
            "Consider data preprocessing before analysis",
        ]

    # Generic recommendations
    return [
        "Check input parameters for correctness",
        "Verify data integrity and format",
        "Review system resources (memory, disk space)",
        "Consult documentation for proper usage",
        "Enable debug logging for more details",
    ]


# Export all exception classes
__all__ = [
    # Base exceptions
    "ENAHOError",
    # Loader exceptions
    "ENAHODownloadError",
    "ENAHOValidationError",
    "ENAHOIntegrityError",
    "ENAHOTimeoutError",
    "ENAHOFileError",
    "FileReaderError",
    "UnsupportedFormatError",
    "ENAHOCacheError",
    # Merger exceptions
    "ENAHOMergeError",
    "GeoMergeError",
    "UbigeoValidationError",
    "TerritorialInconsistencyError",
    "ModuleMergeError",
    "IncompatibleModulesError",
    "MergeKeyError",
    "DuplicateHandlingError",
    # Null analysis exceptions
    "ENAHONullAnalysisError",
    "PatternDetectionError",
    "VisualizationError",
    "ImputationError",
    # System exceptions
    "ENAHOConfigError",
    "DataQualityError",
    # Utility functions
    "format_exception_for_logging",
    "create_error_report",
    "get_error_recommendations",
]

"""
ENAHO Loader Logging Module (Legacy Compatibility)
=================================================

Compatibility wrapper for the old loader logging system.
Now delegates to the centralized ENAHOPY logging system.

This module is deprecated and will be removed in v2.0.
Use enahopy.logging directly instead.
"""

import logging
import warnings
from typing import Optional

# Import from centralized logging system
from ...logging import log_performance as _log_performance
from ...logging import setup_logging as _setup_logging

# Deprecation warning for this module
warnings.warn(
    "enahopy.loader.core.logging is deprecated. Use enahopy.logging instead.",
    DeprecationWarning,
    stacklevel=2,
)


def setup_logging(
    verbose: bool = True, structured: bool = False, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging system (legacy compatibility).

    DEPRECATED: Use enahopy.logging.setup_logging() instead.

    Args:
        verbose: Enable verbose logging
        structured: Use structured (JSON) logging
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    warnings.warn(
        "setup_logging from enahopy.loader.core.logging is deprecated. "
        "Use enahopy.logging.setup_logging() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Delegate to centralized system but maintain backward compatibility
    logger = _setup_logging(
        verbose=verbose,
        structured=structured,
        log_file=log_file,
        log_level="INFO" if verbose else "WARNING",
    )

    # Return a logger with the old name for backward compatibility
    old_logger = logging.getLogger("enaho_downloader")

    # Copy handlers from new logger to old logger name
    if not old_logger.handlers:
        for handler in logger.handlers:
            old_logger.addHandler(handler)
        old_logger.setLevel(logger.level)
        old_logger.propagate = False

    return old_logger


def log_performance(func):
    """
    Performance logging decorator (legacy compatibility).

    DEPRECATED: Use enahopy.logging.log_performance() instead.
    """
    warnings.warn(
        "log_performance from enahopy.loader.core.logging is deprecated. "
        "Use enahopy.logging.log_performance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Delegate to centralized system
    return _log_performance(func)


# Legacy formatter for backward compatibility
class StructuredFormatter:
    """
    Legacy structured formatter (deprecated).

    DEPRECATED: Use enahopy.logging.ENAHOFormatter instead.
    """

    def __init__(self):
        warnings.warn(
            "StructuredFormatter is deprecated. Use enahopy.logging.ENAHOFormatter instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def format(self, record):
        """Legacy format method."""
        from ...logging import ENAHOFormatter

        formatter = ENAHOFormatter(structured=True)
        return formatter.format(record)


__all__ = ["StructuredFormatter", "setup_logging", "log_performance"]

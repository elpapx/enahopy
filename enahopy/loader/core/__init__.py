"""
ENAHO Core Package
=================

Componentes fundamentales de enaho-analyzer:
configuración, excepciones, logging y cache.
"""

import warnings

from .cache import CacheManager
from .config import ENAHOConfig
from .exceptions import (
    ENAHODownloadError,
    ENAHOError,
    ENAHOIntegrityError,
    ENAHOTimeoutError,
    ENAHOValidationError,
    FileReaderError,
    UnsupportedFormatError,
)

# Suppress deprecation warning for internal re-export
# Users should use enahopy.logging directly, but we maintain backward compatibility
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from .logging import StructuredFormatter, log_performance, setup_logging

__all__ = [
    # Config
    "ENAHOConfig",
    # Exceptions
    "ENAHOError",
    "ENAHODownloadError",
    "ENAHOValidationError",
    "ENAHOIntegrityError",
    "ENAHOTimeoutError",
    "FileReaderError",
    "UnsupportedFormatError",
    # Logging
    "StructuredFormatter",
    "setup_logging",
    "log_performance",
    # Cache
    "CacheManager",
]

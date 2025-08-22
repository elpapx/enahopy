"""
ENAHO Core Package
=================

Componentes fundamentales de enaho-analyzer:
configuración, excepciones, logging y cache.
"""

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

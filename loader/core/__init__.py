"""
ENAHO Core Package
=================

Componentes fundamentales de enaho-analyzer:
configuración, excepciones, logging y cache.
"""

from .config import ENAHOConfig
from .exceptions import (
    ENAHOError,
    ENAHODownloadError,
    ENAHOValidationError,
    ENAHOIntegrityError,
    ENAHOTimeoutError,
    FileReaderError,
    UnsupportedFormatError
)
from .logging import (
    StructuredFormatter,
    setup_logging,
    log_performance
)
from .cache import CacheManager

__all__ = [
    # Config
    'ENAHOConfig',

    # Exceptions
    'ENAHOError',
    'ENAHODownloadError',
    'ENAHOValidationError',
    'ENAHOIntegrityError',
    'ENAHOTimeoutError',
    'FileReaderError',
    'UnsupportedFormatError',

    # Logging
    'StructuredFormatter',
    'setup_logging',
    'log_performance',

    # Cache
    'CacheManager'
]
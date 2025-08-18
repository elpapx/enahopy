"""
ENAHO I/O Package
================

Sistema completo de entrada/salida para enaho-analyzer.
Incluye readers, downloaders, validators y la clase principal.
"""

from .base import IReader, DASK_AVAILABLE
from .readers import (
    BaseReader,
    SPSSReader,
    StataReader,
    ParquetReader,
    CSVReader,
    ReaderFactory,
    PYREADSTAT_AVAILABLE
)
from .validators import (
    ColumnValidationResult,
    ColumnValidator,
    ENAHOValidator
)
from .downloaders import (
    NetworkUtils,
    ENAHOExtractor
)
from .local_reader import ENAHOLocalReader
from .main import ENAHODataDownloader

__all__ = [
    # Base interfaces
    'IReader',
    'DASK_AVAILABLE',

    # Readers
    'BaseReader',
    'SPSSReader',
    'StataReader',
    'ParquetReader',
    'CSVReader',
    'ReaderFactory',
    'PYREADSTAT_AVAILABLE',

    # Validators
    'ColumnValidationResult',
    'ColumnValidator',
    'ENAHOValidator',

    # Downloaders
    'NetworkUtils',
    'ENAHOExtractor',

    # Main classes
    'ENAHOLocalReader',
    'ENAHODataDownloader'
]
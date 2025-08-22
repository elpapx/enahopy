"""
ENAHO I/O Package
================

Sistema completo de entrada/salida para enaho-analyzer.
Incluye readers, downloaders, validators y la clase principal.
"""

from .base import DASK_AVAILABLE, IReader
from .downloaders import ENAHOExtractor, NetworkUtils
from .local_reader import ENAHOLocalReader
from .main import ENAHODataDownloader
from .readers import (
    PYREADSTAT_AVAILABLE,
    BaseReader,
    CSVReader,
    ParquetReader,
    ReaderFactory,
    SPSSReader,
    StataReader,
)
from .validators import ColumnValidationResult, ColumnValidator, ENAHOValidator

__all__ = [
    # Base interfaces
    "IReader",
    "DASK_AVAILABLE",
    # Readers
    "BaseReader",
    "SPSSReader",
    "StataReader",
    "ParquetReader",
    "CSVReader",
    "ReaderFactory",
    "PYREADSTAT_AVAILABLE",
    # Validators
    "ColumnValidationResult",
    "ColumnValidator",
    "ENAHOValidator",
    # Downloaders
    "NetworkUtils",
    "ENAHOExtractor",
    # Main classes
    "ENAHOLocalReader",
    "ENAHODataDownloader",
]

"""
ENAHO Readers Package
====================

Lectores especializados para diferentes formatos de archivo.
"""

from .base import BaseReader
from .csv import CSVReader
from .factory import ReaderFactory
from .parquet import ParquetReader
from .spss import SPSSReader
from .stata import StataReader

# Check opcional dependencies
try:
    import pyreadstat

    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False

__all__ = [
    "BaseReader",
    "SPSSReader",
    "StataReader",
    "ParquetReader",
    "CSVReader",
    "ReaderFactory",
    "PYREADSTAT_AVAILABLE",
]

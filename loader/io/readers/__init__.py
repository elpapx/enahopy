"""
ENAHO Readers Package
====================

Readers especializados para diferentes formatos de archivo.
Cada reader implementa la interfaz IReader con optimizaciones
específicas para su formato.
"""

from .base import BaseReader
from .spss import SPSSReader, PYREADSTAT_AVAILABLE
from .stata import StataReader
from .parquet import ParquetReader
from .csv import CSVReader
from .factory import ReaderFactory

__all__ = [
    'BaseReader',
    'SPSSReader',
    'StataReader',
    'ParquetReader',
    'CSVReader',
    'ReaderFactory',
    'PYREADSTAT_AVAILABLE'
]
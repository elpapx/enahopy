"""
ENAHO I/O Base Module
====================

Interfaces y clases base para el sistema de I/O.
Define contratos para lectores, validadores y otros componentes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Union

import pandas as pd

# Imports opcionales
try:
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    if TYPE_CHECKING:
        import dask.dataframe as dd


class IReader(ABC):
    """Interfaz para todos los lectores de archivos. Define los métodos obligatorios."""

    def __init__(self, file_path: Path, logger: logging.Logger):
        self.file_path = file_path
        self.logger = logger

    @abstractmethod
    def read_columns(self, columns: List[str]) -> pd.DataFrame:
        """Lee columnas específicas del archivo y devuelve un DataFrame de Pandas."""
        pass

    @abstractmethod
    def read_in_chunks(
        self, columns: List[str], chunk_size: int
    ) -> Union[dd.DataFrame, Iterator[pd.DataFrame]]:
        """Lee columnas específicas en chunks y devuelve un DataFrame de Dask o iterador."""
        pass

    @abstractmethod
    def get_available_columns(self) -> List[str]:
        """Obtiene la lista de todas las columnas disponibles en el archivo."""
        pass

    @abstractmethod
    def extract_metadata(self) -> Dict:
        """Extrae los metadatos completos del archivo."""
        pass


__all__ = ["IReader", "DASK_AVAILABLE"]
